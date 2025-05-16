
import numpy as np
import struct
import zarr
import subprocess
import tempfile
from pathlib import Path
import logging
import json
from .types import AudioFileBaseFeatures

OGG_PAGE_HEADER_SIZE = 27
OGG_MAX_PAGE_SIZE = 65536

# get the module logger   
logger = logging.getLogger(__name__)

def convert_audio_to_ogg(audio_file: str|Path, 
                         target_codec:str = 'flac', # "flac" or "opus"
                         flac_compression_level: int = 4, # 0...12; 4 is a really good value: fast and low
                                                          # data; higher values does not really reduce 
                                                          # data size but need more time and energy. 
                                                          # Note:
                                                          # The highest values can produce more data
                                                          # than lower values. '12' as maximum must not
                                                          # be the best compression. Check ist: More than
                                                          # 4 is not really less memory consumption but
                                                          # wasted time and energy.
                         opus_bitrate:str = '160k',
                         temp_dir="/tmp") -> tuple[Path, float, str]:
    """
    Konvertiert eine Audiodatei in einen Ogg-Container mit FLAC oder Opus.
    - Unterstützt Ultraschallmodus: PCM-Daten bleiben, Zeitbasis wird manipuliert
    - Rückgabe: Pfad zur Ogg-Datei, Faktor Original-Samplingrate / gespeicherte samplingrate
    """

    def get_source_params(input_file: Path) -> tuple[int, bool]:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate:stream=codec_name",
            "-of", "json", str(input_file)
        ]
        out = subprocess.check_output(args=cmd)
        info = json.loads(out)
        sampling_rate = int(info['streams'][0]['sample_rate'])
        is_opus = info['streams'][0]['codec_name'] == "opus"
        return sampling_rate, is_opus

    assert target_codec in ('flac', 'opus')
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12)

    audio_file = Path(audio_file)

    original_rate, is_opus = get_source_params(audio_file)
    
    Im Weiteren müssen die Parameter des im File-Blob gespeicherten Inhalts auch abgelegt werden:
    Diese werden beim dekodieren benötigt!

    # in order to avoid downsampling, we rescale the time-base and sign this
    # as ultrasonic
    sampling_rescale = 1.0
    is_ultrasonic = (target_codec == "opus") and (original_rate > 48000)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=temp_dir) as tmp_out:
        tmp_file = Path(tmp_out.name)

    if target_codec == 'opus' and is_opus and not is_ultrasonic:
        # copy opus encoded data directly into ogg.opus 
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_file),
            "-c", "copy", "-f", "ogg", str(tmp_file)
        ], check=True)
        return tmp_file, sampling_rescale, target_codec

    ffmpeg_cmd = ["ffmpeg", "-y"]   

    if target_codec == 'opus':
        if is_ultrasonic:
            # we interprete sampling rate as "48000" to can use opus for ultrasonic
            # (for use late, it must be re-intrepreted)
            ffmpeg_cmd += ["-sample_rate", "48000"]
            sampling_rescale = float(original_rate) / 48000.0
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "libopus", "-b:a", opus_bitrate]
        ffmpeg_cmd += ["-vbr", "off"] # constant bitrate is a bit better in quality than VRB=On
        ffmpeg_cmd += ["-apply_phase_inv", "false"] # Phasenrichtige Kodierung: keine Tricks!

    elif target_codec == 'flac':
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
        ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]

    ffmpeg_cmd += ["-f", "ogg", str(tmp_file)]

    subprocess.run(ffmpeg_cmd, check=True)

    return tmp_file, sampling_rescale, target_codec

def decode_ogg_bytes_to_pcm(ogg_bytes: bytes, sampling_rate: int, channels: int = 1, dtype=np.int16):
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-f", "ogg",
        "-i", "pipe:0",
        "-f", "s16le" if dtype == np.int16 else "f32le",
        "-acodec", "pcm_s16le" if dtype == np.int16 else "pcm_f32le",
        "-ac", str(channels),
        "-ar", str(sampling_rate),
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pcm_bytes, _ = proc.communicate(ogg_bytes)

    pcm_array = np.frombuffer(pcm_bytes, dtype=dtype)
    return pcm_array

def parse_ogg_pages_from_array(data: np.ndarray):
    """
    Parsen von Ogg Pages direkt aus einem np.uint8 Array (z.B. Zarr-Blob).
    Gibt Liste von (offset, granule_pos) zurück.
    """
    offset = 0
    data_len = data.shape[0]
    entries = []

    while offset + OGG_PAGE_HEADER_SIZE < data_len:
        # Prüfe auf OggS Sync
        if not np.array_equal(data[offset:offset+4], np.frombuffer(b'OggS', dtype=np.uint8)):
            offset += 1
            continue

        header = data[offset : offset + OGG_PAGE_HEADER_SIZE]
        granule_pos = struct.unpack_from('<Q', header.tobytes(), 6)[0]
        segment_count = header[26]

        # Segment-Tabelle lesen
        seg_table_start = offset + OGG_PAGE_HEADER_SIZE
        seg_table_end = seg_table_start + segment_count
        if seg_table_end > data_len:
            break

        segment_table = data[seg_table_start:seg_table_end]
        page_body_size = int(np.sum(segment_table))

        page_size = OGG_PAGE_HEADER_SIZE + segment_count + page_body_size
        if offset + page_size > data_len:
            break

        entries.append((offset, granule_pos))
        offset += page_size

    return np.array(entries, dtype=np.uint64)  # shape (N, 2)

def create_index_zarr(ogg_file_blob_array: np.ndarray, zarr_original_audio_group: zarr.Group): # zarr_original_audio_group must be open with mode = 'r+'
    """
    Parst Ogg-Pages aus ogg_file_blob_array und speichert Index als 'index' in Zarr-Gruppe.
    """
    index_array = parse_ogg_pages_from_array(ogg_file_blob_array)
    # shape (N, 2), dtype uint64 — Spalten: [file_offset, granule_position]
    with zarr_original_audio_group.create_array(    name="index",
                                                    dtype="uint64",
                                                    shape=index_array.shape,
                                                    chunks=(min(1024, index_array.shape[0]), 2),
                                                    overwrite=True
                                                ) as index:
        index[:,:] = index_array
    print(f"Index mit {index_array.shape[0]} Einträgen gespeichert.")

def find_sample_range_in_index(start_sample: int, end_sample: int, index: np.ndarray):
    """
    Gibt den Start-Offset, End-Offset (im ogg_blob) und relative Sample-Range im dekodierten Signal zurück.
    """
    granule_positions = index[:, 1]
    file_offsets = index[:, 0]

    # Suche Position der letzten Page VOR start_sample
    start_pos = np.searchsorted(granule_positions, start_sample, side='right') - 1
    end_pos = np.searchsorted(granule_positions, end_sample, side='right') - 1

    if start_pos < 0 or end_pos < 0:
        raise ValueError("Samplebereich liegt vor Beginn des Index")

    # Ogg-Bytebereiche
    pages_start_position = int(file_offsets[start_pos])
    pages_end_position = int(file_offsets[end_pos])

    # Absoluter Start in Samples
    page_start_sample = int(granule_positions[start_pos])

    # Relative Sample-Offsets im dekodierten Array
    relative_start = start_sample - page_start_sample
    relative_end = end_sample - page_start_sample

    return pages_start_position, pages_end_position, relative_start, relative_end

def get_pcm_array(zarr_original_audio_group: zarr.Group, start_sample: int, end_sample: int):
    
        index = zarr_original_audio_group["index"] # -> richtig öffnen, damit es als numpy-Array interpretiert werden kann
        pages_start_position, pages_end_position, relative_start, relative_end = find_sample_range_in_index(start_sample, end_sample, index)
        
        ogg_file_blob_array = zarr_original_audio_group["ogg_file_blob"] # -> richtig öffnen, damit es als numpy-Array interpretiert werden kann
        ogg_segment = ogg_file_blob_array[pages_start_position : pages_end_position + OGG_MAX_PAGE_SIZE] # OGG_MAX_PAGE_SIZE ist sicherheitsabstand, da wir die ganze Page brauchen
        
        # Wir gehen davon aus, dass genau ein Stream (Index 0 in AudioFileBaseFeatures._..._PER STREAM) mit 1 oder 2 Kanälen (oder mehr) vorhanden ist.
        Das muss geändert werden: Nicht die original-File-Features, sondern die, mit denen der file blob erstellt wurde!
        base_features: AudioFileBaseFeatures = zarr_original_audio_group.attrs["base_features"]
        channels: int = base_features.NB_STREAMS # unklar of steareo 2 Streams sind oder 2 Channels -> noch klären!
        sampling_rate: int = base_features.SAMPLING_RATE_PER_STREAM[0]
        dtype = base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE
        if base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR:
            raise NotImplementedError("Decoding of planar sample formats is not yet implemented.")
        
        pcm = decode_ogg_bytes_to_pcm(ogg_segment.tobytes(), sampling_rate=sampling_rate, channels=channels, dtype=dtype)[relative_start * channels : relative_end * channels]
        return pcm