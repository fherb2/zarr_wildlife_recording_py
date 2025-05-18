
import numpy as np
import struct
import zarr
import subprocess
import tempfile
from pathlib import Path
import logging
import json
from .utils import file_size
from .config import Config
from .exceptions import OggImportError

OGG_PAGE_HEADER_SIZE = 27
OGG_MAX_PAGE_SIZE = 65536

# get the module logger   
logger = logging.getLogger(__name__)

def import_audio_to_blob(zarr_original_audio_group: zarr.Group, file_to_import: str | Path):
    _create_index_zarr(_import_audio_to_ogg_blob(zarr_original_audio_group,
                                                 file_to_import),
                       zarr_original_audio_group)
    
def _get_ffmpeg_sample_fmt(source_sample_fmt: str, target_codec: str) -> str:
    """
    Gibt das beste sample_fmt-Argument für ffmpeg zurück, basierend auf Quellformat und Zielcodec.
    
    Args:
        source_sample_fmt (str): Sample-Format der Quelle laut ffprobe, z.B. "fltp", "s16", "s32", "flt"
        target_codec (str): "flac" oder "opus"
    
    Returns:
        str: Der passende Wert für "-sample_fmt" bei ffmpeg (z.B. "s32", "flt", "s16")
    """

    # Mappings nach Fähigkeiten der Codecs
    flac_supported = ["s16", "s32", "flt"]
    opus_supported = ["s16", "s24", "flt"]

    # Auflösen von planar zu packed
    normalized_fmt = source_sample_fmt.rstrip("p") if source_sample_fmt.endswith("p") else source_sample_fmt

    if target_codec == "flac":
        if normalized_fmt in flac_supported:
            return normalized_fmt
        # fallback
        return "s32" if normalized_fmt.startswith("s") else "flt"

    elif target_codec == "opus":
        # Opus arbeitet intern mit 16-bit, akzeptiert aber auch float input
        if normalized_fmt in opus_supported:
            return normalized_fmt
        # fallback auf float oder s16
        return "flt" if normalized_fmt.startswith("f") else "s16"

    else:
        raise NotImplementedError(f"Unsupported codec: {target_codec}")

def _import_audio_to_ogg_blob( 
                         original_audio_grp: zarr.Group,
                         audio_file: str|Path, 
                         audio_file_blob_array_name: str = "ogg_file_blob",
                         target_codec:str = 'flac', # "flac" or "opus"
                         flac_compression_level: int = 4, # 0...12; 4 is a really good value: fast and low
                                                          # data; higher values does not really reduce 
                                                          # data size but need more time and energy. 
                                                          # Note:
                                                          # The highest values can produce more data
                                                          # than lower values. '12' as maximum must not
                                                          # be the best compression. Check it: More than
                                                          # 4 is not really less memory consumption but
                                                          # wasted time and energy.
                         opus_bitrate:str = '160k', # formatted as used in ffmpeg
                         temp_dir="/tmp",
                         chunks:int = Config.original_audio_chunk_size,
                         chunks_per_shard:int = Config.original_audio_chunks_per_shard
                         ) -> zarr.Array:
    """
    Konvertiert eine Audiodatei in einen Ogg-Container mit FLAC oder Opus.
    Schreibt auch alle notwendigen Attribute, jedoch erzeugt es keinen Index.
    - Unterstützt Ultraschallmodus: PCM-Daten bleiben, Zeitbasis wird manipuliert
    - Rückgabe: Dieses Blob-Array (kann dann gleich zum indizieren genutzt werden).

    Bisher wird nur ein Audio-Stream unterstützt. Jedoch mit beliebig vielen Kanälen.
    """

    # TODO: Add OggImportError-Exceptions in case of errors. This can be used to remove 
    # data of a started importing!

    assert target_codec in ('flac', 'opus') 
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12)

    audio_file = Path(audio_file)

    source_params = _get_source_params(audio_file)
    target_sample_format = _get_ffmpeg_sample_fmt(source_params["sample_format"], 
                                                  'opus' if source_params["is_opus"] else 'flac')

    # In case, the target codec is 'opus', a lossy compress format:
    # 
    #   In contrast to 'flac', 'opus' codec can only use sampling frequencies of 48kS/s (or lower)! 
    #   If the source has a higher sampling frequency, we would loose information. This is important
    #   insofar as the source could be specialized for ultrasonic frequencies! In order to can compress
    #   such high frequencies, we use a trick: For the compressing algorithm, we say 'source is 48kS/s'.
    #   but we remember the factor of the real sampling frequency to this 48kS/s. If we uncompress
    #   the data later for processing or export, we reinterprete the sampling fequency to the original 
    #   value.
    #   There's only one drawback: If these audio has also very deep frequencies, so the opus comprression
    #   algorithm can interprete these as inaudible and remove all of them. To avoid this, use 'flac'
    #   or downsample the audio source to 48kS/s before.
    #   
    

    with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=temp_dir) as tmp_out:
        tmp_file = Path(tmp_out.name)

        sampling_rescale_factor = _import_to_tempfile(  audio_file=audio_file,
                                                        tmp_file=tmp_file,
                                                        target_codec=target_codec,
                                                        target_sample_format=target_sample_format,
                                                        flac_compression_level=flac_compression_level,
                                                        opus_bitrate=opus_bitrate
                                                        )
        
        # Create the file blob array
        tmp_file_byte_size = file_size(tmp_file)
        ogg_file_blob_array = original_audio_grp.create_array(
                name            = audio_file_blob_array_name,
                shape           = tmp_file_byte_size,
                chunks          = (chunks,),
                shards          = (chunks_per_shard * chunks,),
                dtype           = np.uint8,
                overwrite       = True,
            )

        # copy tmp-file data into the array and remove the temp file
        with open(tmp_file, "rb") as f:
            for offset in range(0, tmp_file_byte_size, chunks):
                buffer = f.read(chunks)
                ogg_file_blob_array[offset : offset + len(buffer)] = np.frombuffer(buffer, dtype="u1")
        tmp_file.unlink()

    # add encoding attributes to this array
    attrs = {
            "container_type": "ogg",
            "codec": target_codec,
            "nb_channels": source_params["nb_channels"],
            }
    if target_codec == "opus":
        attrs.update({
            "opus_bitrate": opus_bitrate,
            "sampling_rescale_factor": sampling_rescale_factor,
            })
    elif target_codec == "flac":
            attrs["compression_level"] = flac_compression_level
    else:
        raise RuntimeError("Unbekannter Codec-Zweig erreicht – dieser Fall ist ein Programmierfehler.")
    ogg_file_blob_array.attrs.update(attrs)

    return ogg_file_blob_array, attrs

def _get_source_params(input_file: Path) -> tuple[int, str, bool]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate:stream=codec_name:stream:sample_fmt,bit_rate,channels",
        "-of", "json", str(input_file)
    ]
    out = subprocess.check_output(args=cmd)
    info = json.loads(out)
    source_params = {
                "sampling_rate": int(info['streams'][0]['sample_rate']),
                "is_opus": info['streams'][0]['codec_name'] == "opus",
                "sample_format": info['streams'][0]['sample_fmt'],
                "bit_rate": int(info['streams'][0]['bit_rate']),
                "nb_channels": int(info['streams'][0]['channels'])
            }
    return source_params

def _import_to_tempfile(audio_file: Path,
                        tmp_file: Path,
                        source_params: dict,
                        target_codec: str,
                        target_sample_format: str,
                        flac_compression_level:int,
                        opus_bitrate: str
                        ) -> float:
    
    sampling_rescale_factor = 1.0
    is_ultrasonic = (target_codec == "opus") and (source_params["sampling_rate"] > 48000)

    if target_codec == 'opus' and source_params["is_opus"] and not is_ultrasonic:
        # copy opus encoded data directly into ogg.opus 
        opus_bitrate = f"{int(source_params["bit_rate"] / 1000.0)}k"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_file),
            "-c", "copy", "-sample_fmt", target_sample_format,
            "-f", "ogg", str(tmp_file)
        ], check=True)
        return sampling_rescale_factor

    ffmpeg_cmd = ["ffmpeg", "-y"]   

    if target_codec == 'opus':
        if is_ultrasonic:
            # we interprete sampling rate as "48000" to can use opus for ultrasonic
            ffmpeg_cmd += ["-sample_rate", "48000"]
            sampling_rescale_factor = float(source_params["bit_rate"]) / 48000.0
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "libopus", "-b:a", opus_bitrate]
        ffmpeg_cmd += ["-vbr", "off"] # constant bitrate is a bit better in quality than VRB=On
        ffmpeg_cmd += ["-apply_phase_inv", "false"] # Phasenrichtige Kodierung: keine Tricks!

    elif target_codec == 'flac':
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
        ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]
    else:
        raise NotImplementedError(f"Target codec {target_codec} is not (yet) implemented.")

    ffmpeg_cmd += ["-sample_fmt", target_sample_format, "-f", "ogg", str(tmp_file)]

    # start encoding into temp file and wait until finished
    subprocess.run(ffmpeg_cmd, check=True)

    return sampling_rescale_factor

def _create_index_zarr(ogg_file_blob_array: np.ndarray, zarr_original_audio_group: zarr.Group):
    """
    Speicheroptimiertes Parsen und Speichern von Ogg-Page-Indexdaten in Zarr-Gruppe.
    Die Indexdaten werden chunkweise direkt ins Zarr-Array 'index' geschrieben.
    """
    data = ogg_file_blob_array
    data_len = data.shape[0]
    offset = 0

    chunk_entries = []
    max_entries_per_chunk = 65536  # entspricht ~1MB RAM

    # Vorab leeres Zarr-Array mit großzügiger Maxshape
    index_zarr = zarr_original_audio_group.create_array(
        name="ogg_page_index",
        shape=(0, 2),
        chunks=(max_entries_per_chunk, 2),
        dtype=np.uint64,
        maxshape=(None, 2),
        overwrite=True
    )

    total_entries = 0

    while offset + OGG_PAGE_HEADER_SIZE < data_len:
        if not np.array_equal(data[offset:offset+4], np.frombuffer(b'OggS', dtype=np.uint8)):
            offset += 1
            continue

        header = data[offset : offset + OGG_PAGE_HEADER_SIZE]
        granule_pos = struct.unpack_from('<Q', header.tobytes(), 6)[0]
        segment_count = header[26]

        seg_table_start = offset + OGG_PAGE_HEADER_SIZE
        seg_table_end = seg_table_start + segment_count
        if seg_table_end > data_len:
            raise OggImportError("Data not parsable during index creation.")
        
        segment_table = data[seg_table_start:seg_table_end]
        page_body_size = int(np.sum(segment_table))

        page_size = OGG_PAGE_HEADER_SIZE + segment_count + page_body_size
        if offset + page_size > data_len:
            raise OggImportError("Data not parsable during index creation.")

        chunk_entries.append((offset, granule_pos))
        offset += page_size

        if len(chunk_entries) >= max_entries_per_chunk:
            chunk_np = np.array(chunk_entries, dtype=np.uint64)
            index_zarr.resize(total_entries + chunk_np.shape[0], axis=0)
            index_zarr[total_entries : total_entries + chunk_np.shape[0], :] = chunk_np
            total_entries += chunk_np.shape[0]
            chunk_entries = []

    if chunk_entries:
        chunk_np = np.array(chunk_entries, dtype=np.uint64)
        index_zarr.resize(total_entries + chunk_np.shape[0], axis=0)
        index_zarr[total_entries : total_entries + chunk_np.shape[0], :] = chunk_np
        total_entries += chunk_np.shape[0]

    return index_zarr

def extract_audio_segment_from_blob(
                                        ogg_file_blob_array,   # zarr.Array (uint8)
                                        index_zarr,            # zarr.Array, shape (N, 2), dtype=uint64
                                        start_sample: int,
                                        last_sample: int,
                                        sample_rate: int,
                                        channels: int,
                                        dtype: np.dtype = np.int16
                                    ) -> np.ndarray:
    """
    Extrahiert ein PCM-Segment (Samples) aus einem Ogg-Zarr-Blob anhand des Index.
    Gibt ein (samples, channels)-Array zurück (dtype = z.B. int16 oder float32).
    """

    if start_sample > last_sample:
        raise ValueError("start_sample darf nicht größer als last_sample sein.")

    # 1. Suche Start- und Endposition im Index (sample-basiert via granule_position)
    granules = index_zarr[:, 1][:]
    start_idx = np.searchsorted(granules, start_sample, side="left")

    if start_idx >= len(granules):
        raise ValueError("Startposition liegt hinter letzter Index-Granule.")

    end_idx = np.searchsorted(granules, last_sample, side="right") - 1
    if end_idx < start_idx:
        raise ValueError("Ungültiger Bereich im Index.")

    # 2. Bestimme Byte-Offsets im Ogg-Blob
    start_byte = int(index_zarr[start_idx, 0])
    end_byte = (
        int(index_zarr[end_idx + 1, 0]) if end_idx + 1 < len(index_zarr) else ogg_file_blob_array.shape[0]
    )
    actual_start_sample = int(index_zarr[start_idx, 1])

    # 3. Lade nur den betroffenen Ausschnitt aus dem Blob
    ogg_slice = ogg_file_blob_array[start_byte:end_byte][:].tobytes()

    # 4. FFMPEG aufrufen, um OGG-Daten zu dekodieren
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-f", "ogg",
        "-i", "pipe:0",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "s16le" if dtype == np.int16 else "f32le",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pcm_bytes, _ = proc.communicate(ogg_slice)

    # 5. PCM-Daten als NumPy-Array interpretieren
    samples = np.frombuffer(pcm_bytes, dtype=dtype)
    if samples.size % channels != 0:
        raise ValueError("Fehlerhafte Kanalanzahl im dekodierten PCM-Strom.")
    samples = samples.reshape(-1, channels)

    # 6. Zielbereich (Samples) aus Gesamtergebnis extrahieren
    rel_start = start_sample - actual_start_sample
    rel_end = last_sample - actual_start_sample + 1  # inklusiv

    if rel_start < 0 or rel_end > samples.shape[0]:
        raise ValueError(
            f"Granule-Offset außerhalb des dekodierten Bereichs: {rel_start=}, {rel_end=}, shape={samples.shape}"
        )

    return samples[rel_start:rel_end]

