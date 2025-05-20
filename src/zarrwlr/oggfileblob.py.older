
import numpy as np
from numpy.typing import DTypeLike
import struct
import zarr
import subprocess
import tempfile
import pathlib
import logging
import json
from .config import Config
from .exceptions import OggImportError

OGG_PAGE_HEADER_SIZE = 27
OGG_MAX_PAGE_SIZE = 65536
STD_FLAC_COMPRESSION_LEVEL = 4  # 0...12; 4 is a really good value: fast and low
                                # data; higher values does not really reduce 
                                # data size but need more time and energy. 
                                # Note:
                                # The highest values can produce more data
                                # than lower values. '12' as maximum must not
                                # be the best compression. Check it: More than
                                # 4 is not really less memory consumption but
                                # wasted time and energy.
STD_OPUS_BITRATE = 160_000      # bit per second


def import_audio_to_blob( 
                         audio_file: str|pathlib.Path, 
                         audio_file_blob_array: zarr.Array,
                         target_codec:str = 'flac', # "flac" or "opus"
                         flac_compression_level: int = STD_FLAC_COMPRESSION_LEVEL,
                         opus_bitrate:int = STD_OPUS_BITRATE,
                         temp_dir=STD_TEMP_DIR,
                         chunk_size:int = Config.original_audio_chunk_size,
                         ):
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

    audio_file = pathlib.Path(audio_file)

    source_params = _get_source_params(audio_file)
    target_sample_format = _get_ffmpeg_sample_fmt(source_params["sample_format"], 
                                                  target_codec)

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
        tmp_file = pathlib.Path(tmp_out.name)
        sampling_rescale_factor = _import_to_tempfile(  audio_file=audio_file,
                                                        tmp_file=tmp_file,
                                                        source_params=source_params,
                                                        target_codec=target_codec,
                                                        target_sample_format=target_sample_format,
                                                        flac_compression_level=flac_compression_level,
                                                        opus_bitrate=opus_bitrate
                                                        )
        
        # copy tmp-file data into the array and remove the temp file
        with open(tmp_file, "rb") as f:
            for buffer in iter(lambda: f.read(chunk_size), b''):
                audio_file_blob_array.append(np.frombuffer(buffer, dtype="u1"))
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
    audio_file_blob_array.attrs.update(attrs)

def _import_to_tempfile(audio_file: pathlib.Path,
                        tmp_file: pathlib.Path,
                        source_params: dict,
                        target_codec: str,
                        target_sample_format: str,
                        flac_compression_level:int = STD_FLAC_COMPRESSION_LEVEL,
                        opus_bitrate:int = STD_OPUS_BITRATE
                        ) -> float:
    """_import_to_tempfile Convert an audio file stream into an ogg container.

    Given audio file data will be converted into opus or flac format inside
    an Ogg container. Source file parameters, needed for ffmpeg, must be
    given as returned from _get_source_params() as dictionary. The target_sample_format
    must be given as returned from _get_ffmpeg_sample_fmt().

    Parameters
    ----------
    audio_file : pathlib.Path
        Source audio file.
    tmp_file : pathlib.Path
        Target file (temporary)
    source_params : dict
        as returned from _get_source_params()
    target_codec : str
        'opus' or 'flac'
    target_sample_format : str
        as returned from _get_ffmpeg_sample_fmt()
    flac_compression_level : int
        0...12; default: 4
    opus_bitrate : int
        int bit/s; default: 160000

    Returns
    -------
    float
        sampling_rescale_factor - In case the original sampling rate is
        higher than 48000 S/s and the target format is 'opus' so the samples
        will be compressed without resampling but uinterpreted as 48kS/s. 
        The factor allows to re-calculate the true sampling rate of the
        original recording:
            original_sampling_rate = sampling_rescale_factor * sampling_rate_in_ogg_blob
        This mode is interesting for ultrasonic records. So also these data
        can be compressed to ogg-opus. For export you need this factor to know
        what was the original sampling rate. Otherwise you will hear the sound too
        slowly and with deeper frequencies.

        In all other cases, the return value of sampling_rescale_factor is
        exactly 1.0 .

    Raises
    ------
    NotImplementedError
        _description_
    """

    # assert (target_codec == 'flac') or (target_codec == 'opus')

    # opus_bitrate_str = str(int(opus_bitrate/1000))+'k'
    # sampling_rescale_factor = 1.0
    # is_ultrasonic = (target_codec == "opus") and (source_params["sampling_rate"] > 48000)
    
    # if target_codec == 'opus' and source_params["is_opus"] and not is_ultrasonic:
    #     # copy opus encoded data directly into ogg.opus 
    #     subprocess.run([
    #         "ffmpeg", "-y", "-i", str(audio_file),
    #         "-c", "copy", "-sample_fmt", target_sample_format,
    #         "-f", "ogg", str(tmp_file)
    #     ], check=True)
    #     return sampling_rescale_factor

    # ffmpeg_cmd = ["ffmpeg", "-y"]   

    # if target_codec == 'opus':
    #     if is_ultrasonic:
    #         # we interprete sampling rate as "48000" to can use opus for ultrasonic
    #         ffmpeg_cmd += ["-sample_rate", "48000"]
    #         sampling_rescale_factor = float(source_params["bit_rate"]) / 48000.0
    #     ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "libopus", "-b:a", opus_bitrate_str]
    #     ffmpeg_cmd += ["-vbr", "off"] # constant bitrate is a bit better in quality than VRB=On
    #     ffmpeg_cmd += ["-apply_phase_inv", "false"] # Phasenrichtige Kodierung: keine Tricks!

    # else:
    #     ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
    #     ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]

    # ffmpeg_cmd += ["-sample_fmt", target_sample_format, "-f", "ogg", str(tmp_file)]

    # # start encoding into temp file and wait until finished
    # subprocess.run(ffmpeg_cmd, check=True)

    # return sampling_rescale_factor

def create_index(audio_file_blob_array: zarr.Array, zarr_original_audio_group: zarr.Group):
    """
    Speicheroptimiertes Parsen und Speichern von Ogg-Page-Indexdaten in Zarr-Gruppe.
    Die Indexdaten werden chunkweise direkt ins Zarr-Array 'index' geschrieben.
    """
    data = audio_file_blob_array
    data_len = data.shape[0]

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
    offset = 0
    invalid_header_count = 0
    while offset + OGG_PAGE_HEADER_SIZE < data_len:
        if not np.array_equal(data[offset:offset+4], np.frombuffer(b'OggS', dtype=np.uint8)):
            offset += 1
            # more robust in case of wrong blob data:
            invalid_header_count += 1
            if invalid_header_count > 1024:
                raise OggImportError("Too many invalid Ogg headers. Aborting.")
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

def extract_audio_segment_from_blob(audio_file_blob_array: zarr.Array, # (uint8)
                                    index_zarr: zarr.Array,            # shape (N, 2), dtype=uint64
                                    start_sample: int,
                                    last_sample: int,
                                    sample_rate: int,
                                    channels: int,
                                    dtype: DTypeLike = np.int16
                                    ) -> np.ndarray:
    """
    Extrahiert ein PCM-Segment (Samples) aus einem Ogg-Zarr-Blob anhand des Index.
    Gibt ein (samples, channels)-Array zurück (dtype = z.B. int16 oder float32).
    """

    if start_sample > last_sample:
        raise ValueError(f"Invalid range: start_idx={start_sample}, end_idx={last_sample}")

    # 1. Suche Start- und Endposition im Index (sample-basiert via granule_position)
    granules = index_zarr[:, 1]
    start_idx = np.searchsorted(granules, start_sample, side="left")

    if start_idx >= len(granules):
        raise ValueError("Startposition liegt hinter letzter Index-Granule.")

    end_idx = np.searchsorted(granules, last_sample, side="right") - 1
    if end_idx < start_idx:
        raise ValueError("Ungültiger Bereich im Index.")

    # 2. Bestimme Byte-Offsets im Ogg-Blob
    start_byte = int(index_zarr[start_idx, 0])
    end_byte = (
        int(index_zarr[end_idx + 1, 0]) if end_idx + 1 < len(index_zarr) else audio_file_blob_array.shape[0]
    )
    actual_start_sample = int(index_zarr[start_idx, 1])

    # 3. Lade nur den betroffenen Ausschnitt aus dem Blob
    ogg_slice = audio_file_blob_array[start_byte:end_byte][:].tobytes()

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
    if proc.returncode != 0 or not pcm_bytes:
        raise RuntimeError("FFmpeg failed to decode audio segment.")
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

