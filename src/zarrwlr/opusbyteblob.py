import numpy as np
import struct
import zarr
import subprocess
import tempfile
from .exceptions import OggImportError

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)

# Konstanten für Ogg-Container
OGG_PAGE_HEADER_SIZE = 27
OGG_MAX_PAGE_SIZE = 65536

def build_opus_index(zarr_group, audio_blob_array):
    """
    Erstellt einen Index für Ogg-Pages, der die Byte-Positionen und Granule-Positionen 
    (Sample-Positionen) enthält
    
    Args:
        zarr_group: Zarr-Gruppe, in der der Index gespeichert werden soll
        audio_blob_array: Array mit den binären Ogg-Opus-Audiodaten
        
    Returns:
        Das erstellte Index-Array
        
    Raises:
        OggImportError: Wenn die Daten nicht vollständig oder ungültig sind
    """
    data = audio_blob_array
    data_len = data.shape[0]

    chunk_entries = []
    max_entries_per_chunk = 65536  # entspricht ~1MB RAM bei typ int64

    # Vorab leeres Zarr-Array mit großzügiger Maxshape
    index_zarr = zarr_group.create_array(
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
        # Überprüfen, ob wir einen Ogg-Page-Header finden
        if not np.array_equal(data[offset:offset+4], np.frombuffer(b'OggS', dtype=np.uint8)):
            offset += 1
            # Robustheit bei fehlerhaften Daten verbessern
            invalid_header_count += 1
            if invalid_header_count > 1024:
                raise OggImportError("Zu viele ungültige Ogg-Header. Import wird abgebrochen.")
            continue

        # Ogg-Page-Header extrahieren
        header = data[offset : offset + OGG_PAGE_HEADER_SIZE]
        # Granule-Position extrahieren (64-bit, little-endian)
        granule_pos = struct.unpack_from('<Q', header.tobytes(), 6)[0]
        # Segment-Anzahl aus dem Header lesen
        segment_count = header[26]

        # Segment-Tabelle lesen
        seg_table_start = offset + OGG_PAGE_HEADER_SIZE
        seg_table_end = seg_table_start + segment_count
        if seg_table_end > data_len:
            raise OggImportError("Daten nicht vollständig beim Indexieren.")
        
        segment_table = data[seg_table_start:seg_table_end]
        # Größe des Page-Body berechnen (Summe aller Segment-Größen)
        page_body_size = int(np.sum(segment_table))

        # Gesamtgröße der Page berechnen
        page_size = OGG_PAGE_HEADER_SIZE + segment_count + page_body_size
        if offset + page_size > data_len:
            raise OggImportError("Daten nicht vollständig beim Indexieren.")

        # Byte-Offset und Granule-Position in Einträge aufnehmen
        chunk_entries.append((offset, granule_pos))
        offset += page_size

        # Wenn genug Einträge gesammelt wurden, zum Zarr-Array hinzufügen
        if len(chunk_entries) >= max_entries_per_chunk:
            chunk_np = np.array(chunk_entries, dtype=np.uint64)
            index_zarr.resize(total_entries + chunk_np.shape[0], axis=0)
            index_zarr[total_entries : total_entries + chunk_np.shape[0], :] = chunk_np
            total_entries += chunk_np.shape[0]
            chunk_entries = []

    # Verbleibende Einträge hinzufügen
    if chunk_entries:
        chunk_np = np.array(chunk_entries, dtype=np.uint64)
        index_zarr.resize(total_entries + chunk_np.shape[0], axis=0)
        index_zarr[total_entries : total_entries + chunk_np.shape[0], :] = chunk_np
        total_entries += chunk_np.shape[0]

    # Metadaten zum Index-Array hinzufügen
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    sampling_rescale_factor = audio_blob_array.attrs.get('sampling_rescale_factor', 1.0)
    
    index_zarr.attrs['sample_rate'] = sample_rate
    index_zarr.attrs['channels'] = channels
    index_zarr.attrs['sampling_rescale_factor'] = sampling_rescale_factor
    
    logger.info(f"Ogg-Opus-Index erstellt mit {total_entries} Seiten")
    return index_zarr


def extract_audio_segment_opus(zarr_group, audio_blob_array, start_sample, end_sample, dtype=np.int16):
    """
    Extrahiert ein Audiosegment aus einer Opus-Datei in einer Zarr-Gruppe
    
    Args:
        zarr_group: Zarr-Gruppe, die den Index enthält
        audio_blob_array: Array mit den Opus-Audiodaten
        start_sample: Erstes Sample, das extrahiert werden soll (inklusive)
        end_sample: Letztes Sample, das extrahiert werden soll (inklusive)
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        
    Returns:
        np.ndarray: Extrahiertes Audiosegment mit Shape (Samples, Channels)
        
    Raises:
        ValueError: Bei ungültigen Parametern oder Problemen mit dem Index
        RuntimeError: Bei Fehlern in der Dekodierung
    """
    ogg_index = zarr_group['ogg_page_index']
    
    if start_sample > end_sample:
        raise ValueError(f"Ungültiger Bereich: start_sample={start_sample}, end_sample={end_sample}")

    # Suche Start- und Endposition im Index (sample-basiert via granule_position)
    granules = ogg_index[:, 1]
    start_idx = np.searchsorted(granules, start_sample, side="left")

    if start_idx >= len(granules):
        raise ValueError("Startposition liegt hinter letzter Index-Granule.")

    end_idx = np.searchsorted(granules, end_sample, side="right") - 1
    if end_idx < start_idx:
        raise ValueError("Ungültiger Bereich im Index.")

    # Bestimme Byte-Offsets im Ogg-Blob
    start_byte = int(ogg_index[start_idx, 0])
    end_byte = (
        int(ogg_index[end_idx + 1, 0]) if end_idx + 1 < len(ogg_index) else audio_blob_array.shape[0]
    )
    actual_start_sample = int(ogg_index[start_idx, 1])

    # Lade nur den betroffenen Ausschnitt aus dem Blob
    ogg_slice = audio_blob_array[start_byte:end_byte].tobytes()

    # Parameter aus Array-Attributen laden
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    
    # FFMPEG aufrufen, um OGG-Daten zu dekodieren
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
    pcm_bytes, _ = proc.communicate(input=ogg_slice)
    
    if proc.returncode != 0 or not pcm_bytes:
        raise RuntimeError("FFmpeg konnte das Audiosegment nicht dekodieren.")

    # PCM-Daten als NumPy-Array interpretieren
    samples = np.frombuffer(pcm_bytes, dtype=dtype)
    if samples.size % channels != 0:
        raise ValueError("Fehlerhafte Kanalanzahl im dekodierten PCM-Strom.")
    samples = samples.reshape(-1, channels)

    # Zielbereich (Samples) aus Gesamtergebnis extrahieren
    rel_start = start_sample - actual_start_sample
    rel_end = rel_start + (end_sample - start_sample + 1)  # inklusiv

    if rel_start < 0 or rel_end > samples.shape[0]:
        # Logger-Warnung, falls der Bereich nicht exakt passt
        logger.warning(
            f"Angeforderte Samples nicht exakt im dekodierten Bereich: {rel_start=}, {rel_end=}, shape={samples.shape}"
        )
        # Sichere Indizierung
        rel_start = max(0, rel_start)
        rel_end = min(rel_end, samples.shape[0])

    return samples[rel_start:rel_end]


def parallel_extract_audio_segments_opus(zarr_group, audio_blob_array, segments, dtype=np.int16, max_workers=4):
    """
    Extrahiert mehrere Audiosegmente parallel aus einer Opus-Datei
    
    Args:
        zarr_group: Zarr-Gruppe, die den Index enthält
        audio_blob_array: Array mit den Opus-Audiodaten
        segments: Liste von (start_sample, end_sample)-Tupeln
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        max_workers: Maximale Anzahl paralleler Worker
        
    Returns:
        Liste von np.ndarray: Extrahierte Audiosegmente für jeden angeforderten Bereich
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(extract_audio_segment_opus, zarr_group, audio_blob_array, start, end, dtype): (start, end) 
            for start, end in segments
        }
        
        results = {}
        for future in future_to_segment:
            segment = future_to_segment[future]
            try:
                results[segment] = future.result()
            except Exception as e:
                logger.error(f"Fehler beim Extrahieren des Segments {segment}: {e}")
                results[segment] = np.array([])
                
        # Sortiere nach der ursprünglichen Segment-Reihenfolge
        return [results[segment] for segment in segments]
