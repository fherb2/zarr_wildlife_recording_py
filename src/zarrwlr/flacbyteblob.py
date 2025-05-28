import zarr
import numpy as np
import io
import soundfile as sf
import tempfile
import os
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)


def _parse_flac_header_and_metadata(audio_bytes: bytes) -> int:
    """
    Parst FLAC-Header und überspringt Metadaten-Blöcke
    
    Args:
        audio_bytes: FLAC-Audiodaten als bytes
        
    Returns:
        Position nach dem letzten Metadaten-Block
    """
    pos = 0
    
    # FLAC-Signature prüfen
    if audio_bytes[:4] != b'fLaC':
        raise ValueError("Keine gültige FLAC-Datei gefunden")
    
    pos = 4
    
    # Metadaten-Blöcke überspringen
    while pos < len(audio_bytes):
        if pos + 4 > len(audio_bytes):
            break
            
        block_header = int.from_bytes(audio_bytes[pos:pos+4], 'big')
        is_last = (block_header & 0x80000000) != 0
        block_size = block_header & 0x7FFFFF
        
        pos += 4 + block_size
        
        if is_last:
            break
    
    return pos


def _find_next_flac_frame_sync(audio_bytes: bytes, start_pos: int, max_search_bytes: int = 65536) -> Optional[int]:
    """
    Sucht das nächste FLAC-Frame-Sync-Pattern
    
    Args:
        audio_bytes: FLAC-Audiodaten
        start_pos: Startposition für die Suche
        max_search_bytes: Maximale Suchreichweite
        
    Returns:
        Position des nächsten Frame-Syncs oder None
    """
    search_end = min(start_pos + max_search_bytes, len(audio_bytes) - 2)
    
    for pos in range(start_pos, search_end):
        if pos + 1 < len(audio_bytes):
            sync_word = int.from_bytes(audio_bytes[pos:pos+2], 'big')
            if (sync_word & 0xFFFE) == 0xFFF8:
                return pos
    
    return None


def _estimate_samples_per_frame_from_header(frame_header_bytes: bytes) -> int:
    """
    Schätzt Samples pro Frame aus FLAC-Frame-Header
    
    Args:
        frame_header_bytes: Erste Bytes des Frame-Headers
        
    Returns:
        Geschätzte Anzahl Samples pro Frame
    """
    # Vereinfachte Schätzung basierend auf häufigen FLAC-Einstellungen
    # Für eine vollständige Implementierung müsste der Frame-Header vollständig geparst werden
    
    # Häufige Block-Größen in FLAC: 1152, 2304, 4608
    # Default auf konservativen Wert
    return 4608


def _parse_flac_frames_from_bytes(audio_bytes: bytes, expected_sample_rate: int = 44100) -> List[dict]:
    """
    Parst FLAC-Frames direkt aus den Byte-Daten
    
    Args:
        audio_bytes: Komplette FLAC-Audiodaten
        expected_sample_rate: Erwartete Sample-Rate aus Metadaten für bessere Frame-Größen-Schätzung
        
    Returns:
        Liste von Frame-Informationen als Dictionaries
    """
    frames_info = []
    
    # Header und Metadaten überspringen
    pos = _parse_flac_header_and_metadata(audio_bytes)
    current_sample = 0
    
    logger.trace(f"Beginne Frame-Analyse ab Position {pos} für {expected_sample_rate}Hz Audio")
    
    # Frame-für-Frame-Analyse
    while pos < len(audio_bytes) - 2:
        # FLAC-Frame-Sync suchen
        sync_word = int.from_bytes(audio_bytes[pos:pos+2], 'big')
        
        if (sync_word & 0xFFFE) == 0xFFF8:  # FLAC Frame Sync Pattern
            frame_start = pos
            
            # Suche nächstes Frame für Größenbestimmung
            next_frame_pos = _find_next_flac_frame_sync(audio_bytes, pos + 16)
            
            if next_frame_pos is not None:
                frame_size = next_frame_pos - frame_start
            else:
                # Letztes Frame - nimm Rest der Datei
                frame_size = len(audio_bytes) - frame_start
            
            # Samples pro Frame schätzen (mit Metadaten-Info)
            header_bytes = audio_bytes[pos:pos+min(16, frame_size)]
            samples_per_frame = _estimate_samples_per_frame_from_header(header_bytes, expected_sample_rate)
            
            frames_info.append({
                'byte_offset': frame_start,
                'frame_size': frame_size,
                'sample_pos': current_sample
            })
            
            current_sample += samples_per_frame
            
            # Springe zum nächsten Frame (oder konservativ weiter)
            if next_frame_pos is not None:
                pos = next_frame_pos
            else:
                pos += max(16, frame_size // 2)
        else:
            pos += 1
    
    logger.trace(f"Gefunden: {len(frames_info)} FLAC-Frames für {expected_sample_rate}Hz Audio")
    return frames_info


def build_flac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Erstellt einen Index für FLAC-Frame-Zugriff durch direkte Byte-Analyse
    
    Args:
        zarr_group: Zarr-Gruppe für Index-Speicherung
        audio_blob_array: Array mit FLAC-Audiodaten
        
    Returns:
        Erstelltes Index-Array
        
    Raises:
        ValueError: Wenn keine FLAC-Frames gefunden werden
    """
    logger.trace("build_flac_index() requested.")
    
    # Metadaten aus Array-Attributen (vom aimport.py Modul gesetzt)
    sample_rate = audio_blob_array.attrs.get('sample_rate', 44100)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    codec = audio_blob_array.attrs.get('codec', 'flac')
    container_type = audio_blob_array.attrs.get('container_type', 'flac-nativ')
    
    # Validierung: Stellen wir sicher, dass es sich um FLAC-Daten handelt
    if codec != 'flac':
        raise ValueError(f"Erwarteter FLAC-Codec, aber gefunden: {codec}")
    
    logger.trace(f"FLAC-Index wird erstellt für: {sample_rate}Hz, {channels} Kanäle, Container: {container_type}")
    
    # Audio-Bytes laden (einmalig)
    logger.trace("Lade FLAC-Audiodaten...")
    audio_bytes = bytes(audio_blob_array[()])
    
    # FLAC-Frames durch direkte Byte-Analyse finden (mit Metadaten-Info)
    logger.trace("Analysiere FLAC-Frames...")
    frames_info = _parse_flac_frames_from_bytes(audio_bytes, sample_rate)
    
    if len(frames_info) < 1:
        raise ValueError("Konnte keine FLAC-Frames im Audio finden")
    
    # Structured Array für Index erstellen
    logger.trace("Erstelle Index-Array...")
    index_array = np.array([
        (f['byte_offset'], f['frame_size'], f['sample_pos']) 
        for f in frames_info
    ], dtype=[
        ('byte_offset', np.int64), 
        ('frame_size', np.int32),
        ('sample_pos', np.int64)
    ])
    
    # Index in Zarr-Gruppe speichern
    flac_index = zarr_group.create_dataset(
        'flac_index', 
        data=index_array, 
        chunks=(min(1000, len(frames_info)),)
    )
    
    # Metadaten am Index speichern (erweitert mit verfügbaren Informationen)
    index_attrs = {
        'sample_rate': sample_rate,
        'channels': channels,
        'total_frames': len(frames_info),
        'codec': codec,
        'container_type': container_type
    }
    
    # Zusätzliche Metadaten aus dem audio_blob_array übernehmen, falls vorhanden
    optional_attrs = [
        'compression_level', 'sampling_rescale_factor', 
        'first_sample_time_stamp', 'last_sample_time_stamp'
    ]
    
    for attr_name in optional_attrs:
        if attr_name in audio_blob_array.attrs:
            index_attrs[attr_name] = audio_blob_array.attrs[attr_name]
    
    flac_index.attrs.update(index_attrs)
    
    logger.success(f"FLAC-Index erstellt mit {len(frames_info)} Frames")
    return flac_index


def _find_frame_range_for_samples(flac_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Findet Frame-Bereich für Sample-Bereich mittels Binary Search
    
    Args:
        flac_index: FLAC-Index-Array
        start_sample: Erstes benötigtes Sample
        end_sample: Letztes benötigtes Sample
        
    Returns:
        Tuple (start_frame_idx, end_frame_idx)
    """
    sample_positions = flac_index['sample_pos']
    
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, len(flac_index) - 1)
    
    return start_idx, end_idx


def _create_temporary_flac_file(audio_bytes: bytes) -> str:
    """
    Erstellt temporäre FLAC-Datei für SoundFile-Zugriff
    
    Args:
        audio_bytes: FLAC-Audiodaten
        
    Returns:
        Pfad zur temporären Datei
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def extract_audio_segment_flac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                              start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """
    Extrahiert Audio-Segment basierend auf FLAC-Index
    
    Args:
        zarr_group: Zarr-Gruppe mit FLAC-Index
        audio_blob_array: Array mit FLAC-Audiodaten
        start_sample: Erstes Sample (inklusive)
        end_sample: Letztes Sample (inklusive)
        dtype: Ausgabe-Datentyp
        
    Returns:
        Dekodierte Audiodaten als numpy array
    """
    try:
        # Index aus Zarr-Gruppe laden
        if 'flac_index' not in zarr_group:
            raise ValueError("FLAC-Index nicht gefunden. Muss erst mit build_flac_index() erstellt werden.")
        
        flac_index = zarr_group['flac_index']
        
        # Frame-Bereich für Sample-Bereich finden
        start_idx, end_idx = _find_frame_range_for_samples(flac_index, start_sample, end_sample)
        
        if start_idx > end_idx:
            raise ValueError(f"Ungültiger Sample-Bereich: start={start_sample}, end={end_sample}")
        
        # Audio-Bytes laden und temporäre Datei erstellen
        audio_bytes = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_flac_file(audio_bytes)
        
        try:
            # Mit SoundFile dekodieren
            with sf.SoundFile(temp_file_path) as sf_file:
                sample_positions = flac_index['sample_pos']
                first_frame_sample = sample_positions[start_idx]
                
                # Zum ersten benötigten Frame springen
                sf_file.seek(first_frame_sample)
                
                # Samples zum Lesen berechnen
                if end_idx < len(flac_index) - 1:
                    last_frame_end = sample_positions[end_idx + 1]
                else:
                    last_frame_end = sf_file.frames
                
                total_samples_to_read = last_frame_end - first_frame_sample
                frames_data = sf_file.read(total_samples_to_read, dtype=dtype)
                
                # Exakt auf angeforderten Bereich zuschneiden
                start_offset = max(0, start_sample - first_frame_sample)
                end_offset = min(
                    start_offset + (end_sample - start_sample + 1), 
                    frames_data.shape[0]
                )
                
                return frames_data[start_offset:end_offset]
                
        finally:
            # Temporäre Datei aufräumen
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren des FLAC-Segments [{start_sample}:{end_sample}]: {e}")
        return np.array([])


def parallel_extract_audio_segments_flac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                                        segments: List[Tuple[int, int]], dtype=np.int16, 
                                        max_workers: int = 4) -> List[np.ndarray]:
    """
    Parallele Extraktion mehrerer Audio-Segmente
    
    Args:
        zarr_group: Zarr-Gruppe mit FLAC-Index
        audio_blob_array: Array mit FLAC-Audiodaten  
        segments: Liste von (start_sample, end_sample) Tupeln
        dtype: Ausgabe-Datentyp
        max_workers: Maximale Anzahl paralleler Worker
        
    Returns:
        Liste von dekodierteren Audio-Arrays in ursprünglicher Reihenfolge
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                extract_audio_segment_flac, 
                zarr_group, audio_blob_array, start, end, dtype
            ): (start, end) 
            for start, end in segments
        }
        
        results = {}
        for future in future_to_segment:
            segment = future_to_segment[future]
            try:
                results[segment] = future.result()
            except Exception as e:
                logger.error(f"Fehler beim parallelen Extrahieren des Segments {segment}: {e}")
                results[segment] = np.array([])
        
        # Ergebnisse in ursprünglicher Reihenfolge zurückgeben
        return [results[segment] for segment in segments]


logger.trace("Module loaded.")