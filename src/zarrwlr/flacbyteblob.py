import zarr
import numpy as np
import io
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

# import and initialize logging
from zarrwlr.logsetup import get_logger
logger = get_logger()

def build_flac_index(zarr_group, audio_blob_array):
    """
    Erstellt einen Index, der Sample-Positionen zu FLAC-Frames zuordnet
    
    Args:
        zarr_group: Zarr-Gruppe, in der der Index gespeichert werden soll
        audio_blob_array: Array mit den binären FLAC-Audiodaten
        
    Returns:
        Das erstellte Index-Array
    
    Raises:
        ValueError: Wenn keine FLAC-Frames gefunden werden
    """
    # Hole Metadaten aus den Attributen
    sample_rate = audio_blob_array.attrs.get('sample_rate', 44100)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    
    audio_bytes = bytes(audio_blob_array[()])
    
    # FLAC-Stream in temporäre Datei umleiten für SoundFile
    with io.BytesIO(audio_bytes) as buf:
        with sf.SoundFile(buf) as f:
            # FLAC-Header und Metadaten überspringen
            frames_info = []
            f.seek(0)  # Aktuelle Position nach dem Header
            
            # Sammle Frame-Informationen
            while True:
                try:
                    # Position des aktuellen Frames speichern
                    frame_offset = f.tell()
                    
                    # Frame-Position in Samples
                    frame_sample_pos = f.seek(0, sf.SEEK_CUR)
                    
                    # Wir lesen einen kleinen Block, um die Frame-Größe zu ermitteln
                    dummy = f.read(1)
                    if len(dummy) == 0:  # EOF erreicht
                        break
                        
                    # Position nach dem Lesen
                    next_frame_offset = f.tell()
                    
                    # Frame-Größe berechnen
                    frame_size = next_frame_offset - frame_offset
                    
                    # Frame-Informationen speichern
                    frames_info.append({
                        'byte_offset': frame_offset,
                        'frame_size': frame_size, 
                        'sample_pos': frame_sample_pos
                    })
                    
                    # Zurück zur Position vor dem nächsten Frame
                    f.seek(next_frame_offset)
                    
                except RuntimeError:
                    # Ende des Streams erreicht
                    break
    
    # Erstelle Index-Array im Zarr-Store
    if frames_info:
        index_array = np.array([(f['byte_offset'], f['frame_size'], f['sample_pos']) 
                               for f in frames_info], 
                              dtype=[('byte_offset', np.int64), 
                                     ('frame_size', np.int32),
                                     ('sample_pos', np.int64)])
        
        flac_index = zarr_group.create_dataset('flac_index', data=index_array, 
                                      chunks=(min(1000, len(frames_info)),))
        
        # Speichere zusätzliche Metadaten
        flac_index.attrs['sample_rate'] = sample_rate
        flac_index.attrs['channels'] = channels
        
        logger.info(f"FLAC-Index erstellt mit {len(frames_info)} Frames")
        return flac_index
    else:
        raise ValueError("Konnte keine FLAC-Frames im Audio finden")


def extract_audio_segment_flac(zarr_group, audio_blob_array, start_sample, end_sample, dtype=np.int16):
    """
    Dekodiert einen spezifischen Bereich von Samples aus einer FLAC-Datei
    
    Args:
        zarr_group: Zarr-Gruppe, die den Index enthält
        audio_blob_array: Array mit den FLAC-Audiodaten
        start_sample: Erstes Sample, das dekodiert werden soll (inklusive)
        end_sample: Letztes Sample, das dekodiert werden soll (inklusive)
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        
    Returns:
        np.ndarray: Dekodierte Audiodaten mit Shape (Samples, Channels)
    """
    flac_index = zarr_group['flac_index']
    
    # Binary Search für den ersten Frame, der start_sample enthält
    sample_positions = flac_index['sample_pos']
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    # Finde alle Frames bis zum letzten benötigten
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, len(flac_index) - 1)
    
    if start_idx > end_idx:
        raise ValueError(f"Ungültiger Bereich: start_idx={start_idx}, end_idx={end_idx}")
    
    # Extrahiere die erforderlichen Frames
    audio_bytes = bytes(audio_blob_array[()])
    
    # FLAC-Daten in temporäre Datei schreiben für SoundFile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name
    
    try:
        with sf.SoundFile(temp_file_path) as sf_file:
            # Setze Position auf den Anfang des ersten benötigten Frames
            first_frame_sample = sample_positions[start_idx]
            sf_file.seek(first_frame_sample)
            
            # Berechne wie viele Samples zu lesen sind
            # Wenn wir den letzten Frame im Index haben, lesen wir bis zum Ende
            if end_idx < len(flac_index) - 1:
                last_frame_end = sample_positions[end_idx + 1]
            else:
                last_frame_end = sf_file.frames
                
            total_samples_to_read = last_frame_end - first_frame_sample
            frames_data = sf_file.read(total_samples_to_read, dtype=dtype)
            
            # Schneide genau auf den angeforderten Bereich zu
            start_offset = start_sample - first_frame_sample
            end_offset = start_offset + (end_sample - start_sample + 1)  # inklusiv
            
            # Stelle sicher, dass wir nicht aus dem Array herausgehen
            if start_offset < 0:
                start_offset = 0
            if end_offset > frames_data.shape[0]:
                end_offset = frames_data.shape[0]
                
            return frames_data[start_offset:end_offset]
                
    except Exception as e:
        logger.error(f"Fehler beim Dekodieren der FLAC-Daten: {e}")
        return np.array([])
    finally:
        # Temporäre Datei entfernen
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def parallel_extract_audio_segments_flac(zarr_group, audio_blob_array, segments, dtype=np.int16, max_workers=4):
    """
    Dekodiert mehrere Bereiche parallel aus einer FLAC-Datei
    
    Args:
        zarr_group: Zarr-Gruppe, die den Index enthält
        audio_blob_array: Array mit den FLAC-Audiodaten
        segments: Liste von (start_sample, end_sample)-Tupeln
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        max_workers: Maximale Anzahl paralleler Worker
        
    Returns:
        Liste von np.ndarray: Dekodierte Audiodaten für jeden angeforderten Bereich
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(extract_audio_segment_flac, zarr_group, audio_blob_array, start, end, dtype): (start, end) 
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
