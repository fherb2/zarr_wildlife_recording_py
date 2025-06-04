

import zarr
import numpy as np
import io
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os


class FLACIndexer:
    """Indexer für native FLAC-Dateien"""
    
    def __init__(self, zarr_path, audio_array_path):
        """
        Initialisiert den Indexer
        
        Args:
            zarr_path: Pfad zum Zarr-Store
            audio_array_path: Pfad zum Array innerhalb des Zarr-Stores, das die Audio-Bytes enthält
        """
        self.zarr_store = zarr.open(zarr_path, mode='r+')
        self.audio_bytes = self.zarr_store[audio_array_path]
        
        # Hole Metadaten aus den Attributen
        self.sample_rate = self.audio_bytes.attrs.get('sample_rate', 44100)
        self.channels = self.audio_bytes.attrs.get('channels', 1)
        
    def build_index(self):
        """
        Erstellt einen Index, der Sample-Positionen zu FLAC-Frames zuordnet
        
        Returns:
            Dictionary mit Indexdaten
        """
        audio_bytes = bytes(self.audio_bytes[()])
        
        # FLAC-Stream in temporäre Datei umleiten für SoundFile
        with io.BytesIO(audio_bytes) as buf:
            with sf.SoundFile(buf) as f:
                # FLAC-Header und Metadaten überspringen
                frames_info = []
                offset = f.seek(0)  # Aktuelle Position nach dem Header
                
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
            
            self.zarr_store.create_dataset('flac_index', data=index_array, 
                                          chunks=(min(1000, len(frames_info)),))
            
            # Speichere zusätzliche Metadaten
            self.zarr_store['flac_index'].attrs['sample_rate'] = self.sample_rate
            self.zarr_store['flac_index'].attrs['channels'] = self.channels
            
            return index_array
        else:
            raise ValueError("Konnte keine FLAC-Frames im Audio finden")
    
    def find_frames_for_samples(self, start_sample, end_sample):
        """
        Findet FLAC-Frames, die die angegebenen Samples enthalten
        
        Args:
            start_sample: Erstes Sample, das dekodiert werden soll
            end_sample: Letztes Sample, das dekodiert werden soll
            
        Returns:
            Liste der Frame-Indizes, die dekodiert werden müssen
        """
        index_array = self.zarr_store['flac_index']
        
        # Binary Search für den ersten Frame, der start_sample enthält oder darüber liegt
        sample_positions = index_array['sample_pos']
        start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
        start_idx = max(0, start_idx)  # Sicherstellen, dass wir nicht unter 0 gehen
        
        # Finde alle Frames bis zum letzten benötigten
        end_idx = np.searchsorted(sample_positions, end_sample, side='right')
        
        # Sichere Obergrenze
        end_idx = min(end_idx, len(index_array) - 1)
        
        # Frames, die dekodiert werden müssen
        required_frames = list(range(start_idx, end_idx + 1))
            
        return required_frames


# Beispiel zur Verwendung
def create_flac_index(zarr_path, audio_array_path):
    """Erstellt einen Index für eine native FLAC-Datei in einem Zarr-Store"""
    indexer = FLACIndexer(zarr_path, audio_array_path)
    index = indexer.build_index()
    print(f"FLAC-Index erstellt mit {len(index)} Frames")
    return index

# Beispielaufruf:
# create_flac_index('mein_zarr_store.zarr', 'audio/flac_bytes')




class FLACDecoder:
    """Decoder für native FLAC-Dateien"""
    
    def __init__(self, zarr_path, audio_array_path='audio/flac_bytes', index_path='flac_index'):
        """
        Initialisiert den Decoder
        
        Args:
            zarr_path: Pfad zum Zarr-Store
            audio_array_path: Pfad zum Array innerhalb des Zarr-Stores, das die Audio-Bytes enthält
            index_path: Pfad zum Index-Array innerhalb des Zarr-Stores
        """
        self.zarr_store = zarr.open(zarr_path, mode='r')
        self.audio_bytes = self.zarr_store[audio_array_path]
        self.index = self.zarr_store[index_path]
        
        # Metadaten aus dem Index holen
        self.sample_rate = self.index.attrs.get('sample_rate', 44100)
        self.channels = self.index.attrs.get('channels', 1)
        
    def decode_frames(self, frame_indices):
        """
        Dekodiert FLAC-Frames basierend auf den bereitgestellten Indizes
        
        Args:
            frame_indices: Liste der Frame-Indizes, die dekodiert werden sollen
            
        Returns:
            Numpy-Array mit dekodierten Audiodaten
        """
        if not frame_indices:
            return np.array([])
            
        audio_data = bytes(self.audio_bytes[()])
        
        # Wir müssen die FLAC-Metadaten (Header) beibehalten, damit die Dekodierung funktioniert
        # Also schreiben wir die komplette Datei in einen temporären Puffer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            with sf.SoundFile(temp_file_path) as sf_file:
                # Sammle alle benötigten Frames
                frames_data = []
                
                for idx in frame_indices:
                    byte_offset = self.index[idx]['byte_offset']
                    sample_pos = self.index[idx]['sample_pos']
                    
                    # Setze Position auf den Anfang des Frames
                    sf_file.seek(sample_pos)
                    
                    # Berechne Anzahl der Samples in diesem Frame
                    if idx < len(self.index) - 1:
                        next_sample_pos = self.index[idx + 1]['sample_pos']
                        num_samples = next_sample_pos - sample_pos
                    else:
                        # Letzter Frame - Lese bis zum Ende
                        num_samples = sf_file.frames - sample_pos
                    
                    # Lese die Audiodaten
                    frame_data = sf_file.read(num_samples)
                    frames_data.append(frame_data)
                
                # Kombiniere alle Frames
                if frames_data:
                    return np.vstack(frames_data) if len(frames_data) > 1 else frames_data[0]
                else:
                    return np.array([])
                    
        except Exception as e:
            print(f"Fehler beim Dekodieren der FLAC-Daten: {e}")
            return np.array([])
        finally:
            # Temporäre Datei entfernen
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def decode_samples(self, start_sample, end_sample):
        """
        Dekodiert einen spezifischen Bereich von Samples
        
        Args:
            start_sample: Erstes Sample, das dekodiert werden soll (inklusive)
            end_sample: Letztes Sample, das dekodiert werden soll (exklusive)
            
        Returns:
            Numpy-Array mit dekodierten Audiodaten
        """
        # Finde die Frames, die die angeforderten Samples enthalten
        sample_positions = self.index['sample_pos']
        
        # Binary Search für den ersten Frame, der start_sample enthält oder darüber liegt
        start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
        start_idx = max(0, start_idx)  # Sicherstellen, dass wir nicht unter 0 gehen
        
        # Finde alle Frames bis zum letzten benötigten
        end_idx = np.searchsorted(sample_positions, end_sample, side='right')
        end_idx = min(end_idx, len(self.index) - 1)
        
        # Frames, die dekodiert werden müssen
        required_frames = list(range(start_idx, end_idx + 1))
        
        # Dekodiere die Frames
        decoded_audio = self.decode_frames(required_frames)
        
        if decoded_audio.size == 0:
            return np.array([])
            
        # Schneide die resultierenden Audiodaten auf den genauen Bereich zu
        first_frame_sample = sample_positions[start_idx]
        start_offset = start_sample - first_frame_sample
        end_offset = start_offset + (end_sample - start_sample)
        
        # Stelle sicher, dass wir nicht aus dem Array herausgehen
        if start_offset < 0:
            start_offset = 0
        if end_offset > decoded_audio.shape[0]:
            end_offset = decoded_audio.shape[0]
            
        trimmed_audio = decoded_audio[start_offset:end_offset]
        return trimmed_audio
    
    def parallel_decode_chunks(self, chunks, max_workers=4):
        """
        Dekodiert mehrere Chunks parallel
        
        Args:
            chunks: Liste von (start_sample, end_sample)-Tupeln
            max_workers: Maximale Anzahl paralleler Worker
            
        Returns:
            Liste von Numpy-Arrays mit dekodierten Audiodaten
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.decode_samples, start, end): (start, end) 
                for start, end in chunks
            }
            
            results = {}
            for future in future_to_chunk:
                chunk = future_to_chunk[future]
                try:
                    results[chunk] = future.result()
                except Exception as e:
                    print(f"Fehler beim Dekodieren des Chunks {chunk}: {e}")
                    results[chunk] = np.array([])
                    
            # Sortiere nach der ursprünglichen Chunk-Reihenfolge
            return [results[chunk] for chunk in chunks]


# Beispiel zur Verwendung
def decode_flac_snippet(zarr_path, start_sample, end_sample):
    """Dekodiert einen bestimmten Bereich aus einer nativen FLAC-Datei"""
    decoder = FLACDecoder(zarr_path)
    audio = decoder.decode_samples(start_sample, end_sample)
    print(f"Dekodierte Audiodaten: {audio.shape}")
    return audio

def parallel_decode_flac_snippets(zarr_path, chunks, max_workers=4):
    """Dekodiert mehrere Bereiche parallel aus einer nativen FLAC-Datei"""
    decoder = FLACDecoder(zarr_path)
    audio_chunks = decoder.parallel_decode_chunks(chunks, max_workers)
    return audio_chunks

# Beispielaufruf:
# audio = decode_flac_snippet('mein_zarr_store.zarr', 10000, 20000)
# 
# # Parallel mehrere Chunks dekodieren
# chunks = [(10000, 20000), (30000, 40000), (50000, 60000)]
# audio_chunks = parallel_decode_flac_snippets('mein_zarr_store.zarr', chunks)

