

import zarr
import numpy as np
import av # is PyAV
import io
from concurrent.futures import ThreadPoolExecutor
      
def build_ogg_opus_index(audio_blob_group: zarr.Group, audio_blob_array: zarr.Array):
    """
    Erstellt einen Index, der Sample-Positionen zu Ogg-Pages zuordnet
    
    Returns:
        Dictionary mit Indexdaten
    """

    # Ogg-Page-Header identifizieren ("OggS")
    page_positions = []
    pos = 0
    while pos < len(audio_blob_array.shape[0]):
        if audio_blob_array[pos:pos+4] == b'OggS':
            # Speichere Position und Parse Header
        #    header_type = self.audio_blob_array[pos+5]
            granule_pos = int.from_bytes(audio_blob_array[pos+6:pos+14], byteorder='little', signed=False)
            
            # Berechne Sampleposition basierend auf Granule Position
            sample_pos = granule_pos
            
            # Anzahl der Segmente in dieser Page
            num_segments = int(audio_blob_array[pos+26])
            
            # Länge der Segmentgrößen-Tabelle
            segment_table_length = num_segments
            
            # Berechne Gesamtlänge der Page
            page_size = 27 + segment_table_length
            for i in range(num_segments):
                page_size += int(audio_blob_array[pos+27+i])
            
            page_positions.append({
                'byte_offset': pos,
                'page_size': page_size,
                'sample_pos': sample_pos,
                'granule_pos': granule_pos
            })
            
            pos += page_size
        else:
            pos += 1
    
    # Erstelle Index-Array
    index_array = np.array([(p['byte_offset'], p['page_size'], p['sample_pos']) 
                            for p in page_positions], 
                            dtype=[('byte_offset', np.int64), 
                                    ('page_size', np.int32),
                                    ('sample_pos', np.int64)])
    
    index_array = audio_blob_group.create_array('index', shape=index_array.shape, dtype=index_array.dtype, 
                        chunks=(min(int(1e6), len(page_positions)),))
    index_array.append(index_array)
    index_array.attrs["index_type"] = "ogg_opus_index"
        
    return index_array
    


def find_pages_for_samples(self, index_array:zarr.Array, start_sample:int, end_sample:int):
    """
    Findet Ogg-Pages, die die angegebenen Samples enthalten
    
    Args:
        start_sample: Erstes Sample, das dekodiert werden soll
        end_sample: Letztes Sample, das dekodiert werden soll
        
    Returns:
        Liste der Page-Indizes, die dekodiert werden müssen
    """
    
    # Binary Search für die erste Page, die start_sample enthält oder darüber liegt
    sample_positions = index_array['sample_pos'] Das ist noch Unsinn.
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)  # Sicherstellen, dass wir nicht unter 0 gehen
    
    # Finde alle Pages bis zur letzten benötigten
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    
    # Sichere Obergrenze
    end_idx = min(end_idx, len(index_array) - 1)
    
    # Pages, die dekodiert werden müssen (inkl. Überlappung für korrekte Dekodierung)
    required_pages = list(range(start_idx, end_idx + 1))
    
    # Für Opus: Wir brauchen möglicherweise eine vorherige Page für die korrekte Dekodierung
    if start_idx > 0:
        required_pages.insert(0, start_idx - 1)
        
    return required_pages



# Beispiel zur Verwendung
def create_opus_index(zarr_audio_blob_grp, zarr_audio_blob_array):
    """Erstellt einen Index für eine Ogg-Opus-Datei in einem Zarr-Store"""
    index = build_ogg_opus_index(zarr_audio_blob_grp, zarr_audio_blob_array)
    print(f"Opus-Index erstellt mit {len(index)} Pages")
    return index

# Beispielaufruf:
# create_opus_index('mein_zarr_store.zarr', 'audio/opus_bytes')


# NACHFOLGEND: Ebenfalls in einzelne Funktionen teilen, wenn für flac funktionstüchtig.


class OggOpusDecoder:
    """Decoder für Ogg-Container mit Opus-Codec"""
    
    def __init__(self, zarr_path, audio_array_path='audio/opus_bytes', index_path='opus_index'):
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
        self.sample_rate = self.index.attrs.get('sample_rate', 48000)
        self.channels = self.index.attrs.get('channels', 1)
    
    def decode_pages(self, page_indices):
        """
        Dekodiert Ogg-Pages basierend auf den bereitgestellten Indizes
        
        Args:
            page_indices: Liste der Page-Indizes, die dekodiert werden sollen
            
        Returns:
            Numpy-Array mit dekodierten Audiodaten
        """
        if not page_indices:
            return np.array([])
            
        # Sammle alle benötigten Pages
        audio_data = bytes(self.audio_bytes[()])
        pages_data = bytearray()
        
        for idx in page_indices:
            byte_offset = self.index[idx]['byte_offset']
            page_size = self.index[idx]['page_size']
            pages_data.extend(audio_data[byte_offset:byte_offset+page_size])
        
        # PyAV für Dekodierung verwenden
        with io.BytesIO(pages_data) as buf:
            output_frames = []
            
            try:
                container = av.open(buf)
                stream = container.streams.audio[0]
                
                for frame in container.decode(stream):
                    # Konvertiere zu einem Numpy-Array
                    frame_array = frame.to_ndarray()
                    output_frames.append(frame_array)
                    
                container.close()
                
                if output_frames:
                    # Kombination aller Frames
                    decoded_audio = np.vstack(output_frames) if len(output_frames) > 1 else output_frames[0]
                    return decoded_audio
                else:
                    return np.array([])
                    
            except Exception as e:
                print(f"Fehler beim Dekodieren der Opus-Daten: {e}")
                return np.array([])
    
    def decode_samples(self, start_sample, end_sample):
        """
        Dekodiert einen spezifischen Bereich von Samples
        
        Args:
            start_sample: Erstes Sample, das dekodiert werden soll (inklusive)
            end_sample: Letztes Sample, das dekodiert werden soll (exklusive)
            
        Returns:
            Numpy-Array mit dekodierten Audiodaten
        """
        # Finde die Pages, die die angeforderten Samples enthalten
        sample_positions = self.index['sample_pos']
        
        # Binary Search für die erste Page, die start_sample enthält oder darüber liegt
        start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
        start_idx = max(0, start_idx)  # Sicherstellen, dass wir nicht unter 0 gehen
        
        # Finde alle Pages bis zur letzten benötigten
        end_idx = np.searchsorted(sample_positions, end_sample, side='right')
        end_idx = min(end_idx, len(self.index) - 1)
        
        # Pages, die dekodiert werden müssen (inkl. Überlappung für korrekte Dekodierung)
        required_pages = list(range(start_idx, end_idx + 1))
        
        # Für Opus: Wir brauchen möglicherweise eine vorherige Page für die korrekte Dekodierung
        if start_idx > 0:
            required_pages.insert(0, start_idx - 1)
        
        # Dekodiere die Pages
        decoded_audio = self.decode_pages(required_pages)
        
        if decoded_audio.size == 0:
            return np.array([])
            
        # Schneide die resultierenden Audiodaten auf den genauen Bereich zu
        first_page_sample = sample_positions[start_idx]
        start_offset = start_sample - first_page_sample
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
def decode_opus_snippet(zarr_path, start_sample, end_sample):
    """Dekodiert einen bestimmten Bereich aus einer Ogg-Opus-Datei"""
    decoder = OggOpusDecoder(zarr_path)
    audio = decoder.decode_samples(start_sample, end_sample)
    print(f"Dekodierte Audiodaten: {audio.shape}")
    return audio

def parallel_decode_opus_snippets(zarr_path, chunks, max_workers=4):
    """Dekodiert mehrere Bereiche parallel aus einer Ogg-Opus-Datei"""
    decoder = OggOpusDecoder(zarr_path)
    audio_chunks = decoder.parallel_decode_chunks(chunks, max_workers)
    return audio_chunks


# Beispielaufruf:
# audio = decode_opus_snippet('mein_zarr_store.zarr', 10000, 20000)

# # Parallel mehrere Chunks dekodieren
# chunks = [(10000, 20000), (30000, 40000), (50000, 60000)]
# audio_chunks = parallel_decode_opus_snippets('mein_zarr_store.zarr', chunks)