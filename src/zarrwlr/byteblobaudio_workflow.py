import zarr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tempfile
import subprocess
import json





# Vollständiges Beispiel, das den gesamten Workflow zeigt

def create_zarr_store_with_audio(audio_file_path, output_zarr_path, compress_format='flac'):
    """
    Erstellt einen neuen Zarr-Store mit komprimierten Audiodaten (FLAC oder Opus)
    
    Args:
        audio_file_path: Pfad zur Eingabe-Audiodatei (z.B. WAV)
        output_zarr_path: Ausgabepfad für den Zarr-Store
        compress_format: 'flac' oder 'opus'
    
    Returns:
        Pfad zum erstellten Zarr-Store
    """
    # Erstelle Zarr-Store
    store = zarr.open(output_zarr_path, mode='w')
    
    # Audiodaten mit FFmpeg komprimieren
    with tempfile.NamedTemporaryFile(suffix=f'.{compress_format}') as temp_file:
        if compress_format == 'flac':
            # Nativ FLAC (ohne Ogg-Container)
            subprocess.run([
                'ffmpeg', '-i', audio_file_path, 
                '-c:a', 'flac', '-compression_level', '8',
                temp_file.name
            ], check=True)
        else:  # opus
            # Opus in Ogg-Container
            subprocess.run([
                'ffmpeg', '-i', audio_file_path, 
                '-c:a', 'libopus', '-b:a', '64k',
                temp_file.name
            ], check=True)
        
        # Metadaten auslesen
        probe_result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', temp_file.name
        ], capture_output=True, text=True, check=True)
        
        info = json.loads(probe_result.stdout)
        
        # Extrahiere Metadaten
        for stream in info['streams']:
            if stream['codec_type'] == 'audio':
                sample_rate = int(stream.get('sample_rate', 44100))
                channels = int(stream.get('channels', 1))
                break
        
        # Komprimierte Audiodaten als Bytes lesen
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
    # Bytes als Zarr-Array speichern
    audio_array_path = f'audio/{compress_format}_bytes'
    audio_array = store.create_dataset(audio_array_path, 
                                      data=np.frombuffer(audio_bytes, dtype=np.uint8),
                                      chunks=(min(1024*1024, len(audio_bytes)),))  # 1MB Chunks
    
    # Metadaten speichern
    audio_array.attrs['sample_rate'] = sample_rate
    audio_array.attrs['channels'] = channels
    audio_array.attrs['format'] = compress_format
    
    # Index erstellen
    if compress_format == 'flac':
        from indexing_code import FLACIndexer
        indexer = FLACIndexer(output_zarr_path, audio_array_path)
    else:  # opus
        from indexing_code import OggOpusIndexer
        indexer = OggOpusIndexer(output_zarr_path, audio_array_path)
    
    index = indexer.build_index()
    print(f"Index für {compress_format} erstellt mit {len(index)} Einträgen")
    
    return output_zarr_path


def demo_complete_workflow():
    """
    Demonstriert den vollständigen Workflow:
    1. Audiodaten in Zarr importieren und komprimieren
    2. Index erstellen
    3. Zufällige Audioschnipsel parallel dekodieren
    """
    # Hier wäre ein Pfad zu einer WAV-Datei für den Test
    input_audio = "/pfad/zu/beispiel.wav"
    
    # Workflow für FLAC
    flac_zarr_path = create_zarr_store_with_audio(
        input_audio, 
        "audio_flac_demo.zarr", 
        compress_format='flac'
    )
    
    # Workflow für Opus
    opus_zarr_path = create_zarr_store_with_audio(
        input_audio, 
        "audio_opus_demo.zarr", 
        compress_format='opus'
    )
    
    # Metadaten und Länge auslesen für Beispielbereiche
    flac_store = zarr.open(flac_zarr_path, mode='r')
    flac_index = flac_store['flac_index']
    flac_total_samples = flac_index[-1]['sample_pos']
    flac_sample_rate = flac_index.attrs['sample_rate']
    
    # 5 zufällige 2-Sekunden-Schnipsel definieren
    import random
    chunks = []
    for _ in range(5):
        start = random.randint(0, int(flac_total_samples - 2 * flac_sample_rate))
        end = start + int(2 * flac_sample_rate)  # 2 Sekunden
        chunks.append((start, end))
    
    print(f"Dekodiere 5 zufällige 2-Sekunden-Schnipsel...")
    
    # Parallel dekodieren mit beiden Formaten
    from decoding_code import FLACDecoder, OggOpusDecoder
    
    flac_decoder = FLACDecoder(flac_zarr_path)
    flac_chunks = flac_decoder.parallel_decode_chunks(chunks, max_workers=4)
    
    opus_decoder = OggOpusDecoder(opus_zarr_path)
    opus_chunks = opus_decoder.parallel_decode_chunks(chunks, max_workers=4)
    
    # Ergebnisse anzeigen
    for i, ((start, end), flac_audio, opus_audio) in enumerate(zip(chunks, flac_chunks, opus_chunks)):
        duration = (end - start) / flac_sample_rate
        print(f"Chunk {i+1}: Samples {start}-{end} ({duration:.2f}s)")
        print(f"  FLAC: {flac_audio.shape} Samples")
        print(f"  Opus: {opus_audio.shape} Samples")
    
    print("Workflow abgeschlossen!")















# Beispiel für einen direkten Anwendungsfall

def beispiel_anwendungsfall():
    """
    Realistisches Anwendungsbeispiel: Eine lange Audiodatei wird analysiert,
    wobei verschiedene Segmente parallel verarbeitet werden.
    """
    zarr_path = "audio_flac_demo.zarr"  # Bereits erstellter Store
    
    # Analyzer-Funktion (Beispiel: Berechnet RMS-Pegel)
    def analyze_audio_segment(audio_data):
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return rms
    
    # FLAC-Decoder initialisieren
    from decoding_code import FLACDecoder
    decoder = FLACDecoder(zarr_path)
    
    # Metadaten abrufen
    store = zarr.open(zarr_path, mode='r')
    index = store['flac_index']
    total_samples = index[-1]['sample_pos']
    sample_rate = index.attrs['sample_rate']
    
    # Gesamte Audiodatei in 3-Sekunden-Segmente unterteilen
    segment_length = 3 * sample_rate
    segments = []
    
    for start in range(0, total_samples, segment_length):
        end = min(start + segment_length, total_samples)
        segments.append((start, end))
    
    print(f"Analysiere {len(segments)} Segmente von je 3 Sekunden...")
    
    # Alle Segmente parallel dekodieren
    audio_segments = decoder.parallel_decode_chunks(segments, max_workers=8)
    
    # Jedes Segment analysieren
    with ThreadPoolExecutor(max_workers=8) as executor:
        analysis_results = list(executor.map(analyze_audio_segment, audio_segments))
    
    # Ergebnisse verarbeiten
    for i, (segment, result) in enumerate(zip(segments, analysis_results)):
        start_time = segment[0] / sample_rate
        end_time = segment[1] / sample_rate
        print(f"Segment {i+1}: {start_time:.1f}s - {end_time:.1f}s, RMS: {result:.6f}")
    
    print("Analyse abgeschlossen!")


if __name__ == "__main__":
    # Uncomment to run the examples
    # demo_complete_workflow()
    # beispiel_anwendungsfall()
    pass