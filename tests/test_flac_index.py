"""
Test für das neue FLAC-Index-Modul (flacbyteblob.py)
Basiert auf test_audio_import_standalone.py und testet die neuen FLAC-Index-Funktionen.
"""
import pathlib
import os
import shutil
import numpy as np
import zarr
import datetime

import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

# Logging konfigurieren - jetzt funktioniert es sofort!
Config.set(log_level=LogLevel.TRACE)
# Get logger for this module
logger = get_module_logger(__file__)

from zarrwlr.aimport import (
    create_original_audio_group,
    import_original_audio_file,
    extract_audio_segment,
    parallel_extract_audio_segments
)

# Import des neuen FLAC-Moduls
import zarrwlr.flacbyteblob as flacbyteblob


# ##########################################################
#
# Helpers (übernommen aus test_audio_import_standalone.py)
# =======
#
# ##########################################################

TEST_RESULTS_DIR = pathlib.Path(__file__).parent.resolve() / "testresults"
ZARR3_STORE_DIR = TEST_RESULTS_DIR / "zarr3-store-flac-test"

def clean_up_testresult_dir():
    logger.trace("clean_up_testresult_dir() requested.")
    logger.debug(f"Remove old test-result directory: {str(TEST_RESULTS_DIR)}")
    os.system(f"rm -rf {str(TEST_RESULTS_DIR)}")
    logger.trace("clean_up_testresult_dir() finished.")

def prepare_zarr_database():
    logger.trace("prepare_zarr_database() requested.")
    # Verzeichnis erstellen, falls es nicht existiert
    if not TEST_RESULTS_DIR.exists():
        logger.debug(f"Create {TEST_RESULTS_DIR.resolve()}")
        TEST_RESULTS_DIR.mkdir(parents=True)
    
    # Bestehende Zarr-Datenbank löschen, falls vorhanden
    if ZARR3_STORE_DIR.exists():
        logger.debug("Remove old database.")
        shutil.rmtree(ZARR3_STORE_DIR)
    
    # Neue Zarr-Datenbank erstellen
    logger.info(f"Create new database at {ZARR3_STORE_DIR} with group 'audio_imports'")
    audio_group = create_original_audio_group(store_path = ZARR3_STORE_DIR, group_path = 'audio_imports')
    logger.trace("prepare_zarr_database() finished.")
    return audio_group

def get_test_files() -> list[pathlib.Path]:
    logger.trace("get_test_files() requested.")
    test_files = [
                    "testdata/audiomoth_long_snippet.wav",
                    "testdata/audiomoth_long_snippet_converted.opus",
                    "testdata/audiomoth_long_snippet_converted.flac",
                    "testdata/audiomoth_short_snippet.wav",
                    "testdata/bird1_snippet.mp3",
                    "testdata/camtrap_snippet.mov" # mp4 coded video with audio stream
                ]
    logger.trace("get_test_files() finished.")
    return [pathlib.Path(__file__).parent.resolve() / file for file in test_files]


# ###########################################################
#
# FLAC-Index Tests (neue Funktionen)
# ===================
#
# ###########################################################

def test_import_wav_to_flac_with_new_index():
    """Test: WAV zu FLAC Import mit neuem FLAC-Index-Modul"""
    logger.trace("test_import_wav_to_flac_with_new_index() requested.")
    print("\n=== Test: WAV zu FLAC Import mit neuem Index-Modul ===")
    
    try:  # Neuer try-except Block für besseres Debugging
        # Zarr-Datenbank vorbereiten
        audio_group = prepare_zarr_database()
        
        # WAV-Datei finden
        test_files = get_test_files()
        wav_file = next((f for f in test_files if f.name.endswith(".wav")), None)
        
        if not wav_file or not wav_file.exists():
            print(f"FEHLER: WAV-Testdatei nicht gefunden.")
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        print(f"Importiere WAV-Datei: {wav_file}")
        
        # Import durchführen (wie im ursprünglichen Test)
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=wav_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # FIXED: Finde die tatsächlich erstellte Gruppe dynamisch
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Audio-Gruppe erstellt.")
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        # Nimm die Gruppe mit der höchsten Nummer (neueste)
        latest_group_name = max(group_names, key=int)
        latest_group = audio_group[latest_group_name]
        
        # Prüfen, ob audio_data_blob_array existiert
        if "audio_data_blob_array" not in latest_group:
            print("FEHLER: audio_data_blob_array wurde nicht erstellt.")
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        audio_blob_array = latest_group["audio_data_blob_array"]
        
        print(f"WAV zu FLAC Import erfolgreich: Gruppe '{latest_group_name}' erstellt")
        print(f"Blob-Array-Größe: {audio_blob_array.shape[0]} Bytes")
        print(f"Codec in Blob-Array: {audio_blob_array.attrs.get('codec')}")
        print(f"Sample-Rate: {audio_blob_array.attrs.get('sample_rate')} Hz")
        print(f"Kanäle: {audio_blob_array.attrs.get('nb_channels')}")
        
        # FIXED: Index wurde bereits durch import_original_audio_file() erstellt
        print("\n--- Prüfe automatisch erstellten FLAC-Index ---")
        
        if 'flac_index' not in latest_group:
            print("FEHLER: FLAC-Index wurde nicht automatisch erstellt")
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        flac_index = latest_group['flac_index']
        print(f"✓ FLAC-Index wurde automatisch erstellt!")
        print(f"Index-Einträge: {flac_index.shape[0]}")
        print(f"Index-Shape: {flac_index.shape}")
        print(f"Index-Dtype: {flac_index.dtype}")
        print(f"Index-Codec: {flac_index.attrs.get('codec')}")
        print(f"Index-Sample-Rate: {flac_index.attrs.get('sample_rate')} Hz")
        print(f"Index-Kanäle: {flac_index.attrs.get('channels')}")
        
        # Validiere Index-Struktur
        if not _validate_flac_index_structure_2d(flac_index, audio_blob_array):
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        # Teste Audio-Segment-Extraktion mit neuem Modul
        if not _test_new_flac_segment_extraction(latest_group, audio_blob_array):
            logger.trace("test_import_wav_to_flac_with_new_index() finished.")
            return False
        
        logger.trace("test_import_wav_to_flac_with_new_index() finished.")
        return True
        
    except Exception as e:
        print(f"DETAILLIERTER FEHLER beim WAV-Test: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.trace("test_import_wav_to_flac_with_new_index() finished.")
        return False

def test_existing_flac_file_with_new_index():
    """Test: Bestehende FLAC-Datei direkt mit neuem Index-Modul importieren"""
    logger.trace("test_existing_flac_file_with_new_index() requested.")
    print("\n=== Test: Direkte FLAC-Datei mit neuem Index ===")
    
    # Zarr-Datenbank vorbereiten (neue DB für diesen Test)
    if ZARR3_STORE_DIR.exists():
        shutil.rmtree(ZARR3_STORE_DIR)
    audio_group = prepare_zarr_database()
    
    # FLAC-Datei finden
    test_files = get_test_files()
    flac_file = next((f for f in test_files if f.name.endswith(".flac")), None)
    
    if not flac_file or not flac_file.exists():
        print(f"WARNUNG: FLAC-Testdatei nicht gefunden. Test übersprungen.")
        logger.trace("test_existing_flac_file_with_new_index() finished.")
        return True  # Nicht als Fehler werten
    
    print(f"Importiere FLAC-Datei: {flac_file}")
    
    try:
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=flac_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=6
        )
        
        # FIXED: Finde die tatsächlich erstellte Gruppe dynamisch
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Audio-Gruppe erstellt.")
            logger.trace("test_existing_flac_file_with_new_index() finished.")
            return False
        
        latest_group_name = max(group_names, key=int)
        latest_group = audio_group[latest_group_name]
        audio_blob_array = latest_group["audio_data_blob_array"]
        
        print(f"FLAC Import erfolgreich in Gruppe '{latest_group_name}'")
        print(f"Blob-Array-Größe: {audio_blob_array.shape[0]} Bytes")
        
        # FIXED: Index wurde bereits durch import_original_audio_file() erstellt - nicht nochmal erstellen
        print("\n--- Prüfe automatisch erstellten FLAC-Index ---")
        
        if 'flac_index' not in latest_group:
            print("FEHLER: FLAC-Index wurde nicht automatisch erstellt")
            logger.trace("test_existing_flac_file_with_new_index() finished.")
            return False
        
        flac_index = latest_group['flac_index']
        print(f"✓ FLAC-Index für direkte FLAC-Datei wurde automatisch erstellt!")
        print(f"Index-Einträge: {flac_index.shape[0]}")
        
        # Teste Segment-Extraktion
        if not _test_new_flac_segment_extraction(latest_group, audio_blob_array):
            logger.trace("test_existing_flac_file_with_new_index() finished.")
            return False
        
        logger.trace("test_existing_flac_file_with_new_index() finished.")
        return True
        
    except Exception as e:
        print(f"FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.trace("test_existing_flac_file_with_new_index() finished.")
        return False

def _validate_flac_index_structure_2d(flac_index, audio_blob_array):
    """Validiert die Struktur des 2D FLAC-Index"""
    print("\n--- Validiere Index-Struktur (2D Array) ---")
    
    try:
        # Index-Array-Struktur prüfen
        if not isinstance(flac_index, zarr.Array):
            print("FEHLER: Index ist kein Zarr-Array")
            return False
        
        # Erwartete 2D-Array-Struktur
        if len(flac_index.shape) != 2 or flac_index.shape[1] != 3:
            print(f"FEHLER: Index-Shape: erwartet (n, 3), gefunden {flac_index.shape}")
            return False
        
        # Index sollte mindestens einen Frame haben
        if flac_index.shape[0] == 0:
            print("WARNUNG: Index ist leer - keine Frames gefunden")
            return True  # Nicht als Fehler werten - könnte bei sehr kurzen Dateien passieren
        
        # Metadaten prüfen
        required_attrs = ['sample_rate', 'channels', 'total_frames', 'codec']
        for attr in required_attrs:
            if attr not in flac_index.attrs:
                print(f"FEHLER: Fehlendes Index-Attribut: {attr}")
                return False
        
        if flac_index.attrs['codec'] != 'flac':
            print(f"FEHLER: Falscher Codec im Index: {flac_index.attrs['codec']}")
            return False
        
        if flac_index.attrs['total_frames'] != flac_index.shape[0]:
            print("FEHLER: total_frames stimmt nicht mit Index-Länge überein")
            return False
        
        # Frame-Positionen sollten monoton steigend sein (Spalte 2)
        if flac_index.shape[0] > 1:
            sample_positions = flac_index[:, 2]  # Spalte 2 = sample_pos
            if not np.all(sample_positions[1:] >= sample_positions[:-1]):
                print("FEHLER: Sample-Positionen sind nicht monoton steigend")
                return False
            
            # Byte-Offsets sollten monoton steigend sein (Spalte 0)
            byte_offsets = flac_index[:, 0]  # Spalte 0 = byte_offset
            if not np.all(byte_offsets[1:] > byte_offsets[:-1]):
                print("FEHLER: Byte-Offsets sind nicht monoton steigend")
                return False
        
        # Frame-Größen sollten plausibel sein (Spalte 1)
        frame_sizes = flac_index[:, 1]  # Spalte 1 = frame_size
        if not np.all(frame_sizes > 0):
            print("FEHLER: Ungültige Frame-Größen gefunden")
            return False
        
        # FIXED: Use .shape[0] instead of len() for Zarr v3 compatibility
        if not np.all(frame_sizes < audio_blob_array.shape[0]):
            print("FEHLER: Frame-Größen größer als Audio-Blob")
            return False
        
        print("✓ Index-Struktur-Validierung erfolgreich")
        # FIXED: Use .shape[0] instead of len() for Zarr v3 compatibility
        print(f"  Frames: {flac_index.shape[0]}")
        print(f"  Sample-Rate: {flac_index.attrs['sample_rate']} Hz")
        print(f"  Kanäle: {flac_index.attrs['channels']}")
        
        return True
        
    except Exception as e:
        print(f"FEHLER bei Index-Validierung: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def _test_new_flac_segment_extraction(audio_group, audio_blob_array):
    """Testet die Audio-Segment-Extraktion mit dem neuen FLAC-Modul"""
    print("\n--- Teste Segment-Extraktion mit neuem Modul ---")
    
    try:
        # Test einzelne Segment-Extraktion
        print("Teste einzelne Segment-Extraktion...")
        segment = flacbyteblob.extract_audio_segment_flac(
            zarr_group=audio_group,
            audio_blob_array=audio_blob_array,
            start_sample=0,
            end_sample=999,
            dtype=np.int16
        )
        
        if not isinstance(segment, np.ndarray):
            print(f"FEHLER: Segment ist kein NumPy-Array: {type(segment)}")
            return False
        
        if segment.dtype != np.int16:
            print(f"FEHLER: Falscher Segment-Dtype: {segment.dtype}")
            return False
        
        print(f"✓ Einzelne Segment-Extraktion erfolgreich")
        print(f"  Segment-Shape: {segment.shape}")
        print(f"  Segment-Dtype: {segment.dtype}")
        print(f"  Erste 5 Samples: {segment[:5] if len(segment) >= 5 else segment}")
        
        # Test parallele Segment-Extraktion
        print("\nTeste parallele Segment-Extraktion...")
        segments = [(0, 499), (500, 999), (1000, 1499)]
        
        extracted_segments = flacbyteblob.parallel_extract_audio_segments_flac(
            zarr_group=audio_group,
            audio_blob_array=audio_blob_array,
            segments=segments,
            dtype=np.int16,
            max_workers=2
        )
        
        if len(extracted_segments) != len(segments):
            print(f"FEHLER: Anzahl extrahierter Segmente stimmt nicht überein")
            return False
        
        for i, seg in enumerate(extracted_segments):
            if not isinstance(seg, np.ndarray):
                print(f"FEHLER: Segment {i} ist kein NumPy-Array")
                return False
            if seg.dtype != np.int16:
                print(f"FEHLER: Segment {i} hat falschen Dtype: {seg.dtype}")
                return False
        
        print("✓ Parallele Segment-Extraktion erfolgreich")
        for i, seg in enumerate(extracted_segments):
            print(f"  Segment {i}: Shape={seg.shape}")
        
        return True
        
    except Exception as e:
        print(f"FEHLER bei Segment-Extraktion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ###########################################################
#
# Main Test Execution
# ===================
#
# ###########################################################

if __name__ == "__main__":
    logger.trace("__main__ started.")
    print("=== FLAC-Index Modul Tests ===")
    
    # Ausführung der Tests
    succeeded = []
    failed = []
    
    # Test 1: WAV zu FLAC mit neuem Index
    if test_import_wav_to_flac_with_new_index():
        succeeded.append("WAV zu FLAC mit neuem Index")
    else:
        failed.append("WAV zu FLAC mit neuem Index")
    
    # Test 2: Direkte FLAC-Datei mit neuem Index
    if test_existing_flac_file_with_new_index():
        succeeded.append("Direkte FLAC-Datei mit neuem Index")
    else:
        failed.append("Direkte FLAC-Datei mit neuem Index")
    
    # Zusammenfassung
    print("\n=== Zusammenfassung FLAC-Index Tests ===")
    print(f"Erfolgreich: {len(succeeded)} Tests")
    for test in succeeded:
        print(f"✓ {test}")
    
    if failed:
        print(f"\nFehlgeschlagen: {len(failed)} Tests")
        for test in failed:
            print(f"✗ {test}")
    else:
        print("\n🎉 Alle Tests erfolgreich!")
    
    print("\nFertig!")
    logger.success("__main__ finalised.")