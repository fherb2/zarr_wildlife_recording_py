"""
Einfachere Standalone-Tests für das Audio-Import-System.
Dieses Skript führt grundlegende Tests ohne das unittest-Framework durch.
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


# ##########################################################
#
# Helpers
# =======
#
# ##########################################################

TEST_RESULTS_DIR = pathlib.Path(__file__).parent.resolve() / "testresults"
ZARR3_STORE_DIR = TEST_RESULTS_DIR / "zarr3-store"

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
# Tests
# =====
#
# ###########################################################

def test_import_wav_to_flac():
    """Test: WAV-Datei zu FLAC konvertieren und in Zarr importieren"""
    logger.trace("test_import_wav_to_flac() requested.")
    print("\n=== Test: WAV zu FLAC Import ===")
    
    # Zarr-Datenbank vorbereiten
    audio_group = prepare_zarr_database()
    
    # WAV-Datei finden
    test_files = get_test_files()
    wav_file = next((f for f in test_files if f.name.endswith(".wav")), None)
    
    if not wav_file or not wav_file.exists():
        print(f"FEHLER: WAV-Testdatei nicht gefunden.")
        logger.trace("test_import_wav_to_flac() finished.")
        return False
    
    print(f"Importiere WAV-Datei: {wav_file}")
    
    # Import durchführen
    try:
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=wav_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # Prüfen, ob die Gruppe erstellt wurde
        if "0" not in audio_group:
            print("FEHLER: Gruppe '0' wurde nicht erstellt.")
            logger.trace("test_import_wav_to_flac() finished.")
            return False
        
        group_0 = audio_group["0"]
        
        # Prüfen, ob der Index erstellt wurde
        if "flac_index" not in group_0:
            print("FEHLER: FLAC-Index wurde nicht erstellt.")
            logger.trace("test_import_wav_to_flac() finished.")
            return False
        
        # Prüfen, ob audio_data_blob_array existiert
        if "audio_data_blob_array" not in group_0:
            print("FEHLER: audio_data_blob_array wurde nicht erstellt.")
            logger.trace("test_import_wav_to_flac() finished.")
            return False
        
        # Detaillierte Informationen ausgeben
        flac_index = group_0['flac_index']
        blob_array = group_0['audio_data_blob_array']
        
        print(f"WAV zu FLAC Import erfolgreich: Gruppe '0' erstellt")
        print(f"Blob-Array-Größe: {blob_array.shape[0]} Bytes")
        print(f"FLAC-Index-Einträge: {flac_index.shape[0]}")
        print(f"Index-Shape: {flac_index.shape}")
        print(f"Codec: {blob_array.attrs.get('codec', 'unbekannt')}")
        print(f"Sample Rate: {blob_array.attrs.get('sample_rate', 'unbekannt')}")
        print(f"Channels: {blob_array.attrs.get('nb_channels', 'unbekannt')}")
        print(f"Compression Level: {blob_array.attrs.get('compression_level', 'unbekannt')}")
        
        logger.trace("test_import_wav_to_flac() finished.")
        return True
        
    except Exception as e:
        print(f"FEHLER beim Import: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.trace("test_import_wav_to_flac() finished.")
        return False


def test_extract_segment():
    """Test: Extrahieren eines Audiosegments aus der importierten Datei"""
    logger.trace("test_extract_segment() requested.")
    print("\n=== Test: Audiosegment extrahieren ===")
    
    # Zarr-Datenbank öffnen (setzt voraus, dass test_import_wav_to_flac vorher lief)
    if not ZARR3_STORE_DIR.exists():
        print("FEHLER: Zarr-Datenbank nicht gefunden. Führe zuerst test_import_wav_to_flac aus.")
        logger.trace("test_extract_segment() finished.")
        return False
    
    try:
        store = zarr.storage.LocalStore(str(ZARR3_STORE_DIR))
        root = zarr.open_group(store, mode='r')
        zarr_group = root['audio_imports']
    except Exception as e:
        print(f"FEHLER beim Öffnen der Zarr-Datenbank: {str(e)}")
        logger.trace("test_extract_segment() finished.")
        return False
    
    if "0" not in zarr_group:
        print("FEHLER: Gruppe '0' nicht gefunden. Führe zuerst test_import_wav_to_flac aus.")
        logger.trace("test_extract_segment() finished.")
        return False
    
    group_0 = zarr_group["0"]
    
    try:
        # Prüfe Index-Informationen
        if "flac_index" not in group_0:
            print("FEHLER: FLAC-Index nicht gefunden.")
            return False
            
        flac_index = group_0["flac_index"]
        blob_array = group_0["audio_data_blob_array"]
        
        print(f"Index verfügbar: {flac_index.shape[0]} Frames")
        print(f"Codec: {blob_array.attrs.get('codec', 'unbekannt')}")
        
        # Extrahiere ein kurzes Segment (ersten 1000 Samples)
        print("Extrahiere Samples 0-999...")
        segment = extract_audio_segment(group_0, 0, 999)
        
        if not isinstance(segment, np.ndarray):
            print(f"FEHLER: Ergebnis ist kein numpy-Array: {type(segment)}")
            logger.trace("test_extract_segment() finished.")
            return False
        
        if segment.size == 0:
            print("FEHLER: Extrahiertes Segment ist leer")
            logger.trace("test_extract_segment() finished.")
            return False
        
        print(f"Segment extrahiert: Shape={segment.shape}, Dtype={segment.dtype}")
        if segment.size > 0:
            print(f"Erste 5 Samples: {segment[:5] if segment.size >= 5 else segment}")
            print(f"Min/Max Werte: {segment.min()}/{segment.max()}")
        
        # Extrahiere mehrere Segmente parallel
        print("\nExtrahiere mehrere Segmente parallel...")
        segments = [(0, 999), (1000, 1999), (2000, 2999)]
        multi_segments = parallel_extract_audio_segments(group_0, segments)
        
        if len(multi_segments) != 3:
            print(f"FEHLER: Falsche Anzahl an extrahierten Segmenten: {len(multi_segments)}")
            logger.trace("test_extract_segment() finished.")
            return False
        
        for i, seg in enumerate(multi_segments):
            if isinstance(seg, np.ndarray) and seg.size > 0:
                print(f"Segment {i}: Shape={seg.shape}, Dtype={seg.dtype}, Samples={seg.size}")
            else:
                print(f"Segment {i}: Leer oder Fehler")
        
        logger.trace("test_extract_segment() finished.")
        return True
        
    except Exception as e:
        print(f"FEHLER beim Extrahieren: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.trace("test_extract_segment() finished.")
        return False


def test_import_mp3_to_opus():
    """Test: MP3-Datei zu Opus konvertieren und in Zarr importieren"""
    logger.trace("test_import_mp3_to_opus() requested.")
    print("\n=== Test: MP3 zu Opus Import ===")
    
    # Für diesen Test eine separate Zarr-Datenbank verwenden
    opus_test_dir = TEST_RESULTS_DIR / "zarr3-store-opus"
    if opus_test_dir.exists():
        shutil.rmtree(opus_test_dir)
    
    zarr_group = create_original_audio_group(store_path=opus_test_dir, group_path='audio_imports')
    print(f"Verwende separate Opus-Testdatenbank: {opus_test_dir}")
    
    # MP3-Datei finden
    test_files = get_test_files()
    mp3_file = next((f for f in test_files if f.name.endswith(".mp3")), None)
    
    if not mp3_file or not mp3_file.exists():
        print(f"FEHLER: MP3-Testdatei nicht gefunden.")
        logger.trace("test_import_mp3_to_opus() finished.")
        return False
    
    print(f"Importiere MP3-Datei: {mp3_file}")
    
    # Import durchführen
    try:
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=mp3_file,
            zarr_original_audio_group=zarr_group,
            first_sample_time_stamp=timestamp,
            target_codec='opus',
            opus_bitrate=128000  # 128 kbps
        )
        
        # Ermittle die neue Gruppe (könnte "0" oder "1" sein, je nach vorherigen Tests)
        group_names = [name for name in zarr_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Gruppe erstellt.")
            logger.trace("test_import_mp3_to_opus() finished.")
            return False
        
        # Nimm die Gruppe mit der höchsten Nummer
        latest_group_name = max(group_names, key=int)
        latest_group = zarr_group[latest_group_name]
        
        # Prüfen, ob der Index erstellt wurde
        if "ogg_page_index" not in latest_group:
            print("FEHLER: Ogg-Page-Index wurde nicht erstellt.")
            logger.trace("test_import_mp3_to_opus() finished.")
            return False
        
        # Prüfen, ob audio_data_blob_array existiert
        if "audio_data_blob_array" not in latest_group:
            print("FEHLER: audio_data_blob_array wurde nicht erstellt.")
            logger.trace("test_import_mp3_to_opus() finished.")
            return False
        
        blob_array = latest_group["audio_data_blob_array"]
        ogg_index = latest_group["ogg_page_index"]
        
        print(f"MP3 zu Opus Import erfolgreich: Gruppe '{latest_group_name}' erstellt")
        print(f"Blob-Array-Größe: {blob_array.shape[0]} Bytes")
        print(f"Ogg-Page-Index-Einträge: {ogg_index.shape[0]}")
        print(f"Codec: {blob_array.attrs.get('codec')}")
        print(f"Bitrate: {blob_array.attrs.get('opus_bitrate')} bit/s")
        print(f"Sample Rate: {blob_array.attrs.get('sample_rate')}")
        print(f"Channels: {blob_array.attrs.get('nb_channels')}")
        
        logger.trace("test_import_mp3_to_opus() finished.")
        return True
        
    except Exception as e:
        print(f"FEHLER beim Import: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.trace("test_import_mp3_to_opus() finished.")
        return False


def test_flac_index_structure():
    """Test: Prüfe die Struktur des FLAC-Index"""
    logger.trace("test_flac_index_structure() requested.")
    print("\n=== Test: FLAC-Index Struktur ===")
    
    if not ZARR3_STORE_DIR.exists():
        print("FEHLER: Zarr-Datenbank nicht gefunden. Führe zuerst test_import_wav_to_flac aus.")
        return False
    
    try:
        store = zarr.storage.LocalStore(str(ZARR3_STORE_DIR))
        root = zarr.open_group(store, mode='r')
        zarr_group = root['audio_imports']
        
        if "0" not in zarr_group:
            print("FEHLER: Gruppe '0' nicht gefunden.")
            return False
        
        group_0 = zarr_group["0"]
        
        if "flac_index" not in group_0:
            print("FEHLER: FLAC-Index nicht gefunden.")
            return False
        
        flac_index = group_0["flac_index"]
        
        print(f"FLAC-Index Shape: {flac_index.shape}")
        print(f"FLAC-Index Dtype: {flac_index.dtype}")
        print(f"Anzahl Frames: {flac_index.shape[0]}")
        print(f"Spalten: {flac_index.shape[1] if len(flac_index.shape) > 1 else 'N/A'}")
        
        # Zeige erste paar Einträge
        if flac_index.shape[0] > 0:
            print("\nErste 5 Index-Einträge:")
            for i in range(min(5, flac_index.shape[0])):
                entry = flac_index[i]
                if len(flac_index.shape) > 1 and flac_index.shape[1] >= 3:
                    print(f"  Frame {i}: Offset={entry[0]}, Size={entry[1]}, Sample={entry[2]}")
                else:
                    print(f"  Frame {i}: {entry}")
        
        # Index-Attribute prüfen
        print(f"\nIndex-Attribute:")
        for attr_name, attr_value in flac_index.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        return True
        
    except Exception as e:
        print(f"FEHLER beim Index-Check: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.trace("__main__ started.")
    print("=== Audio Import Tests ===")
    
    # Ausführung der Tests
    succeeded = []
    failed = []
    
    # Test 1: WAV zu FLAC
    if test_import_wav_to_flac():
        succeeded.append("WAV zu FLAC Import")
    else:
        failed.append("WAV zu FLAC Import")
    
    # Test 2: FLAC-Index Struktur
    if test_flac_index_structure():
        succeeded.append("FLAC-Index Struktur")
    else:
        failed.append("FLAC-Index Struktur")
    
    # Test 3: Segment Extraktion
    if test_extract_segment():
        succeeded.append("Segment Extraktion")
    else:
        failed.append("Segment Extraktion")
    
    # # Test 4: MP3 zu Opus
    # if test_import_mp3_to_opus():
    #     succeeded.append("MP3 zu Opus Import")
    # else:
    #     failed.append("MP3 zu Opus Import")
    
    # Zusammenfassung
    print("\n=== Zusammenfassung ===")
    print(f"Erfolgreich: {len(succeeded)} Tests")
    for test in succeeded:
        print(f"✓ {test}")
    
    if failed:
        print(f"\nFehlgeschlagen: {len(failed)} Tests")
        for test in failed:
            print(f"✗ {test}")
    
    print("\nFertig!")
    logger.success("__main__ finalized.")