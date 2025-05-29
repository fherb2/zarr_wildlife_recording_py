"""
Test für das neue FLAC-Index-Modul (flac_access.py + flac_index_backend.py)
Basiert auf test_audio_import_standalone.py und testet die neuen FLAC-Index-Funktionen.
"""
import pathlib
import os
import shutil
import numpy as np
import zarr
import datetime
import tempfile
import soundfile as sf

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

# Import des neuen FLAC-Moduls (KORRIGIERT)
from zarrwlr import flac_access
from zarrwlr import flac_index_backend


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
# FLAC-Index Validierungsfunktionen
# =================================
#
# ###########################################################

def validate_flac_index_comprehensive(zarr_group, audio_blob_array, original_audio_file=None):
    """
    Umfassende Validierung des FLAC-Index mit mehreren Prüfebenen
    
    Args:
        zarr_group: Zarr-Gruppe mit FLAC-Daten und Index
        audio_blob_array: Das Audio-Blob-Array
        original_audio_file: Optional - Pfad zur ursprünglichen Audiodatei für Vergleich
        
    Returns:
        dict: Validierungsergebnis mit Details
    """
    print("\n=== UMFASSENDE FLAC-INDEX VALIDIERUNG ===")
    
    validation_result = {
        'overall_valid': True,
        'tests_passed': 0,
        'tests_total': 0,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    def add_test(name, passed, error_msg=None, warning_msg=None):
        validation_result['tests_total'] += 1
        if passed:
            validation_result['tests_passed'] += 1
            print(f"✓ {name}")
        else:
            validation_result['overall_valid'] = False
            validation_result['errors'].append(error_msg or f"{name} fehlgeschlagen")
            print(f"✗ {name}: {error_msg}")
        
        if warning_msg:
            validation_result['warnings'].append(warning_msg)
            print(f"⚠ {name}: {warning_msg}")
    
    try:
        # Test 1: Index-Existenz
        if 'flac_index' not in zarr_group:
            add_test("Index-Existenz", False, "FLAC-Index nicht gefunden")
            return validation_result
        
        flac_index = zarr_group['flac_index']
        add_test("Index-Existenz", True)
        
        # Test 2: Index-Struktur (2D Array mit 3 Spalten)
        correct_structure = (len(flac_index.shape) == 2 and flac_index.shape[1] == 3)
        add_test("Index-Struktur (2D, 3 Spalten)", correct_structure, 
                f"Erwartete Shape (n, 3), gefunden {flac_index.shape}")
        
        if not correct_structure:
            return validation_result
        
        # Test 3: Index nicht leer
        has_frames = flac_index.shape[0] > 0
        add_test("Index nicht leer", has_frames, 
                f"Index hat {flac_index.shape[0]} Frames")
        
        if not has_frames:
            return validation_result
        
        # Test 4: Erforderliche Metadaten
        required_attrs = ['sample_rate', 'channels', 'total_frames', 'codec']
        for attr in required_attrs:
            has_attr = attr in flac_index.attrs
            add_test(f"Attribut '{attr}' vorhanden", has_attr, 
                    f"Fehlendes Attribut: {attr}")
        
        # Test 5: Codec-Konsistenz
        codec_correct = flac_index.attrs.get('codec') == 'flac'
        add_test("Codec korrekt (flac)", codec_correct, 
                f"Erwarteter Codec 'flac', gefunden '{flac_index.attrs.get('codec')}'")
        
        # Test 6: Frame-Anzahl-Konsistenz
        total_frames_consistent = flac_index.attrs.get('total_frames') == flac_index.shape[0]
        add_test("Frame-Anzahl konsistent", total_frames_consistent,
                f"total_frames ({flac_index.attrs.get('total_frames')}) != Index-Länge ({flac_index.shape[0]})")
        
        # Test 7: Monotonie der Byte-Offsets
        byte_offsets = flac_index[:, 0]
        byte_offsets_monotonic = np.all(byte_offsets[1:] > byte_offsets[:-1]) if len(byte_offsets) > 1 else True
        add_test("Byte-Offsets monoton steigend", byte_offsets_monotonic,
                "Byte-Offsets sind nicht monoton steigend")
        
        # Test 8: Monotonie der Sample-Positionen
        sample_positions = flac_index[:, 2]
        sample_positions_monotonic = np.all(sample_positions[1:] >= sample_positions[:-1]) if len(sample_positions) > 1 else True
        add_test("Sample-Positionen monoton steigend", sample_positions_monotonic,
                "Sample-Positionen sind nicht monoton steigend")
        
        # Test 9: Plausible Frame-Größen
        frame_sizes = flac_index[:, 1]
        reasonable_frame_sizes = np.all(frame_sizes > 0) and np.all(frame_sizes < audio_blob_array.shape[0])
        add_test("Frame-Größen plausibel", reasonable_frame_sizes,
                "Frame-Größen sind nicht plausibel (≤0 oder größer als Audio-Blob)")
        
        # Test 10: Letzte Frame-Position im Audio-Blob
        last_frame_offset = byte_offsets[-1]
        last_frame_size = frame_sizes[-1]
        last_frame_end = last_frame_offset + last_frame_size
        frame_positions_valid = last_frame_end <= audio_blob_array.shape[0]
        add_test("Frame-Positionen im Audio-Blob", frame_positions_valid,
                f"Letzter Frame endet bei Byte {last_frame_end}, Audio-Blob hat {audio_blob_array.shape[0]} Bytes")
        
        # Test 11: Funktionale Validierung - Segment-Extraktion
        try:
            test_segment = flac_access.extract_audio_segment_flac(
                zarr_group, audio_blob_array, 0, min(999, sample_positions[-1]), np.int16
            )
            extraction_works = isinstance(test_segment, np.ndarray) and test_segment.size > 0
            add_test("Segment-Extraktion funktional", extraction_works,
                    "Segment-Extraktion fehlgeschlagen oder leer")
            
            validation_result['details']['test_segment_shape'] = test_segment.shape if extraction_works else None
        except Exception as e:
            add_test("Segment-Extraktion funktional", False, f"Exception: {str(e)}")
        
        # Test 12: Sample-Rate Konsistenz mit Audio-Blob
        blob_sample_rate = audio_blob_array.attrs.get('sample_rate')
        index_sample_rate = flac_index.attrs.get('sample_rate')
        sample_rate_consistent = blob_sample_rate == index_sample_rate
        add_test("Sample-Rate Konsistenz", sample_rate_consistent,
                f"Blob Sample-Rate ({blob_sample_rate}) != Index Sample-Rate ({index_sample_rate})")
        
        # Test 13: Kanal-Konsistenz
        blob_channels = audio_blob_array.attrs.get('nb_channels')
        index_channels = flac_index.attrs.get('channels')
        channels_consistent = blob_channels == index_channels
        add_test("Kanal-Konsistenz", channels_consistent,
                f"Blob Kanäle ({blob_channels}) != Index Kanäle ({index_channels})")
        
        # Zusätzliche Statistiken sammeln
        validation_result['details'].update({
            'total_frames': flac_index.shape[0],
            'sample_rate': index_sample_rate,
            'channels': index_channels,
            'total_samples': sample_positions[-1] if len(sample_positions) > 0 else 0,
            'avg_frame_size': np.mean(frame_sizes) if len(frame_sizes) > 0 else 0,
            'min_frame_size': np.min(frame_sizes) if len(frame_sizes) > 0 else 0,
            'max_frame_size': np.max(frame_sizes) if len(frame_sizes) > 0 else 0,
            'audio_blob_size': audio_blob_array.shape[0]
        })
        
        # Optional: Vergleich mit ursprünglicher Datei
        if original_audio_file and pathlib.Path(original_audio_file).exists():
            try:
                validation_result['details'].update(
                    _validate_against_original_file(original_audio_file, validation_result['details'])
                )
            except Exception as e:
                add_test("Vergleich mit Originaldatei", False, f"Exception: {str(e)}")
        
    except Exception as e:
        validation_result['overall_valid'] = False
        validation_result['errors'].append(f"Validierungsfehler: {str(e)}")
        print(f"✗ Validierungsfehler: {str(e)}")
    
    # Zusammenfassung ausgeben
    print(f"\n=== VALIDIERUNGSERGEBNIS ===")
    print(f"Tests bestanden: {validation_result['tests_passed']}/{validation_result['tests_total']}")
    print(f"Gesamt-Status: {'✓ BESTANDEN' if validation_result['overall_valid'] else '✗ FEHLGESCHLAGEN'}")
    
    if validation_result['errors']:
        print(f"\nFehler ({len(validation_result['errors'])}):")
        for error in validation_result['errors']:
            print(f"  - {error}")
    
    if validation_result['warnings']:
        print(f"\nWarnungen ({len(validation_result['warnings'])}):")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    
    if validation_result['details']:
        print(f"\nStatistiken:")
        details = validation_result['details']
        print(f"  - Frames: {details.get('total_frames', 'N/A')}")
        print(f"  - Sample-Rate: {details.get('sample_rate', 'N/A')} Hz")
        print(f"  - Kanäle: {details.get('channels', 'N/A')}")
        print(f"  - Geschätzte Samples: {details.get('total_samples', 'N/A')}")
        print(f"  - Durchschnittliche Frame-Größe: {details.get('avg_frame_size', 'N/A'):.1f} Bytes")
        print(f"  - Frame-Größe Min/Max: {details.get('min_frame_size', 'N/A')}/{details.get('max_frame_size', 'N/A')} Bytes")
        print(f"  - Audio-Blob-Größe: {details.get('audio_blob_size', 'N/A')} Bytes")
    
    return validation_result


def _validate_against_original_file(original_file, current_details):
    """Validiert Index gegen ursprüngliche Audiodatei"""
    print(f"\n--- Vergleich mit Originaldatei: {pathlib.Path(original_file).name} ---")
    
    comparison_details = {}
    
    try:
        with sf.SoundFile(str(original_file)) as sf_file:
            original_sample_rate = sf_file.samplerate
            original_channels = sf_file.channels
            original_frames = sf_file.frames
            
            comparison_details.update({
                'original_sample_rate': original_sample_rate,
                'original_channels': original_channels,
                'original_frames': original_frames,
                'sample_rate_match': current_details.get('sample_rate') == original_sample_rate,
                'channels_match': current_details.get('channels') == original_channels,
                'estimated_frames_close': abs(current_details.get('total_samples', 0) - original_frames) < 1000
            })
            
            print(f"  Original: {original_sample_rate}Hz, {original_channels}ch, {original_frames} samples")
            print(f"  Index: {current_details.get('sample_rate')}Hz, {current_details.get('channels')}ch, ~{current_details.get('total_samples')} samples")
            print(f"  Sample-Rate Match: {'✓' if comparison_details['sample_rate_match'] else '✗'}")
            print(f"  Kanäle Match: {'✓' if comparison_details['channels_match'] else '✗'}")
            print(f"  Sample-Anzahl ähnlich: {'✓' if comparison_details['estimated_frames_close'] else '✗'}")
            
    except Exception as e:
        comparison_details['validation_error'] = str(e)
        print(f"  Fehler beim Dateienvergleich: {str(e)}")
    
    return comparison_details


# ###########################################################
#
# FLAC-Index Tests (neue Funktionen)
# ===================
#
# ###########################################################

def test_import_wav_to_flac_with_validation():
    """Test: WAV zu FLAC Import mit umfassender Index-Validierung"""
    logger.trace("test_import_wav_to_flac_with_validation() requested.")
    print("\n=== Test: WAV zu FLAC Import mit umfassender Validierung ===")
    
    try:
        # Zarr-Datenbank vorbereiten
        audio_group = prepare_zarr_database()
        
        # WAV-Datei finden
        test_files = get_test_files()
        wav_file = next((f for f in test_files if f.name.endswith(".wav")), None)
        
        if not wav_file or not wav_file.exists():
            print(f"FEHLER: WAV-Testdatei nicht gefunden.")
            return False
        
        print(f"Importiere WAV-Datei: {wav_file}")
        
        # Import durchführen
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=wav_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # Finde die erstellte Gruppe
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Audio-Gruppe erstellt.")
            return False
        
        latest_group_name = max(group_names, key=int)
        latest_group = audio_group[latest_group_name]
        audio_blob_array = latest_group["audio_data_blob_array"]
        
        print(f"Import erfolgreich in Gruppe '{latest_group_name}'")
        print(f"Blob-Array-Größe: {audio_blob_array.shape[0]} Bytes")
        print(f"Codec: {audio_blob_array.attrs.get('codec')}")
        
        # UMFASSENDE VALIDIERUNG
        validation_result = validate_flac_index_comprehensive(
            latest_group, audio_blob_array, wav_file
        )
        
        return validation_result['overall_valid']
        
    except Exception as e:
        print(f"FEHLER beim WAV-Test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_flac_file_with_validation():
    """Test: Direkte FLAC-Datei mit umfassender Index-Validierung"""
    logger.trace("test_existing_flac_file_with_validation() requested.")
    print("\n=== Test: Direkte FLAC-Datei mit umfassender Validierung ===")
    
    # Neue DB für diesen Test
    if ZARR3_STORE_DIR.exists():
        shutil.rmtree(ZARR3_STORE_DIR)
    audio_group = prepare_zarr_database()
    
    # FLAC-Datei finden
    test_files = get_test_files()
    flac_file = next((f for f in test_files if f.name.endswith(".flac")), None)
    
    if not flac_file or not flac_file.exists():
        print(f"WARNUNG: FLAC-Testdatei nicht gefunden. Test übersprungen.")
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
        
        # Finde die erstellte Gruppe
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Audio-Gruppe erstellt.")
            return False
        
        latest_group_name = max(group_names, key=int)
        latest_group = audio_group[latest_group_name]
        audio_blob_array = latest_group["audio_data_blob_array"]
        
        print(f"Import erfolgreich in Gruppe '{latest_group_name}'")
        
        # UMFASSENDE VALIDIERUNG
        validation_result = validate_flac_index_comprehensive(
            latest_group, audio_blob_array, flac_file
        )
        
        return validation_result['overall_valid']
        
    except Exception as e:
        print(f"FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_segment_extraction_accuracy():
    """Test: Genauigkeit der Segment-Extraktion"""
    print("\n=== Test: Segment-Extraktion Genauigkeit ===")
    
    if not ZARR3_STORE_DIR.exists():
        print("FEHLER: Zarr-Datenbank nicht gefunden. Führe zuerst Import-Test aus.")
        return False
    
    try:
        store = zarr.storage.LocalStore(str(ZARR3_STORE_DIR))
        root = zarr.open_group(store, mode='r')
        zarr_group = root['audio_imports']
        
        # Finde letzte Gruppe
        group_names = [name for name in zarr_group.keys() if name.isdigit()]
        if not group_names:
            print("FEHLER: Keine Audio-Gruppe gefunden.")
            return False
        
        latest_group_name = max(group_names, key=int)
        latest_group = zarr_group[latest_group_name]
        
        if "flac_index" not in latest_group:
            print("FEHLER: FLAC-Index nicht gefunden.")
            return False
        
        flac_index = latest_group["flac_index"]
        audio_blob_array = latest_group["audio_data_blob_array"]
        
        # Test verschiedene Segment-Größen
        test_segments = [
            (0, 999),      # 1000 Samples
            (1000, 1999),  # 1000 Samples
            (500, 1499),   # 1000 Samples (überlappend)
            (0, 99),       # 100 Samples (klein)
            (0, 9999)      # 10000 Samples (groß)
        ]
        
        all_passed = True
        
        for i, (start, end) in enumerate(test_segments):
            try:
                segment = flac_access.extract_audio_segment_flac(
                    latest_group, audio_blob_array, start, end, np.int16
                )
                
                expected_length = end - start + 1
                actual_length = segment.shape[0] if len(segment.shape) == 1 else segment.shape[0]
                
                # Bei Stereo: Länge ist Anzahl Frames, nicht Samples
                channels = flac_index.attrs.get('channels', 1)
                if len(segment.shape) > 1:
                    actual_length = segment.shape[0]  # Frames bei Stereo
                
                # Toleranz für letzte Segmente (könnten kürzer sein)
                tolerance = 50  # 50 Samples Toleranz
                length_ok = abs(actual_length - expected_length) <= tolerance
                
                if length_ok:
                    print(f"✓ Segment {i+1} ({start}-{end}): {actual_length} Samples")
                else:
                    print(f"✗ Segment {i+1} ({start}-{end}): Erwartet ~{expected_length}, erhalten {actual_length}")
                    all_passed = False
                
            except Exception as e:
                print(f"✗ Segment {i+1} ({start}-{end}): Exception {str(e)}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"FEHLER bei Genauigkeits-Test: {str(e)}")
        return False


# ###########################################################
#
# Main Test Execution
# ===================
#
# ###########################################################

if __name__ == "__main__":
    logger.trace("__main__ started.")
    print("=== FLAC-Index Modul Tests mit umfassender Validierung ===")
    
    # Ausführung der Tests
    succeeded = []
    failed = []
    
    # Test 1: WAV zu FLAC mit umfassender Validierung
    if test_import_wav_to_flac_with_validation():
        succeeded.append("WAV zu FLAC mit umfassender Validierung")
    else:
        failed.append("WAV zu FLAC mit umfassender Validierung")
    
    # Test 2: Direkte FLAC-Datei mit umfassender Validierung
    if test_existing_flac_file_with_validation():
        succeeded.append("Direkte FLAC-Datei mit umfassender Validierung")
    else:
        failed.append("Direkte FLAC-Datei mit umfassender Validierung")
    
    # Test 3: Segment-Extraktion Genauigkeit
    if test_segment_extraction_accuracy():
        succeeded.append("Segment-Extraktion Genauigkeit")
    else:
        failed.append("Segment-Extraktion Genauigkeit")
    
    # Zusammenfassung
    print("\n=== ZUSAMMENFASSUNG FLAC-INDEX TESTS ===")
    print(f"Erfolgreich: {len(succeeded)} Tests")
    for test in succeeded:
        print(f"✓ {test}")
    
    if failed:
        print(f"\nFehlgeschlagen: {len(failed)} Tests")
        for test in failed:
            print(f"✗ {test}")
        print(f"\n❌ {len(failed)} von {len(succeeded) + len(failed)} Tests fehlgeschlagen")
    else:
        print("\n🎉 Alle Tests erfolgreich!")
    
    print("\nFertig!")
    logger.success("__main__ finalised.")
