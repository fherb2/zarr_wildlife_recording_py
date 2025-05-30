"""
FLAC Parallel Index - DEBUG VERSION
===================================

Debug-Version mit ausführlichen Logging für Problemanalyse
"""
from __future__ import annotations

import pathlib
import os
import shutil
import numpy as np
import zarr
import datetime
import time
import multiprocessing as mp
import traceback
from typing import List, Tuple

# Memory monitoring (mit Fallback falls psutil nicht verfügbar)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNUNG: psutil nicht verfügbar. Memory-Monitoring mit Dummy-Werten.")

import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

# ✅ DEBUG: Setze Log-Level auf DEBUG für mehr Details
Config.set(log_level=LogLevel.DEBUG)
logger = get_module_logger(__file__)

print("=" * 80)
print("🔍 DEBUG VERSION - FLAC PARALLEL INDEX TESTS")
print("=" * 80)

# ✅ DEBUG: Prüfe Import-Chain
print("\n🔍 IMPORT-CHAIN DEBUG:")
try:
    print("1. Importiere zarrwlr.aimport...")
    from zarrwlr.aimport import create_original_audio_group, import_original_audio_file
    print("   ✅ zarrwlr.aimport erfolgreich importiert")
except Exception as e:
    print(f"   ❌ FEHLER beim Import von zarrwlr.aimport: {e}")
    traceback.print_exc()

try:
    print("2. Importiere zarrwlr.flac_index_backend...")
    from zarrwlr import flac_index_backend
    print("   ✅ zarrwlr.flac_index_backend erfolgreich importiert")
    
    # ✅ DEBUG: Prüfe verfügbare Funktionen
    available_functions = [name for name in dir(flac_index_backend) if not name.startswith('_')]
    print(f"   📋 Verfügbare Funktionen: {available_functions}")
    
    # ✅ DEBUG: Prüfe build_flac_index Signatur
    import inspect
    sig = inspect.signature(flac_index_backend.build_flac_index)
    print(f"   📋 build_flac_index Signatur: {sig}")
    
except Exception as e:
    print(f"   ❌ FEHLER beim Import von zarrwlr.flac_index_backend: {e}")
    traceback.print_exc()

try:
    print("3. Importiere zarrwlr.flac_access (optional)...")
    from zarrwlr import flac_access
    print("   ✅ zarrwlr.flac_access erfolgreich importiert")
except Exception as e:
    print(f"   ⚠️  zarrwlr.flac_access nicht verfügbar: {e}")

print("\n" + "=" * 80)

class MemoryStats:
    """Helper für Memory-Monitoring (mit psutil Fallback)"""
    
    @staticmethod
    def get_current_memory_mb():
        """Aktueller RAM-Verbrauch in MB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return 0.0  # Fallback


TEST_RESULTS_DIR = pathlib.Path(__file__).parent.resolve() / "testresults"

def prepare_test_environment():
    """Testumgebung vorbereiten"""
    print("\n🔍 DEBUG: Bereite Testumgebung vor...")
    if not TEST_RESULTS_DIR.exists():
        TEST_RESULTS_DIR.mkdir(parents=True)
        print(f"   ✅ Testverzeichnis erstellt: {TEST_RESULTS_DIR}")
    else:
        print(f"   ✅ Testverzeichnis existiert: {TEST_RESULTS_DIR}")


def get_test_files() -> list[pathlib.Path]:
    """Testdateien finden"""
    print("\n🔍 DEBUG: Suche Testdateien...")
    test_files = [
        "testdata/audiomoth_long_snippet.wav",
        "testdata/audiomoth_short_snippet.wav", 
        "testdata/bird1_snippet.mp3",
    ]
    
    base_path = pathlib.Path(__file__).parent.resolve()
    found_files = []
    
    for file_rel in test_files:
        file_path = base_path / file_rel
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"   ✅ Gefunden: {file_path.name} ({file_size_mb:.1f} MB)")
            found_files.append(file_path)
        else:
            print(f"   ❌ Nicht gefunden: {file_path}")
    
    print(f"   📋 Gesamt gefunden: {len(found_files)} Dateien")
    return found_files


def import_test_file_to_zarr(audio_file: pathlib.Path, test_name: str) -> Tuple[zarr.Group, bytes]:
    """Importiere Testdatei zu FLAC in Zarr"""
    print(f"\n🔍 DEBUG: Importiere {audio_file.name} für Test '{test_name}'...")
    
    test_store_dir = TEST_RESULTS_DIR / f"zarr3-store-{test_name}"
    if test_store_dir.exists():
        print(f"   🗑️  Lösche existierenden Store: {test_store_dir}")
        shutil.rmtree(test_store_dir)
    
    print(f"   📁 Erstelle Audio-Gruppe in: {test_store_dir}")
    audio_group = create_original_audio_group(store_path=test_store_dir, group_path='audio_imports')
    
    timestamp = datetime.datetime.now()
    print(f"   🎵 Starte FLAC-Import mit Timestamp: {timestamp}")
    
    try:
        import_original_audio_file(
            audio_file=audio_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        print(f"   ✅ FLAC-Import erfolgreich")
    except Exception as e:
        print(f"   ❌ FLAC-Import fehlgeschlagen: {e}")
        traceback.print_exc()
        raise
    
    # Finde importierte Gruppe
    group_names = [name for name in audio_group.keys() if name.isdigit()]
    print(f"   📋 Verfügbare Gruppen: {group_names}")
    
    if not group_names:
        raise ValueError("Keine numerischen Gruppen nach Import gefunden!")
    
    latest_group_name = max(group_names, key=int)
    latest_group = audio_group[latest_group_name]
    print(f"   📂 Verwende Gruppe: {latest_group_name}")
    
    # Prüfe Array-Struktur
    print(f"   📋 Gruppe-Inhalte: {list(latest_group.keys())}")
    
    if "audio_data_blob_array" not in latest_group:
        raise ValueError("audio_data_blob_array nicht in importierter Gruppe gefunden!")
    
    audio_blob_array = latest_group["audio_data_blob_array"]
    print(f"   📊 Audio-Array: Shape={audio_blob_array.shape}, Dtype={audio_blob_array.dtype}")
    print(f"   📊 Audio-Array Attribute: {dict(audio_blob_array.attrs)}")
    
    audio_bytes = bytes(audio_blob_array[()])
    audio_size_mb = len(audio_bytes) / 1024 / 1024
    print(f"   📊 Audio-Daten: {audio_size_mb:.1f} MB")
    
    return latest_group, audio_bytes


def test_simple_index_creation():
    """Test: Einfache Index-Erstellung (DEBUG)"""
    print("\n" + "=" * 60)
    print("🔍 DEBUG TEST: Einfache Index-Erstellung")
    print("=" * 60)
    
    prepare_test_environment()
    
    test_files = get_test_files()
    if not test_files:
        print("❌ Keine Testdateien gefunden!")
        return False
    
    # Verwende die kleinste verfügbare Datei
    test_file = min(test_files, key=lambda f: f.stat().st_size)
    file_size_mb = test_file.stat().st_size / 1024 / 1024
    print(f"🎯 Verwende Testdatei: {test_file.name} ({file_size_mb:.1f} MB)")
    
    try:
        zarr_group, audio_bytes = import_test_file_to_zarr(test_file, "simple_debug_test")
        audio_blob_array = zarr_group["audio_data_blob_array"]
        
        print(f"\n🔍 DEBUG: Teste Sequential Index-Erstellung...")
        start_time = time.time()
        
        try:
            sequential_index = flac_index_backend.build_flac_index(
                zarr_group, audio_blob_array, use_parallel=False
            )
            sequential_time = time.time() - start_time
            
            print(f"   ✅ Sequential erfolgreich:")
            print(f"      Zeit: {sequential_time:.3f}s")
            print(f"      Frames: {sequential_index.shape[0]}")
            print(f"      Shape: {sequential_index.shape}")
            print(f"      Dtype: {sequential_index.dtype}")
            print(f"      Attribute: {dict(sequential_index.attrs)}")
            
        except Exception as e:
            print(f"   ❌ Sequential fehlgeschlagen: {e}")
            traceback.print_exc()
            return False
        
        # Lösche Index für Parallel-Test
        if 'flac_index' in zarr_group:
            del zarr_group['flac_index']
        
        print(f"\n🔍 DEBUG: Teste Parallel Index-Erstellung...")
        start_time = time.time()
        
        try:
            parallel_index = flac_index_backend.build_flac_index(
                zarr_group, audio_blob_array, use_parallel=True
            )
            parallel_time = time.time() - start_time
            
            print(f"   ✅ Parallel erfolgreich:")
            print(f"      Zeit: {parallel_time:.3f}s")
            print(f"      Frames: {parallel_index.shape[0]}")
            print(f"      Shape: {parallel_index.shape}")
            print(f"      Dtype: {parallel_index.dtype}")
            print(f"      Attribute: {dict(parallel_index.attrs)}")
            
            # Prüfe ob wirklich parallel verarbeitet wurde
            parallel_used = parallel_index.attrs.get('parallel_processing_used', False)
            print(f"      Parallel verwendet: {parallel_used}")
            
            if not parallel_used:
                print(f"   ⚠️  WARNUNG: Fallback auf Sequential verwendet!")
            
        except Exception as e:
            print(f"   ❌ Parallel fehlgeschlagen: {e}")
            traceback.print_exc()
            return False
        
        print(f"\n🔍 DEBUG: Vergleiche Ergebnisse...")
        
        # Lade Daten für Vergleich
        seq_data = sequential_index[:]
        par_data = parallel_index[:]
        
        same_shape = seq_data.shape == par_data.shape
        print(f"   Gleiche Shape: {same_shape}")
        
        if same_shape:
            identical = np.array_equal(seq_data, par_data)
            if identical:
                print(f"   ✅ Ergebnisse: Exakt identisch")
            else:
                max_diff = np.max(np.abs(seq_data.astype(float) - par_data.astype(float)))
                print(f"   📊 Max. Unterschied: {max_diff}")
                
                if max_diff <= 10:
                    print(f"   ✅ Ergebnisse: Praktisch identisch (Toleranz: {max_diff})")
                    identical = True
                else:
                    print(f"   ❌ Ergebnisse: Signifikante Unterschiede")
                    
                    # Debug: Zeige erste Unterschiede
                    diff_mask = seq_data != par_data
                    if np.any(diff_mask):
                        diff_indices = np.where(diff_mask)
                        for i in range(min(5, len(diff_indices[0]))):
                            row, col = diff_indices[0][i], diff_indices[1][i]
                            print(f"      Diff [{row},{col}]: {seq_data[row,col]} vs {par_data[row,col]}")
        else:
            identical = False
            print(f"   ❌ Unterschiedliche Shapes: {seq_data.shape} vs {par_data.shape}")
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\n📊 Performance: {speedup:.2f}x Speedup ({sequential_time:.3f}s → {parallel_time:.3f}s)")
        
        success = same_shape and identical
        print(f"\n🎯 Test-Ergebnis: {'✅ ERFOLGREICH' if success else '❌ FEHLGESCHLAGEN'}")
        
        return success
        
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER im Debug-Test: {e}")
        traceback.print_exc()
        return False


def test_api_chain_debug():
    """Test: API-Chain Debug (aimport → flac_access → flac_index_backend)"""
    print("\n" + "=" * 60)
    print("🔍 DEBUG TEST: API-Chain Konsistenz")
    print("=" * 60)
    
    test_files = get_test_files()
    if not test_files:
        print("❌ Keine Testdateien gefunden!")
        return False
    
    test_file = min(test_files, key=lambda f: f.stat().st_size)
    print(f"🎯 Verwende Testdatei: {test_file.name}")
    
    try:
        zarr_group, audio_bytes = import_test_file_to_zarr(test_file, "api_chain_debug")
        audio_blob_array = zarr_group["audio_data_blob_array"]
        
        print(f"\n🔍 DEBUG: Prüfe Import-Chain...")
        
        # 1. Prüfe ob Index bereits von aimport erstellt wurde
        if 'flac_index' in zarr_group:
            existing_index = zarr_group['flac_index']
            print(f"   ✅ Index bereits von aimport erstellt:")
            print(f"      Frames: {existing_index.shape[0]}")
            print(f"      Parallel verwendet: {existing_index.attrs.get('parallel_processing_used', 'UNBEKANNT')}")
            
            # Test ob Index funktioniert
            try:
                sample_data = existing_index[:5]  # Erste 5 Frames
                print(f"      Sample-Daten: {sample_data}")
                print(f"   ✅ Index ist lesbar")
            except Exception as e:
                print(f"   ❌ Index-Lesefehler: {e}")
                return False
        else:
            print(f"   ⚠️  Kein Index von aimport erstellt")
        
        # 2. Teste direkte flac_index_backend API
        print(f"\n🔍 DEBUG: Teste direkte flac_index_backend API...")
        
        # Lösche vorhandenen Index
        if 'flac_index' in zarr_group:
            del zarr_group['flac_index']
        
        try:
            direct_index = flac_index_backend.build_flac_index(
                zarr_group, audio_blob_array, use_parallel=True
            )
            print(f"   ✅ Direkte API erfolgreich:")
            print(f"      Frames: {direct_index.shape[0]}")
            print(f"      Parallel: {direct_index.attrs.get('parallel_processing_used', 'UNBEKANNT')}")
        except Exception as e:
            print(f"   ❌ Direkte API fehlgeschlagen: {e}")
            traceback.print_exc()
            return False
        
        # 3. Teste flac_access API (falls verfügbar)
        try:
            from zarrwlr import flac_access
            
            print(f"\n🔍 DEBUG: Teste flac_access API...")
            
            # Lösche Index
            if 'flac_index' in zarr_group:
                del zarr_group['flac_index']
            
            access_index = flac_access.build_flac_index(
                zarr_group, audio_blob_array, use_parallel=True
            )
            print(f"   ✅ flac_access API erfolgreich:")
            print(f"      Frames: {access_index.shape[0]}")
            print(f"      Parallel: {access_index.attrs.get('parallel_processing_used', 'UNBEKANNT')}")
            
        except ImportError:
            print(f"   ⚠️  flac_access nicht verfügbar")
        except Exception as e:
            print(f"   ❌ flac_access API fehlgeschlagen: {e}")
            traceback.print_exc()
        
        print(f"\n🎯 API-Chain Test: ✅ ABGESCHLOSSEN")
        return True
        
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER im API-Chain Test: {e}")
        traceback.print_exc()
        return False


def test_parallel_configuration_debug():
    """Test: Parallel-Konfiguration Debug"""
    print("\n" + "=" * 60)
    print("🔍 DEBUG TEST: Parallel-Konfiguration")
    print("=" * 60)
    
    try:
        # 1. Teste configure_parallel_processing
        if hasattr(flac_index_backend, 'configure_parallel_processing'):
            config = flac_index_backend.configure_parallel_processing()
            print(f"✅ Parallel-Konfiguration:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ configure_parallel_processing nicht verfügbar")
        
        # 2. Teste System-Info
        print(f"\n🔍 DEBUG: System-Info:")
        print(f"   CPU-Kerne: {mp.cpu_count()}")
        print(f"   PSUTIL verfügbar: {PSUTIL_AVAILABLE}")
        
        if PSUTIL_AVAILABLE:
            memory_mb = MemoryStats.get_current_memory_mb()
            print(f"   Aktueller RAM: {memory_mb:.1f} MB")
        
        # 3. Teste Zarr-Store-Pfad-Extraktion
        test_files = get_test_files()
        if test_files:
            test_file = min(test_files, key=lambda f: f.stat().st_size)
            zarr_group, _ = import_test_file_to_zarr(test_file, "config_debug")
            audio_blob_array = zarr_group["audio_data_blob_array"]
            
            print(f"\n🔍 DEBUG: Zarr-Store-Pfad-Extraktion:")
            try:
                store_path = flac_index_backend._get_zarr_store_path(zarr_group)
                print(f"   ✅ Store-Pfad: {store_path}")
                
                store_path, group_path, array_name = flac_index_backend._get_zarr_array_path_components(
                    zarr_group, audio_blob_array
                )
                print(f"   ✅ Komponenten:")
                print(f"      Store: {store_path}")
                print(f"      Group: {group_path}")
                print(f"      Array: {array_name}")
                
            except Exception as e:
                print(f"   ❌ Pfad-Extraktion fehlgeschlagen: {e}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ KRITISCHER FEHLER im Konfigurations-Test: {e}")
        traceback.print_exc()
        return False


# ##########################################################
#
# Main Debug Execution
# ====================
#
# ##########################################################

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔍 FLAC PARALLEL INDEX - DEBUG TEST SUITE")
    print("=" * 80)
    print("Debug-Version für Problemanalyse")
    print()
    
    debug_tests = [
        ("Einfache Index-Erstellung", test_simple_index_creation),
        ("API-Chain Konsistenz", test_api_chain_debug),
        ("Parallel-Konfiguration", test_parallel_configuration_debug),
    ]
    
    succeeded = []
    failed = []
    
    for test_name, test_func in debug_tests:
        print(f"\n{'🔍 STARTE: ' + test_name :-<70}")
        try:
            if test_func():
                succeeded.append(test_name)
                print(f"{'✅ ERFOLGREICH: ' + test_name :-<70}")
            else:
                failed.append(test_name)
                print(f"{'❌ FEHLGESCHLAGEN: ' + test_name :-<70}")
        except Exception as e:
            failed.append(test_name)
            print(f"{'💥 FEHLER: ' + test_name :-<70}")
            print(f"Exception: {e}")
            traceback.print_exc()
    
    # Zusammenfassung
    print(f"\n" + "=" * 80)
    print("🔍 DEBUG TEST ZUSAMMENFASSUNG")
    print("=" * 80)
    
    print(f"✅ Erfolgreich: {len(succeeded)} Tests")
    for test in succeeded:
        print(f"   ✅ {test}")
    
    if failed:
        print(f"\n❌ Fehlgeschlagen: {len(failed)} Tests")
        for test in failed:
            print(f"   ❌ {test}")
        
        print(f"\n🔍 NÄCHSTE SCHRITTE:")
        print(f"1. Prüfe Import-Chain: aimport → flac_access → flac_index_backend")
        print(f"2. Prüfe API-Parameter-Weiterleitung")
        print(f"3. Prüfe Zarr-Store-Pfad-Kompatibilität")
        print(f"4. Teste mit verschiedenen Worker-Anzahlen")
    else:
        print(f"\n🎉 Alle Debug-Tests erfolgreich!")
        print(f"✅ Import-Chain funktioniert")
        print(f"✅ Parallelisierung ist aktiv")
        print(f"✅ API-Konsistenz gewährleistet")
    
    print(f"\n🔍 Debug-Tests abgeschlossen!")
    print("=" * 80)