#!/usr/bin/env python3
"""
Combined Test Runner for zarrwlr Import Modules

Führt alle Tests für import_utils.py und aimport.py aus und gibt eine 
umfassende Zusammenfassung.

Verwendung:
    python run_all_tests.py
    pytest run_all_tests.py -v  # Lädt Tests als pytest module
"""

import sys
import pathlib
import time
import subprocess
from typing import Dict, List, Tuple

# Füge Projekt-Root zum Python-Pfad hinzu
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_pytest_if_available():
    """Versuche pytest-Tests auszuführen falls pytest verfügbar ist"""
    try:
        import pytest
        
        test_files = [
            "test_import_utils.py",
            "test_aimport.py"
        ]
        
        print("🔬 Running tests with pytest...")
        print("=" * 60)
        
        # Führe pytest aus
        args = ["-v", "--tb=short"] + test_files
        result = pytest.main(args)
        
        return result == 0
        
    except ImportError:
        print("ℹ️  pytest not available, running direct tests...")
        return None

def run_direct_tests():
    """Führe Tests direkt aus ohne pytest"""
    print("🧪 Running Direct Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Test import_utils.py
    print("\n" + "🔬 TESTING import_utils.py" + "🔬")
    print("=" * 50)
    
    try:
        from test_import_utils import run_tests as run_import_utils_tests
        start_time = time.time()
        success = run_import_utils_tests()
        duration = time.time() - start_time
        test_results['import_utils'] = {
            'success': success,
            'duration': duration,
            'error': None
        }
    except Exception as e:
        test_results['import_utils'] = {
            'success': False,
            'duration': 0,
            'error': str(e)
        }
        print(f"❌ Error running import_utils tests: {e}")
    
    # Test aimport.py
    print("\n" + "🚀 TESTING aimport.py" + "🚀")
    print("=" * 50)
    
    try:
        from test_aimport import run_tests as run_aimport_tests
        start_time = time.time()
        success = run_aimport_tests()
        duration = time.time() - start_time
        test_results['aimport'] = {
            'success': success,
            'duration': duration,
            'error': None
        }
    except Exception as e:
        test_results['aimport'] = {
            'success': False,
            'duration': 0,
            'error': str(e)
        }
        print(f"❌ Error running aimport tests: {e}")
    
    return test_results

def print_comprehensive_summary(test_results: Dict):
    """Drucke umfassende Test-Zusammenfassung"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    total_modules = len(test_results)
    successful_modules = sum(1 for result in test_results.values() if result['success'])
    total_duration = sum(result['duration'] for result in test_results.values())
    
    print(f"\n📈 Overall Results:")
    print(f"   Modules Tested: {total_modules}")
    print(f"   Successful: {successful_modules} ✅")
    print(f"   Failed: {total_modules - successful_modules} ❌")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    # Test-Ergebnisse-Verzeichnis Info
    testresults_dir = pathlib.Path(__file__).parent / "testresults"
    if testresults_dir.exists():
        print(f"   Test Results: {testresults_dir}")
        try:
            # Zeige Größe der Test-Ergebnisse
            total_size = sum(f.stat().st_size for f in testresults_dir.rglob('*') if f.is_file())
            print(f"   Results Size: {total_size / 1024 / 1024:.2f} MB")
        except Exception:
            pass
    
    print(f"\n📋 Module Details:")
    for module_name, result in test_results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"   {module_name}: {status} ({result['duration']:.2f}s)")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    if successful_modules == total_modules:
        print(f"\n🎉 ALL MODULES PASSED! 🎉")
        print(f"🔬 The zarrwlr import system is ready for use!")
        print(f"📁 Test artifacts preserved in: {testresults_dir}")
    else:
        print(f"\n💥 {total_modules - successful_modules} MODULE(S) FAILED")
        print(f"🔧 Review the errors above and fix the issues.")
        print(f"📁 Test artifacts available for debugging in: {testresults_dir}")
    
    return successful_modules == total_modules

def check_test_environment():
    """Überprüfe Test-Umgebung und Abhängigkeiten"""
    print("🔍 Checking Test Environment...")
    print("-" * 40)
    
    # Überprüfe Python-Version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Überprüfe wichtige Module
    required_modules = [
        'zarr',
        'numpy', 
        'pathlib',
        'subprocess',
        'tempfile'
    ]
    
    optional_modules = [
        'pytest',
        'av',  # PyAV für AAC
        'soundfile',  # für Audio-Analyse
    ]
    
    print(f"\nRequired Modules:")
    missing_required = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING!")
            missing_required.append(module)
    
    print(f"\nOptional Modules:")
    missing_optional = []
    for module in optional_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ⚠️  {module} - not available")
            missing_optional.append(module)
    
    # Überprüfe Test-Verzeichnisse
    testdata_dir = pathlib.Path(__file__).parent / "testdata"
    testresults_dir = pathlib.Path(__file__).parent / "testresults"
    
    print(f"\nTest Directories:")
    if testdata_dir.exists():
        audio_files = list(testdata_dir.glob("*.wav")) + list(testdata_dir.glob("*.mp3")) + list(testdata_dir.glob("*.flac"))
        print(f"  ✅ Testdata: {testdata_dir} ({len(audio_files)} audio files)")
        if len(audio_files) < 5:
            print(f"  ⚠️  Limited test files available")
    else:
        print(f"  ❌ Testdata directory not found: {testdata_dir}")
        return False
    
    # Erstelle/säubere testresults
    if testresults_dir.exists():
        import shutil
        shutil.rmtree(testresults_dir)
    testresults_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Test Results: {testresults_dir} (cleaned)")
    
    # Überprüfe ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n✅ ffmpeg available")
        else:
            print(f"\n❌ ffmpeg not working properly")
            return False
    except FileNotFoundError:
        print(f"\n❌ ffmpeg not found - REQUIRED for audio processing!")
        return False
    
    if missing_required:
        print(f"\n❌ Missing required modules: {', '.join(missing_required)}")
        return False
    
    print(f"\n✅ Test environment ready!")
    return True

def main():
    """Haupt-Test-Runner"""
    print("🧪 zarrwlr Import Modules Test Suite")
    print("=" * 80)
    
    # Überprüfe Test-Umgebung
    if not check_test_environment():
        print("❌ Test environment check failed!")
        return 1
    
    # Versuche pytest falls verfügbar
    pytest_result = run_pytest_if_available()
    
    if pytest_result is not None:
        # pytest war verfügbar
        if pytest_result:
            print("\n🎉 All pytest tests passed!")
            return 0
        else:
            print("\n❌ Some pytest tests failed!")
            return 1
    
    # Fallback zu direkten Tests
    test_results = run_direct_tests()
    
    # Umfassende Zusammenfassung
    all_passed = print_comprehensive_summary(test_results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


# ============================================================================
# pytest Integration (wird automatisch erkannt wenn pytest verfügbar ist)
# ============================================================================

def test_import_utils_module():
    """pytest-kompatible Wrapper für import_utils Tests"""
    from test_import_utils import run_tests
    assert run_tests() == True

def test_aimport_module():
    """pytest-kompatible Wrapper für aimport Tests"""
    from test_aimport import run_tests
    assert run_tests() == True

# Alle Test-Klassen für pytest-Discovery importieren
try:
    from test_import_utils import (
        TestEnumConversions,
        TestCodecClassification, 
        TestCopyModeFunctions,
        TestQualityAnalyzer,
        TestConflictAnalyzer,
        TestFileParameterWithRealFiles,
        TestIntegrationScenarios,
        TestPerformance as ImportUtilsPerformance,
        TestErrorHandling as ImportUtilsErrorHandling
    )
    
    from test_aimport import (
        TestAGroupWrapper,
        TestZarrAudioGroupManagement,
        TestImportStatusChecking,
        TestAudioImportOperations,
        TestImportWorker,
        TestSpecialScenarios,
        TestPerformance as AimportPerformance,
        TestErrorHandling as AimportErrorHandling,
        TestIntegration,
        TestConfigurationIntegration,
        TestMockAndSubprocess,
        TestStressScenarios
    )
    
except ImportError as e:
    # Falls Module nicht gefunden werden, erstelle Dummy-Tests
    def test_module_import_error():
        """Fallback-Test falls Module nicht importiert werden können"""
        pytest.skip(f"Could not import test modules: {e}")
