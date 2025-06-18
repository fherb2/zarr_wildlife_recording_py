#!/usr/bin/env python3
"""
Test Utilities for zarrwlr Import Tests

Hilfsskript für Test-Verwaltung, Cleanup und Debugging.

Verwendung:
    python test_utils.py --help
    python test_utils.py clean
    python test_utils.py list-results
    python test_utils.py analyze-results
    python test_utils.py debug <test_name>
"""

import argparse
import pathlib
import shutil
import sys
import time
from typing import List, Dict

# Test-Verzeichnisse
TESTS_DIR = pathlib.Path(__file__).parent
TESTDATA_DIR = TESTS_DIR / "testdata"
TESTRESULTS_DIR = TESTS_DIR / "testresults"

def clean_test_results():
    """Räume Test-Ergebnisse-Verzeichnis auf"""
    if TESTRESULTS_DIR.exists():
        print(f"🧹 Cleaning test results directory: {TESTRESULTS_DIR}")
        
        # Zeige Größe vor dem Löschen
        try:
            total_size = sum(f.stat().st_size for f in TESTRESULTS_DIR.rglob('*') if f.is_file())
            print(f"   Removing {total_size / 1024 / 1024:.2f} MB of test data")
        except Exception:
            pass
        
        shutil.rmtree(TESTRESULTS_DIR)
        print("   ✅ Test results cleaned")
    else:
        print("   ℹ️  Test results directory doesn't exist")
    
    # Erstelle leeres Verzeichnis
    TESTRESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   📁 Empty test results directory created")

def list_test_results():
    """Liste Test-Ergebnisse auf"""
    if not TESTRESULTS_DIR.exists():
        print("❌ No test results directory found")
        return
    
    print(f"📋 Test Results in {TESTRESULTS_DIR}:")
    print("-" * 50)
    
    test_dirs = sorted([d for d in TESTRESULTS_DIR.iterdir() if d.is_dir()])
    
    if not test_dirs:
        print("   📭 No test results found")
        return
    
    total_size = 0
    for test_dir in test_dirs:
        try:
            # Berechne Größe des Test-Verzeichnisses
            dir_size = sum(f.stat().st_size for f in test_dir.rglob('*') if f.is_file())
            total_size += dir_size
            
            # Zeige Verzeichnis-Info
            file_count = len(list(test_dir.rglob('*')))
            mod_time = test_dir.stat().st_mtime
            mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
            
            print(f"   📁 {test_dir.name}")
            print(f"      Size: {dir_size / 1024:.1f} KB, Files: {file_count}, Modified: {mod_time_str}")
            
            # Zeige wichtige Dateien
            zarr_stores = list(test_dir.glob("*/zarr.json"))
            if zarr_stores:
                print(f"      🗄️  Zarr stores: {len(zarr_stores)}")
            
            log_files = list(test_dir.glob("*.log"))
            if log_files:
                print(f"      📝 Log files: {len(log_files)}")
                
        except Exception as e:
            print(f"   ❌ Error reading {test_dir.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Total test directories: {len(test_dirs)}")
    print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")

def analyze_test_results():
    """Analysiere Test-Ergebnisse detailliert"""
    if not TESTRESULTS_DIR.exists():
        print("❌ No test results directory found")
        return
    
    print(f"🔍 Analyzing Test Results in {TESTRESULTS_DIR}:")
    print("=" * 60)
    
    test_dirs = sorted([d for d in TESTRESULTS_DIR.iterdir() if d.is_dir()])
    
    if not test_dirs:
        print("   📭 No test results found")
        return
    
    # Kategorisiere Tests
    categories = {
        'zarr_operations': [],
        'file_analysis': [],
        'import_operations': [],
        'error_tests': [],
        'performance_tests': [],
        'other': []
    }
    
    for test_dir in test_dirs:
        test_name = test_dir.name.lower()
        
        if 'zarr' in test_name or 'store' in test_name:
            categories['zarr_operations'].append(test_dir)
        elif 'analysis' in test_name or 'parameter' in test_name:
            categories['file_analysis'].append(test_dir)
        elif 'import' in test_name or 'aimport' in test_name:
            categories['import_operations'].append(test_dir)
        elif 'error' in test_name or 'corrupted' in test_name or 'invalid' in test_name:
            categories['error_tests'].append(test_dir)
        elif 'performance' in test_name or 'stress' in test_name:
            categories['performance_tests'].append(test_dir)
        else:
            categories['other'].append(test_dir)
    
    # Zeige Kategorien
    for category, dirs in categories.items():
        if dirs:
            print(f"\n📂 {category.replace('_', ' ').title()}: ({len(dirs)} tests)")
            
            total_size = 0
            for test_dir in dirs:
                try:
                    dir_size = sum(f.stat().st_size for f in test_dir.rglob('*') if f.is_file())
                    total_size += dir_size
                    print(f"   📁 {test_dir.name} ({dir_size / 1024:.1f} KB)")
                    
                    # Zeige spezielle Dateien
                    zarr_files = list(test_dir.rglob("zarr.json"))
                    audio_files = list(test_dir.rglob("*.wav")) + list(test_dir.rglob("*.mp3")) + list(test_dir.rglob("*.flac"))
                    
                    if zarr_files:
                        print(f"      🗄️  Zarr archives: {len(zarr_files)}")
                    if audio_files:
                        print(f"      🎵 Audio files: {len(audio_files)}")
                
                except Exception as e:
                    print(f"   ❌ {test_dir.name}: Error - {e}")
            
            print(f"   Total category size: {total_size / 1024 / 1024:.2f} MB")

def debug_test_result(test_name: str):
    """Debug spezifisches Test-Ergebnis"""
    # Finde Test-Verzeichnis
    matching_dirs = [d for d in TESTRESULTS_DIR.iterdir() 
                    if d.is_dir() and test_name.lower() in d.name.lower()]
    
    if not matching_dirs:
        print(f"❌ No test results found matching '{test_name}'")
        available = [d.name for d in TESTRESULTS_DIR.iterdir() if d.is_dir()]
        if available:
            print(f"Available test results: {', '.join(available[:5])}")
            if len(available) > 5:
                print(f"... and {len(available) - 5} more")
        return
    
    if len(matching_dirs) > 1:
        print(f"🔍 Multiple matches found for '{test_name}':")
        for i, d in enumerate(matching_dirs):
            print(f"   {i+1}. {d.name}")
        
        try:
            choice = int(input("Select test result (number): ")) - 1
            if 0 <= choice < len(matching_dirs):
                test_dir = matching_dirs[choice]
            else:
                print("❌ Invalid selection")
                return
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid selection")
            return
    else:
        test_dir = matching_dirs[0]
    
    print(f"🔍 Debugging Test Result: {test_dir.name}")
    print("=" * 60)
    
    # Zeige Verzeichnisstruktur
    print("📁 Directory Structure:")
    for item in sorted(test_dir.rglob('*')):
        if item.is_file():
            relative_path = item.relative_to(test_dir)
            size = item.stat().st_size
            print(f"   📄 {relative_path} ({size} bytes)")
        elif item.is_dir() and item != test_dir:
            relative_path = item.relative_to(test_dir)
            file_count = len(list(item.iterdir()))
            print(f"   📁 {relative_path}/ ({file_count} items)")
    
    # Suche nach Zarr-Stores
    zarr_stores = list(test_dir.rglob("zarr.json"))
    if zarr_stores:
        print(f"\n🗄️  Zarr Stores Found: {len(zarr_stores)}")
        for zarr_file in zarr_stores:
            zarr_dir = zarr_file.parent
            print(f"   📦 {zarr_dir.relative_to(test_dir)}")
            
            # Zeige Zarr-Inhalt
            try:
                import zarr
                store = zarr.storage.LocalStore(str(zarr_dir))
                root = zarr.open_group(store, mode='r')
                
                print(f"      Groups: {list(root.keys())}")
                if hasattr(root, 'attrs'):
                    attrs = dict(root.attrs)
                    if attrs:
                        print(f"      Attrs: {attrs}")
                        
            except Exception as e:
                print(f"      ❌ Error reading Zarr: {e}")
    
    # Suche nach Log-Dateien
    log_files = list(test_dir.rglob("*.log"))
    if log_files:
        print(f"\n📝 Log Files Found: {len(log_files)}")
        for log_file in log_files:
            print(f"   📄 {log_file.relative_to(test_dir)}")
            try:
                content = log_file.read_text()
                lines = content.split('\n')
                print(f"      Lines: {len(lines)}")
                
                # Zeige letzte paar Zeilen
                if lines:
                    print("      Last 3 lines:")
                    for line in lines[-3:]:
                        if line.strip():
                            print(f"        {line.strip()}")
            except Exception as e:
                print(f"      ❌ Error reading log: {e}")

def show_test_data():
    """Zeige verfügbare Testdaten"""
    print(f"📊 Test Data in {TESTDATA_DIR}:")
    print("-" * 40)
    
    if not TESTDATA_DIR.exists():
        print("❌ Test data directory not found!")
        return
    
    # Audio-Dateien
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        files = list(TESTDATA_DIR.glob(f"*{ext}"))
        if files:
            audio_files.extend(files)
    
    if not audio_files:
        print("❌ No audio files found in test data directory!")
        return
    
    total_size = 0
    for audio_file in sorted(audio_files):
        size = audio_file.stat().st_size
        total_size += size
        print(f"   🎵 {audio_file.name} ({size / 1024 / 1024:.2f} MB)")
    
    print(f"\n📊 Summary:")
    print(f"   Total audio files: {len(audio_files)}")
    print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
    
    # Zeige auch Analysis-Report falls vorhanden
    analysis_report = TESTDATA_DIR / "TESTDATA_ANALYSIS.md"
    if analysis_report.exists():
        print(f"   📋 Analysis report available: {analysis_report.name}")

def main():
    """Haupt-CLI-Interface"""
    parser = argparse.ArgumentParser(
        description="Test Utilities for zarrwlr Import Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Clean command
    subparsers.add_parser('clean', help='Clean test results directory')
    
    # List command
    subparsers.add_parser('list-results', help='List test result directories')
    
    # Analyze command
    subparsers.add_parser('analyze-results', help='Analyze test results in detail')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug specific test result')
    debug_parser.add_argument('test_name', help='Name or pattern of test to debug')
    
    # Show test data command
    subparsers.add_parser('show-testdata', help='Show available test data files')
    
    args = parser.parse_args()
    
    if args.command == 'clean':
        clean_test_results()
    elif args.command == 'list-results':
        list_test_results()
    elif args.command == 'analyze-results':
        analyze_test_results()
    elif args.command == 'debug':
        debug_test_result(args.test_name)
    elif args.command == 'show-testdata':
        show_test_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
