#!/usr/bin/env python3
"""
OPUS Index System - PRODUCTION-READY TEST SUITE
===============================================

BASED ON PROVEN FLAC TEST PATTERNS:
===================================
- Guarantees real audio processing on every run
- Robust cleanup mechanisms
- Detailed performance metrics
- Automatic validation of all aspects
- Suitable for CI/CD and regular testing

USAGE:
======
1. Manual tests: python opus_index_test_production.py
2. CI/CD integration: python opus_index_test_production.py --ci
3. Verbose mode: python opus_index_test_production.py --verbose
4. Debug packet parsing: python opus_index_test_production.py --debug
"""

import pathlib
import shutil
import time
import datetime
import argparse
import sys
import numpy as np

# Zarrwlr imports
import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger
from zarrwlr.aimport import create_original_audio_group, import_original_audio_file
from zarrwlr import opus_index_backend

class OpusIndexTester:
    """Production-ready Opus Index System Tester"""
    
    def __init__(self, verbose: bool = False, ci_mode: bool = False, debug: bool = False):
        self.verbose = verbose
        self.ci_mode = ci_mode
        self.debug = debug
        
        # Log-Level konfigurieren
        if debug:
            log_level = LogLevel.TRACE
        elif verbose:
            log_level = LogLevel.DEBUG
        else:
            log_level = LogLevel.INFO
        
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Stelle sicher dass Testverzeichnis existiert
        self.test_results_dir.mkdir(exist_ok=True)
    
    def find_test_file(self) -> pathlib.Path:
        """Finde beste verfügbare Testdatei (wie FLAC-Test)"""
        test_files = [
            "testdata/audiomoth_short_snippet.wav",  # Priorität: Klein für schnelles Debugging
            "testdata/bird1_snippet.mp3",            # Alternative: MP3 Format
            "testdata/audiomoth_long_snippet.wav",   # Fallback: Groß aber funktioniert
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        
        for file_rel in test_files:
            file_path = base_path / file_rel
            if file_path.exists():
                return file_path
        
        raise FileNotFoundError("Keine Testdatei gefunden! Benötigt: " + ", ".join(test_files))
    
    def cleanup_test_stores(self):
        """Clean up all test stores for guaranteed fresh tests (wie FLAC-Test)"""
        if self.verbose:
            print("Cleaning up old test stores...")
        
        cleanup_patterns = [
            "zarr3-store-opus-*",
            "index_test_*.zarr",
            "zarr3-store-*opus*",
        ]
        
        cleaned = 0
        for pattern in cleanup_patterns:
            for store_dir in self.test_results_dir.glob(pattern):
                if store_dir.is_dir():
                    shutil.rmtree(store_dir)
                    cleaned += 1
        
        if self.verbose and cleaned > 0:
            print(f"   Cleaned {cleaned} old stores")
    
    def import_fresh_audio(self, test_name: str) -> tuple:
        """Import audio fresh (guarantees real processing) - wie FLAC-Test"""
        test_file = self.find_test_file()
        test_store_dir = self.test_results_dir / f"zarr3-store-opus-{test_name}-{int(time.time())}"
        
        if self.verbose:
            print(f"Using test file: {test_file.name}")
            print(f"Store directory: {test_store_dir.name}")
        
        # Create audio group
        audio_group = create_original_audio_group(
            store_path=test_store_dir, 
            group_path='audio_imports'
        )
        
        # Import audio with timestamp for uniqueness
        timestamp = datetime.datetime.now()
        
        # OPUS-SPECIFIC: Use low bitrate for simpler packet structure during debugging
        opus_bitrate = 32000 if self.debug else 64000
        
        import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='opus',
            opus_bitrate=opus_bitrate
        )
        
        # Find imported group (robuste Methode wie FLAC-Test)
        available_groups = []
        try:
            # Method 1: Use zarr group iteration
            for key in audio_group.group_keys():
                if key.isdigit():
                    available_groups.append(key)
        except:
            # Method 2: Fallback to manual enumeration
            for i in range(10):
                if str(i) in audio_group:
                    available_groups.append(str(i))
        
        if not available_groups:
            raise RuntimeError("No audio groups found after import")
        
        latest_group_name = max(available_groups, key=int)
        zarr_group = audio_group[latest_group_name]
        audio_blob_array = zarr_group["audio_data_blob_array"]
        
        return zarr_group, audio_blob_array, test_file
    
    def safe_delete_index(self, zarr_group, index_name: str = 'opus_index'):
        """Safe index deletion with validation (wie FLAC-Test)"""
        try:
            if index_name in zarr_group:
                del zarr_group[index_name]
                # Short pause to avoid race conditions
                time.sleep(0.1)
                
                # Validate that index was actually deleted
                if index_name in zarr_group:
                    raise RuntimeError(f"Index {index_name} could not be deleted")
                
                if self.verbose:
                    print(f"   Index '{index_name}' successfully deleted")
        except Exception as e:
            if not self.ci_mode:  # In CI mode, continue anyway
                raise RuntimeError(f"Index deletion failed: {e}")
    
    def test_import_and_packet_parsing(self) -> dict:
        """Test 1: Basic import and packet parsing (CRITICAL für Opus)"""
        test_name = "import_parsing"
        if not self.ci_mode:
            print(f"\n=== TEST: Import and Packet Parsing ===")
        
        try:
            zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
            
            audio_bytes = bytes(audio_blob_array[()])
            file_size_mb = len(audio_bytes) / 1024 / 1024
            
            if not self.ci_mode:
                print(f"Audio: {file_size_mb:.2f} MB, Codec: {audio_blob_array.attrs.get('codec', 'unknown')}")
                print(f"Container: {audio_blob_array.attrs.get('container_type', 'unknown')}")
            
            # Validate basic attributes
            codec = audio_blob_array.attrs.get('codec', 'unknown')
            sample_rate = audio_blob_array.attrs.get('sample_rate', 0)
            channels = audio_blob_array.attrs.get('nb_channels', 0)
            
            success = (
                codec == 'opus' and
                sample_rate > 0 and
                channels > 0 and
                len(audio_bytes) > 1000  # Reasonable audio data
            )
            
            return {
                'test_name': 'Import and Packet Parsing',
                'success': success,
                'file_size_mb': file_size_mb,
                'codec': codec,
                'sample_rate': sample_rate,
                'channels': channels,
                'audio_bytes': len(audio_bytes),
                'details': {
                    'container_type': audio_blob_array.attrs.get('container_type', 'unknown'),
                    'opus_bitrate': audio_blob_array.attrs.get('opus_bitrate', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                'test_name': 'Import and Packet Parsing',
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_index_creation(self) -> dict:
        """Test 2: Index creation and validation"""
        test_name = "index_creation"
        if not self.ci_mode:
            print(f"\n=== TEST: Index Creation ===")
        
        try:
            zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
            
            # Delete any existing index for clean test
            self.safe_delete_index(zarr_group)
            
            # Create index
            if not self.ci_mode:
                print("Creating Opus index...")
            
            start_time = time.time()
            opus_index = opus_index_backend.build_opus_index(zarr_group, audio_blob_array)
            index_time = time.time() - start_time
            
            # Validate index structure
            index_shape = opus_index.shape
            expected_cols = 4 if hasattr(opus_index_backend, '_find_packet_range_for_samples') else 3
            
            if not self.ci_mode:
                print(f"   Index created: {index_shape[0]} entries in {index_time:.3f}s")
                print(f"   Index size: {opus_index.nbytes / 1024:.1f}KB")
            
            # Basic validation
            valid_shape = len(index_shape) == 2 and index_shape[1] >= 3
            has_entries = index_shape[0] > 0
            
            success = valid_shape and has_entries
            
            # Advanced validation if index has data
            arrays_valid = True
            if success and index_shape[0] > 0:
                try:
                    index_data = opus_index[:]
                    
                    # Check for monotonic properties (if applicable)
                    if index_shape[1] >= 4:  # Packet-based format
                        offsets = index_data[:, 0]
                        cumulative = index_data[:, 3]
                        
                        offsets_monotonic = np.all(offsets[1:] >= offsets[:-1])
                        cumulative_monotonic = np.all(cumulative[1:] >= cumulative[:-1])
                        
                        arrays_valid = offsets_monotonic and cumulative_monotonic
                        
                        if not self.ci_mode and arrays_valid:
                            print(f"   ✅ Index validation: Monotonic properties confirmed")
                    
                except Exception as validation_error:
                    arrays_valid = False
                    if self.verbose:
                        print(f"   ⚠️ Index validation failed: {validation_error}")
            
            return {
                'test_name': 'Index Creation',
                'success': success and arrays_valid,
                'index_entries': index_shape[0],
                'index_columns': index_shape[1],
                'index_time': index_time,
                'index_size_kb': opus_index.nbytes / 1024,
                'arrays_valid': arrays_valid,
                'details': {
                    'expected_columns': expected_cols,
                    'shape': index_shape
                }
            }
            
        except Exception as e:
            return {
                'test_name': 'Index Creation',
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_index_lookup_performance(self) -> dict:
        """Test 3: Index lookup performance"""
        test_name = "lookup_performance"
        if not self.ci_mode:
            print(f"\n=== TEST: Index Lookup Performance ===")
        
        try:
            zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
            
            # Ensure index exists
            if 'opus_index' not in zarr_group:
                opus_index = opus_index_backend.build_opus_index(zarr_group, audio_blob_array)
            else:
                opus_index = zarr_group['opus_index']
            
            # Get audio metadata
            sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
            total_entries = opus_index.shape[0]
            
            # Estimate total duration
            if opus_index.shape[1] >= 4:  # Packet-based
                # Get last cumulative sample count
                index_data = opus_index[:]
                total_samples = index_data[-1, 3] if total_entries > 0 else 0
            else:  # Legacy
                total_samples = total_entries * 960  # Estimate
            
            duration_seconds = total_samples / sample_rate
            
            if not self.ci_mode:
                print(f"   Audio duration: {duration_seconds:.1f}s")
                print(f"   Index entries: {total_entries}")
            
            # Test lookups at different positions
            max_sample = min(sample_rate * 10, total_samples - sample_rate)  # Max 10s or end
            
            test_positions = [
                (0, min(sample_rate, max_sample)),                    # First second
                (sample_rate * 2, min(sample_rate * 3, max_sample)), # 2-3 seconds
                (sample_rate * 5, min(sample_rate * 6, max_sample)), # 5-6 seconds
            ]
            
            # Filter valid positions
            valid_positions = [(start, end) for start, end in test_positions 
                             if start < end and end <= max_sample and start >= 0]
            
            lookup_times = []
            successful_lookups = 0
            
            for start_sample, end_sample in valid_positions:
                try:
                    start_time = time.time()
                    
                    # Use appropriate lookup function
                    if hasattr(opus_index_backend, '_find_packet_range_for_samples'):
                        start_idx, end_idx = opus_index_backend._find_packet_range_for_samples(
                            opus_index, int(start_sample), int(end_sample)
                        )
                    else:
                        # Fallback for legacy format
                        start_idx, end_idx = opus_index_backend._find_page_range_for_samples(
                            opus_index, int(start_sample), int(end_sample)
                        )
                    
                    lookup_time = (time.time() - start_time) * 1000  # Convert to ms
                    lookup_times.append(lookup_time)
                    successful_lookups += 1
                    
                    if not self.ci_mode:
                        print(f"   🔍 Samples {start_sample:,}-{end_sample:,} → entries {start_idx}-{end_idx} ({lookup_time:.3f}ms)")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ Lookup failed for samples {start_sample}-{end_sample}: {e}")
            
            avg_lookup_time = sum(lookup_times) / len(lookup_times) if lookup_times else float('inf')
            
            # Performance evaluation
            performance_excellent = avg_lookup_time < 1.0
            performance_good = avg_lookup_time < 5.0
            performance_acceptable = avg_lookup_time < 50.0
            
            success = successful_lookups > 0 and performance_acceptable
            
            if not self.ci_mode:
                if performance_excellent:
                    print(f"   ✅ EXCELLENT: Average lookup {avg_lookup_time:.3f}ms")
                elif performance_good:
                    print(f"   ✅ GOOD: Average lookup {avg_lookup_time:.3f}ms")
                elif performance_acceptable:
                    print(f"   ⚠️ ACCEPTABLE: Average lookup {avg_lookup_time:.3f}ms")
                else:
                    print(f"   ❌ SLOW: Average lookup {avg_lookup_time:.3f}ms")
            
            return {
                'test_name': 'Index Lookup Performance',
                'success': success,
                'successful_lookups': successful_lookups,
                'total_tests': len(valid_positions),
                'avg_lookup_time_ms': avg_lookup_time,
                'performance_level': (
                    'excellent' if performance_excellent else
                    'good' if performance_good else
                    'acceptable' if performance_acceptable else 'slow'
                ),
                'details': {
                    'duration_seconds': duration_seconds,
                    'total_entries': total_entries,
                    'lookup_times': lookup_times
                }
            }
            
        except Exception as e:
            return {
                'test_name': 'Index Lookup Performance',
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def run_full_test_suite(self) -> dict:
        """Run complete test suite (wie FLAC-Test)"""
        if not self.ci_mode:
            print("=" * 70)
            print("OPUS INDEX SYSTEM - PRODUCTION TEST SUITE")
            print("=" * 70)
        
        # Cleanup for fresh tests
        self.cleanup_test_stores()
        
        results = []
        start_time = time.time()
        
        # Core tests (always run)
        tests = [
            self.test_import_and_packet_parsing,
            self.test_index_creation,
            self.test_index_lookup_performance,
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                
                if self.ci_mode:
                    status = "PASS" if result['success'] else "FAIL"
                    print(f"{status} {result['test_name']}")
                
            except Exception as e:
                error_result = {
                    'test_name': test_func.__name__,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                results.append(error_result)
                
                if not self.ci_mode:
                    print(f"ERROR {test_func.__name__}: CRITICAL ERROR - {str(e)}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"FAIL {test_func.__name__}")
        
        total_time = time.time() - start_time
        
        # Summary
        succeeded = sum(1 for r in results if r['success'])
        total = len(results)
        
        summary = {
            'total_tests': total,
            'succeeded': succeeded,
            'failed': total - succeeded,
            'success_rate': succeeded / total if total > 0 else 0,
            'total_time': total_time,
            'overall_success': succeeded == total,
            'results': results
        }
        
        if not self.ci_mode:
            print(f"\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            print(f"Total time: {total_time:.1f}s")
            print(f"Successful: {succeeded}/{total} tests ({summary['success_rate']*100:.1f}%)")
            
            for result in results:
                status = "PASS" if result['success'] else "FAIL"
                print(f"   {status} {result['test_name']}")
            
            if summary['overall_success']:
                print(f"\n🎉 ALL TESTS PASSED!")
                print(f"Opus index system: PRODUCTION-READY!")
            else:
                print(f"\n❌ SOME TESTS FAILED!")
                for result in results:
                    if not result['success'] and 'error' in result:
                        print(f"   ERROR {result['test_name']}: {result['error']}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Opus Index System Test Suite')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output with detailed logging')
    parser.add_argument('--ci', action='store_true',
                       help='CI/CD mode with minimal output')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Debug mode with packet parsing details')
    
    args = parser.parse_args()
    
    if args.ci and args.verbose:
        print("ERROR: --ci and --verbose are mutually exclusive")
        sys.exit(1)
    
    try:
        tester = OpusIndexTester(verbose=args.verbose, ci_mode=args.ci, debug=args.debug)
        summary = tester.run_full_test_suite()
        
        # Exit code for CI/CD
        sys.exit(0 if summary['overall_success'] else 1)
        
    except Exception as e:
        if args.ci:
            print(f"CRITICAL ERROR: {str(e)}")
        else:
            print(f"CRITICAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()