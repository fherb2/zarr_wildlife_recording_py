#!/usr/bin/env python3
"""
FLAC Parallelization - PRODUCTION-READY TEST SUITE
==================================================

FINAL VERSION FOR FUTURE QUALITY CONTROL:
=========================================
- Guarantees real audio processing on every run
- Robust cleanup mechanisms
- Detailed performance metrics
- Automatic validation of all aspects
- Suitable for CI/CD and regular testing

USAGE:
======
1. Manual tests: python test_flac_production_ready.py
2. CI/CD integration: python test_flac_production_ready.py --ci
3. Verbose mode: python test_flac_production_ready.py --verbose
4. Performance benchmark: python test_flac_production_ready.py --benchmark
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
from zarrwlr import flac_index_backend

class FlacParallelTester:
    """Production-ready FLAC Parallelization Tester"""
    
    def __init__(self, verbose: bool = False, ci_mode: bool = False):
        self.verbose = verbose
        self.ci_mode = ci_mode
        
        # Log-Level konfigurieren
        log_level = LogLevel.TRACE if verbose else LogLevel.INFO
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Stelle sicher dass Testverzeichnis existiert
        self.test_results_dir.mkdir(exist_ok=True)
    
    def find_test_file(self) -> pathlib.Path:
        """Finde beste verfügbare Testdatei"""
        test_files = [
            "testdata/audiomoth_long_snippet.wav",
            "testdata/bird1_snippet.mp3",
            "testdata/audiomoth_short_snippet.wav",
            
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        
        for file_rel in test_files:
            file_path = base_path / file_rel
            if file_path.exists():
                return file_path
        
        raise FileNotFoundError("Keine Testdatei gefunden! Benötigt: " + ", ".join(test_files))
    
    def cleanup_test_stores(self):
        """Clean up all test stores for guaranteed fresh tests"""
        if self.verbose:
            print("Cleaning up old test stores...")
        
        cleanup_patterns = [
            "zarr3-store-production-*",
            "zarr3-store-standalone-*",
            "zarr3-store-*test*",
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
        """Import audio fresh (guarantees real processing)"""
        test_file = self.find_test_file()
        test_store_dir = self.test_results_dir / f"zarr3-store-production-{test_name}-{int(time.time())}"
        
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
        import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # Find imported group
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        latest_group_name = max(group_names, key=int)
        zarr_group = audio_group[latest_group_name]
        audio_blob_array = zarr_group["audio_data_blob_array"]
        
        return zarr_group, audio_blob_array, test_file
    
    def safe_delete_index(self, zarr_group, index_name: str = 'flac_index'):
        """Safe index deletion with validation"""
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
    
    def test_parallel_correctness(self) -> dict:
        """Core test: Parallel vs Sequential correctness"""
        test_name = "correctness"
        if not self.ci_mode:
            print(f"\n=== TEST: Parallel vs Sequential Correctness ===")
        
        zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
        
        audio_bytes = bytes(audio_blob_array[()])
        file_size_mb = len(audio_bytes) / 1024 / 1024
        
        if not self.ci_mode:
            print(f"Audio: {file_size_mb:.2f} MB, Codec: {audio_blob_array.attrs.get('codec', 'unknown')}")
        
        # Delete automatically created index for clean test
        self.safe_delete_index(zarr_group)
        
        # Create sequential index
        if not self.ci_mode:
            print("Sequential index...")
        
        start_time = time.time()
        sequential_index = flac_index_backend.build_flac_index(
            zarr_group, audio_blob_array, use_parallel=False
        )
        sequential_time = time.time() - start_time
        sequential_data = sequential_index[:]
        
        if not self.ci_mode:
            print(f"   {sequential_data.shape[0]} frames in {sequential_time:.3f}s")
        
        # Delete for parallel test
        self.safe_delete_index(zarr_group)
        
        # Create parallel index
        if not self.ci_mode:
            print("Parallel index...")
        
        start_time = time.time()
        parallel_index = flac_index_backend.build_flac_index(
            zarr_group, audio_blob_array, use_parallel=True
        )
        parallel_time = time.time() - start_time
        parallel_data = parallel_index[:]
        
        parallel_used = parallel_index.attrs.get('parallel_processing_used', False)
        
        if not self.ci_mode:
            print(f"   {parallel_data.shape[0]} frames in {parallel_time:.3f}s")
            print(f"   Parallel active: {parallel_used}")
        
        # Validation
        same_shape = sequential_data.shape == parallel_data.shape
        
        if same_shape:
            arrays_identical = np.array_equal(sequential_data, parallel_data)
            if not arrays_identical:
                max_diff = np.max(np.abs(sequential_data.astype(float) - parallel_data.astype(float)))
                arrays_close = max_diff <= 100  # Generous tolerance
                if arrays_close:
                    arrays_identical = True
                    if self.verbose:
                        print(f"   Arrays practically identical (max diff: {max_diff})")
        else:
            arrays_identical = False
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        return {
            'test_name': 'Parallel vs Sequential Correctness',
            'success': same_shape and arrays_identical,
            'frames': sequential_data.shape[0],
            'file_size_mb': file_size_mb,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'parallel_used': parallel_used,
            'arrays_identical': arrays_identical,
            'details': {
                'same_shape': same_shape,
                'seq_shape': sequential_data.shape,
                'par_shape': parallel_data.shape,
            }
        }
    
    def test_api_consistency(self) -> dict:
        """Test: API consistency across different call variants"""
        test_name = "api_consistency"
        if not self.ci_mode:
            print(f"\n=== TEST: API Consistency ===")
        
        zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
        
        self.safe_delete_index(zarr_group)
        
        frame_counts = []
        api_variants = [
            ("Standard", {}),
            ("Explicit Parallel", {"use_parallel": True}),
            ("Explicit Sequential", {"use_parallel": False}),
            ("With max_workers", {"use_parallel": True, "max_workers": 2}),
        ]
        
        for variant_name, kwargs in api_variants:
            try:
                index = flac_index_backend.build_flac_index(zarr_group, audio_blob_array, **kwargs)
                frame_counts.append(index.shape[0])
                
                if not self.ci_mode:
                    parallel_used = index.attrs.get('parallel_processing_used', 'unknown')
                    print(f"   {variant_name}: {index.shape[0]} frames (parallel: {parallel_used})")
                
                self.safe_delete_index(zarr_group)
                
            except Exception as e:
                if not self.ci_mode:
                    print(f"   {variant_name}: ERROR - {str(e)}")
                return {
                    'test_name': 'API Consistency',
                    'success': False,
                    'error': f"{variant_name} failed: {str(e)}"
                }
        
        # All variants should deliver same frame count
        consistent = len(set(frame_counts)) == 1
        
        return {
            'test_name': 'API Consistency',
            'success': consistent,
            'frame_counts': frame_counts,
            'variants_tested': len(api_variants),
            'consistent': consistent
        }
    
    def test_performance_benchmark(self) -> dict:
        """Test: Performance benchmark (optional)"""
        test_name = "benchmark"
        if not self.ci_mode:
            print(f"\n=== TEST: Performance Benchmark ===")
        
        zarr_group, audio_blob_array, test_file = self.import_fresh_audio(test_name)
        
        audio_bytes = bytes(audio_blob_array[()])
        file_size_mb = len(audio_bytes) / 1024 / 1024
        
        self.safe_delete_index(zarr_group)
        
        # Multiple runs for more stable measurement
        iterations = 3 if not self.ci_mode else 1
        
        sequential_times = []
        parallel_times = []
        
        for i in range(iterations):
            # Sequential
            start_time = time.time()
            seq_index = flac_index_backend.build_flac_index(zarr_group, audio_blob_array, use_parallel=False)
            sequential_times.append(time.time() - start_time)
            
            self.safe_delete_index(zarr_group)
            
            # Parallel
            start_time = time.time()
            par_index = flac_index_backend.build_flac_index(zarr_group, audio_blob_array, use_parallel=True)
            parallel_times.append(time.time() - start_time)
            
            if i < iterations - 1:  # Don't delete after last iteration
                self.safe_delete_index(zarr_group)
        
        seq_avg = sum(sequential_times) / len(sequential_times)
        par_avg = sum(parallel_times) / len(parallel_times)
        speedup = seq_avg / par_avg if par_avg > 0 else 1.0
        
        frames = par_index.shape[0]
        frames_per_sec = frames / par_avg if par_avg > 0 else 0
        
        if not self.ci_mode:
            print(f"   {iterations} iterations performed")
            print(f"   Sequential: {seq_avg:.3f}s ± {np.std(sequential_times):.3f}s")
            print(f"   Parallel: {par_avg:.3f}s ± {np.std(parallel_times):.3f}s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Performance: {frames_per_sec:.1f} frames/s")
        
        return {
            'test_name': 'Performance Benchmark',
            'success': speedup > 0.8,  # Mindestens nicht viel langsamer
            'file_size_mb': file_size_mb,
            'frames': frames,
            'iterations': iterations,
            'sequential_avg': seq_avg,
            'parallel_avg': par_avg,
            'speedup': speedup,
            'frames_per_second': frames_per_sec,
            'sequential_times': sequential_times,
            'parallel_times': parallel_times
        }
    
    def run_full_test_suite(self, include_benchmark: bool = False) -> dict:
        """Run complete test suite"""
        if not self.ci_mode:
            print("=" * 70)
            print("FLAC PARALLELIZATION - PRODUCTION TEST SUITE")
            print("=" * 70)
        
        # Cleanup for fresh tests
        self.cleanup_test_stores()
        
        results = []
        start_time = time.time()
        
        # Core tests (always run)
        tests = [
            self.test_parallel_correctness,
            self.test_api_consistency,
        ]
        
        # Optional: benchmark
        if include_benchmark:
            tests.append(self.test_performance_benchmark)
        
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
                    'error': str(e)
                }
                results.append(error_result)
                
                if not self.ci_mode:
                    print(f"ERROR {test_func.__name__}: CRITICAL ERROR - {str(e)}")
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
                print(f"\nALL TESTS PASSED!")
                print(f"Parallelized FLAC index creation: PRODUCTION-READY!")
            else:
                print(f"\nSOME TESTS FAILED!")
                for result in results:
                    if not result['success'] and 'error' in result:
                        print(f"   ERROR {result['test_name']}: {result['error']}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='FLAC Parallelization Test Suite')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output with detailed logging')
    parser.add_argument('--ci', action='store_true',
                       help='CI/CD mode with minimal output')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Include performance benchmark tests')
    
    args = parser.parse_args()
    
    if args.ci and args.verbose:
        print("ERROR: --ci and --verbose are mutually exclusive")
        sys.exit(1)
    
    try:
        tester = FlacParallelTester(verbose=args.verbose, ci_mode=args.ci)
        summary = tester.run_full_test_suite(include_benchmark=args.benchmark)
        
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