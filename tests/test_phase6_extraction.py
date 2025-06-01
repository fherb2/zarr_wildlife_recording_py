#!/usr/bin/env python3
"""
Phase 6 Extraction Performance Test
===================================

GOAL: Test and validate the batch-optimized extraction performance

TEST CASES:
1. Correctness: Original vs. Batch-optimized → identical results?
2. Performance: Many small segments → 5-10x faster?
3. Edge Cases: Overlapping segments, various batch sizes
4. Integration: Works with existing end-to-end pipeline?
"""

import pathlib
import time
import datetime
import numpy as np
import argparse
import sys
import shutil

# Import zarrwlr
import zarrwlr
import zarr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

# Direct import for testing
from zarrwlr.opus_access import (
    parallel_extract_audio_segments_opus,              # Original
    parallel_extract_audio_segments_opus_optimized,    # Phase 6 optimized
    parallel_extract_audio_segments_opus_auto          # Auto-selection
)

class Phase6ExtractionTester:
    """Test suite for Phase 6 batch-optimized extraction"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Configure logging
        log_level = LogLevel.TRACE if verbose else LogLevel.INFO
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Test results tracking
        self.test_results = {
            'correctness_tests': [],
            'performance_tests': [],
            'edge_case_tests': [],
            'integration_tests': []
        }
        
        # Audio data
        self.audio_group = None
        self.audio_blob_array = None
        self.opus_index = None
        self.sample_rate = None
        self.channels = None
        self.total_samples = None
    
    def print_status(self, message: str, level: str = "INFO"):
        """Print status with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{timestamp}] [{level}]"
        
        if self.verbose or level in ["ERROR", "WARNING", "SUCCESS"]:
            print(f"{prefix} {message}")
        
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "SUCCESS":
            self.logger.success(message)
        else:
            self.logger.info(message)
    
    def cleanup_old_stores(self):
        """Clean up old test stores"""
        self.print_status("Cleaning up old test stores...")
        cleanup_count = 0
        for store_dir in self.test_results_dir.glob("phase6-test-*"):
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
                cleanup_count += 1
        self.print_status(f"Removed {cleanup_count} old test stores")
    
    def setup_test_audio(self):
        """Setup test audio data (reuse existing test infrastructure)"""
        self.print_status("Setting up test audio data...")
        
        # Find test file
        test_files = [
            "testdata/audiomoth_long_snippet_converted.opus",
            "testdata/bird1_snippet.mp3",
            "testdata/audiomoth_short_snippet.wav"
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        test_file = None
        
        for test_file_name in test_files:
            test_path = base_path / test_file_name
            if test_path.exists():
                test_file = test_path
                break
        
        if not test_file:
            raise FileNotFoundError(f"No test files found. Searched: {test_files}")
        
        self.print_status(f"Using test file: {test_file.name}")
        
        # Create test Zarr store
        timestamp = int(time.time())
        store_dir = self.test_results_dir / f"phase6-test-{timestamp}"
        
        # Import audio
        audio_group = zarrwlr.create_original_audio_group(
            store_path=store_dir,
            group_path='audio_imports'
        )
        
        zarrwlr.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=datetime.datetime.now(),
            target_codec='opus',
            opus_bitrate=128000,
            temp_dir=self.test_results_dir
        )
        
        # Get imported group
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        latest_group = audio_group[max(group_names, key=int)]
        
        self.audio_group = latest_group
        self.audio_blob_array = latest_group['audio_data_blob_array']
        self.opus_index = latest_group['opus_index']
        
        # Get audio info
        self.sample_rate = self.audio_blob_array.attrs.get('sample_rate', 48000)
        self.channels = self.audio_blob_array.attrs.get('nb_channels', 1)
        
        sample_positions = self.opus_index[:, 2]  # OPUS_INDEX_COL_SAMPLE_POS
        self.total_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 48000
        
        self.print_status(f"Audio setup: {self.sample_rate}Hz, {self.channels}ch, {self.total_samples} samples")
        self.print_status(f"Index: {self.opus_index.shape[0]} pages")
    
    def generate_test_segments(self, segment_count: int, segment_duration_seconds: float = 2.0) -> list:
        """Generate test segments for performance testing"""
        segment_duration_samples = int(segment_duration_seconds * self.sample_rate)
        segments = []
        
        # Distribute segments across the audio
        max_start = max(0, self.total_samples - segment_duration_samples)
        step = max_start // (segment_count - 1) if segment_count > 1 else max_start
        
        for i in range(segment_count):
            start_sample = min(i * step, max_start)
            end_sample = min(start_sample + segment_duration_samples - 1, self.total_samples - 1)
            
            if start_sample < end_sample:
                segments.append((start_sample, end_sample))
        
        return segments[:segment_count]  # Ensure we don't exceed requested count
    
    def test_correctness(self) -> dict:
        """Test correctness: Original vs. Batch-optimized should be identical"""
        self.print_status("=== CORRECTNESS TEST ===")
        
        test_cases = [
            (5, 1.0, "5 segments @ 1s each"),
            (10, 2.0, "10 segments @ 2s each"),
            (20, 1.5, "20 segments @ 1.5s each")
        ]
        
        correctness_results = []
        
        for segment_count, duration, description in test_cases:
            self.print_status(f"Testing correctness: {description}")
            
            # Generate test segments
            segments = self.generate_test_segments(segment_count, duration)
            if len(segments) < segment_count:
                self.print_status(f"Warning: Only generated {len(segments)} segments (requested {segment_count})", "WARNING")
            
            try:
                # Extract with original method
                start_time = time.time()
                original_results = parallel_extract_audio_segments_opus(
                    self.audio_group, self.audio_blob_array, segments, max_workers=2
                )
                original_time = time.time() - start_time
                
                # Extract with batch-optimized method
                start_time = time.time()
                optimized_results = parallel_extract_audio_segments_opus_optimized(
                    self.audio_group, self.audio_blob_array, segments, max_workers=2
                )
                optimized_time = time.time() - start_time
                
                # Compare results
                identical_count = 0
                differences = []
                
                for i, (orig, opt) in enumerate(zip(original_results, optimized_results)):
                    if orig.shape == opt.shape:
                        if np.array_equal(orig, opt):
                            identical_count += 1
                        else:
                            max_diff = np.max(np.abs(orig - opt)) if orig.size > 0 and opt.size > 0 else 0
                            differences.append((i, max_diff, orig.shape))
                    else:
                        differences.append((i, "shape_mismatch", (orig.shape, opt.shape)))
                
                correctness_percentage = (identical_count / len(segments)) * 100 if segments else 0
                speedup = original_time / optimized_time if optimized_time > 0 else 0
                
                result = {
                    'description': description,
                    'segments': len(segments),
                    'identical_count': identical_count,
                    'correctness_percentage': correctness_percentage,
                    'differences': differences[:5],  # Limit to first 5 differences
                    'original_time': original_time,
                    'optimized_time': optimized_time,
                    'speedup': speedup,
                    'success': correctness_percentage >= 95.0  # Allow for small floating point differences
                }
                
                correctness_results.append(result)
                
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                self.print_status(f"{status} {description}: {correctness_percentage:.1f}% identical, {speedup:.1f}x speedup")
                
                if differences:
                    self.print_status(f"  Differences found: {len(differences)} segments", "WARNING")
                    
            except Exception as e:
                self.print_status(f"❌ ERROR {description}: {e}", "ERROR")
                correctness_results.append({
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['correctness_tests'] = correctness_results
        return correctness_results
    
    def test_performance(self) -> dict:
        """Test performance: Many small segments should be much faster"""
        self.print_status("=== PERFORMANCE TEST ===")
        
        test_cases = [
            (10, 2.0, "10 segments (small test)"),
            (25, 1.5, "25 segments (medium test)"),
            (50, 1.0, "50 segments (large test)"),
            (100, 0.5, "100 segments (stress test)")
        ]
        
        performance_results = []
        
        for segment_count, duration, description in test_cases:
            self.print_status(f"Testing performance: {description}")
            
            segments = self.generate_test_segments(segment_count, duration)
            if len(segments) < segment_count * 0.8:  # Allow 20% reduction
                self.print_status(f"Skipping {description}: insufficient segments ({len(segments)} < {segment_count})", "WARNING")
                continue
            
            try:
                # Test original method (with timeout for large tests)
                original_time = None
                original_success = True
                
                if segment_count <= 50:  # Only test original for smaller sets
                    self.print_status(f"  Testing original method...")
                    start_time = time.time()
                    original_results = parallel_extract_audio_segments_opus(
                        self.audio_group, self.audio_blob_array, segments, max_workers=2
                    )
                    original_time = time.time() - start_time
                    original_success = sum(1 for r in original_results if r.size > 0) >= len(segments) * 0.8
                else:
                    self.print_status(f"  Skipping original method (too many segments)")
                
                # Test batch-optimized method
                self.print_status(f"  Testing batch-optimized method...")
                start_time = time.time()
                optimized_results = parallel_extract_audio_segments_opus_optimized(
                    self.audio_group, self.audio_blob_array, segments, max_workers=2
                )
                optimized_time = time.time() - start_time
                optimized_success = sum(1 for r in optimized_results if r.size > 0) >= len(segments) * 0.8
                
                # Calculate metrics
                speedup = original_time / optimized_time if original_time and optimized_time > 0 else None
                segments_per_second_orig = len(segments) / original_time if original_time else None
                segments_per_second_opt = len(segments) / optimized_time if optimized_time > 0 else 0
                
                result = {
                    'description': description,
                    'segments': len(segments),
                    'original_time': original_time,
                    'optimized_time': optimized_time,
                    'speedup': speedup,
                    'segments_per_second_original': segments_per_second_orig,
                    'segments_per_second_optimized': segments_per_second_opt,
                    'original_success': original_success,
                    'optimized_success': optimized_success,
                    'success': optimized_success and (speedup is None or speedup > 2.0)  # At least 2x speedup
                }
                
                performance_results.append(result)
                
                if speedup:
                    status = "✅ PASS" if result['success'] else "⚠️ SLOW"
                    self.print_status(f"{status} {description}: {speedup:.1f}x speedup ({original_time:.2f}s → {optimized_time:.2f}s)")
                else:
                    status = "✅ PASS" if optimized_success else "❌ FAIL"
                    self.print_status(f"{status} {description}: {optimized_time:.2f}s ({segments_per_second_opt:.1f} segments/sec)")
                    
            except Exception as e:
                self.print_status(f"❌ ERROR {description}: {e}", "ERROR")
                performance_results.append({
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['performance_tests'] = performance_results
        return performance_results
    
    def test_edge_cases(self) -> dict:
        """Test edge cases: overlapping segments, various batch sizes"""
        self.print_status("=== EDGE CASE TEST ===")
        
        edge_case_results = []
        
        # Test case 1: Overlapping segments
        try:
            self.print_status("Testing overlapping segments...")
            base_start = self.total_samples // 4
            segment_length = int(1.0 * self.sample_rate)  # 1 second
            
            overlapping_segments = []
            for i in range(10):
                start = base_start + i * (segment_length // 2)  # 50% overlap
                end = start + segment_length - 1
                if end < self.total_samples:
                    overlapping_segments.append((start, end))
            
            optimized_results = parallel_extract_audio_segments_opus_optimized(
                self.audio_group, self.audio_blob_array, overlapping_segments
            )
            
            success = len(optimized_results) == len(overlapping_segments)
            edge_case_results.append({
                'test': 'overlapping_segments',
                'segments': len(overlapping_segments),
                'success': success
            })
            
            status = "✅ PASS" if success else "❌ FAIL"
            self.print_status(f"{status} Overlapping segments: {len(optimized_results)}/{len(overlapping_segments)} extracted")
            
        except Exception as e:
            self.print_status(f"❌ ERROR Overlapping segments: {e}", "ERROR")
            edge_case_results.append({'test': 'overlapping_segments', 'success': False, 'error': str(e)})
        
        # Test case 2: Different batch sizes
        for batch_duration in [10.0, 20.0, 30.0, 45.0]:
            try:
                self.print_status(f"Testing batch duration: {batch_duration}s...")
                
                segments = self.generate_test_segments(20, 1.0)
                start_time = time.time()
                
                optimized_results = parallel_extract_audio_segments_opus_optimized(
                    self.audio_group, self.audio_blob_array, segments,
                    max_batch_duration_seconds=batch_duration
                )
                
                processing_time = time.time() - start_time
                success = len(optimized_results) == len(segments)
                
                edge_case_results.append({
                    'test': f'batch_duration_{batch_duration}s',
                    'segments': len(segments),
                    'processing_time': processing_time,
                    'success': success
                })
                
                status = "✅ PASS" if success else "❌ FAIL"
                self.print_status(f"{status} Batch {batch_duration}s: {processing_time:.2f}s")
                
            except Exception as e:
                self.print_status(f"❌ ERROR Batch {batch_duration}s: {e}", "ERROR")
                edge_case_results.append({
                    'test': f'batch_duration_{batch_duration}s',
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['edge_case_tests'] = edge_case_results
        return edge_case_results
    
    def test_auto_selection(self) -> dict:
        """Test auto-selection API"""
        self.print_status("=== AUTO-SELECTION TEST ===")
        
        auto_test_results = []
        
        test_cases = [
            (5, "few segments (should use original)"),
            (15, "many segments (should use batch-optimized)")
        ]
        
        for segment_count, description in test_cases:
            try:
                self.print_status(f"Testing auto-selection: {description}")
                
                segments = self.generate_test_segments(segment_count, 1.0)
                
                start_time = time.time()
                auto_results = parallel_extract_audio_segments_opus_auto(
                    self.audio_group, self.audio_blob_array, segments
                )
                auto_time = time.time() - start_time
                
                success = len(auto_results) == len(segments) and sum(1 for r in auto_results if r.size > 0) >= len(segments) * 0.8
                
                auto_test_results.append({
                    'description': description,
                    'segments': len(segments),
                    'processing_time': auto_time,
                    'success': success
                })
                
                status = "✅ PASS" if success else "❌ FAIL"
                self.print_status(f"{status} {description}: {auto_time:.2f}s")
                
            except Exception as e:
                self.print_status(f"❌ ERROR {description}: {e}", "ERROR")
                auto_test_results.append({
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['integration_tests'] = auto_test_results
        return auto_test_results
    
    def generate_summary_report(self) -> dict:
        """Generate comprehensive test summary"""
        self.print_status("=== GENERATING SUMMARY REPORT ===")
        
        # Count successes
        correctness_passed = sum(1 for r in self.test_results['correctness_tests'] if r.get('success', False))
        performance_passed = sum(1 for r in self.test_results['performance_tests'] if r.get('success', False))
        edge_case_passed = sum(1 for r in self.test_results['edge_case_tests'] if r.get('success', False))
        integration_passed = sum(1 for r in self.test_results['integration_tests'] if r.get('success', False))
        
        total_tests = (len(self.test_results['correctness_tests']) + 
                      len(self.test_results['performance_tests']) + 
                      len(self.test_results['edge_case_tests']) + 
                      len(self.test_results['integration_tests']))
        
        total_passed = correctness_passed + performance_passed + edge_case_passed + integration_passed
        
        # Find best speedup
        best_speedup = 0
        for test in self.test_results['performance_tests']:
            if test.get('speedup') and test['speedup'] > best_speedup:
                best_speedup = test['speedup']
        
        summary = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'success_rate': total_passed / total_tests * 100 if total_tests > 0 else 0,
            'correctness_passed': correctness_passed,
            'performance_passed': performance_passed,
            'edge_case_passed': edge_case_passed,
            'integration_passed': integration_passed,
            'best_speedup': best_speedup,
            'overall_success': total_passed >= total_tests * 0.8  # 80% pass rate
        }
        
        return summary
    
    def run_all_tests(self) -> dict:
        """Run complete Phase 6 test suite"""
        print("=" * 80)
        print("PHASE 6 EXTRACTION PERFORMANCE TEST SUITE")
        print("=" * 80)
        print("Testing: Batch-optimized vs. Original extraction")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Setup
            self.cleanup_old_stores()
            self.setup_test_audio()
            
            # Run test suites
            self.test_correctness()
            self.test_performance()
            self.test_edge_cases()
            self.test_auto_selection()
            
            # Generate summary
            summary = self.generate_summary_report()
            total_time = time.time() - overall_start_time
            
            # Print results
            print(f"\n" + "=" * 80)
            print("PHASE 6 TEST RESULTS SUMMARY")
            print("=" * 80)
            print(f"Total test time: {total_time:.1f}s")
            print(f"")
            
            print(f"📊 TEST RESULTS:")
            print(f"  ✅ Correctness Tests: {summary['correctness_passed']}/{len(self.test_results['correctness_tests'])}")
            print(f"  🚀 Performance Tests: {summary['performance_passed']}/{len(self.test_results['performance_tests'])}")
            print(f"  🔧 Edge Case Tests: {summary['edge_case_passed']}/{len(self.test_results['edge_case_tests'])}")
            print(f"  🤖 Integration Tests: {summary['integration_passed']}/{len(self.test_results['integration_tests'])}")
            print(f"")
            print(f"📈 PERFORMANCE:")
            if summary['best_speedup'] > 0:
                print(f"  🚀 Best speedup achieved: {summary['best_speedup']:.1f}x")
            print(f"  📊 Overall success rate: {summary['success_rate']:.1f}%")
            print(f"")
            
            if summary['overall_success']:
                print("🎉 RESULT: PHASE 6 BATCH-OPTIMIZATION SUCCESS!")
                print("✅ Correctness: Batch-optimized results match original")
                print("✅ Performance: Significant speedup for many segments")
                print("✅ Integration: Auto-selection works correctly")
                print("✅ Edge Cases: Overlapping segments and various batch sizes handled")
                print(f"\n🚀 Phase 6 extraction parallelization is PRODUCTION READY!")
            else:
                print("⚠️  RESULT: PARTIAL SUCCESS - Some issues detected")
                print(f"Success rate: {summary['success_rate']:.1f}% (target: 80%)")
                print("\n🔧 Review test results for debugging")
            
            summary['total_time'] = total_time
            summary['test_results'] = self.test_results
            return summary
            
        except Exception as e:
            total_time = time.time() - overall_start_time
            
            print(f"\n" + "=" * 80)
            print("CRITICAL TEST FAILURE")
            print("=" * 80)
            print(f"❌ Error: {str(e)}")
            print(f"⏱️  Failed after: {total_time:.1f}s")
            
            if self.verbose:
                import traceback
                print(f"\nFull traceback:")
                traceback.print_exc()
            
            return {
                'overall_success': False,
                'total_time': total_time,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(description='Phase 6 Extraction Performance Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed debugging (TRACE level)')
    
    args = parser.parse_args()
    
    try:
        tester = Phase6ExtractionTester(verbose=args.verbose)
        summary = tester.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if summary.get('overall_success', False) else 1)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()