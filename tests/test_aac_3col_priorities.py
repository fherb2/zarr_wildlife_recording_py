"""
Test Suite for AAC 3-Column Index Next Priorities
=================================================

Pytest-compatible tests for the three key next development priorities:
1. PyAV-Extraktion mit 3-Spalten-Index testen
2. Random Access Performance messen  
3. Overlap-Handling validieren

These tests validate the optimized 3-column index functionality and
measure performance improvements achieved through the optimization.

Run with: pytest test_aac_3column_priorities.py -v -s
"""

import pytest
import pathlib
import shutil
import tempfile
import numpy as np
import time
import statistics
from typing import List, Tuple
import zarr

# Import the modules to test
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.config import Config
from zarrwlr.packagetypes import LogLevel
from zarrwlr import aimport
from zarrwlr.aac_access import (
    extract_audio_segment_aac,
    parallel_extract_audio_segments_aac
)
from zarrwlr.aac_index_backend import (
    get_index_statistics,
    benchmark_aac_access,
    _find_frame_range_for_samples,
    get_aac_frame_samples,
    calculate_timestamp_ms
)


class TestAAC_3ColumnPriorities:
    """Test class for AAC 3-column index priority validation"""
    
    @pytest.fixture(scope="class")
    def test_environment(self):
        """Set up test environment"""
        print("\n🎯 Setting up AAC 3-Column Index Priority Test Environment")
        
        # Configure for detailed testing
        Config.set(
            log_level=LogLevel.DEBUG,  # Less verbose than TRACE for performance tests
            aac_default_bitrate=160000,
            aac_enable_pyav_native=True,
            aac_fallback_to_ffmpeg=True,
            aac_max_workers=4
        )
        
        # Create test directories
        test_dir = pathlib.Path(__file__).parent / "testresults" / "aac_3col_priorities"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'test_dir': test_dir,
            'zarr_store_path': test_dir / "aac_priorities_store"
        }
    
    @pytest.fixture
    def test_files(self) -> List[pathlib.Path]:
        """Get available test audio files"""
        test_data_dir = pathlib.Path(__file__).parent / "testdata"
        
        # Priority order: prefer longer files for better performance testing
        candidate_files = [
            "audiomoth_long_snippet.wav",    # Prefer longer files
            "audiomoth_short_snippet.wav", 
            "bird1_snippet.mp3",
            "camtrap_snippet.mov"
        ]
        
        available_files = []
        for filename in candidate_files:
            filepath = test_data_dir / filename
            if filepath.exists() and filepath.stat().st_size > 1000:
                available_files.append(filepath)
                
        print(f"📁 Found {len(available_files)} test files: {[f.name for f in available_files]}")
        return available_files
    
    @pytest.fixture
    def imported_aac_data(self, test_files, test_environment):
        """Create imported AAC data for testing"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]  # Use first (preferably longest) file
        store_path = test_environment['zarr_store_path']
        
        # Remove if exists
        if store_path.exists():
            shutil.rmtree(store_path)
            
        # Create store and import
        store = zarr.storage.LocalStore(root=str(store_path))
        root = zarr.create_group(store=store, overwrite=True)
        audio_import_grp = root.create_group('audio_imports')
        
        audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_import_grp.attrs["version"] = Config.original_audio_group_version
        
        print(f"📥 Importing test file: {test_file.name} ({test_file.stat().st_size} bytes)")
        
        # Import the file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_import_grp,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=160000
        )
        
        # Get imported data
        group_name = list(audio_import_grp.keys())[0]
        imported_group = audio_import_grp[group_name]
        audio_array = imported_group['audio_data_blob_array']
        aac_index = imported_group['aac_index']
        
        # Validate 3-column structure
        assert aac_index.shape[1] == 3, f"Expected 3-column index, got {aac_index.shape[1]}"
        assert aac_index.attrs.get('index_format_version') == '3-column-optimized'
        
        print(f"✅ Import completed: {aac_index.shape[0]} frames, 3-column index")
        
        return {
            'zarr_group': imported_group,
            'audio_array': audio_array,
            'aac_index': aac_index,
            'test_file': test_file
        }
    

    # ====================================================================
    # Priority 1: PyAV-Extraktion mit 3-Spalten-Index testen
    # ====================================================================
    
    def test_priority1_pyav_extraction_basic(self, imported_aac_data):
        """Priority 1: Test basic PyAV extraction with 3-column index"""
        print("\n🎯 Priority 1: Testing PyAV extraction with 3-column index")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        
        # Get audio parameters
        sample_rate = audio_array.attrs['sample_rate']
        total_frames = aac_index.shape[0]
        total_samples = aac_index.attrs['total_samples']
        
        print(f"📊 Audio info: {sample_rate}Hz, {total_frames} frames, {total_samples} samples")
        
        # Test small extraction from beginning
        start_sample = 0
        end_sample = 4410  # 100ms at 44.1kHz (or equivalent)
        
        print(f"🔍 Testing extraction: samples [{start_sample}:{end_sample}]")
        
        start_time = time.time()
        audio_data = extract_audio_segment_aac(
            zarr_group, audio_array, start_sample, end_sample
        )
        extraction_time = time.time() - start_time
        
        # Validate results
        assert isinstance(audio_data, np.ndarray), f"Expected ndarray, got {type(audio_data)}"
        assert audio_data.dtype == np.int16, f"Expected int16, got {audio_data.dtype}"
        assert len(audio_data) > 0, "Extraction returned empty array"
        
        expected_samples = end_sample - start_sample + 1
        actual_samples = len(audio_data)
        
        # Allow some tolerance for frame boundaries
        sample_tolerance = 1024  # One AAC frame
        assert abs(actual_samples - expected_samples) <= sample_tolerance, \
            f"Sample count mismatch: expected ~{expected_samples}, got {actual_samples}"
        
        print(f"✅ Basic extraction successful:")
        print(f"   Extracted: {actual_samples} samples")
        print(f"   Time: {extraction_time*1000:.2f}ms")
        print(f"   Data type: {audio_data.dtype}")
        print(f"   Data range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        
        # Test extraction from middle
        mid_start = total_samples // 2
        mid_end = mid_start + 2205  # 50ms
        
        print(f"🔍 Testing middle extraction: samples [{mid_start}:{mid_end}]")
        
        start_time = time.time()
        mid_audio = extract_audio_segment_aac(
            zarr_group, audio_array, mid_start, mid_end
        )
        mid_extraction_time = time.time() - start_time
        
        assert len(mid_audio) > 0, "Middle extraction returned empty array"
        print(f"✅ Middle extraction: {len(mid_audio)} samples in {mid_extraction_time*1000:.2f}ms")
    
    def test_priority1_pyav_extraction_edge_cases(self, imported_aac_data):
        """Priority 1: Test PyAV extraction edge cases with 3-column index"""
        print("\n🎯 Priority 1: Testing PyAV extraction edge cases")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        # Test very small extraction (single sample)
        print("🔍 Testing single sample extraction")
        single_sample = extract_audio_segment_aac(
            zarr_group, audio_array, 1000, 1000
        )
        assert len(single_sample) >= 1, "Single sample extraction failed"
        print(f"✅ Single sample: {len(single_sample)} samples extracted")
        
        # Test extraction near end
        print("🔍 Testing extraction near end of file")
        end_start = max(0, total_samples - 1024)
        end_end = total_samples - 1
        
        end_audio = extract_audio_segment_aac(
            zarr_group, audio_array, end_start, end_end
        )
        assert len(end_audio) > 0, "End extraction failed"
        print(f"✅ End extraction: {len(end_audio)} samples")
        
        # Test large extraction (multiple frames)
        print("🔍 Testing large extraction (multiple frames)")
        large_start = 0
        large_end = min(44100, total_samples - 1)  # 1 second or available
        
        start_time = time.time()
        large_audio = extract_audio_segment_aac(
            zarr_group, audio_array, large_start, large_end
        )
        large_time = time.time() - start_time
        
        assert len(large_audio) > 0, "Large extraction failed"
        print(f"✅ Large extraction: {len(large_audio)} samples in {large_time*1000:.2f}ms")
    
    def test_priority1_different_dtypes(self, imported_aac_data):
        """Priority 1: Test PyAV extraction with different data types"""
        print("\n🎯 Priority 1: Testing different output data types")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        
        start_sample = 1000
        end_sample = 5000
        
        # Test different output types
        dtypes_to_test = [np.int16, np.int32, np.float32]
        
        for dtype in dtypes_to_test:
            print(f"🔍 Testing dtype: {dtype}")
            
            audio_data = extract_audio_segment_aac(
                zarr_group, audio_array, start_sample, end_sample, dtype=dtype
            )
            
            assert audio_data.dtype == dtype, f"Expected {dtype}, got {audio_data.dtype}"
            assert len(audio_data) > 0, f"Empty result for dtype {dtype}"
            
            print(f"✅ {dtype}: {len(audio_data)} samples, range [{np.min(audio_data):.2f}, {np.max(audio_data):.2f}]")


    # ====================================================================
    # Priority 2: Random Access Performance messen
    # ====================================================================
    
    def test_priority2_random_access_performance(self, imported_aac_data):
        """Priority 2: Measure random access performance with 3-column index"""
        print("\n🎯 Priority 2: Measuring random access performance")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        print(f"📊 Performance test setup: {total_samples} total samples, {aac_index.shape[0]} frames")
        
        # Generate random extraction points
        num_tests = 50
        segment_length = 2205  # ~50ms at 44.1kHz
        
        np.random.seed(42)  # Reproducible results
        test_segments = []
        
        for _ in range(num_tests):
            start = np.random.randint(0, max(1, total_samples - segment_length))
            end = min(start + segment_length, total_samples - 1)
            test_segments.append((start, end))
        
        print(f"🔍 Running {num_tests} random extractions...")
        
        # Measure extraction times
        extraction_times = []
        successful_extractions = 0
        
        overall_start = time.time()
        
        for i, (start_sample, end_sample) in enumerate(test_segments):
            start_time = time.time()
            
            try:
                audio_data = extract_audio_segment_aac(
                    zarr_group, audio_array, start_sample, end_sample
                )
                
                extraction_time = time.time() - start_time
                
                if len(audio_data) > 0:
                    extraction_times.append(extraction_time)
                    successful_extractions += 1
                
            except Exception as e:
                print(f"⚠️ Extraction {i} failed: {e}")
        
        total_time = time.time() - overall_start
        
        # Calculate statistics
        if extraction_times:
            min_time = min(extraction_times) * 1000
            max_time = max(extraction_times) * 1000
            mean_time = statistics.mean(extraction_times) * 1000
            median_time = statistics.median(extraction_times) * 1000
            std_time = statistics.stdev(extraction_times) * 1000 if len(extraction_times) > 1 else 0
            
            print(f"\n📈 Performance Results:")
            print(f"   Total tests: {num_tests}")
            print(f"   Successful: {successful_extractions}")
            print(f"   Success rate: {successful_extractions/num_tests*100:.1f}%")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Extraction times (ms):")
            print(f"     Min:    {min_time:.2f}ms")
            print(f"     Max:    {max_time:.2f}ms")
            print(f"     Mean:   {mean_time:.2f}ms")
            print(f"     Median: {median_time:.2f}ms")
            print(f"     Std:    {std_time:.2f}ms")
            
            # Performance targets validation
            target_max_time = 25.0  # 25ms target
            assert mean_time <= target_max_time, f"Mean extraction time {mean_time:.2f}ms exceeds target {target_max_time}ms"
            assert successful_extractions >= num_tests * 0.95, f"Success rate too low: {successful_extractions}/{num_tests}"
            
            print(f"✅ Performance targets met: {mean_time:.2f}ms mean < {target_max_time}ms target")
        else:
            pytest.fail("No successful extractions for performance measurement")
    
    def test_priority2_index_lookup_performance(self, imported_aac_data):
        """Priority 2: Measure 3-column index lookup performance specifically"""
        print("\n🎯 Priority 2: Testing 3-column index lookup performance")
        
        zarr_group = imported_aac_data['zarr_group']
        aac_index = imported_aac_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        # Test index lookup speed
        num_lookups = 1000
        np.random.seed(42)
        
        lookup_times = []
        
        print(f"🔍 Testing {num_lookups} index lookups...")
        
        for _ in range(num_lookups):
            start_sample = np.random.randint(0, max(1, total_samples - 1000))
            end_sample = start_sample + 1000
            
            start_time = time.time()
            start_idx, end_idx = _find_frame_range_for_samples(aac_index, start_sample, end_sample)
            lookup_time = time.time() - start_time
            
            lookup_times.append(lookup_time)
            
            # Validate lookup results
            assert 0 <= start_idx < aac_index.shape[0], f"Invalid start_idx: {start_idx}"
            assert 0 <= end_idx < aac_index.shape[0], f"Invalid end_idx: {end_idx}"
            assert start_idx <= end_idx, f"start_idx > end_idx: {start_idx} > {end_idx}"
        
        # Calculate lookup statistics
        min_lookup = min(lookup_times) * 1000000  # microseconds
        max_lookup = max(lookup_times) * 1000000
        mean_lookup = statistics.mean(lookup_times) * 1000000
        
        print(f"📈 Index Lookup Performance:")
        print(f"   Lookups tested: {num_lookups}")
        print(f"   Min time: {min_lookup:.2f}μs")
        print(f"   Max time: {max_lookup:.2f}μs")
        print(f"   Mean time: {mean_lookup:.2f}μs")
        
        # Validate O(log n) performance
        target_max_lookup = 100.0  # 100μs should be more than enough for O(log n)
        assert mean_lookup <= target_max_lookup, f"Index lookup too slow: {mean_lookup:.2f}μs > {target_max_lookup}μs"
        
        print(f"✅ Index lookup performance: {mean_lookup:.2f}μs mean < {target_max_lookup}μs target")
    
    def test_priority2_parallel_extraction_performance(self, imported_aac_data):
        """Priority 2: Test parallel extraction performance"""
        print("\n🎯 Priority 2: Testing parallel extraction performance")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        total_samples = imported_aac_data['aac_index'].attrs['total_samples']
        
        # Generate test segments
        num_segments = 20
        segment_length = 2205
        np.random.seed(42)
        
        segments = []
        for _ in range(num_segments):
            start = np.random.randint(0, max(1, total_samples - segment_length))
            end = min(start + segment_length, total_samples - 1)
            segments.append((start, end))
        
        # Test sequential vs parallel extraction
        print(f"🔍 Comparing sequential vs parallel extraction ({num_segments} segments)")
        
        # Sequential extraction
        start_time = time.time()
        sequential_results = []
        for start, end in segments:
            result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Parallel extraction
        start_time = time.time()
        parallel_results = parallel_extract_audio_segments_aac(
            zarr_group, audio_array, segments, max_workers=4
        )
        parallel_time = time.time() - start_time
        
        # Validate results
        assert len(sequential_results) == len(parallel_results), "Result count mismatch"
        assert len(sequential_results) == num_segments, f"Expected {num_segments} results"
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        print(f"📈 Parallel Performance:")
        print(f"   Sequential time: {sequential_time:.3f}s")
        print(f"   Parallel time: {parallel_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        # Validate parallel extraction is faster (or at least not much slower)
        assert speedup >= 0.8, f"Parallel extraction too slow: {speedup:.2f}x speedup"
        
        print(f"✅ Parallel extraction performance validated")


    # ====================================================================
    # Priority 3: Overlap-Handling validieren
    # ====================================================================
    
    def test_priority3_overlap_handling_accuracy(self, imported_aac_data):
        """Priority 3: Validate overlap handling produces accurate results"""
        print("\n🎯 Priority 3: Validating overlap handling accuracy")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        sample_rate = audio_array.attrs['sample_rate']
        
        # Test overlapping extractions to ensure consistency
        print("🔍 Testing overlapping extractions for consistency")
        
        # Extract three overlapping segments
        base_start = 5000
        segment_length = 3000
        
        # Segment A: [5000:8000]
        # Segment B: [6000:9000] (overlaps with A)
        # Segment C: [7000:10000] (overlaps with B)
        
        seg_a = extract_audio_segment_aac(zarr_group, audio_array, base_start, base_start + segment_length)
        seg_b = extract_audio_segment_aac(zarr_group, audio_array, base_start + 1000, base_start + segment_length + 1000)
        seg_c = extract_audio_segment_aac(zarr_group, audio_array, base_start + 2000, base_start + segment_length + 2000)
        
        assert len(seg_a) > 0 and len(seg_b) > 0 and len(seg_c) > 0, "Empty segments"
        
        print(f"   Segment A: {len(seg_a)} samples")
        print(f"   Segment B: {len(seg_b)} samples")  
        print(f"   Segment C: {len(seg_c)} samples")
        
        # Test that overlapping regions produce similar results
        # Compare last part of A with first part of B
        overlap_length = min(len(seg_a) - 1000, len(seg_b), 1000)
        if overlap_length > 100:  # Only test if significant overlap
            a_end = seg_a[-overlap_length:]
            b_start = seg_b[:overlap_length]
            
            # Allow some difference due to frame boundaries, but should be very similar
            correlation = np.corrcoef(a_end.astype(float), b_start.astype(float))[0, 1]
            
            print(f"   Overlap correlation: {correlation:.4f}")
            assert correlation > 0.9, f"Overlap correlation too low: {correlation:.4f}"
            
        print("✅ Overlap consistency validated")
    
    def test_priority3_frame_boundary_handling(self, imported_aac_data):
        """Priority 3: Test overlap handling at frame boundaries"""
        print("\n🎯 Priority 3: Testing frame boundary overlap handling")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        
        # Test extractions that start exactly at frame boundaries
        print("🔍 Testing extractions at frame boundaries")
        
        # Get sample positions of first few frames
        sample_positions = aac_index[:, 2]  # 3rd column is sample_pos
        frame_samples = get_aac_frame_samples()  # Should be 1024
        
        for i in range(min(5, len(sample_positions))):
            frame_start = int(sample_positions[i])
            
            # Test extraction starting exactly at frame boundary
            exact_start = frame_start
            exact_end = exact_start + 512  # Half a frame
            
            print(f"   Testing frame {i} boundary: sample {exact_start}")
            
            start_time = time.time()
            boundary_audio = extract_audio_segment_aac(
                zarr_group, audio_array, exact_start, exact_end
            )
            extraction_time = time.time() - start_time
            
            assert len(boundary_audio) > 0, f"Empty extraction at frame {i} boundary"
            
            # Validate that we get reasonable amount of samples
            expected_samples = exact_end - exact_start + 1
            actual_samples = len(boundary_audio)
            
            # Allow tolerance for frame boundary effects
            tolerance = frame_samples  # One frame tolerance
            assert abs(actual_samples - expected_samples) <= tolerance, \
                f"Frame {i}: expected ~{expected_samples} samples, got {actual_samples}"
            
            print(f"     Frame {i}: {actual_samples} samples in {extraction_time*1000:.2f}ms")
        
        print("✅ Frame boundary handling validated")
    
    def test_priority3_overlap_frame_calculation(self, imported_aac_data):
        """Priority 3: Test that overlap calculation works correctly"""
        print("\n🎯 Priority 3: Testing overlap frame calculation")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        
        # Test the overlap frame calculation directly
        print("🔍 Testing _find_frame_range_for_samples with overlap")
        
        sample_positions = aac_index[:, 2]
        
        # Test case: request samples that start in the middle of frame 3
        if len(sample_positions) > 5:
            frame_3_start = int(sample_positions[3])
            frame_4_start = int(sample_positions[4])
            
            # Request samples starting halfway through frame 3
            request_start = frame_3_start + 512
            request_end = frame_4_start + 512
            
            print(f"   Request: samples [{request_start}:{request_end}]")
            print(f"   Frame 3 starts at: {frame_3_start}")
            print(f"   Frame 4 starts at: {frame_4_start}")
            
            # Get frame range with overlap
            start_idx, end_idx = _find_frame_range_for_samples(aac_index, request_start, request_end)
            
            print(f"   Frame range: [{start_idx}:{end_idx}]")
            print(f"   Start frame sample pos: {int(sample_positions[start_idx])}")
            print(f"   End frame sample pos: {int(sample_positions[end_idx])}")
            
            # Validate overlap handling:
            # Should start at frame 2 (one before frame 3) for overlap
            expected_start_idx = max(0, 3 - 1)  # Frame 3 - 1 for overlap
            assert start_idx == expected_start_idx, \
                f"Expected overlap start at frame {expected_start_idx}, got {start_idx}"
            
            # Should end at frame 4 or later
            assert end_idx >= 4, f"End frame should be >= 4, got {end_idx}"
            
            print("✅ Overlap frame calculation correct")
        else:
            print("⚠️ Not enough frames for overlap calculation test")
    
    def test_priority3_sample_accuracy_with_overlap(self, imported_aac_data):
        """Priority 3: Test sample accuracy when using overlap handling"""
        print("\n🎯 Priority 3: Testing sample accuracy with overlap handling")
        
        zarr_group = imported_aac_data['zarr_group']
        audio_array = imported_aac_data['audio_array']
        aac_index = imported_aac_data['aac_index']
        sample_rate = audio_array.attrs['sample_rate']
        
        # Test precise sample extraction
        print("🔍 Testing precise sample extraction with overlap")
        
        # Test multiple precise extractions
        test_cases = [
            (1000, 1999),   # 1000 samples exactly
            (5000, 5499),   # 500 samples
            (10000, 10099), # 100 samples  
        ]
        
        for start_sample, end_sample in test_cases:
            if end_sample < aac_index.attrs['total_samples']:
                print(f"   Testing precise range: [{start_sample}:{end_sample}]")
                
                expected_samples = end_sample - start_sample + 1
                
                audio_data = extract_audio_segment_aac(
                    zarr_group, audio_array, start_sample, end_sample
                )
                
                actual_samples = len(audio_data)
                
                # With good overlap handling, we should get very close to exact samples
                sample_tolerance = 50  # Very tight tolerance
                sample_diff = abs(actual_samples - expected_samples)
                
                print(f"     Expected: {expected_samples} samples")
                print(f"     Actual: {actual_samples} samples")
                print(f"     Difference: {sample_diff} samples")
                
                assert sample_diff <= sample_tolerance, \
                    f"Sample accuracy poor: {sample_diff} > {sample_tolerance} tolerance"
        
        print("✅ Sample accuracy with overlap validated")


# Additional utility functions for manual testing
def run_manual_priority_tests():
    """Manual test runner for debugging - not pytest"""
    print("🎯 Running manual priority tests...")
    
    # Set up environment
    test_dir = pathlib.Path(__file__).parent / "testresults" / "manual_priorities"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure for verbose output
    Config.set(
        log_level=LogLevel.DEBUG,
        aac_default_bitrate=160000
    )
    
    # Find test file
    test_data_dir = pathlib.Path(__file__).parent / "testdata"
    test_file = None
    for candidate in ["audiomoth_long_snippet.wav", "audiomoth_short_snippet.wav", "bird1_snippet.mp3"]:
        candidate_path = test_data_dir / candidate
        if candidate_path.exists():
            test_file = candidate_path
            break
    
    if not test_file:
        print("❌ No test files found")
        return
        
    print(f"🎵 Using test file: {test_file}")
    
    try:
        # Create Zarr group and import
        store = zarr.storage.LocalStore(root=str(test_dir / "zarr_store"))
        root = zarr.create_group(store=store, overwrite=True)
        audio_group = root.create_group('audio_imports')
        audio_group.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_group.attrs["version"] = Config.original_audio_group_version
        
        # Import test file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=160000
        )
        
        # Get imported data
        imported_group = audio_group[list(audio_group.keys())[0]]
        audio_array = imported_group['audio_data_blob_array']
        aac_index = imported_group['aac_index']
        
        print("✅ Import completed successfully")
        print(f"   Index shape: {aac_index.shape}")
        print(f"   Total samples: {aac_index.attrs.get('total_samples', 'unknown')}")
        
        # Priority 1: Test basic extraction
        print("\n🎯 Priority 1: Testing basic PyAV extraction")
        start_sample = 1000
        end_sample = 5000
        
        start_time = time.time()
        audio_data = extract_audio_segment_aac(
            imported_group, audio_array, start_sample, end_sample
        )
        extraction_time = time.time() - start_time
        
        print(f"   Extracted: {len(audio_data)} samples")
        print(f"   Time: {extraction_time*1000:.2f}ms")
        print(f"   Data type: {audio_data.dtype}")
        
        # Priority 2: Test performance
        print("\n🎯 Priority 2: Testing random access performance")
        
        # Quick performance test
        num_tests = 10
        extraction_times = []
        
        for i in range(num_tests):
            start = 1000 + i * 1000
            end = start + 500
            
            start_time = time.time()
            test_audio = extract_audio_segment_aac(
                imported_group, audio_array, start, end
            )
            test_time = time.time() - start_time
            extraction_times.append(test_time)
        
        mean_time = statistics.mean(extraction_times) * 1000
        print(f"   Mean extraction time: {mean_time:.2f}ms")
        print(f"   Target: <25ms")
        
        if mean_time <= 25.0:
            print("   ✅ Performance target met")
        else:
            print("   ⚠️ Performance target not met")
        
        # Priority 3: Test overlap handling
        print("\n🎯 Priority 3: Testing overlap handling")
        
        # Test frame boundary extraction
        sample_positions = aac_index[:, 2]  # sample_pos column
        if len(sample_positions) > 3:
            frame_start = int(sample_positions[2])
            boundary_start = frame_start + 512  # Middle of frame
            boundary_end = boundary_start + 1000
            
            boundary_audio = extract_audio_segment_aac(
                imported_group, audio_array, boundary_start, boundary_end
            )
            
            print(f"   Frame boundary extraction: {len(boundary_audio)} samples")
            print(f"   Frame starts at sample: {frame_start}")
            print(f"   Extraction started at: {boundary_start}")
            
            if len(boundary_audio) > 0:
                print("   ✅ Overlap handling working")
            else:
                print("   ⚠️ Overlap handling issue")
        
        print("\n✅ Manual priority tests completed successfully")
        
    except Exception as e:
        print(f"❌ Manual priority tests failed: {e}")
        import traceback
        traceback.print_exc()


def quick_performance_benchmark():
    """Quick performance benchmark for development"""
    print("🚀 Quick performance benchmark...")
    
    # This function can be used for quick performance checks during development
    # without running the full test suite
    
    test_dir = pathlib.Path(__file__).parent / "testresults" / "quick_benchmark"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    Config.set(log_level=LogLevel.WARNING)  # Minimal logging for clean output
    
    # Find shortest test file for quick testing
    test_data_dir = pathlib.Path(__file__).parent / "testdata"
    test_file = None
    for candidate in ["audiomoth_short_snippet.wav", "bird1_snippet.mp3"]:
        candidate_path = test_data_dir / candidate
        if candidate_path.exists():
            test_file = candidate_path
            break
    
    if not test_file:
        print("❌ No test files found")
        return
    
    try:
        # Quick import
        store = zarr.storage.LocalStore(root=str(test_dir / "zarr_store"))
        root = zarr.create_group(store=store, overwrite=True)
        audio_group = root.create_group('audio_imports')
        audio_group.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_group.attrs["version"] = Config.original_audio_group_version
        
        import_start = time.time()
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=128000  # Lower bitrate for speed
        )
        import_time = time.time() - import_start
        
        # Get data
        imported_group = audio_group[list(audio_group.keys())[0]]
        audio_array = imported_group['audio_data_blob_array']
        aac_index = imported_group['aac_index']
        
        # Quick extraction test
        extraction_start = time.time()
        audio_data = extract_audio_segment_aac(
            imported_group, audio_array, 1000, 3000
        )
        extraction_time = time.time() - extraction_start
        
        # Results
        print(f"📊 Quick Benchmark Results:")
        print(f"   File: {test_file.name}")
        print(f"   Import time: {import_time:.2f}s")
        print(f"   Frames: {aac_index.shape[0]}")
        print(f"   Index format: {aac_index.attrs.get('index_format_version', 'unknown')}")
        print(f"   Extraction time: {extraction_time*1000:.2f}ms")
        print(f"   Extracted samples: {len(audio_data)}")
        
        if extraction_time * 1000 <= 25:
            print("   ✅ Performance: GOOD")
        else:
            print("   ⚠️ Performance: NEEDS IMPROVEMENT")
            
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")


if __name__ == "__main__":
    # Run manual tests if called directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_performance_benchmark()
    else:
        run_manual_priority_tests()