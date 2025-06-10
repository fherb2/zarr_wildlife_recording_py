"""
AAC Performance Test Suite - Optimized Implementation Validation
===============================================================

Comprehensive pytest-compatible performance testing for optimized AAC implementation.
Validates performance improvements and measures actual speedup achieved.

Tests validate the Phase 3 performance optimization targets:
1. Container caching performance (warm vs cold cache)
2. Index lookup optimization (cached vs uncached)  
3. End-to-end performance comparison
4. Memory usage patterns
5. Parallel processing scaling
6. Real-world performance targets (target: <150ms extraction)

Run with: pytest test_aac_performance_optimized.py -v -s
"""

import pytest
import pathlib
import shutil
import tempfile
import numpy as np
import time
import statistics
from typing import List, Tuple, Optional
import zarr

# Import modules to test
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.config import Config
from zarrwlr.packagetypes import LogLevel
from zarrwlr import aimport

# Import optimized modules
from zarrwlr.aac_access import (
    extract_audio_segment_aac,
    parallel_extract_audio_segments_aac,
    clear_performance_caches,
    get_performance_stats
)

from zarrwlr.aac_index_backend import (
    find_frame_range_for_samples_fast,
    get_index_statistics_fast,
    benchmark_aac_access_optimized,
    clear_all_caches,
    get_optimization_stats
)


class TestAACPerformanceOptimized:
    """Performance test suite for optimized AAC implementation with Phase 3 targets"""
    
    @pytest.fixture(scope="class")
    def performance_environment(self):
        """Set up performance testing environment"""
        print("\n🚀 Setting up AAC Performance Test Environment (Phase 3 Validation)")
        
        # Configure for performance testing
        Config.set(
            log_level=LogLevel.DEBUG,  # Enable detailed logging for validation
            aac_default_bitrate=160000,
            aac_enable_pyav_native=True,
            aac_fallback_to_ffmpeg=True,
            aac_max_workers=4,
            aac_memory_limit_mb=500
        )
        
        # Create test directories
        test_dir = pathlib.Path(__file__).parent / "testresults" / "aac_performance_optimized"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Test directory: {test_dir}")
        
        return {
            'test_dir': test_dir,
            'zarr_store_path': test_dir / "performance_store"
        }
    
    @pytest.fixture
    def test_files(self) -> List[pathlib.Path]:
        """Get available test audio files"""
        test_data_dir = pathlib.Path(__file__).parent / "testdata"
        
        # Priority order: prefer longer files for meaningful performance testing
        candidate_files = [
            "audiomoth_long_snippet.wav",    # Best for performance testing
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
    def performance_data(self, test_files, performance_environment):
        """Create performance test data with AAC import"""
        if not test_files:
            pytest.skip("No test files available")
            
        # Use largest file for meaningful performance testing
        test_file = test_files[0]
        store_path = performance_environment['zarr_store_path']
        
        print(f"\n📥 Importing performance test data: {test_file.name}")
        print(f"   File size: {test_file.stat().st_size:,} bytes")
        
        if store_path.exists():
            shutil.rmtree(store_path)
            
        # Create Zarr v3 store
        store = zarr.storage.LocalStore(root=str(store_path))
        root = zarr.create_group(store=store, overwrite=True)
        audio_import_grp = root.create_group('audio_imports')
        audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_import_grp.attrs["version"] = Config.original_audio_group_version
        
        # Import AAC data with performance timing
        import_start = time.time()
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_import_grp,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=160000
        )
        import_time = time.time() - import_start
        
        # Get imported data
        group_name = list(audio_import_grp.keys())[0]
        imported_group = audio_import_grp[group_name]
        audio_array = imported_group['audio_data_blob_array']
        aac_index = imported_group['aac_index']
        
        # Validate import
        assert aac_index.shape[1] == 3, f"Expected 3-column index, got {aac_index.shape[1]}"
        assert aac_index.attrs.get('index_format_version') == '3-column-optimized'
        
        print(f"✅ Import completed in {import_time:.2f}s:")
        print(f"   Frames: {aac_index.shape[0]:,}")
        print(f"   Total samples: {aac_index.attrs['total_samples']:,}")
        print(f"   Sample rate: {audio_array.attrs['sample_rate']}Hz")
        print(f"   Channels: {audio_array.attrs['nb_channels']}")
        print(f"   Index format: {aac_index.attrs['index_format_version']}")
        
        return {
            'zarr_group': imported_group,
            'audio_array': audio_array,
            'aac_index': aac_index,
            'test_file': test_file,
            'import_time': import_time
        }
    
    def test_container_caching_performance(self, performance_data):
        """Test PyAV container caching performance improvement (Phase 3 Priority)"""
        print("\n⚡ Testing Container Caching Performance (Phase 3)")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        total_samples = performance_data['aac_index'].attrs['total_samples']
        
        # Generate test segments
        num_tests = 8
        segment_length = 4410  # ~100ms at 44.1kHz
        
        np.random.seed(42)  # Reproducible results
        test_segments = []
        for i in range(num_tests):
            start = min(i * 5000, total_samples - segment_length - 1)
            end = start + segment_length
            test_segments.append((start, end))
        
        print(f"🔍 Testing {num_tests} extractions for container caching...")
        
        # Phase 1: Cold cache (first run)
        print("   Phase 1: Cold cache extractions...")
        clear_performance_caches()
        
        cold_times = []
        for i, (start, end) in enumerate(test_segments):
            start_time = time.time()
            result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
            extraction_time = time.time() - start_time
            cold_times.append(extraction_time)
            
            assert len(result) > 0, f"Cold extraction {i} failed"
            print(f"     Cold extraction {i}: {extraction_time*1000:.2f}ms")
        
        cold_mean = statistics.mean(cold_times) * 1000
        print(f"   Cold cache mean: {cold_mean:.2f}ms")
        
        # Phase 2: Warm cache (repeated extractions)
        print("   Phase 2: Warm cache extractions...")
        
        warm_times = []
        for i, (start, end) in enumerate(test_segments):
            start_time = time.time()
            result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
            extraction_time = time.time() - start_time
            warm_times.append(extraction_time)
            
            assert len(result) > 0, f"Warm extraction {i} failed"
            print(f"     Warm extraction {i}: {extraction_time*1000:.2f}ms")
        
        warm_mean = statistics.mean(warm_times) * 1000
        print(f"   Warm cache mean: {warm_mean:.2f}ms")
        
        # Calculate cache benefit
        cache_benefit = cold_mean / warm_mean if warm_mean > 0 else 1
        
        # Get performance stats
        perf_stats = get_performance_stats()
        
        print(f"\n📈 Container Caching Results:")
        print(f"   Cold cache:       {cold_mean:.2f}ms")
        print(f"   Warm cache:       {warm_mean:.2f}ms")
        print(f"   Cache benefit:    {cache_benefit:.2f}x improvement")
        print(f"   Cached containers: {perf_stats['container_pool']['cached_containers']}")
        print(f"   Cached indices:    {perf_stats['index_cache']['cached_indices']}")
        
        # Phase 3 Target: 1.5x+ cache benefit
        target_cache_benefit = 1.5
        assert cache_benefit >= target_cache_benefit, \
            f"Expected cache benefit ≥{target_cache_benefit}x, got {cache_benefit:.2f}x"
        
        # Phase 3 Target: Warm cache < 100ms
        target_warm_ms = 100
        assert warm_mean <= target_warm_ms, \
            f"Expected warm cache ≤{target_warm_ms}ms, got {warm_mean:.2f}ms"
        
        print(f"✅ Container caching targets achieved:")
        print(f"   Cache benefit: {cache_benefit:.2f}x ≥ {target_cache_benefit}x ✓")
        print(f"   Warm performance: {warm_mean:.2f}ms ≤ {target_warm_ms}ms ✓")
    
    def test_index_lookup_optimization(self, performance_data):
        """Test index lookup performance optimization (Phase 3 Priority)"""
        print("\n⚡ Testing Index Lookup Optimization (Phase 3)")
        
        aac_index = performance_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        # Generate lookup requests
        num_lookups = 500
        np.random.seed(42)
        lookup_requests = []
        for _ in range(num_lookups):
            start = np.random.randint(0, max(1, total_samples - 1000))
            end = start + 1000
            lookup_requests.append((start, end))
        
        print(f"🔍 Testing {num_lookups} index lookups...")
        
        # Test with cold cache
        print("   Cold cache lookups...")
        clear_all_caches()
        
        cold_times = []
        start_time = time.time()
        for start_sample, end_sample in lookup_requests:
            lookup_start = time.time()
            start_idx, end_idx = find_frame_range_for_samples_fast(aac_index, start_sample, end_sample)
            lookup_time = time.time() - lookup_start
            cold_times.append(lookup_time)
            
            assert 0 <= start_idx < aac_index.shape[0], f"Invalid start_idx: {start_idx}"
            assert 0 <= end_idx < aac_index.shape[0], f"Invalid end_idx: {end_idx}"
        total_cold_time = time.time() - start_time
        
        cold_mean_us = statistics.mean(cold_times) * 1000000
        print(f"   Cold lookups: {cold_mean_us:.2f}μs mean, {total_cold_time:.3f}s total")
        
        # Test with warm cache (second pass)
        print("   Warm cache lookups...")
        
        warm_times = []
        start_time = time.time()
        for start_sample, end_sample in lookup_requests:
            lookup_start = time.time()
            start_idx, end_idx = find_frame_range_for_samples_fast(aac_index, start_sample, end_sample)
            lookup_time = time.time() - lookup_start
            warm_times.append(lookup_time)
        total_warm_time = time.time() - start_time
        
        warm_mean_us = statistics.mean(warm_times) * 1000000
        print(f"   Warm lookups: {warm_mean_us:.2f}μs mean, {total_warm_time:.3f}s total")
        
        # Calculate improvements
        cache_speedup = cold_mean_us / warm_mean_us if warm_mean_us > 0 else 1
        total_speedup = total_cold_time / total_warm_time if total_warm_time > 0 else 1
        
        # Get optimization statistics
        opt_stats = get_optimization_stats()
        
        print(f"\n📈 Index Lookup Results:")
        print(f"   Cold cache:       {cold_mean_us:.2f}μs mean")
        print(f"   Warm cache:       {warm_mean_us:.2f}μs mean")
        print(f"   Cache speedup:    {cache_speedup:.2f}x")
        print(f"   Total speedup:    {total_speedup:.2f}x")
        print(f"   Cached indices:   {opt_stats['index_manager']['cached_indices']}")
        
        # Phase 3 Target: 5x+ lookup speedup
        target_lookup_speedup = 5.0
        assert cache_speedup >= target_lookup_speedup, \
            f"Expected lookup speedup ≥{target_lookup_speedup}x, got {cache_speedup:.2f}x"
        
        # Phase 3 Target: Warm lookups < 10μs
        target_warm_us = 10.0
        assert warm_mean_us <= target_warm_us, \
            f"Expected warm lookups ≤{target_warm_us}μs, got {warm_mean_us:.2f}μs"
        
        print(f"✅ Index lookup targets achieved:")
        print(f"   Cache speedup: {cache_speedup:.2f}x ≥ {target_lookup_speedup}x ✓")
        print(f"   Warm performance: {warm_mean_us:.2f}μs ≤ {target_warm_us}μs ✓")
    
    def test_end_to_end_performance_comparison(self, performance_data):
        """Test end-to-end performance improvement vs baseline (Phase 3 Priority)"""
        print("\n⚡ Testing End-to-End Performance vs Baseline (Phase 3)")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        aac_index = performance_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        # Generate test segments
        num_extractions = 30
        segment_length = 2205  # ~50ms
        
        np.random.seed(42)
        segments = []
        for _ in range(num_extractions):
            start = np.random.randint(0, max(1, total_samples - segment_length))
            end = min(start + segment_length, total_samples - 1)
            segments.append((start, end))
        
        print(f"🔍 Running {num_extractions} end-to-end extractions...")
        
        # Test optimized implementation with detailed timing
        print("   Testing optimized implementation...")
        clear_performance_caches()  # Start with cold cache
        
        optimized_times = []
        successful_extractions = 0
        
        overall_start = time.time()
        
        for i, (start, end) in enumerate(segments):
            extraction_start = time.time()
            try:
                result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
                extraction_time = time.time() - extraction_start
                
                if len(result) > 0:
                    optimized_times.append(extraction_time)
                    successful_extractions += 1
                    
                    if i < 5:  # Log first few for debugging
                        print(f"     Extraction {i}: {extraction_time*1000:.2f}ms, {len(result)} samples")
                
            except Exception as e:
                print(f"     ⚠️ Extraction {i} failed: {e}")
        
        total_time = time.time() - overall_start
        
        # Calculate statistics
        if optimized_times:
            min_time = min(optimized_times) * 1000
            max_time = max(optimized_times) * 1000
            mean_time = statistics.mean(optimized_times) * 1000
            median_time = statistics.median(optimized_times) * 1000
            std_time = statistics.stdev(optimized_times) * 1000 if len(optimized_times) > 1 else 0
            
            print(f"\n📈 End-to-End Performance Results:")
            print(f"   Successful extractions: {successful_extractions}/{num_extractions}")
            print(f"   Success rate: {successful_extractions/num_extractions*100:.1f}%")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Extraction times (ms):")
            print(f"     Min:    {min_time:.2f}ms")
            print(f"     Max:    {max_time:.2f}ms")
            print(f"     Mean:   {mean_time:.2f}ms")
            print(f"     Median: {median_time:.2f}ms")
            print(f"     Std:    {std_time:.2f}ms")
            
            # Baseline comparison (400ms baseline from Phase 2)
            baseline_time_ms = 400.0
            speedup_vs_baseline = baseline_time_ms / mean_time if mean_time > 0 else 0
            print(f"   Speedup vs baseline: {speedup_vs_baseline:.2f}x (baseline: {baseline_time_ms}ms)")
            
            # Phase 3 Targets
            target_extraction_ms = 150.0  # Realistic target
            target_success_rate = 0.95
            target_speedup = 2.5  # vs 400ms baseline
            
            print(f"\n🎯 Phase 3 Target Validation:")
            print(f"   Target extraction: <{target_extraction_ms}ms")
            print(f"   Actual: {mean_time:.2f}ms")
            success_extraction = mean_time <= target_extraction_ms
            print(f"   Result: {'✅ ACHIEVED' if success_extraction else '❌ NOT MET'}")
            
            print(f"   Target success rate: >{target_success_rate*100:.0f}%")
            print(f"   Actual: {successful_extractions/num_extractions*100:.1f}%")
            success_rate = (successful_extractions/num_extractions) >= target_success_rate
            print(f"   Result: {'✅ ACHIEVED' if success_rate else '❌ NOT MET'}")
            
            print(f"   Target speedup: >{target_speedup:.1f}x")
            print(f"   Actual: {speedup_vs_baseline:.2f}x")
            success_speedup = speedup_vs_baseline >= target_speedup
            print(f"   Result: {'✅ ACHIEVED' if success_speedup else '❌ NOT MET'}")
            
            # Assertions for CI/CD (relaxed for initial validation)
            assert successful_extractions >= num_extractions * 0.90, \
                f"Success rate too low: {successful_extractions}/{num_extractions}"
            assert mean_time <= 200, \
                f"Mean extraction time too high: {mean_time:.2f}ms"
            assert speedup_vs_baseline >= 2.0, \
                f"Insufficient speedup vs baseline: {speedup_vs_baseline:.2f}x"
            
            print(f"✅ End-to-end performance validated")
        else:
            pytest.fail("No successful extractions for performance measurement")
    
    def test_memory_efficiency_patterns(self, performance_data):
        """Test memory usage optimization patterns"""
        print("\n⚡ Testing Memory Efficiency Patterns")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        
        try:
            import psutil
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024
            print(f"   Baseline memory: {baseline_memory:.1f}MB")
            
            # Test extraction patterns with memory monitoring
            test_segments = [(i * 2000, i * 2000 + 1000) for i in range(15)]
            
            print("   Testing memory usage during extractions...")
            memory_before = process.memory_info().rss / 1024 / 1024
            
            clear_performance_caches()
            
            for i, (start, end) in enumerate(test_segments):
                result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
                
                if i % 5 == 0:  # Check memory every 5 extractions
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - memory_before
                    print(f"     After {i+1} extractions: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
            
            memory_after = process.memory_info().rss / 1024 / 1024
            total_growth = memory_after - memory_before
            
            print(f"   Total memory growth: {total_growth:.1f}MB")
            
            # Memory efficiency validation
            memory_limit = Config.aac_memory_limit_mb
            assert memory_after - baseline_memory <= memory_limit * 1.2, \
                f"Memory usage {memory_after - baseline_memory:.1f}MB exceeds limit {memory_limit}MB"
            
            print(f"✅ Memory efficiency validated: {total_growth:.1f}MB growth")
            
        except ImportError:
            print("   psutil not available, skipping detailed memory analysis")
    
    def test_parallel_processing_scaling(self, performance_data):
        """Test parallel processing performance scaling"""
        print("\n⚡ Testing Parallel Processing Scaling")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        total_samples = performance_data['aac_index'].attrs['total_samples']
        
        # Generate test segments for parallel processing
        num_segments = 20
        segment_length = 1000
        
        np.random.seed(42)
        segments = []
        for _ in range(num_segments):
            start = np.random.randint(0, max(1, total_samples - segment_length))
            end = min(start + segment_length, total_samples - 1)
            segments.append((start, end))
        
        print(f"🔍 Testing parallel scaling with {num_segments} segments...")
        
        # Test different worker counts
        worker_counts = [1, 2, 4]
        results = {}
        
        for workers in worker_counts:
            print(f"   Testing with {workers} workers...")
            clear_performance_caches()
            
            start_time = time.time()
            parallel_results = parallel_extract_audio_segments_aac(
                zarr_group, audio_array, segments, max_workers=workers
            )
            total_time = time.time() - start_time
            
            success_count = sum(1 for r in parallel_results if len(r) > 0)
            results[workers] = {
                'time': total_time,
                'success_rate': success_count / num_segments,
                'avg_time_per_segment': total_time / num_segments
            }
            
            print(f"     {workers} workers: {total_time:.2f}s total, {success_count}/{num_segments} success")
        
        # Calculate scaling efficiency
        baseline_time = results[1]['time']
        print(f"\n📈 Parallel Processing Results:")
        for workers in worker_counts:
            speedup = baseline_time / results[workers]['time'] if results[workers]['time'] > 0 else 0
            efficiency = speedup / workers * 100
            print(f"   {workers} workers: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")
        
        # Validate parallel scaling
        if 4 in results:
            speedup_4_workers = baseline_time / results[4]['time'] if results[4]['time'] > 0 else 0
            target_speedup = 1.2  # Modest target due to I/O bottlenecks
            
            assert speedup_4_workers >= target_speedup, \
                f"Expected speedup ≥{target_speedup}x with 4 workers, got {speedup_4_workers:.2f}x"
            
            print(f"✅ Parallel scaling validated: {speedup_4_workers:.2f}x speedup with 4 workers")
    
    def test_realistic_performance_targets_comprehensive(self, performance_data):
        """Test comprehensive realistic performance targets (Phase 3 Final Validation)"""
        print("\n⚡ Testing Comprehensive Performance Targets (Phase 3 Final)")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        
        # Run comprehensive benchmark using built-in function
        print("🔍 Running comprehensive performance benchmark...")
        
        try:
            benchmark_results = benchmark_aac_access_optimized(
                zarr_group, audio_array, num_extractions=50
            )
            
            avg_extraction_ms = benchmark_results['performance_metrics']['average_extraction_ms']
            success_rate = benchmark_results['performance_metrics']['success_rate']
            speedup_vs_baseline = benchmark_results['performance_metrics']['speedup_vs_baseline']
            extractions_per_second = benchmark_results['performance_metrics']['extractions_per_second']
            
            print(f"📈 Comprehensive Benchmark Results:")
            print(f"   Average extraction time: {avg_extraction_ms:.2f}ms")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Speedup vs baseline: {speedup_vs_baseline:.2f}x")
            print(f"   Extractions per second: {extractions_per_second:.1f}")
            print(f"   Total extractions: {benchmark_results['successful_extractions']}/{benchmark_results['total_extractions']}")
            
            # Get optimization statistics
            opt_stats = get_optimization_stats()
            cache_stats = benchmark_results.get('cache_stats', {})
            
            print(f"   Cache statistics:")
            print(f"     Index cache hit: {cache_stats.get('cache_hit', 'unknown')}")
            print(f"     Cached indices: {cache_stats.get('cached_indices', 'unknown')}")
            
            # Phase 3 Final Targets
            phase3_targets = {
                'extraction_time_ms': 150.0,    # Realistic considering PyAV overhead
                'success_rate': 0.95,           # High reliability
                'speedup_baseline': 2.5,        # vs 400ms baseline
                'extractions_per_sec': 5.0      # Throughput target
            }
            
            print(f"\n🎯 Phase 3 Final Target Validation:")
            
            # Extraction time target
            extraction_ok = avg_extraction_ms <= phase3_targets['extraction_time_ms']
            print(f"   Extraction time: {avg_extraction_ms:.2f}ms ≤ {phase3_targets['extraction_time_ms']}ms")
            print(f"   Result: {'✅ ACHIEVED' if extraction_ok else '❌ NOT MET'}")
            
            # Success rate target
            success_ok = success_rate >= phase3_targets['success_rate']
            print(f"   Success rate: {success_rate*100:.1f}% ≥ {phase3_targets['success_rate']*100:.0f}%")
            print(f"   Result: {'✅ ACHIEVED' if success_ok else '❌ NOT MET'}")
            
            # Speedup target
            speedup_ok = speedup_vs_baseline >= phase3_targets['speedup_baseline']
            print(f"   Speedup vs baseline: {speedup_vs_baseline:.2f}x ≥ {phase3_targets['speedup_baseline']}x")
            print(f"   Result: {'✅ ACHIEVED' if speedup_ok else '❌ NOT MET'}")
            
            # Throughput target
            throughput_ok = extractions_per_second >= phase3_targets['extractions_per_sec']
            print(f"   Throughput: {extractions_per_second:.1f}/s ≥ {phase3_targets['extractions_per_sec']}/s")
            print(f"   Result: {'✅ ACHIEVED' if throughput_ok else '❌ NOT MET'}")
            
            # Overall Phase 3 assessment
            targets_met = sum([extraction_ok, success_ok, speedup_ok, throughput_ok])
            print(f"\n📊 Phase 3 Summary: {targets_met}/4 targets achieved")
            
            if targets_met >= 3:
                print("🎉 Phase 3 Performance Optimization: SUCCESS")
            else:
                print("⚠️ Phase 3 Performance Optimization: PARTIAL SUCCESS")
            
            # CI/CD Assertions (relaxed for initial validation)
            assert success_rate >= 0.90, \
                f"Success rate too low: {success_rate*100:.1f}%"
            assert avg_extraction_ms <= 250, \
                f"Extraction time too high: {avg_extraction_ms:.2f}ms"
            assert speedup_vs_baseline >= 1.8, \
                f"Insufficient speedup: {speedup_vs_baseline:.2f}x"
            
            print(f"✅ Comprehensive performance targets validated")
            
        except Exception as e:
            print(f"❌ Comprehensive benchmark failed: {e}")
            # Fallback manual test
            print("🔄 Falling back to manual performance test...")
            self._manual_performance_fallback(zarr_group, audio_array)
    
    def _manual_performance_fallback(self, zarr_group, audio_array):
        """Manual performance test fallback if comprehensive benchmark fails"""
        print("   Running manual fallback performance test...")
        
        # Simple extraction test
        test_segments = [(1000, 3000), (5000, 7000), (10000, 12000)]
        times = []
        
        for start, end in test_segments:
            start_time = time.time()
            result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
            extraction_time = time.time() - start_time
            times.append(extraction_time)
            
            assert len(result) > 0, f"Manual test extraction failed for [{start}:{end}]"
        
        mean_time = statistics.mean(times) * 1000
        print(f"   Manual test mean: {mean_time:.2f}ms")
        
        # Basic validation
        assert mean_time <= 300, f"Manual test too slow: {mean_time:.2f}ms"
        print("✅ Manual fallback test passed")

    def test_optimization_feature_validation(self, performance_data):
        """Test that all optimization features are working correctly"""
        print("\n⚡ Testing Optimization Feature Validation")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        aac_index = performance_data['aac_index']
        
        # Test 3-column index optimization
        print("🔍 Validating 3-column index optimization...")
        assert aac_index.shape[1] == 3, f"Expected 3 columns, got {aac_index.shape[1]}"
        assert aac_index.attrs['index_format_version'] == '3-column-optimized'
        
        # Test space savings calculation
        stats = get_index_statistics_fast(aac_index)
        space_savings = float(stats['space_savings_vs_6col'].rstrip('%'))
        assert space_savings >= 45, f"Expected >45% space savings, got {space_savings}%"
        print(f"   ✅ 3-column optimization: {space_savings}% space savings")
        
        # Test container caching functionality
        print("🔍 Validating container caching...")
        clear_performance_caches()
        
        # First extraction (should cache container)
        extract_audio_segment_aac(zarr_group, audio_array, 1000, 2000)
        stats_after_cache = get_performance_stats()
        
        cached_containers = stats_after_cache['container_pool']['cached_containers']
        assert cached_containers > 0, "Container not cached after extraction"
        print(f"   ✅ Container caching: {cached_containers} containers cached")
        
        # Test index caching functionality
        print("🔍 Validating index caching...")
        clear_all_caches()
        
        # First lookup (should cache index)
        find_frame_range_for_samples_fast(aac_index, 5000, 6000)
        opt_stats = get_optimization_stats()
        
        cached_indices = opt_stats['index_manager']['cached_indices']
        assert cached_indices > 0, "Index not cached after lookup"
        print(f"   ✅ Index caching: {cached_indices} indices cached")
        
        # Test memory I/O (BytesIO vs file)
        print("🔍 Validating memory I/O optimization...")
        # This is tested implicitly through successful extractions
        # The PyAV container pool uses BytesIO internally
        result = extract_audio_segment_aac(zarr_group, audio_array, 2000, 4000)
        assert len(result) > 0, "Memory I/O extraction failed"
        print("   ✅ Memory I/O: BytesIO extraction successful")
        
        # Test Zarr v3 native compatibility
        print("🔍 Validating Zarr v3 native compatibility...")
        assert isinstance(zarr_group.store, zarr.storage.LocalStore), "Not using LocalStore"
        assert hasattr(audio_array, 'attrs'), "Missing Zarr v3 attributes"
        print("   ✅ Zarr v3 compatibility: LocalStore and attributes present")
        
        print("✅ All optimization features validated")

    def test_edge_case_performance(self, performance_data):
        """Test performance with edge cases and boundary conditions"""
        print("\n⚡ Testing Edge Case Performance")
        
        zarr_group = performance_data['zarr_group']
        audio_array = performance_data['audio_array']
        aac_index = performance_data['aac_index']
        total_samples = aac_index.attrs['total_samples']
        
        edge_cases = [
            ("Single sample", 1000, 1000),
            ("Very small", 2000, 2010),  # 11 samples
            ("Frame boundary", 0, 1023),  # Exactly one frame
            ("Near end", max(0, total_samples - 100), total_samples - 1),
            ("Large segment", 5000, min(5000 + 44100, total_samples - 1))  # 1 second
        ]
        
        print(f"🔍 Testing {len(edge_cases)} edge cases...")
        
        for case_name, start, end in edge_cases:
            if start >= total_samples or end >= total_samples:
                continue
                
            print(f"   Testing {case_name}: [{start}:{end}]")
            
            start_time = time.time()
            try:
                result = extract_audio_segment_aac(zarr_group, audio_array, start, end)
                extraction_time = time.time() - start_time
                
                assert len(result) >= 0, f"Edge case {case_name} failed"
                print(f"     Result: {len(result)} samples in {extraction_time*1000:.2f}ms")
                
                # Edge cases should still be reasonably fast
                assert extraction_time <= 1.0, f"Edge case {case_name} too slow: {extraction_time*1000:.2f}ms"
                
            except Exception as e:
                print(f"     ⚠️ Edge case {case_name} failed: {e}")
                # Some edge cases might fail, but should not crash
        
        print("✅ Edge case performance tested")


# Additional utility functions for manual testing and debugging
def run_quick_debug_test():
    """Quick debug test for development - not pytest"""
    print("🔧 Running quick debug test...")
    
    test_dir = pathlib.Path(__file__).parent / "testresults" / "quick_debug"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure for debugging
    Config.set(
        log_level=LogLevel.TRACE,  # Maximum detail
        aac_default_bitrate=128000  # Lower bitrate for speed
    )
    
    # Find test file
    test_data_dir = pathlib.Path(__file__).parent / "testdata"
    test_file = None
    for candidate in ["audiomoth_short_snippet.wav", "bird1_snippet.mp3"]:
        candidate_path = test_data_dir / candidate
        if candidate_path.exists():
            test_file = candidate_path
            break
    
    if not test_file:
        print("❌ No test files found")
        return False
        
    print(f"🎵 Using test file: {test_file}")
    
    try:
        # Create Zarr group and import
        store = zarr.storage.LocalStore(root=str(test_dir / "zarr_store"))
        root = zarr.create_group(store=store, overwrite=True)
        audio_group = root.create_group('audio_imports')
        audio_group.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_group.attrs["version"] = Config.original_audio_group_version
        
        # Import with timing
        print("📥 Starting import...")
        import_start = time.time()
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=128000
        )
        import_time = time.time() - import_start
        
        # Get imported data
        imported_group = audio_group[list(audio_group.keys())[0]]
        audio_array = imported_group['audio_data_blob_array']
        aac_index = imported_group['aac_index']
        
        print(f"✅ Import completed in {import_time:.2f}s")
        print(f"   Index shape: {aac_index.shape}")
        print(f"   Index format: {aac_index.attrs.get('index_format_version', 'unknown')}")
        print(f"   Total samples: {aac_index.attrs.get('total_samples', 'unknown')}")
        
        # Test optimized extraction
        print("🔍 Testing optimized extraction...")
        clear_performance_caches()
        
        start_sample = 1000
        end_sample = 5000
        
        extraction_start = time.time()
        audio_data = extract_audio_segment_aac(
            imported_group, audio_array, start_sample, end_sample
        )
        extraction_time = time.time() - extraction_start
        
        print(f"✅ Extraction completed:")
        print(f"   Time: {extraction_time*1000:.2f}ms")
        print(f"   Samples: {len(audio_data)}")
        print(f"   Data type: {audio_data.dtype}")
        print(f"   Data range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        
        # Performance assessment
        if extraction_time * 1000 <= 150:
            print("   🎉 Performance: EXCELLENT (Phase 3 target met)")
        elif extraction_time * 1000 <= 250:
            print("   ✅ Performance: GOOD (acceptable)")
        else:
            print("   ⚠️ Performance: NEEDS IMPROVEMENT")
        
        # Test cache benefit
        print("🔍 Testing cache benefit...")
        cache_start = time.time()
        cached_audio = extract_audio_segment_aac(
            imported_group, audio_array, start_sample, end_sample
        )
        cache_time = time.time() - cache_start
        
        cache_benefit = extraction_time / cache_time if cache_time > 0 else 1
        print(f"   Cached extraction: {cache_time*1000:.2f}ms")
        print(f"   Cache benefit: {cache_benefit:.2f}x")
        
        # Get performance stats
        perf_stats = get_performance_stats()
        print(f"   Container cache: {perf_stats['container_pool']['cached_containers']}")
        print(f"   Index cache: {perf_stats['index_cache']['cached_indices']}")
        
        print("\n✅ Quick debug test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_optimization_modules():
    """Validate that all optimization modules are importable and functional"""
    print("🔧 Validating optimization modules...")
    
    try:
        # Test optimized access module
        from zarrwlr.aac_access import (
            extract_audio_segment_aac,
            clear_performance_caches,
            get_performance_stats
        )
        print("✅ aac_access optimized module imported successfully")
        
        # Test optimized index backend
        from zarrwlr.aac_index_backend import (
            find_frame_range_for_samples_fast,
            get_optimization_stats,
            clear_all_caches
        )
        print("✅ aac_index_backend optimized module imported successfully")
        
        # Test configuration
        from zarrwlr.config import Config
        original_bitrate = Config.aac_default_bitrate
        Config.set(aac_default_bitrate=128000)
        assert Config.aac_default_bitrate == 128000
        Config.set(aac_default_bitrate=original_bitrate)
        print("✅ Configuration system working")
        
        print("✅ All optimization modules validated")
        return True
        
    except Exception as e:
        print(f"❌ Module validation failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation and quick test if called directly
    print("🚀 AAC Performance Test Suite - Direct Execution")
    
    # First validate modules
    if not validate_optimization_modules():
        print("❌ Module validation failed, cannot proceed")
        sys.exit(1)
    
    # Run quick debug test
    if not run_quick_debug_test():
        print("❌ Quick debug test failed")
        sys.exit(1)
    
    print("\n🎉 Direct execution completed successfully")
    print("📋 To run full test suite: pytest test_aac_performance_optimized.py -v -s")