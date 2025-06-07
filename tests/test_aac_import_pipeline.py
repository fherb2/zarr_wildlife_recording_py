"""
Test Suite for AAC Import Pipeline - Updated for 3-Column Index Optimization
============================================================================

Pytest-compatible comprehensive testing of the AAC import pipeline with
optimized 3-column index structure for minimal overhead.

Tests are designed to be CI-ready with proper fixtures, assertions, and cleanup.

Run with: pytest test_aac_import_pipeline.py -v -s
"""

import pytest
import pathlib
import shutil
import tempfile
import numpy as np
import subprocess
from typing import List, Optional
import zarr

# Import the modules to test
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.config import Config
from zarrwlr.packagetypes import LogLevel
from zarrwlr import aimport
from zarrwlr.aac_access import (
    import_aac_to_zarr,
    _convert_to_aac_ffmpeg,
    extract_audio_segment_aac
)
from zarrwlr.aac_index_backend import (
    build_aac_index,
    get_index_statistics,
    validate_aac_index,
    diagnose_aac_data,
    _analyze_real_aac_frames,
    get_aac_frame_samples,
    calculate_timestamp_ms,
    get_sample_position_for_frame
)


class TestAAC_ImportPipeline:
    """Test class for AAC import pipeline validation with 3-column index optimization"""
    
    @pytest.fixture(scope="class")
    def test_environment(self):
        """Set up test environment - runs once per test class"""
        print("\n🔧 Setting up AAC Import Pipeline Test Environment (3-Column Optimized)")
        
        # Configure logging for debugging
        Config.set(
            log_level=LogLevel.TRACE,
            aac_default_bitrate=160000,
            aac_enable_pyav_native=True,
            aac_fallback_to_ffmpeg=True,
            aac_max_workers=2
        )
        
        # Create test directories
        test_dir = pathlib.Path(__file__).parent / "testresults" / "aac_import_test_3col"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'test_dir': test_dir,
            'zarr_store_path': test_dir / "aac_test_store_3col"
        }
    
    @pytest.fixture
    def test_files(self) -> List[pathlib.Path]:
        """Get available test audio files"""
        test_data_dir = pathlib.Path(__file__).parent / "testdata"
        
        # Priority order: prefer WAV for predictable results
        candidate_files = [
            "audiomoth_short_snippet.wav",
            "audiomoth_long_snippet.wav", 
            "bird1_snippet.mp3",
            "camtrap_snippet.mov"
        ]
        
        available_files = []
        for filename in candidate_files:
            filepath = test_data_dir / filename
            if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                available_files.append(filepath)
                
        print(f"📁 Found {len(available_files)} test files: {[f.name for f in available_files]}")
        return available_files
    
    @pytest.fixture
    def zarr_group(self, test_environment):
        """Create fresh Zarr group for each test"""
        store_path = test_environment['zarr_store_path']
        
        # Remove if exists
        if store_path.exists():
            shutil.rmtree(store_path)
            
        # Create new store
        store = zarr.storage.LocalStore(root=str(store_path))
        root = zarr.create_group(store=store, overwrite=True)
        audio_import_grp = root.create_group('audio_imports')
        
        # Set required attributes
        audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_import_grp.attrs["version"] = Config.original_audio_group_version
        
        return audio_import_grp
    
    def test_ffmpeg_availability(self):
        """CI-Critical: Verify ffmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            assert result.returncode == 0, "ffmpeg not available or returned error"
            print(f"✅ ffmpeg available: {result.stdout.split()[2]}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.fail(f"ffmpeg not available: {e}")
    
    def test_3column_index_constants(self):
        """Test 3-column index optimization constants"""
        print("🔧 Testing 3-column index constants...")
        
        # Test samples per frame constant
        frame_samples = get_aac_frame_samples()
        assert frame_samples == 1024, f"Expected 1024 samples per frame, got {frame_samples}"
        print(f"✅ Frame samples constant: {frame_samples}")
        
        # Test sample position calculation
        for frame_idx in [0, 1, 10, 100]:
            sample_pos = get_sample_position_for_frame(frame_idx)
            expected_pos = frame_idx * 1024
            assert sample_pos == expected_pos, f"Sample position mismatch for frame {frame_idx}"
        print("✅ Sample position calculations correct")
        
        # Test timestamp calculation
        sample_rates = [44100, 48000, 96000]
        for sample_rate in sample_rates:
            # Test 1 second worth of samples
            samples_per_second = sample_rate
            timestamp_ms = calculate_timestamp_ms(samples_per_second, sample_rate)
            assert abs(timestamp_ms - 1000) <= 1, f"Timestamp calculation error at {sample_rate}Hz"
        print("✅ Timestamp calculations correct")
    
    def test_ffmpeg_aac_conversion(self, test_files, test_environment):
        """Test ffmpeg AAC conversion step in isolation"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]  # Use first available file
        print(f"🎵 Testing ffmpeg conversion with: {test_file.name}")
        
        # Get source parameters (simplified)
        source_params = {
            "nb_channels": 2,
            "sampling_rate": 48000,
            "sample_format": "s16",
            "bit_rate": None
        }
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp_file:
            tmp_path = pathlib.Path(tmp_file.name)
        
        try:
            # Test the conversion
            _convert_to_aac_ffmpeg(
                input_file=test_file,
                output_file=tmp_path,
                bitrate=128000,  # Lower bitrate for faster test
                source_params=source_params
            )
            
            # Validate output
            assert tmp_path.exists(), "Output file was not created"
            assert tmp_path.stat().st_size > 100, f"Output file too small: {tmp_path.stat().st_size} bytes"
            
            # Check for ADTS header (0xFFF sync pattern)
            with open(tmp_path, 'rb') as f:
                header_bytes = f.read(10)
                
            # Look for ADTS sync pattern in first few bytes
            found_sync = False
            for i in range(len(header_bytes) - 1):
                sync_word = int.from_bytes(header_bytes[i:i+2], 'big')
                if (sync_word & 0xFFF0) == 0xFFF0:
                    found_sync = True
                    print(f"✅ Found ADTS sync pattern at offset {i}: 0x{sync_word:04X}")
                    break
            
            assert found_sync, f"No ADTS sync pattern found in header: {header_bytes.hex()}"
            
            print(f"✅ ffmpeg conversion successful: {tmp_path.stat().st_size} bytes")
            
        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_adts_frame_analysis_3column(self, test_files, test_environment):
        """Test ADTS frame analysis with real AAC data for 3-column optimization"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]
        print(f"🔍 Testing ADTS frame analysis (3-column) with: {test_file.name}")
        
        # Convert to AAC first
        source_params = {"nb_channels": 2, "sampling_rate": 48000}
        
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp_file:
            tmp_path = pathlib.Path(tmp_file.name)
        
        try:
            _convert_to_aac_ffmpeg(test_file, tmp_path, 128000, source_params)
            
            # Read AAC data
            with open(tmp_path, 'rb') as f:
                aac_data = f.read()
            
            print(f"📊 AAC data size: {len(aac_data)} bytes")
            
            # Test frame analysis for 3-column optimization
            frames = _analyze_real_aac_frames(aac_data, 48000)
            
            # Validate results
            assert len(frames) > 0, "No AAC frames found"
            assert len(frames) >= 1, f"Expected at least 1 frame, got {len(frames)}"
            
            print(f"✅ Found {len(frames)} AAC frames")
            
            # Validate 3-column frame structure
            for i, frame in enumerate(frames[:3]):  # Check first 3 frames
                assert 'byte_offset' in frame, f"Frame {i} missing byte_offset"
                assert 'frame_size' in frame, f"Frame {i} missing frame_size"
                assert 'sample_pos' in frame, f"Frame {i} missing sample_pos (3-column format)"
                
                # Validate ranges
                assert 0 <= frame['byte_offset'] < len(aac_data), f"Invalid byte_offset: {frame['byte_offset']}"
                assert 7 <= frame['frame_size'] <= 16384, f"Invalid frame_size: {frame['frame_size']}"
                
                # Validate 3-column optimization: sample_pos should be frame_idx * 1024
                expected_sample_pos = i * 1024  # 1024 samples per frame
                assert frame['sample_pos'] == expected_sample_pos, f"Frame {i} sample_pos mismatch: got {frame['sample_pos']}, expected {expected_sample_pos}"
                
                print(f"  Frame {i}: offset={frame['byte_offset']}, size={frame['frame_size']}, sample_pos={frame['sample_pos']}")
            
            # Check frame progression
            if len(frames) > 1:
                assert frames[1]['byte_offset'] > frames[0]['byte_offset'], "Frame offsets not increasing"
                assert frames[1]['sample_pos'] > frames[0]['sample_pos'], "Sample positions not increasing"
                
                # Validate sample position increment is exactly 1024
                pos_diff = frames[1]['sample_pos'] - frames[0]['sample_pos']
                assert pos_diff == 1024, f"Sample position increment should be 1024, got {pos_diff}"
                
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_full_import_pipeline_3column(self, test_files, zarr_group):
        """Test complete AAC import pipeline end-to-end with 3-column optimization"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]
        print(f"🚀 Testing full import pipeline (3-column) with: {test_file.name}")
        print(f"📁 File size: {test_file.stat().st_size} bytes")
        
        # Import using the main interface
        try:
            aimport.import_original_audio_file(
                audio_file=test_file,
                zarr_original_audio_group=zarr_group,
                first_sample_time_stamp=None,
                target_codec='aac',
                aac_bitrate=160000
            )
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            raise
        
        # Validate import results
        group_names = list(zarr_group.keys())
        assert len(group_names) > 0, "No groups created after import"
        print(f"✅ Created group: {group_names[0]}")
        
        # Get the imported group
        imported_group = zarr_group[group_names[0]]
        
        # Check required arrays exist
        assert 'audio_data_blob_array' in imported_group, "audio_data_blob_array missing"
        assert 'aac_index' in imported_group, "aac_index missing"
        
        # Validate audio array
        audio_array = imported_group['audio_data_blob_array']
        assert audio_array.shape[0] > 0, "Audio array is empty"
        assert audio_array.dtype == np.uint8, f"Wrong dtype: {audio_array.dtype}"
        
        # Check metadata
        assert audio_array.attrs.get('codec') == 'aac', f"Wrong codec: {audio_array.attrs.get('codec')}"
        assert audio_array.attrs.get('aac_bitrate') == 160000, f"Wrong bitrate: {audio_array.attrs.get('aac_bitrate')}"
        assert audio_array.attrs.get('sample_rate') > 0, "Invalid sample rate"
        assert audio_array.attrs.get('nb_channels') > 0, "Invalid channel count"
        
        print(f"✅ Audio array: {audio_array.shape[0]} bytes, {audio_array.attrs.get('sample_rate')}Hz, {audio_array.attrs.get('nb_channels')} channels")
        
        # Validate index (3-column optimized format)
        aac_index = imported_group['aac_index']
        assert aac_index.shape[1] == 3, f"Index should have 3 columns (optimized), got {aac_index.shape[1]}"
        assert aac_index.shape[0] > 0, "Index has no frames"
        
        # Verify 3-column structure
        assert 'index_format_version' in aac_index.attrs, "Missing index format version"
        assert aac_index.attrs['index_format_version'] == '3-column-optimized', "Wrong index format"
        
        # Test index validation
        assert validate_aac_index(aac_index, audio_array), "Index validation failed"
        
        # Get and validate statistics
        stats = get_index_statistics(aac_index)
        assert stats['total_frames'] > 0, "No frames in statistics"
        assert stats['total_samples'] > 0, "No samples in statistics"
        assert stats['duration_ms'] > 0, "Invalid duration"
        assert stats['index_format'] == '3-column-optimized', "Wrong index format in stats"
        
        print(f"✅ Index: {stats['total_frames']} frames, {stats['total_samples']} samples, {stats['duration_ms']}ms")
        print(f"   Frame size: {stats['frame_size_stats']['min']}-{stats['frame_size_stats']['max']} bytes")
        print(f"   Index format: {stats['index_format']} ({stats['space_savings_vs_6col']} space savings)")
        
        # Test data diagnosis with 3-column optimization
        diagnosis = diagnose_aac_data(audio_array)
        assert diagnosis['has_adts_headers'], "ADTS headers not detected"
        assert diagnosis['sync_patterns_found'] > 0, "No sync patterns found"
        assert diagnosis['optimization_format'] == '3-column-index', "Wrong optimization format"
        assert len(diagnosis['issues']) == 0, f"Data issues found: {diagnosis['issues']}"
        
        # Validate index overhead comparison
        if 'index_overhead_comparison' in diagnosis:
            overhead = diagnosis['index_overhead_comparison']
            assert overhead['savings_percent'] >= 45, f"Expected >45% index savings, got {overhead['savings_percent']:.1f}%"
            print(f"✅ Index overhead: {overhead['savings_percent']:.1f}% reduction vs 6-column")
        
        print(f"✅ Diagnosis: {diagnosis['sync_patterns_found']} sync patterns, {diagnosis['size_mb']:.2f}MB")
        
        return imported_group, audio_array, aac_index
    
    def test_3column_index_optimization(self, test_files, zarr_group):
        """Test 3-column index optimization specifically"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]
        print(f"🔧 Testing 3-column index optimization with: {test_file.name}")
        
        # Import the file
        imported_group, audio_array, aac_index = self.test_full_import_pipeline_3column(test_files, zarr_group)
        
        # Validate 3-column structure
        assert aac_index.shape[1] == 3, f"Expected 3 columns, got {aac_index.shape[1]}"
        
        # Test calculated values
        frame_samples = get_aac_frame_samples()
        assert frame_samples == 1024, f"Expected 1024 samples per frame, got {frame_samples}"
        
        # Test timestamp calculation
        sample_rate = aac_index.attrs['sample_rate']
        timestamp_ms = calculate_timestamp_ms(48000, sample_rate)  # 1 second worth of samples
        expected_ms = 1000 if sample_rate == 48000 else int(48000 * 1000 / sample_rate)
        assert abs(timestamp_ms - expected_ms) <= 1, f"Timestamp calculation error: {timestamp_ms} vs {expected_ms}"
        
        # Test sample position calculation
        for frame_idx in [0, 1, 10]:
            if frame_idx < aac_index.shape[0]:
                calculated_pos = get_sample_position_for_frame(frame_idx)
                stored_pos = aac_index[frame_idx, 2]  # 3rd column is sample_pos
                assert calculated_pos == stored_pos, f"Sample position mismatch at frame {frame_idx}"
        
        # Validate space savings
        stats = get_index_statistics(aac_index)
        space_savings = float(stats['space_savings_vs_6col'].rstrip('%'))
        assert space_savings >= 45, f"Expected >45% space savings, got {space_savings}%"
        
        print(f"✅ 3-column index optimization validated: {space_savings}% space savings")
    
    def test_import_data_integrity_3column(self, test_files, zarr_group):
        """Test data integrity and compression ratios with 3-column optimization"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]
        original_size = test_file.stat().st_size
        print(f"📊 Testing data integrity (3-column) for: {test_file.name} ({original_size} bytes)")
        
        # Import the file
        imported_group, audio_array, aac_index = self.test_full_import_pipeline_3column(test_files, zarr_group)
        
        # Calculate compression
        compressed_size = audio_array.shape[0]
        index_size = aac_index.nbytes
        total_size = compressed_size + index_size
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        overhead_percentage = (index_size / compressed_size) * 100 if compressed_size > 0 else 0
        
        # Calculate 3-column index optimization
        index_overhead_old = aac_index.shape[0] * 6 * 8  # Old 6-column format
        index_overhead_new = aac_index.nbytes  # New 3-column format
        space_savings = index_overhead_old - index_overhead_new
        space_savings_percent = (space_savings / index_overhead_old) * 100 if index_overhead_old > 0 else 0
        
        print(f"📈 Compression Analysis (3-Column Optimized):")
        print(f"   Original size: {original_size:,} bytes")
        print(f"   Compressed size: {compressed_size:,} bytes")
        print(f"   Index size (3-col): {index_size:,} bytes")
        print(f"   Total size: {total_size:,} bytes")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Index overhead: {overhead_percentage:.2f}%")
        
        print(f"📈 Index Optimization:")
        print(f"   Old 6-column size: {index_overhead_old:,} bytes")
        print(f"   New 3-column size: {index_overhead_new:,} bytes") 
        print(f"   Space savings: {space_savings:,} bytes ({space_savings_percent:.1f}%)")
        
        # Validate compression expectations
        assert compressed_size < original_size, "No compression achieved"
        assert compression_ratio >= 1.1, f"Insufficient compression: {compression_ratio:.2f}x"
        assert overhead_percentage < 10.0, f"Index overhead too high: {overhead_percentage:.2f}%"
        
        # Validate 3-column optimization
        assert space_savings_percent >= 45, f"Expected >45% index space savings, got {space_savings_percent:.1f}%"
        
        # Validate frame data consistency
        stats = get_index_statistics(aac_index)
        expected_duration_ms = (stats['total_samples'] / audio_array.attrs['sample_rate']) * 1000
        actual_duration_ms = stats['duration_ms']
        
        duration_diff = abs(expected_duration_ms - actual_duration_ms)
        duration_tolerance = max(50, expected_duration_ms * 0.05)  # 5% or 50ms
        
        assert duration_diff <= duration_tolerance, f"Duration mismatch: expected {expected_duration_ms:.0f}ms, got {actual_duration_ms}ms"
        
        print(f"✅ Duration consistency: {actual_duration_ms:.0f}ms (expected {expected_duration_ms:.0f}ms)")
        print(f"✅ 3-column optimization: {space_savings_percent:.1f}% index space reduction achieved")
    
    def test_multiple_bitrates_3column(self, test_files, test_environment):
        """Test import with different bitrates using 3-column optimization"""
        if not test_files:
            pytest.skip("No test files available")
            
        test_file = test_files[0]
        bitrates = [128000, 192000]  # Test range
        
        print(f"🎛️ Testing multiple bitrates (3-column) with: {test_file.name}")
        
        for bitrate in bitrates:
            print(f"   Testing {bitrate} bps...")
            
            # Create separate store for each bitrate
            store_path = test_environment['zarr_store_path'] / f"bitrate_{bitrate}_3col"
            if store_path.exists():
                shutil.rmtree(store_path)
                
            store = zarr.storage.LocalStore(root=str(store_path))
            root = zarr.create_group(store=store)
            test_group = root.create_group('audio_imports')
            test_group.attrs["magic_id"] = Config.original_audio_group_magic_id
            test_group.attrs["version"] = Config.original_audio_group_version
            
            # Import with specific bitrate
            aimport.import_original_audio_file(
                audio_file=test_file,
                zarr_original_audio_group=test_group,
                first_sample_time_stamp=None,
                target_codec='aac',
                aac_bitrate=bitrate
            )
            
            # Validate bitrate in metadata
            imported_group = test_group[list(test_group.keys())[0]]
            audio_array = imported_group['audio_data_blob_array']
            aac_index = imported_group['aac_index']
            
            assert audio_array.attrs['aac_bitrate'] == bitrate, f"Bitrate mismatch: expected {bitrate}, got {audio_array.attrs['aac_bitrate']}"
            
            # Validate 3-column structure
            assert aac_index.shape[1] == 3, f"Expected 3 columns, got {aac_index.shape[1]}"
            assert aac_index.attrs['index_format_version'] == '3-column-optimized', "Wrong index format"
            
            print(f"   ✅ {bitrate} bps: {audio_array.shape[0]} bytes, 3-column index with {aac_index.shape[0]} frames")


# Additional utility functions for manual testing
def run_manual_debug_test_3column():
    """Manual test function for debugging 3-column optimization - not pytest"""
    print("🔧 Running manual debug test (3-column optimization)...")
    
    # Set up environment
    test_dir = pathlib.Path(__file__).parent / "testresults" / "manual_debug_3col"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure for verbose output
    Config.set(
        log_level=LogLevel.TRACE,
        aac_default_bitrate=128000
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
        return
        
    print(f"🎵 Using test file: {test_file}")
    
    # Create Zarr group
    store = zarr.storage.LocalStore(root=str(test_dir / "zarr_store"))
    root = zarr.create_group(store=store, overwrite=True)
    audio_group = root.create_group('audio_imports')
    audio_group.attrs["magic_id"] = Config.original_audio_group_magic_id
    audio_group.attrs["version"] = Config.original_audio_group_version
    
    try:
        # Test import
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=128000
        )
        
        # Validate 3-column optimization
        imported_group = audio_group[list(audio_group.keys())[0]]
        aac_index = imported_group['aac_index']
        
        print(f"✅ Manual test completed successfully")
        print(f"   Index shape: {aac_index.shape}")
        print(f"   Index format: {aac_index.attrs.get('index_format_version', 'unknown')}")
        
        stats = get_index_statistics(aac_index)
        print(f"   Space savings: {stats.get('space_savings_vs_6col', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run manual debug if called directly
    run_manual_debug_test_3column()