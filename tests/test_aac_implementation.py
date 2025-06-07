"""
Test Suite for AAC Implementation
=================================

Comprehensive testing of AAC import, indexing, and extraction functionality.
Based on the existing test patterns and follows the project's testing guidelines.

Test Structure:
- Unit tests for AAC modules
- Integration tests with Zarr storage
- Performance benchmarks
- Error handling validation
- Compatibility with existing FLAC infrastructure
"""

import unittest
import pathlib
import shutil
import tempfile
import numpy as np
import time
from typing import List
import zarr

# Test setup (based on project structure)
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.config import Config
from zarrwlr import aimport
from zarrwlr.aac_access import (
    import_aac_to_zarr,
    extract_audio_segment_aac,
    parallel_extract_audio_segments_aac,
    build_aac_index
)
from zarrwlr.aac_index_backend import (
    build_aac_index,
    AACStreamAnalyzer,  # Falls implementiert
    get_index_statistics,
    validate_aac_index,
    benchmark_aac_access,
    diagnose_aac_data
)

# Test constants
ZARR3_STORE_DIR = pathlib.Path(__file__).parent / "testresults" / "aac_test_zarr3_store"


class TestAACIntegration(unittest.TestCase):
    """Integration tests for AAC import and extraction
    
    Architecture:
    - Import: ffmpeg (universal, subprocess-based)
    - Random Access: PyAV (native Python, fast extraction)
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Clean test results directory
        if ZARR3_STORE_DIR.exists():
            shutil.rmtree(ZARR3_STORE_DIR)
        ZARR3_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Import LogLevel from packagetypes
        from zarrwlr.packagetypes import LogLevel
        
        # Configure for testing
        Config.set(
            log_level=LogLevel.TRACE,  # Use correct LogLevel import
            aac_default_bitrate=160000,
            aac_enable_pyav_native=True,
            aac_fallback_to_ffmpeg=True,
            aac_max_workers=2  # Limit for test environment
        )
    
    def setUp(self):
        """Set up individual test"""
        self.test_files = self.get_test_files()
        self.zarr_group = self.prepare_zarr_database()
    
    def get_test_files(self) -> List[pathlib.Path]:
        """Get list of test audio files"""
        test_files = [
            "testdata/audiomoth_long_snippet.wav",
            "testdata/audiomoth_short_snippet.wav", 
            "testdata/bird1_snippet.mp3",
            "testdata/camtrap_snippet.mov"  # mp4 with audio
        ]
        base_path = pathlib.Path(__file__).parent
        return [base_path / file for file in test_files if (base_path / file).exists()]
    
    def prepare_zarr_database(self) -> zarr.Group:
        """Prepare Zarr database for testing"""
        store = zarr.storage.LocalStore(root=str(ZARR3_STORE_DIR))
        root = zarr.create_group(store=store, overwrite=True)
        audio_import_grp = root.create_group('audio_imports')
        audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_import_grp.attrs["version"] = Config.original_audio_group_version
        return audio_import_grp
    
    def test_aac_import_basic(self):
        """Test basic AAC import functionality (ffmpeg-based)"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        # Import with AAC using ffmpeg
        try:
            aimport.import_original_audio_file(
                audio_file=test_file,
                zarr_original_audio_group=self.zarr_group,
                first_sample_time_stamp=None,
                target_codec='aac',
                aac_bitrate=160000
            )
            
            # Verify import results
            group_names = list(self.zarr_group.keys())
            self.assertGreater(len(group_names), 0, "No groups created after import")
            
            # Check imported content
            imported_group = self.zarr_group[group_names[0]]
            self.assertIn('audio_data_blob_array', imported_group)
            self.assertIn('aac_index', imported_group)
            
            # Verify metadata
            audio_array = imported_group['audio_data_blob_array']
            self.assertEqual(audio_array.attrs['codec'], 'aac')
            self.assertEqual(audio_array.attrs['aac_bitrate'], 160000)
            
            print(f"✓ AAC import successful for {test_file.name} (ffmpeg-based)")
            
        except Exception as e:
            self.fail(f"AAC import failed: {e}")


    def test_aac_index_creation(self):
        """Test AAC index creation and validation"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        # Import AAC file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=128000  # Lower bitrate for testing
        )
        
        imported_group = self.zarr_group[list(self.zarr_group.keys())[0]]
        aac_index = imported_group['aac_index']
        audio_blob = imported_group['audio_data_blob_array']
        
        # Validate index structure
        self.assertEqual(aac_index.shape[1], 6, "AAC index should have 6 columns")
        self.assertGreater(aac_index.shape[0], 0, "AAC index should have frames")
        
        # Validate index integrity
        self.assertTrue(validate_aac_index(aac_index, audio_blob))
        
        # Get index statistics
        stats = get_index_statistics(aac_index)
        self.assertGreater(stats['total_frames'], 0)
        self.assertGreater(stats['total_samples'], 0)
        self.assertGreater(stats['duration_ms'], 0)
        
        print(f"✓ AAC index validation successful: {stats['total_frames']} frames, {stats['duration_ms']}ms")
    
    def test_aac_extraction(self):
        """Test AAC audio segment extraction"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        # Import AAC file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=None,
            target_codec='aac'
        )
        
        imported_group = self.zarr_group[list(self.zarr_group.keys())[0]]
        aac_index = imported_group['aac_index']
        audio_blob = imported_group['audio_data_blob_array']
        
        # Get total samples
        total_samples = int(aac_index[-1, 2] + aac_index[-1, 4])  # sample_pos + sample_count
        
        if total_samples < 1000:
            self.skipTest("Audio too short for extraction test")
        
        # Test single extraction
        start_sample = 1000
        end_sample = 2000
        
        extracted = extract_audio_segment_aac(
            imported_group, audio_blob, start_sample, end_sample
        )
        
        self.assertIsInstance(extracted, np.ndarray)
        self.assertGreater(len(extracted), 0, "Extracted audio should not be empty")
        expected_length = end_sample - start_sample + 1
        self.assertLessEqual(abs(len(extracted) - expected_length), 1024, 
                            "Extracted length should be approximately correct")
        
        print(f"✓ AAC extraction successful: {len(extracted)} samples extracted")
    
    def test_aac_parallel_extraction(self):
        """Test parallel AAC extraction"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        # Import AAC file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=None,
            target_codec='aac'
        )
        
        imported_group = self.zarr_group[list(self.zarr_group.keys())[0]]
        aac_index = imported_group['aac_index']
        audio_blob = imported_group['audio_data_blob_array']
        
        # Get total samples
        total_samples = int(aac_index[-1, 2] + aac_index[-1, 4])
        
        if total_samples < 10000:
            self.skipTest("Audio too short for parallel extraction test")
        
        # Create multiple segments
        segments = [
            (1000, 2000),
            (3000, 4000),
            (5000, 6000)
        ]
        
        # Test parallel extraction
        start_time = time.time()
        extracted_segments = parallel_extract_audio_segments_aac(
            imported_group, audio_blob, segments, max_workers=2
        )
        parallel_time = time.time() - start_time
        
        self.assertEqual(len(extracted_segments), len(segments))
        
        for i, segment in enumerate(extracted_segments):
            self.assertIsInstance(segment, np.ndarray)
            self.assertGreater(len(segment), 0, f"Segment {i} should not be empty")
        
        print(f"✓ AAC parallel extraction successful: {len(segments)} segments in {parallel_time:.3f}s")
    
    def test_aac_configuration(self):
        """Test AAC configuration parameters"""
        # Test different bitrates
        test_bitrates = [128000, 160000, 192000]
        
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        for bitrate in test_bitrates:
            with self.subTest(bitrate=bitrate):
                # Create separate group for each bitrate test
                test_group = self.zarr_group.create_group(f'test_bitrate_{bitrate}')
                
                Config.set(aac_default_bitrate=bitrate)
                
                try:
                    aimport.import_original_audio_file(
                        audio_file=test_file,
                        zarr_original_audio_group=test_group,
                        first_sample_time_stamp=None,
                        target_codec='aac',
                        aac_bitrate=bitrate
                    )
                    
                    # Verify bitrate in metadata
                    imported_group = test_group[list(test_group.keys())[0]]
                    audio_array = imported_group['audio_data_blob_array']
                    self.assertEqual(audio_array.attrs['aac_bitrate'], bitrate)
                    
                    print(f"✓ AAC bitrate test successful: {bitrate} bps")
                    
                except Exception as e:
                    self.fail(f"AAC import failed with bitrate {bitrate}: {e}")


class TestAACPerformance(unittest.TestCase):
    """Performance tests for AAC implementation"""
    
    def setUp(self):
        """Set up performance test"""
        self.test_files = self.get_test_files()
        if ZARR3_STORE_DIR.exists():
            shutil.rmtree(ZARR3_STORE_DIR)
        ZARR3_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.zarr_group = self.prepare_zarr_database()
    
    def get_test_files(self) -> List[pathlib.Path]:
        """Get test files for performance testing"""
        test_files = [
            "testdata/audiomoth_long_snippet.wav",  # Prefer longer files for performance tests
            "testdata/audiomoth_short_snippet.wav"
        ]
        base_path = pathlib.Path(__file__).parent
        return [base_path / file for file in test_files if (base_path / file).exists()]
    
    def prepare_zarr_database(self) -> zarr.Group:
        """Prepare Zarr database"""
        store = zarr.storage.LocalStore(root=str(ZARR3_STORE_DIR))
        root = zarr.create_group(store=store, overwrite=True)
        audio_import_grp = root.create_group('audio_imports')
        audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        audio_import_grp.attrs["version"] = Config.original_audio_group_version
        return audio_import_grp
    

    def test_aac_import_performance(self):
        """Test AAC import performance (ffmpeg subprocess)"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        file_size_mb = test_file.stat().st_size / 1024 / 1024
        
        start_time = time.time()
        
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=None,
            target_codec='aac',
            aac_bitrate=160000
        )
        
        import_time = time.time() - start_time
        throughput = file_size_mb / import_time
        
        print(f"✓ AAC import performance (ffmpeg): {file_size_mb:.1f}MB in {import_time:.2f}s ({throughput:.1f} MB/s)")
        
        # Adjust expectations for ffmpeg subprocess overhead
        # ffmpeg ist langsamer als PyAV aber sollte trotzdem vernünftig sein
        self.assertLess(import_time, file_size_mb * 10, "Import should be faster than 10s per MB (ffmpeg)")
    
    
    def test_aac_random_access_benchmark(self):
        """Benchmark AAC random access performance"""
        if not self.test_files:
            self.skipTest("No test files available")
        
        test_file = self.test_files[0]
        
        # Import AAC file
        aimport.import_original_audio_file(
            audio_file=test_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=None,
            target_codec='aac'
        )
        
        imported_group = self.zarr_group[list(self.zarr_group.keys())[0]]
        audio_blob = imported_group['audio_data_blob_array']
        
        # Run benchmark
        benchmark_results = benchmark_aac_access(
            imported_group, audio_blob, num_extractions=50
        )
        
        self.assertIn('performance_metrics', benchmark_results)
        
        avg_extraction_ms = benchmark_results['performance_metrics']['average_extraction_ms']
        success_rate = benchmark_results['performance_metrics']['success_rate']
        
        print(f"✓ AAC random access benchmark:")
        print(f"  Average extraction time: {avg_extraction_ms:.2f}ms")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Extractions per second: {benchmark_results['performance_metrics']['extractions_per_second']:.1f}")
        
        # Performance assertions
        self.assertLess(avg_extraction_ms, 50, "Average extraction should be under 50ms")
        self.assertGreater(success_rate, 0.9, "Success rate should be over 90%")


class TestAACErrorHandling(unittest.TestCase):
    """Error handling and edge case tests"""
    
    def test_invalid_aac_data(self):
        """Test handling of invalid AAC data"""
        # Create dummy invalid data
        invalid_data = np.random.randint(0, 255, 1000, dtype=np.uint8)
        
        # Create temporary Zarr group
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        test_group = root.create_group('test')
        
        # Create invalid audio array using correct Zarr API
        audio_array = test_group.create_array(
            name='audio_data_blob_array',
            shape=invalid_data.shape,
            dtype=np.uint8
        )
        audio_array[:] = invalid_data  # Set data after creation
        audio_array.attrs['codec'] = 'aac'
        audio_array.attrs['sample_rate'] = 48000
        
        # Test diagnosis
        diagnosis = diagnose_aac_data(audio_array)
        self.assertIn('issues', diagnosis)
        self.assertGreater(len(diagnosis['issues']), 0, "Should detect issues with invalid data")
        
        print(f"✓ Invalid AAC data handling: {len(diagnosis['issues'])} issues detected")
    
    def test_missing_index_error(self):
        """Test error handling when AAC index is missing"""
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        test_group = root.create_group('test')
        
        # Create audio array without index using correct Zarr API
        audio_array = test_group.create_array(
            name='audio_data_blob_array',
            shape=(1000,),
            dtype=np.uint8
        )
        audio_array[:] = np.zeros(1000, dtype=np.uint8)
        
        # Should raise error when trying to extract without index
        with self.assertRaises(ValueError):
            extract_audio_segment_aac(test_group, audio_array, 0, 100)
        
        print("✓ Missing index error handling works correctly")


def run_all_aac_tests():
    """Run all AAC tests with proper setup"""
    print("🔧 Starting AAC Implementation Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestAACIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestAACPerformance))
    suite.addTest(loader.loadTestsFromTestCase(TestAACErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All AAC tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} test failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_aac_tests()
    exit(0 if success else 1)