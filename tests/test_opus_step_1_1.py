#!/usr/bin/env python3
"""
Test Suite for Step 1.1: Raw Opus Detection & Parsing
=====================================================

Tests the new OpusContainerDetector and RawOpusParser classes
"""

import pathlib
import tempfile
import subprocess
import pytest
import sys

# Add zarrwlr to path for testing
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.opus_access import OpusContainerDetector, RawOpusParser
from zarrwlr.logsetup import get_module_logger
from zarrwlr.config import Config

logger = get_module_logger(__file__)

class TestOpusStep1_1:
    """Test suite for Step 1.1 components"""
    
    def setup_method(self):
        """Setup for each test"""
        self.test_data_dir = pathlib.Path(__file__).parent / "testdata"
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        
        # Ensure test data directory exists
        self.test_data_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def find_test_audio_file(self) -> pathlib.Path:
        """Find any available test audio file"""
        test_candidates = [
            self.test_data_dir / "audiomoth_long_snippet.wav",
            self.test_data_dir / "bird1_snippet.mp3", 
            self.test_data_dir / "audiomoth_short_snippet.wav",
            # Look for any audio file in testdata
            *list(self.test_data_dir.glob("*.wav")),
            *list(self.test_data_dir.glob("*.mp3")),
            *list(self.test_data_dir.glob("*.ogg")),
        ]
        
        for candidate in test_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        
        pytest.skip("No test audio files found in testdata/")
    
    def create_test_opus_file(self, source_audio: pathlib.Path, format_type: str) -> pathlib.Path:
        """Create test Opus file in specified format"""
        if format_type == 'raw_opus':
            output_file = self.temp_dir / f"test_{source_audio.stem}.opus"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(source_audio),
                '-c:a', 'libopus', '-b:a', '128k',
                '-f', 'opus', str(output_file)
            ]
        elif format_type == 'ogg':
            output_file = self.temp_dir / f"test_{source_audio.stem}.ogg"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(source_audio),
                '-c:a', 'libopus', '-b:a', '128k', 
                '-f', 'ogg', str(output_file)
            ]
        elif format_type == 'webm':
            output_file = self.temp_dir / f"test_{source_audio.stem}.webm"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(source_audio),
                '-c:a', 'libopus', '-b:a', '128k',
                '-f', 'webm', str(output_file)
            ]
        else:
            raise ValueError(f"Unsupported test format: {format_type}")
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not output_file.exists():
                pytest.skip(f"Failed to create {format_type} test file")
            
            return output_file
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip(f"Cannot create {format_type} test file (ffmpeg issue)")
    
    def test_container_format_detection_by_extension(self):
        """Test format detection by file extension"""
        
        # Test extension-based detection
        test_cases = [
            ('test.opus', 'raw_opus'),
            ('test.ogg', 'ogg'),
            ('test.oga', 'ogg'), 
            ('test.webm', 'webm'),
            ('test.mkv', 'mkv'),
            ('test.mp4', 'mp4'),
            ('test.m4a', 'mp4'),
            ('test.unknown', 'other'),  # Should fallback to header detection
        ]
        
        for filename, expected in test_cases:
            test_path = pathlib.Path(filename)
            detected = OpusContainerDetector.detect_format(test_path)
            
            if expected == 'other':
                # For unknown extensions, it will try header detection and likely return 'other'
                assert detected in ['other'], f"Expected fallback for {filename}, got {detected}"
            else:
                assert detected == expected, f"Expected {expected} for {filename}, got {detected}"
    
    def test_can_extract_directly(self):
        """Test direct extraction capability detection"""
        
        # Mock source_params for testing
        opus_params = {'is_opus': True, 'codec_name': 'opus'}
        non_opus_params = {'is_opus': False, 'codec_name': 'mp3'}
        
        test_cases = [
            ('test.opus', opus_params, True),   # Raw Opus - can extract
            ('test.ogg', opus_params, True),    # OGG Opus - can extract  
            ('test.webm', opus_params, True),   # WebM Opus - can extract
            ('test.mkv', opus_params, True),    # MKV Opus - can extract
            ('test.mp4', opus_params, False),   # MP4 Opus - not yet supported
            ('test.opus', non_opus_params, False),  # Not Opus - cannot extract
        ]
        
        for filename, params, expected in test_cases:
            test_path = pathlib.Path(filename)
            can_extract = OpusContainerDetector.can_extract_directly(test_path, params)
            assert can_extract == expected, f"Expected {expected} for {filename} with Opus={params['is_opus']}"
    
    def test_raw_opus_file_extraction(self):
        """Test extraction from raw .opus file"""
        
        # Find source audio and create raw Opus test file
        source_audio = self.find_test_audio_file()
        raw_opus_file = self.create_test_opus_file(source_audio, 'raw_opus')
        
        # Test extraction
        try:
            opus_data = OpusContainerDetector.extract_raw_opus_stream(raw_opus_file)
            
            assert len(opus_data) > 0, "Extracted data should not be empty"
            assert len(opus_data) > 100, "Extracted data should be substantial"
            
            # Verify it's valid Opus data by checking for OpusHead
            assert b'OpusHead' in opus_data, "Extracted data should contain OpusHead"
            
            logger.info(f"Successfully extracted {len(opus_data)} bytes from raw Opus file")
            
        except Exception as e:
            pytest.fail(f"Raw Opus extraction failed: {e}")
    
    def test_ogg_opus_file_extraction(self):
        """Test extraction from OGG Opus file"""
        
        # Find source audio and create OGG Opus test file
        source_audio = self.find_test_audio_file()
        ogg_opus_file = self.create_test_opus_file(source_audio, 'ogg')
        
        # Test extraction  
        try:
            opus_data = OpusContainerDetector.extract_raw_opus_stream(ogg_opus_file)
            
            assert len(opus_data) > 0, "Extracted data should not be empty"
            assert len(opus_data) > 100, "Extracted data should be substantial"
            
            # Verify it's valid raw Opus data
            assert b'OpusHead' in opus_data, "Extracted data should contain OpusHead"
            
            logger.info(f"Successfully extracted {len(opus_data)} bytes from OGG Opus file")
            
        except Exception as e:
            pytest.fail(f"OGG Opus extraction failed: {e}")
    
    def test_raw_opus_parser_basic(self):
        """Test basic RawOpusParser functionality"""
        
        # Create test Opus file and extract raw data
        source_audio = self.find_test_audio_file()
        raw_opus_file = self.create_test_opus_file(source_audio, 'raw_opus')
        
        try:
            # Extract raw Opus data
            opus_data = OpusContainerDetector.extract_raw_opus_stream(raw_opus_file)
            
            # Parse with RawOpusParser
            parser = RawOpusParser(opus_data)
            packets, opus_header, total_samples = parser.extract_packets()
            
            # Validate results
            assert opus_header is not None, "OpusHead should be found"
            assert len(opus_header) > 0, "OpusHead should not be empty"
            assert opus_header.startswith(b'OpusHead'), "OpusHead should start with magic"
            
            assert len(packets) > 0, "Should extract audio packets"
            assert total_samples > 0, "Should estimate total samples"
            
            # Basic sanity checks
            assert len(packets) < 10000, "Packet count should be reasonable"
            assert total_samples < 10000000, "Sample count should be reasonable"
            
            logger.info(f"RawOpusParser extracted {len(packets)} packets, {total_samples} samples")
            
        except Exception as e:
            pytest.fail(f"RawOpusParser test failed: {e}")
    
    def test_raw_opus_parser_header_validation(self):
        """Test RawOpusParser header validation"""
        
        # Test with invalid data
        invalid_data = b"This is not Opus data"
        parser = RawOpusParser(invalid_data)
        
        try:
            packets, opus_header, total_samples = parser.extract_packets()
            pytest.fail("Should have failed with invalid data")
        except ValueError as e:
            assert "OpusHead" in str(e), "Error should mention missing OpusHead"
    
    def test_integration_workflow(self):
        """Test complete workflow: detect → extract → parse"""
        
        source_audio = self.find_test_audio_file()
        
        # Test with different formats
        for format_type in ['raw_opus', 'ogg']:
            opus_file = self.create_test_opus_file(source_audio, format_type)
            
            try:
                # Step 1: Detect format
                detected_format = OpusContainerDetector.detect_format(opus_file)
                assert detected_format in ['raw_opus', 'ogg'], f"Should detect {format_type} format"
                
                # Step 2: Check if direct extraction possible
                opus_params = {'is_opus': True, 'codec_name': 'opus'}
                can_extract = OpusContainerDetector.can_extract_directly(opus_file, opus_params)
                assert can_extract, f"Should be able to extract from {format_type}"
                
                # Step 3: Extract raw Opus stream
                opus_data = OpusContainerDetector.extract_raw_opus_stream(opus_file)
                assert len(opus_data) > 0, "Should extract non-empty data"
                
                # Step 4: Parse packets
                parser = RawOpusParser(opus_data)
                packets, header, samples = parser.extract_packets()
                
                assert header is not None, "Should extract OpusHead"
                assert len(packets) > 0, "Should extract packets"
                assert samples > 0, "Should estimate samples"
                
                logger.info(f"Integration test passed for {format_type}: "
                          f"{len(packets)} packets, {samples} samples")
                
            except Exception as e:
                pytest.fail(f"Integration test failed for {format_type}: {e}")


def run_step_1_1_tests():
    """Run Step 1.1 tests standalone"""
    print("=" * 60)
    print("STEP 1.1 TEST SUITE: Raw Opus Detection & Parsing")
    print("=" * 60)
    
    # Run tests with pytest
    test_file = __file__
    result = pytest.main(['-v', test_file])
    
    if result == 0:
        print("\n✅ ALL STEP 1.1 TESTS PASSED!")
        print("Raw Opus Detection & Parsing: IMPLEMENTATION SUCCESSFUL")
    else:
        print("\n❌ SOME STEP 1.1 TESTS FAILED")
        print("Review implementation before proceeding to Step 1.2")
    
    return result == 0


if __name__ == "__main__":
    success = run_step_1_1_tests()
    sys.exit(0 if success else 1)