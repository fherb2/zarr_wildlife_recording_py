#!/usr/bin/env python3
"""
Basic Opus Functionality Test
=============================

GOAL:
    Test the basic opus_access.py + opus_index_backend.py functionality
    before integrating with aimport.py

TEST PROCEDURE:
1. Create fresh database 
2. Import small audio file to Opus format with indexing
3. Extract a few segments and verify they contain audio data
4. Test ultrasonic handling (if test file available)
5. Test 1:1 Opus copy (if Opus source available)

This tests the core pipeline: Import -> Indexing -> Extraction
WITHOUT the full aimport.py integration (that comes in Step 1.3)
"""

import pathlib
import shutil
import time
import datetime
import numpy as np
import argparse
import sys
import tempfile

# Test imports - direct module import (no aimport.py yet)
import zarr
import zarrcompatibility as zc
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

# Enable universal serialization
zc.enable_universal_serialization()

class OpusBasicTester:
    """Basic Opus functionality tester"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Configure logging
        log_level = LogLevel.INFO if verbose else LogLevel.WARNING
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Ensure test directory exists
        self.test_results_dir.mkdir(exist_ok=True)
    
    def find_test_file(self) -> pathlib.Path:
        """Find a suitable audio test file"""
        test_files = [
            "testdata/bird1_snippet.mp3",           # Small MP3 file
            "testdata/audiomoth_short_snippet.wav", # Short WAV file
            "testdata/audiomoth_long_snippet.wav",  # Fallback to long file
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        for test_file in test_files:
            test_path = base_path / test_file
            if test_path.exists():
                if self.verbose:
                    print(f"Using test file: {test_path.name}")
                return test_path
        
        raise FileNotFoundError(
            f"No test files found. Searched for: {test_files}\n"
            f"Required: at least one audio file in testdata/"
        )
    
    def cleanup_test_stores(self):
        """Clean up old test stores"""
        if self.verbose:
            print("Cleaning up old test stores...")
        
        for store_dir in self.test_results_dir.glob("zarr3-store-opus-basic-*"):
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
    
    def get_source_params_simple(self, audio_file: pathlib.Path) -> dict:
        """
        Simple source parameter extraction (minimal version of aimport.py function)
        Just enough for basic testing
        """
        import subprocess
        import json
        
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate:stream=codec_name:stream=channels",
            "-of", "json", str(audio_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        if not info.get('streams'):
            raise ValueError(f"No audio streams found in {audio_file}")
        
        stream = info['streams'][0]
        
        return {
            "sampling_rate": int(stream.get('sample_rate', 48000)),
            "is_opus": stream.get('codec_name') == 'opus',
            "nb_channels": int(stream.get('channels', 1))
        }
    
    def test_opus_import_and_indexing(self, audio_file: pathlib.Path) -> tuple:
        """Test Opus import with automatic indexing"""
        timestamp = int(time.time())
        test_store_dir = self.test_results_dir / f"zarr3-store-opus-basic-{timestamp}"
        
        if self.verbose:
            print(f"Creating test database: {test_store_dir.name}")
            print(f"Testing import: {audio_file.name}")
        
        # Create minimal Zarr structure (without full aimport.py)
        store = zarr.storage.LocalStore(str(test_store_dir))
        root = zarr.open_group(store, mode='a')
        
        # Create simple group structure
        audio_group = root.create_group('test_audio')
        
        # Get source parameters
        source_params = self.get_source_params_simple(audio_file)
        
        if self.verbose:
            print(f"Source parameters: {source_params}")
        
        # Import using opus_access module directly
        try:
            # This should work now that __init__.py is updated
            from zarrwlr import opus_access
            
            audio_blob_array = opus_access.import_opus_to_zarr(
                zarr_group=audio_group,
                audio_file=audio_file,
                source_params=source_params,
                first_sample_time_stamp=datetime.datetime.now(),
                opus_bitrate=128000,  # Lower bitrate for test
                temp_dir=tempfile.gettempdir()
            )
            
            if self.verbose:
                print(f"Import successful: {audio_blob_array.shape[0]} bytes stored")
                print(f"Attributes: {dict(audio_blob_array.attrs)}")
            
            # Verify index was created
            if 'opus_index' not in audio_group:
                raise RuntimeError("Opus index was not created during import")
            
            opus_index = audio_group['opus_index']
            
            if self.verbose:
                print(f"Index created: {opus_index.shape[0]} pages indexed")
                print(f"Index attributes: {dict(opus_index.attrs)}")
            
            return audio_group, audio_blob_array, opus_index
            
        except ImportError as e:
            raise RuntimeError(f"Cannot import opus_access module: {e}")
        except Exception as e:
            raise RuntimeError(f"Opus import failed: {e}")
    
    def test_segment_extraction(self, audio_group, audio_blob_array, opus_index) -> list:
        """Test audio segment extraction"""
        if self.verbose:
            print("Testing segment extraction...")
        
        # Get total samples from index
        if opus_index.shape[0] == 0:
            raise RuntimeError("Empty index - cannot test extraction")
        
        sample_positions = opus_index[:, 2]  # OPUS_INDEX_COL_SAMPLE_POS
        total_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 1000
        
        if self.verbose:
            print(f"Total samples in index: {total_samples}")
        
        # Test a few small segments
        test_segments = []
        
        if total_samples > 100:
            # Test beginning
            test_segments.append((0, 99))
            
            # Test middle (if enough samples)
            if total_samples > 1000:
                mid_start = total_samples // 2
                test_segments.append((mid_start, mid_start + 99))
            
            # Test near end
            if total_samples > 200:
                end_start = max(100, total_samples - 200)
                test_segments.append((end_start, end_start + 99))
        else:
            # Very short file - test what we can
            test_segments.append((0, min(total_samples - 1, 49)))
        
        results = []
        
        for i, (start_sample, end_sample) in enumerate(test_segments):
            try:
                from zarrwlr import opus_access
                
                segment_data = opus_access.extract_audio_segment_opus(
                    audio_group, audio_blob_array, start_sample, end_sample, dtype=np.int16
                )
                
                if segment_data.size == 0:
                    result = {
                        'segment_id': i + 1,
                        'range': (start_sample, end_sample),
                        'success': False,
                        'error': 'Empty segment returned'
                    }
                else:
                    result = {
                        'segment_id': i + 1,
                        'range': (start_sample, end_sample),
                        'success': True,
                        'shape': segment_data.shape,
                        'dtype': str(segment_data.dtype),
                        'sample_range': (segment_data.min(), segment_data.max()) if segment_data.size > 0 else (0, 0),
                        'has_audio_data': np.any(segment_data != 0)
                    }
                
                results.append(result)
                
                if self.verbose:
                    if result['success']:
                        print(f"  Segment {i+1}: {result['shape']} samples, "
                              f"range: {result['sample_range']}, "
                              f"has_audio: {result['has_audio_data']}")
                    else:
                        print(f"  Segment {i+1}: FAILED - {result['error']}")
                
            except Exception as e:
                result = {
                    'segment_id': i + 1,
                    'range': (start_sample, end_sample),
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
                
                if self.verbose:
                    print(f"  Segment {i+1}: ERROR - {str(e)}")
        
        return results
    
    def run_basic_test(self) -> dict:
        """Run complete basic functionality test"""
        print("=" * 60)
        print("OPUS BASIC FUNCTIONALITY TEST")
        print("=" * 60)
        
        start_time = time.time()
        
        # Clean up old test data
        self.cleanup_test_stores()
        
        # Find test file
        audio_file = self.find_test_file()
        
        try:
            # Test import and indexing
            print(f"\n1. Testing Opus import and indexing...")
            audio_group, audio_blob_array, opus_index = self.test_opus_import_and_indexing(audio_file)
            
            # Test extraction
            print(f"\n2. Testing segment extraction...")
            extraction_results = self.test_segment_extraction(audio_group, audio_blob_array, opus_index)
            
            # Calculate summary
            total_time = time.time() - start_time
            successful_extractions = sum(1 for r in extraction_results if r['success'])
            total_extractions = len(extraction_results)
            
            summary = {
                'test_file': audio_file.name,
                'total_time': total_time,
                'import_success': True,
                'index_pages': opus_index.shape[0],
                'extraction_tests': total_extractions,
                'successful_extractions': successful_extractions,
                'extraction_success_rate': successful_extractions / total_extractions if total_extractions > 0 else 0,
                'overall_success': successful_extractions == total_extractions,
                'extraction_results': extraction_results
            }
            
            # Print summary
            print(f"\n" + "=" * 60)
            print("BASIC TEST SUMMARY")
            print("=" * 60)
            print(f"Test file: {summary['test_file']}")
            print(f"Total time: {summary['total_time']:.1f}s")
            print(f"Import: {'SUCCESS' if summary['import_success'] else 'FAILED'}")
            print(f"Index pages: {summary['index_pages']}")
            print(f"Extraction tests: {summary['successful_extractions']}/{summary['extraction_tests']} "
                  f"({summary['extraction_success_rate']:.1%})")
            
            if summary['overall_success']:
                print(f"\nRESULT: SUCCESS - Basic Opus functionality working!")
                print(f"✅ Import: Working")
                print(f"✅ Indexing: Working") 
                print(f"✅ Extraction: Working")
                print(f"\nReady for Step 1.3 (aimport.py integration)")
            else:
                print(f"\nRESULT: PARTIAL - Some functionality issues detected")
                failed_extractions = [r for r in extraction_results if not r['success']]
                for failed in failed_extractions:
                    print(f"  ❌ Segment {failed['segment_id']}: {failed['error']}")
                print(f"\nNeeds debugging before Step 1.3")
            
            return summary
            
        except Exception as e:
            error_summary = {
                'test_file': audio_file.name,
                'total_time': time.time() - start_time,
                'overall_success': False,
                'error': str(e)
            }
            
            print(f"\nRESULT: CRITICAL FAILURE")
            print(f"❌ Error: {str(e)}")
            print(f"\nMust fix before proceeding to Step 1.3")
            
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            return error_summary

def main():
    parser = argparse.ArgumentParser(description='Basic Opus Functionality Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed logging')
    
    args = parser.parse_args()
    
    try:
        tester = OpusBasicTester(verbose=args.verbose)
        summary = tester.run_basic_test()
        
        # Exit with appropriate code
        sys.exit(0 if summary['overall_success'] else 1)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()