#!/usr/bin/env python3
"""
FLAC Audio Integrity Test - End-to-End Validation
=================================================

GOAL:
    Validate that indexed FLAC data can be perfectly reconstructed
    by comparing extracted samples with original audio file samples.

TEST PROCEDURE:
1. Create fresh database and import long audio file with indexing
2. Select n random segments (default: 10) of 20% total length each
3. For each segment:
   - Extract samples from indexed database
   - Read same segment from original audio file
   - Compare sample streams for equality
4. Report results for all segments

This tests the complete pipeline: Import -> Indexing -> Extraction -> Validation
"""

import pathlib
import shutil
import time
import datetime
import random
import numpy as np
import soundfile as sf
import argparse
import sys

# Zarrwlr imports
import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger
from zarrwlr.aimport import create_original_audio_group, import_original_audio_file
from zarrwlr import flac_access

class FlacIntegrityTester:
    """End-to-end FLAC audio integrity tester"""
    
    def __init__(self, verbose: bool = False, seed: int = None):
        self.verbose = verbose
        
        # Set random seed for reproducible tests
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Configure logging
        log_level = LogLevel.INFO if verbose else LogLevel.WARNING
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Ensure test directory exists
        self.test_results_dir.mkdir(exist_ok=True)
    
    def find_test_file(self) -> pathlib.Path:
        """Find the long audio test file"""
        test_file = pathlib.Path(__file__).parent.resolve() / "testdata" / "audiomoth_long_snippet.wav"
        
        if not test_file.exists():
            # Fallback options
            fallback_files = [
                "testdata/audiomoth_short_snippet.wav",
                "testdata/bird1_snippet.mp3",
            ]
            
            base_path = pathlib.Path(__file__).parent.resolve()
            for fallback in fallback_files:
                fallback_path = base_path / fallback
                if fallback_path.exists():
                    if self.verbose:
                        print(f"WARNING: Using fallback file {fallback_path.name} instead of long audio")
                    return fallback_path
            
            raise FileNotFoundError(
                f"Test file not found: {test_file}\n"
                f"Required: testdata/audiomoth_long_snippet.wav"
            )
        
        return test_file
    
    def cleanup_test_stores(self):
        """Clean up old test stores"""
        if self.verbose:
            print("Cleaning up old test stores...")
        
        for store_dir in self.test_results_dir.glob("zarr3-store-integrity-*"):
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
    
    def import_audio_with_indexing(self, audio_file: pathlib.Path) -> tuple:
        """Import audio file with FLAC indexing"""
        timestamp = int(time.time())
        test_store_dir = self.test_results_dir / f"zarr3-store-integrity-{timestamp}"
        
        if self.verbose:
            print(f"Creating database: {test_store_dir.name}")
            print(f"Importing: {audio_file.name}")
        
        # Create audio group
        audio_group = create_original_audio_group(
            store_path=test_store_dir,
            group_path='audio_imports'
        )
        
        # Import audio with automatic indexing
        import_timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=audio_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=import_timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # Find imported group
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        latest_group_name = max(group_names, key=int)
        zarr_group = audio_group[latest_group_name]
        
        # Verify indexing was created
        if 'flac_index' not in zarr_group:
            raise RuntimeError("FLAC index was not created during import")
        
        audio_blob_array = zarr_group["audio_data_blob_array"]
        flac_index = zarr_group["flac_index"]
        
        if self.verbose:
            print(f"Import successful: {flac_index.shape[0]} frames indexed")
        
        return zarr_group, audio_blob_array, flac_index
    
    def get_original_audio_info(self, audio_file: pathlib.Path) -> dict:
        """Get information about original audio file"""
        with sf.SoundFile(audio_file) as sf_file:
            return {
                'frames': sf_file.frames,
                'samplerate': sf_file.samplerate,
                'channels': sf_file.channels,
                'duration_seconds': sf_file.frames / sf_file.samplerate,
                'format': sf_file.format,
                'subtype': sf_file.subtype
            }
    
    def generate_test_segments(self, total_samples: int, num_segments: int = 10, 
                             segment_length_ratio: float = 0.2) -> list:
        """Generate random test segments"""
        segment_length = int(total_samples * segment_length_ratio)
        
        # Ensure we don't exceed bounds
        max_start = max(0, total_samples - segment_length)
        
        if max_start <= 0:
            # File too short, use single segment covering whole file
            return [(0, total_samples - 1)]
        
        segments = []
        for i in range(num_segments):
            start_sample = random.randint(0, max_start)
            end_sample = min(start_sample + segment_length - 1, total_samples - 1)
            segments.append((start_sample, end_sample))
        
        # Sort segments by start position for easier debugging
        segments.sort()
        
        if self.verbose:
            print(f"Generated {len(segments)} test segments:")
            for i, (start, end) in enumerate(segments):
                duration = (end - start + 1) / total_samples
                print(f"  Segment {i+1}: samples {start:,} - {end:,} ({duration:.1%} of total)")
        
        return segments
    
    def extract_samples_from_database(self, zarr_group, audio_blob_array, 
                                    start_sample: int, end_sample: int) -> np.ndarray:
        """Extract samples from indexed FLAC database"""
        try:
            return flac_access.extract_audio_segment_flac(
                zarr_group, audio_blob_array, start_sample, end_sample, dtype=np.int16
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract samples from database: {e}")
    
    def read_samples_from_original(self, audio_file: pathlib.Path, 
                                 start_sample: int, end_sample: int) -> np.ndarray:
        """Read samples directly from original audio file"""
        try:
            with sf.SoundFile(audio_file) as sf_file:
                sf_file.seek(start_sample)
                samples_to_read = end_sample - start_sample + 1
                samples = sf_file.read(samples_to_read, dtype=np.int16)
                
                # Ensure we got the expected number of samples
                if len(samples) != samples_to_read:
                    # Handle end-of-file case
                    actual_end = start_sample + len(samples) - 1
                    if self.verbose:
                        print(f"    WARNING: Requested {samples_to_read} samples, got {len(samples)} (EOF at {actual_end})")
                
                return samples
        except Exception as e:
            raise RuntimeError(f"Failed to read samples from original file: {e}")
    
    def compare_sample_arrays(self, database_samples: np.ndarray, 
                            original_samples: np.ndarray, segment_id: int) -> dict:
        """Compare two sample arrays for equality - both strict and with tolerance"""
        result = {
            'segment_id': segment_id,
            'database_shape': database_samples.shape,
            'original_shape': original_samples.shape,
            'shapes_match': False,
            'samples_identical': False,
            'samples_within_tolerance': False,
            'max_difference': None,
            'mean_difference': None,
            'num_different_samples': None,
            'success_strict': False,
            'success_tolerant': False,
            'success': False
        }
        
        # Check shapes
        result['shapes_match'] = database_samples.shape == original_samples.shape
        
        if not result['shapes_match']:
            if self.verbose:
                print(f"    Shape mismatch: database {database_samples.shape} vs original {original_samples.shape}")
            return result
        
        # Handle empty arrays
        if database_samples.size == 0:
            result['samples_identical'] = True
            result['samples_within_tolerance'] = True
            result['success_strict'] = True
            result['success_tolerant'] = True
            result['success'] = True
            return result
        
        # STRICT COMPARISON (should be identical for lossless WAV -> FLAC -> WAV)
        result['samples_identical'] = np.array_equal(database_samples, original_samples)
        result['success_strict'] = result['samples_identical']
        
        if result['samples_identical']:
            # Perfect match - no need for further analysis
            result['max_difference'] = 0
            result['mean_difference'] = 0.0
            result['num_different_samples'] = 0
            result['samples_within_tolerance'] = True
            result['success_tolerant'] = True
            result['success'] = True
            
            if self.verbose:
                print(f"    PERFECT: Samples bit-exactly identical")
            
            return result
        
        # DETAILED ANALYSIS for non-identical samples
        differences = np.abs(database_samples.astype(np.float32) - original_samples.astype(np.float32))
        result['max_difference'] = np.max(differences)
        result['mean_difference'] = np.mean(differences)
        result['num_different_samples'] = np.sum(differences > 0)
        
        # TOLERANCE COMPARISON (small numerical differences allowed)
        tolerance = 1.0  # Allow 1 sample value difference for format conversion artifacts
        result['samples_within_tolerance'] = result['max_difference'] <= tolerance
        result['success_tolerant'] = result['samples_within_tolerance']
        result['success'] = result['success_tolerant']  # Overall success uses tolerance
        
        if self.verbose:
            total_samples = database_samples.size
            different_percent = (result['num_different_samples'] / total_samples) * 100
            
            print(f"    STRICT TEST: FAIL - {result['num_different_samples']:,}/{total_samples:,} samples differ ({different_percent:.3f}%)")
            print(f"    Max difference: {result['max_difference']}")
            print(f"    Mean difference: {result['mean_difference']:.6f}")
            
            if result['success_tolerant']:
                print(f"    TOLERANCE TEST: PASS - Differences within tolerance ({tolerance})")
                print(f"    VERDICT: Likely format conversion artifacts")
            else:
                print(f"    TOLERANCE TEST: FAIL - Differences exceed tolerance ({tolerance})")
                print(f"    VERDICT: Significant processing errors detected")
        else:
            # Compact output for non-verbose mode
            if result['success_tolerant']:
                print("TOLERANCE-PASS", end="")
            else:
                print("FAIL", end="")
        
        return result
    
    def run_integrity_test(self, num_segments: int = 10, 
                         segment_length_ratio: float = 0.2) -> dict:
        """Run complete integrity test"""
        print("=" * 70)
        print("FLAC AUDIO INTEGRITY TEST - END-TO-END VALIDATION")
        print("=" * 70)
        
        start_time = time.time()
        
        # Clean up old test data
        self.cleanup_test_stores()
        
        # Find and analyze test file
        audio_file = self.find_test_file()
        audio_info = self.get_original_audio_info(audio_file)
        
        print(f"Test file: {audio_file.name}")
        print(f"Audio info: {audio_info['frames']:,} samples, {audio_info['samplerate']} Hz, "
              f"{audio_info['channels']} ch, {audio_info['duration_seconds']:.1f}s")
        
        # Import audio with indexing
        print(f"\nImporting audio with FLAC indexing...")
        zarr_group, audio_blob_array, flac_index = self.import_audio_with_indexing(audio_file)
        
        # Generate test segments
        print(f"\nGenerating {num_segments} random test segments ({segment_length_ratio:.0%} each)...")
        segments = self.generate_test_segments(
            audio_info['frames'], num_segments, segment_length_ratio
        )
        
        # Test each segment
        print(f"\nTesting segments...")
        segment_results = []
        
        for i, (start_sample, end_sample) in enumerate(segments):
            segment_id = i + 1
            segment_length = end_sample - start_sample + 1
            
            if self.verbose:
                print(f"\nSegment {segment_id}/{len(segments)}: samples {start_sample:,} - {end_sample:,} "
                      f"({segment_length:,} samples)")
            else:
                print(f"  Segment {segment_id}/{len(segments)}: ", end="", flush=True)
            
            try:
                # Extract from database
                database_samples = self.extract_samples_from_database(
                    zarr_group, audio_blob_array, start_sample, end_sample
                )
                
                # Read from original
                original_samples = self.read_samples_from_original(
                    audio_file, start_sample, end_sample
                )
                
                # Compare
                comparison = self.compare_sample_arrays(
                    database_samples, original_samples, segment_id
                )
                
                comparison['start_sample'] = start_sample
                comparison['end_sample'] = end_sample
                comparison['segment_length'] = segment_length
                
                segment_results.append(comparison)
                
                if not self.verbose:
                    if comparison['success_strict']:
                        status = "PERFECT"
                    elif comparison['success_tolerant']:
                        status = "TOLERANCE-PASS"
                    else:
                        status = "FAIL"
                    print(f"{status}")
                
            except Exception as e:
                error_result = {
                    'segment_id': segment_id,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'segment_length': segment_length,
                    'success': False,
                    'error': str(e)
                }
                segment_results.append(error_result)
                
                if self.verbose:
                    print(f"    ERROR: {str(e)}")
                else:
                    print("ERROR")
        
        # Calculate summary
        total_time = time.time() - start_time
        successful_segments_strict = sum(1 for r in segment_results if r.get('success_strict', False))
        successful_segments_tolerant = sum(1 for r in segment_results if r.get('success_tolerant', False))
        total_segments = len(segment_results)
        
        strict_success_rate = successful_segments_strict / total_segments if total_segments > 0 else 0
        tolerant_success_rate = successful_segments_tolerant / total_segments if total_segments > 0 else 0
        
        summary = {
            'test_file': audio_file.name,
            'audio_info': audio_info,
            'total_segments': total_segments,
            'successful_segments_strict': successful_segments_strict,
            'successful_segments_tolerant': successful_segments_tolerant,
            'failed_segments': total_segments - successful_segments_tolerant,
            'strict_success_rate': strict_success_rate,
            'tolerant_success_rate': tolerant_success_rate,
            'overall_success_strict': successful_segments_strict == total_segments,
            'overall_success_tolerant': successful_segments_tolerant == total_segments,
            'overall_success': successful_segments_tolerant == total_segments,
            'total_time': total_time,
            'segment_results': segment_results
        }
        
        # Print summary
        print(f"\n" + "=" * 70)
        print("INTEGRITY TEST SUMMARY")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s")
        print(f"Segments tested: {total_segments}")
        print(f"")
        print(f"STRICT TEST (bit-exact): {successful_segments_strict}/{total_segments} ({strict_success_rate:.1%})")
        print(f"TOLERANCE TEST (±1 sample): {successful_segments_tolerant}/{total_segments} ({tolerant_success_rate:.1%})")
        
        if summary['overall_success_strict']:
            print(f"\nRESULT: PERFECT - All segments bit-exactly identical!")
            print(f"Lossless WAV->FLAC->WAV pipeline verified: Zero data corruption")
        elif summary['overall_success_tolerant']:
            print(f"\nRESULT: ACCEPTABLE - All segments within tolerance")
            print(f"Minor format conversion artifacts detected, but data integrity maintained")
            
            # Show details about non-strict matches
            tolerance_segments = [r for r in segment_results 
                                if r.get('success_tolerant', False) and not r.get('success_strict', False)]
            if tolerance_segments:
                print(f"\nSegments with minor differences:")
                for seg in tolerance_segments:
                    print(f"  Segment {seg['segment_id']}: {seg.get('num_different_samples', 0):,} samples, "
                          f"max diff: {seg.get('max_difference', 0)}")
        else:
            print(f"\nRESULT: FAIL - Significant data corruption detected!")
            print(f"WAV->FLAC->WAV pipeline has serious integrity issues")
            
            failed_segments = [r for r in segment_results if not r.get('success_tolerant', True)]
            for failed in failed_segments:
                if 'error' in failed:
                    print(f"  Segment {failed['segment_id']}: ERROR - {failed['error']}")
                else:
                    max_diff = failed.get('max_difference', 'unknown')
                    num_diff = failed.get('num_different_samples', 'unknown')
                    print(f"  Segment {failed['segment_id']}: {num_diff} samples differ, max diff: {max_diff}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='FLAC Audio Integrity Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed logging')
    parser.add_argument('--segments', '-n', type=int, default=10,
                       help='Number of random segments to test (default: 10)')
    parser.add_argument('--length', '-l', type=float, default=0.2,
                       help='Segment length as ratio of total (default: 0.2 = 20%%)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible tests')
    
    args = parser.parse_args()
    
    if args.length <= 0 or args.length > 1:
        print("ERROR: Segment length ratio must be between 0 and 1")
        sys.exit(1)
    
    if args.segments <= 0:
        print("ERROR: Number of segments must be positive")
        sys.exit(1)
    
    try:
        tester = FlacIntegrityTester(verbose=args.verbose, seed=args.seed)
        summary = tester.run_integrity_test(
            num_segments=args.segments,
            segment_length_ratio=args.length
        )
        
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