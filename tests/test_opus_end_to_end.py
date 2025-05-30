#!/usr/bin/env python3
"""
Opus End-to-End Integration Test
===============================

GOAL:
    Test the complete integrated pipeline after Step 1.3:
    aimport.py -> opus_access.py -> opus_index_backend.py

TEST PROCEDURE:
1. Use the original aimport.py API (import_original_audio_file)
2. Import audio with target_codec='opus' 
3. Extract segments using generic extract_audio_segment()
4. Validate complete pipeline with detailed debugging info

This tests the COMPLETE integration - not just individual modules.
"""

import pathlib
import shutil
import time
import datetime
import numpy as np
import argparse
import sys
import traceback

# Full zarrwlr import (testing complete integration)
import zarrwlr
import zarr  # Direct zarr import for type hints
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

class OpusEndToEndTester:
    """Complete Opus pipeline integration tester with extensive debugging"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Configure logging for maximum debugging info
        log_level = LogLevel.TRACE if verbose else LogLevel.INFO
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Ensure test directory exists
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Track pipeline stages for debugging
        self.pipeline_stages = {
            'audio_group_creation': {'status': 'pending', 'duration': 0, 'details': {}},
            'file_analysis': {'status': 'pending', 'duration': 0, 'details': {}},
            'import_process': {'status': 'pending', 'duration': 0, 'details': {}},
            'index_creation': {'status': 'pending', 'duration': 0, 'details': {}},
            'extraction_tests': {'status': 'pending', 'duration': 0, 'details': {}},
            'validation': {'status': 'pending', 'duration': 0, 'details': {}}
        }
    
    def debug_print(self, stage: str, message: str, level: str = "INFO"):
        """Enhanced debug printing with stage tracking"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{timestamp}] [{level}] [{stage}]"
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix} {message}")
        
        # Log to logger as well
        if level == "ERROR":
            self.logger.error(f"{stage}: {message}")
        elif level == "WARNING":
            self.logger.warning(f"{stage}: {message}")
        elif level == "DEBUG":
            self.logger.debug(f"{stage}: {message}")
        else:
            self.logger.info(f"{stage}: {message}")
    
    def update_stage_status(self, stage: str, status: str, duration: float = 0, **details):
        """Update pipeline stage status with debugging info"""
        self.pipeline_stages[stage]['status'] = status
        self.pipeline_stages[stage]['duration'] = duration
        self.pipeline_stages[stage]['details'].update(details)
        
        self.debug_print(stage, f"Status: {status} (took {duration:.3f}s)", 
                        "DEBUG" if status == "success" else "WARNING")
    
    def find_test_file(self) -> pathlib.Path:
        """Find a suitable audio test file with detailed analysis"""
        self.debug_print("SETUP", "Searching for test audio files...")
        
        test_files = [
            "testdata/audiomoth_long_snippet_converted.opus",  # PREFERRED: Native Opus for 1:1 copy test
            "testdata/bird1_snippet.mp3",                     # Good: small, stereo, needs encoding
            "testdata/audiomoth_short_snippet.wav",           # Fallback: short WAV
            "testdata/audiomoth_long_snippet.wav",            # Fallback: long WAV
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        for test_file in test_files:
            test_path = base_path / test_file
            if test_path.exists():
                file_size_mb = test_path.stat().st_size / 1024 / 1024
                self.debug_print("SETUP", f"Found test file: {test_path.name} ({file_size_mb:.2f} MB)")
                
                # Analyze file with ffprobe for debugging
                try:
                    import subprocess
                    import json
                    cmd = ["ffprobe", "-v", "error", "-print_format", "json", 
                           "-show_format", "-show_streams", str(test_path)]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    info = json.loads(result.stdout)
                    
                    if info.get('streams'):
                        stream = info['streams'][0]
                        codec_name = stream.get('codec_name', 'unknown')
                        sample_rate = stream.get('sample_rate', 'unknown')
                        channels = stream.get('channels', 'unknown')
                        duration = info.get('format', {}).get('duration', 'unknown')
                        
                        self.debug_print("SETUP", f"Audio properties: {codec_name} "
                                        f"{sample_rate}Hz "
                                        f"{channels}ch "
                                        f"duration: {duration}s")
                        
                        # Special handling for Opus sources
                        if codec_name == 'opus':
                            self.debug_print("SETUP", "🎯 NATIVE OPUS SOURCE - Will test 1:1 copy mode!", "INFO")
                        elif test_path.suffix.lower() == '.opus':
                            self.debug_print("SETUP", "⚠️  Opus file but codec detection failed", "WARNING")
                            
                except Exception as e:
                    self.debug_print("SETUP", f"Could not analyze file: {e}", "WARNING")
                
                return test_path
        
        raise FileNotFoundError(
            f"No test files found. Searched for: {test_files}\n"
            f"Required: at least one audio file in testdata/"
        )
    
    def cleanup_test_stores(self):
        """Clean up old test stores with logging"""
        self.debug_print("CLEANUP", "Cleaning up old test stores...")
        
        cleanup_count = 0
        for store_dir in self.test_results_dir.glob("zarr3-store-opus-e2e-*"):
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
                cleanup_count += 1
        
        self.debug_print("CLEANUP", f"Removed {cleanup_count} old test stores")
    
    def test_audio_group_creation(self) -> zarr.Group:
        """Test audio group creation with detailed monitoring"""
        stage = "audio_group_creation"
        start_time = time.time()
        
        try:
            timestamp = int(time.time())
            test_store_dir = self.test_results_dir / f"zarr3-store-opus-e2e-{timestamp}"
            
            self.debug_print(stage, f"Creating Zarr store: {test_store_dir.name}")
            
            # Create audio group using original aimport.py API
            audio_group = zarrwlr.create_original_audio_group(
                store_path=test_store_dir,
                group_path='audio_imports'
            )
            
            duration = time.time() - start_time
            
            # Validate group attributes
            required_attrs = ['magic_id', 'version']
            missing_attrs = [attr for attr in required_attrs if attr not in audio_group.attrs]
            
            if missing_attrs:
                raise ValueError(f"Missing required attributes: {missing_attrs}")
            
            self.update_stage_status(stage, "success", duration,
                                   store_path=str(test_store_dir),
                                   group_path='audio_imports',
                                   magic_id=audio_group.attrs.get('magic_id'),
                                   version=audio_group.attrs.get('version'))
            
            return audio_group
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_stage_status(stage, "failed", duration, error=str(e))
            raise RuntimeError(f"Audio group creation failed: {e}")
    
    def test_file_analysis(self, audio_file: pathlib.Path) -> tuple:
        """Test file analysis with base_features extraction"""
        stage = "file_analysis"
        start_time = time.time()
        
        try:
            self.debug_print(stage, f"Analyzing file: {audio_file.name}")
            
            # Test base features extraction (aimport.py function)
            base_features = zarrwlr.base_features_from_audio_file(audio_file)
            
            self.debug_print(stage, f"Container format: {base_features.get(base_features.CONTAINER_FORMAT, 'unknown')}")
            self.debug_print(stage, f"Codecs: {base_features.get(base_features.CODEC_PER_STREAM, [])}")
            self.debug_print(stage, f"Sample rates: {base_features.get(base_features.SAMPLING_RATE_PER_STREAM, [])}")
            self.debug_print(stage, f"Channels: {base_features.get(base_features.CHANNELS_PER_STREAM, [])}")
            self.debug_print(stage, f"Has audio: {base_features.get(base_features.HAS_AUDIO_STREAM, False)}")
            self.debug_print(stage, f"File size: {base_features.get(base_features.SIZE_BYTES, 0)} bytes")
            
            if not base_features.get(base_features.HAS_AUDIO_STREAM, False):
                raise ValueError("File has no audio streams")
            
            duration = time.time() - start_time
            self.update_stage_status(stage, "success", duration,
                                   container_format=base_features.get(base_features.CONTAINER_FORMAT),
                                   codecs=base_features.get(base_features.CODEC_PER_STREAM),
                                   sample_rates=base_features.get(base_features.SAMPLING_RATE_PER_STREAM),
                                   channels=base_features.get(base_features.CHANNELS_PER_STREAM),
                                   file_size=base_features.get(base_features.SIZE_BYTES))
            
            return base_features
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_stage_status(stage, "failed", duration, error=str(e))
            raise RuntimeError(f"File analysis failed: {e}")
    
    def test_import_process(self, audio_file: pathlib.Path, audio_group: zarr.Group, base_features: dict) -> zarr.Group:
        """Test the complete import process using aimport.py API"""
        stage = "import_process"
        start_time = time.time()
        
        try:
            self.debug_print(stage, "Starting import using aimport.py API...")
            self.debug_print(stage, f"Target codec: opus")
            self.debug_print(stage, f"Opus bitrate: 128000 bps")
            
            # Use the original aimport.py API (this tests the complete integration!)
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',  # ← This should now use opus_access.py
                opus_bitrate=128000,  # Lower bitrate for faster testing
                temp_dir=self.test_results_dir
            )
            
            # Find the imported group
            group_names = [name for name in audio_group.keys() if name.isdigit()]
            if not group_names:
                raise RuntimeError("No audio groups found after import")
            
            latest_group_name = max(group_names, key=int)
            imported_group = audio_group[latest_group_name]
            
            self.debug_print(stage, f"Import completed. Created group: {latest_group_name}")
            
            # Validate imported group structure
            required_items = ['audio_data_blob_array', 'opus_index']
            missing_items = [item for item in required_items if item not in imported_group]
            
            if missing_items:
                raise ValueError(f"Missing required items in imported group: {missing_items}")
            
            # Analyze imported data
            audio_blob_array = imported_group['audio_data_blob_array']
            opus_index = imported_group['opus_index']
            
            self.debug_print(stage, f"Audio blob size: {audio_blob_array.shape[0]} bytes")
            self.debug_print(stage, f"Index pages: {opus_index.shape[0]}")
            self.debug_print(stage, f"Audio codec: {audio_blob_array.attrs.get('codec', 'unknown')}")
            self.debug_print(stage, f"Sample rate: {audio_blob_array.attrs.get('sample_rate', 'unknown')}")
            self.debug_print(stage, f"Channels: {audio_blob_array.attrs.get('nb_channels', 'unknown')}")
            self.debug_print(stage, f"Container: {audio_blob_array.attrs.get('container_type', 'unknown')}")
            
            # Check for 1:1 Opus copy mode
            is_opus_source = audio_blob_array.attrs.get('codec') == 'opus' and \
                           base_features.get(base_features.CODEC_PER_STREAM, [None])[0] == 'opus'
            
            if is_opus_source:
                self.debug_print(stage, "🎯 1:1 OPUS COPY MODE detected - no re-encoding used!")
            
            # Check for ultrasonic handling
            sampling_rescale_factor = audio_blob_array.attrs.get('sampling_rescale_factor', 1.0)
            is_ultrasonic = audio_blob_array.attrs.get('is_ultrasonic', False)
            if is_ultrasonic:
                self.debug_print(stage, f"🔊 Ultrasonic file detected - rescale factor: {sampling_rescale_factor}")
            
            duration = time.time() - start_time
            self.update_stage_status(stage, "success", duration,
                                   group_name=latest_group_name,
                                   audio_size_bytes=audio_blob_array.shape[0],
                                   index_pages=opus_index.shape[0],
                                   codec=audio_blob_array.attrs.get('codec'),
                                   sample_rate=audio_blob_array.attrs.get('sample_rate'),
                                   channels=audio_blob_array.attrs.get('nb_channels'),
                                   is_ultrasonic=is_ultrasonic,
                                   rescale_factor=sampling_rescale_factor,
                                   is_opus_source=is_opus_source,
                                   copy_mode_used=is_opus_source)
            
            return imported_group
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_stage_status(stage, "failed", duration, error=str(e), traceback=traceback.format_exc())
            raise RuntimeError(f"Import process failed: {e}")
    
    def test_extraction_with_generic_api(self, imported_group: zarr.Group) -> list:
        """Test extraction using the generic aimport.py API (codec-agnostic)"""
        stage = "extraction_tests"
        start_time = time.time()
        
        try:
            self.debug_print(stage, "Testing extraction using generic aimport.py API...")
            
            # Get audio info for test planning
            opus_index = imported_group['opus_index']
            audio_blob_array = imported_group['audio_data_blob_array']
            
            if opus_index.shape[0] == 0:
                raise RuntimeError("Empty index - cannot test extraction")
            
            # Calculate test segments
            sample_positions = opus_index[:, 2]  # OPUS_INDEX_COL_SAMPLE_POS
            total_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 1000
            
            self.debug_print(stage, f"Total samples available: {total_samples}")
            
            # Plan test segments
            test_segments = []
            if total_samples > 100:
                test_segments.append((0, 99, "beginning"))
                if total_samples > 1000:
                    mid_start = total_samples // 2
                    test_segments.append((mid_start, mid_start + 99, "middle"))
                if total_samples > 200:
                    end_start = max(100, total_samples - 200)
                    test_segments.append((end_start, end_start + 99, "end"))
            else:
                test_segments.append((0, min(total_samples - 1, 49), "full_short"))
            
            self.debug_print(stage, f"Planned {len(test_segments)} test extractions")
            
            # Test each segment using generic API
            extraction_results = []
            
            for i, (start_sample, end_sample, description) in enumerate(test_segments):
                segment_start = time.time()
                
                try:
                    self.debug_print(stage, f"Extracting segment {i+1}: {description} [{start_sample}:{end_sample}]")
                    
                    # Use generic extract_audio_segment() - should detect codec automatically
                    segment_data = zarrwlr.extract_audio_segment(
                        imported_group, start_sample, end_sample, dtype=np.int16
                    )
                    
                    segment_duration = time.time() - segment_start
                    
                    if segment_data.size == 0:
                        result = {
                            'segment_id': i + 1,
                            'description': description,
                            'range': (start_sample, end_sample),
                            'success': False,
                            'error': 'Empty segment returned',
                            'duration': segment_duration
                        }
                        self.debug_print(stage, f"Segment {i+1} FAILED: Empty result", "WARNING")
                    else:
                        # Analyze extracted data
                        sample_range = (segment_data.min(), segment_data.max()) if segment_data.size > 0 else (0, 0)
                        has_audio_data = np.any(segment_data != 0)
                        dynamic_range = sample_range[1] - sample_range[0] if segment_data.size > 0 else 0
                        
                        result = {
                            'segment_id': i + 1,
                            'description': description,
                            'range': (start_sample, end_sample),
                            'success': True,
                            'shape': segment_data.shape,
                            'dtype': str(segment_data.dtype),
                            'sample_range': sample_range,
                            'has_audio_data': has_audio_data,
                            'dynamic_range': dynamic_range,
                            'duration': segment_duration
                        }
                        
                        self.debug_print(stage, f"Segment {i+1} SUCCESS: {result['shape']} samples, "
                                              f"range: {result['sample_range']}, "
                                              f"dynamic: {dynamic_range}, "
                                              f"has_audio: {has_audio_data}")
                    
                    extraction_results.append(result)
                    
                except Exception as e:
                    segment_duration = time.time() - segment_start
                    result = {
                        'segment_id': i + 1,
                        'description': description,
                        'range': (start_sample, end_sample),
                        'success': False,
                        'error': str(e),
                        'duration': segment_duration
                    }
                    extraction_results.append(result)
                    
                    self.debug_print(stage, f"Segment {i+1} ERROR: {str(e)}", "ERROR")
            
            # Calculate summary
            successful_extractions = sum(1 for r in extraction_results if r['success'])
            total_extractions = len(extraction_results)
            total_extraction_time = sum(r['duration'] for r in extraction_results)
            
            duration = time.time() - start_time
            self.update_stage_status(stage, "success" if successful_extractions == total_extractions else "partial",
                                   duration,
                                   successful_extractions=successful_extractions,
                                   total_extractions=total_extractions,
                                   extraction_time=total_extraction_time,
                                   avg_extraction_time=total_extraction_time / total_extractions if total_extractions > 0 else 0)
            
            return extraction_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_stage_status(stage, "failed", duration, error=str(e))
            raise RuntimeError(f"Extraction testing failed: {e}")
    
    def validate_pipeline_integrity(self, extraction_results: list) -> dict:
        """Validate overall pipeline integrity"""
        stage = "validation"
        start_time = time.time()
        
        try:
            # Check all pipeline stages
            all_stages_passed = all(
                stage_info['status'] == 'success' 
                for stage_info in self.pipeline_stages.values()
                if stage_info['status'] != 'pending'
            )
            
            # Check extraction quality
            successful_extractions = sum(1 for r in extraction_results if r['success'])
            extraction_success_rate = successful_extractions / len(extraction_results) if extraction_results else 0
            
            # Check for audio data quality
            segments_with_audio = sum(1 for r in extraction_results 
                                    if r['success'] and r.get('has_audio_data', False))
            audio_quality_rate = segments_with_audio / len(extraction_results) if extraction_results else 0
            
            # Overall validation
            pipeline_healthy = (
                all_stages_passed and 
                extraction_success_rate >= 0.8 and  # At least 80% extraction success
                audio_quality_rate >= 0.5  # At least 50% segments have actual audio data
            )
            
            validation_result = {
                'pipeline_healthy': pipeline_healthy,
                'all_stages_passed': all_stages_passed,
                'extraction_success_rate': extraction_success_rate,
                'audio_quality_rate': audio_quality_rate,
                'total_pipeline_time': sum(stage['duration'] for stage in self.pipeline_stages.values()),
                'stage_details': self.pipeline_stages.copy()
            }
            
            duration = time.time() - start_time
            self.update_stage_status(stage, "success", duration, **validation_result)
            
            return validation_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.update_stage_status(stage, "failed", duration, error=str(e))
            raise RuntimeError(f"Pipeline validation failed: {e}")
    
    def run_end_to_end_test(self) -> dict:
        """Run complete end-to-end integration test"""
        print("=" * 80)
        print("OPUS END-TO-END INTEGRATION TEST - COMPLETE PIPELINE")
        print("=" * 80)
        print("Testing: aimport.py -> opus_access.py -> opus_index_backend.py")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Clean up old test data
            self.cleanup_test_stores()
            
            # Find and analyze test file
            audio_file = self.find_test_file()
            
            # Test pipeline stages
            self.debug_print("PIPELINE", "Starting end-to-end pipeline test...")
            
            # Stage 1: Audio group creation
            audio_group = self.test_audio_group_creation()
            
            # Stage 2: File analysis
            base_features = self.test_file_analysis(audio_file)
            
            # Stage 3: Import process (the main integration test!)
            imported_group = self.test_import_process(audio_file, audio_group, base_features)
            
            # Stage 4: Extraction tests
            extraction_results = self.test_extraction_with_generic_api(imported_group)
            
            # Stage 5: Pipeline validation
            validation_result = self.validate_pipeline_integrity(extraction_results)
            
            # Final summary
            total_time = time.time() - overall_start_time
            
            print(f"\n" + "=" * 80)
            print("END-TO-END TEST SUMMARY")
            print("=" * 80)
            print(f"Test file: {audio_file.name}")
            print(f"Total time: {total_time:.1f}s")
            print(f"")
            
            # Stage summary
            for stage_name, stage_info in self.pipeline_stages.items():
                status = stage_info['status']
                duration = stage_info['duration']
                emoji = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
                print(f"{emoji} {stage_name.replace('_', ' ').title()}: {status.upper()} ({duration:.3f}s)")
            
            print(f"")
            
            # Overall result
            if validation_result['pipeline_healthy']:
                print(f"🎉 RESULT: SUCCESS - Complete Opus pipeline working!")
                print(f"✅ Import: Working ({self.pipeline_stages['import_process']['details'].get('index_pages', 0)} pages indexed)")
                print(f"✅ Integration: Working (aimport.py ↔ opus_access.py)")
                print(f"✅ Extraction: Working ({validation_result['extraction_success_rate']:.1%} success rate)")
                print(f"✅ Audio Quality: Good ({validation_result['audio_quality_rate']:.1%} segments with audio)")
                
                # Special indicators for advanced features
                if self.pipeline_stages['import_process']['details'].get('is_opus_source', False):
                    print(f"🎯 1:1 Opus Copy: Verified (no re-encoding)")
                if self.pipeline_stages['import_process']['details'].get('is_ultrasonic', False):
                    print(f"🔊 Ultrasonic Handling: Verified (rescale factor applied)")
                
                print(f"\n🚀 Ready for Phase 2 (Parallelization)!")
            else:
                print(f"⚠️  RESULT: PARTIAL SUCCESS - Some issues detected")
                print(f"Pipeline stages: {validation_result['all_stages_passed']}")
                print(f"Extraction rate: {validation_result['extraction_success_rate']:.1%}")
                print(f"Audio quality: {validation_result['audio_quality_rate']:.1%}")
                print(f"\n🔧 Needs debugging before Phase 2")
            
            return {
                'overall_success': validation_result['pipeline_healthy'],
                'total_time': total_time,
                'validation_result': validation_result,
                'extraction_results': extraction_results,
                'pipeline_stages': self.pipeline_stages
            }
            
        except Exception as e:
            total_time = time.time() - overall_start_time
            
            print(f"\n" + "=" * 80)
            print("CRITICAL PIPELINE FAILURE")
            print("=" * 80)
            print(f"❌ Error: {str(e)}")
            print(f"⏱️  Failed after: {total_time:.1f}s")
            
            # Show which stages completed
            for stage_name, stage_info in self.pipeline_stages.items():
                if stage_info['status'] != 'pending':
                    emoji = "✅" if stage_info['status'] == "success" else "❌"
                    print(f"{emoji} {stage_name}: {stage_info['status']}")
            
            if self.verbose:
                print(f"\nFull traceback:")
                traceback.print_exc()
            
            return {
                'overall_success': False,
                'total_time': total_time,
                'error': str(e),
                'pipeline_stages': self.pipeline_stages
            }

def main():
    parser = argparse.ArgumentParser(description='Opus End-to-End Integration Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed debugging (TRACE level)')
    
    args = parser.parse_args()
    
    try:
        tester = OpusEndToEndTester(verbose=args.verbose)
        summary = tester.run_end_to_end_test()
        
        # Exit with appropriate code
        sys.exit(0 if summary['overall_success'] else 1)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()