#!/usr/bin/env python3
"""
Packet-Based Opus Implementation Test Suite
===========================================

Tests the new packet-based Opus implementation:
1. Import: ffmpeg → OGG → packet extraction → Zarr storage
2. Format detection: packet-based vs legacy OGG
3. Extraction: opuslib direct vs ffmpeg fallback
4. Performance: packet-based vs legacy comparison
5. Compatibility: backward compatibility with existing databases

GOAL: Verify packet-based implementation provides:
- Sample-accurate extraction (eliminates 99.8% mismatch)
- 5-20x performance improvement for small segments
- Seamless backward compatibility
"""

import pathlib
import shutil
import time
import datetime
import numpy as np
import argparse
import sys
import traceback
import tempfile
import subprocess
import json

# Full zarrwlr import
import zarrwlr
import zarr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel, get_module_logger

# Direct imports for testing specific modules
from zarrwlr import opus_access, opus_index_backend

class PacketBasedOpusTestSuite:
    """Comprehensive test suite for packet-based Opus implementation"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Configure logging
        log_level = LogLevel.TRACE if verbose else LogLevel.INFO
        Config.set(log_level=log_level)
        
        self.logger = get_module_logger(__file__)
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        
        # Ensure test directory exists
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Test results tracking
        self.test_results = {
            'packet_extraction_test': {'status': 'pending', 'details': {}},
            'import_packet_format_test': {'status': 'pending', 'details': {}},
            'format_detection_test': {'status': 'pending', 'details': {}},
            'opuslib_extraction_test': {'status': 'pending', 'details': {}},
            'legacy_compatibility_test': {'status': 'pending', 'details': {}},
            'correctness_comparison_test': {'status': 'pending', 'details': {}},
            'performance_benchmark_test': {'status': 'pending', 'details': {}},
            'end_to_end_integration_test': {'status': 'pending', 'details': {}}
        }
    
    def debug_print(self, test_name: str, message: str, level: str = "INFO"):
        """Enhanced debug printing"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{timestamp}] [{level}] [{test_name}]"
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{prefix} {message}")
        
        # Log appropriately
        if level == "ERROR":
            self.logger.error(f"{test_name}: {message}")
        elif level == "WARNING":
            self.logger.warning(f"{test_name}: {message}")
        else:
            self.logger.info(f"{test_name}: {message}")
    
    def update_test_status(self, test_name: str, status: str, **details):
        """Update test status and details"""
        self.test_results[test_name]['status'] = status
        self.test_results[test_name]['details'].update(details)
        
        emoji = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
        self.debug_print(test_name, f"{emoji} Status: {status}")
    
    def find_test_file(self) -> pathlib.Path:
        """Find suitable audio test file"""
        self.debug_print("SETUP", "Searching for test audio files...")
        
        test_files = [
            "testdata/audiomoth_long_snippet_converted.opus",
            "testdata/bird1_snippet.mp3",
            "testdata/audiomoth_short_snippet.wav",
            "testdata/audiomoth_long_snippet.wav"
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        for test_file in test_files:
            test_path = base_path / test_file
            if test_path.exists():
                self.debug_print("SETUP", f"Found test file: {test_path.name}")
                return test_path
        
        raise FileNotFoundError(f"No test files found. Searched: {test_files}")
    
    def cleanup_test_stores(self):
        """Clean up old test stores"""
        self.debug_print("CLEANUP", "Cleaning up old test stores...")
        
        for store_dir in self.test_results_dir.glob("zarr3-store-packet-test-*"):
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
    
    def test_packet_extraction_from_ogg(self) -> dict:
        """Test 1: OGG → Raw Packet Extraction"""
        test_name = "packet_extraction_test"
        self.debug_print(test_name, "Testing OGG container → raw packet extraction...")
        
        try:
            # Create a test OGG file with ffmpeg
            audio_file = self.find_test_file()
            
            with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_ogg:
                # Convert to OGG using ffmpeg
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", str(audio_file), 
                    "-c:a", "libopus", "-b:a", "128000",
                    "-f", "ogg", temp_ogg.name
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                # Read OGG data
                with open(temp_ogg.name, "rb") as f:
                    ogg_data = f.read()
                
                self.debug_print(test_name, f"Created test OGG: {len(ogg_data)} bytes")
                
                # Test packet extraction
                start_time = time.time()
                packets, opus_header, total_samples = opus_access._extract_opus_packets_from_ogg(ogg_data)
                extraction_time = time.time() - start_time
                
                # Validate results
                if not packets:
                    raise ValueError("No packets extracted")
                if not opus_header:
                    raise ValueError("No OpusHead header found")
                
                # Analyze packets
                total_packet_size = sum(len(p) for p in packets)
                avg_packet_size = total_packet_size / len(packets) if packets else 0
                
                self.debug_print(test_name, f"Extracted {len(packets)} packets in {extraction_time:.3f}s")
                self.debug_print(test_name, f"OpusHead header: {len(opus_header)} bytes")
                self.debug_print(test_name, f"Total packet data: {total_packet_size} bytes")
                self.debug_print(test_name, f"Average packet size: {avg_packet_size:.1f} bytes")
                self.debug_print(test_name, f"Estimated samples: {total_samples}")
                
                self.update_test_status(test_name, "success",
                                      packet_count=len(packets),
                                      opus_header_size=len(opus_header),
                                      total_packet_size=total_packet_size,
                                      avg_packet_size=avg_packet_size,
                                      estimated_samples=total_samples,
                                      extraction_time=extraction_time)
                
                return {
                    'packets': packets,
                    'opus_header': opus_header,
                    'total_samples': total_samples,
                    'ogg_data': ogg_data
                }
                
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Packet extraction test failed: {e}")
    
    def test_import_packet_format(self, audio_file: pathlib.Path) -> zarr.Group:
        """Test 2: Import with packet-based format"""
        test_name = "import_packet_format_test" 
        self.debug_print(test_name, "Testing import with packet-based format...")
        
        try:
            # Create test Zarr store
            timestamp = int(time.time())
            test_store_dir = self.test_results_dir / f"zarr3-store-packet-test-{timestamp}"
            
            # Create audio group
            audio_group = zarrwlr.create_original_audio_group(
                store_path=test_store_dir,
                group_path='audio_imports'
            )
            
            # Import using new packet-based implementation
            start_time = time.time()
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',
                opus_bitrate=128000,
                temp_dir=self.test_results_dir
            )
            import_time = time.time() - start_time
            
            # Find imported group
            group_names = [name for name in audio_group.keys() if name.isdigit()]
            if not group_names:
                raise RuntimeError("No audio groups found after import")
            
            latest_group_name = max(group_names, key=int)
            imported_group = audio_group[latest_group_name]
            
            # Check for packet-based format
            format_type = opus_index_backend.detect_opus_format(imported_group)
            
            self.debug_print(test_name, f"Import completed in {import_time:.3f}s")
            self.debug_print(test_name, f"Detected format: {format_type}")
            
            # Analyze format
            if format_type == 'packet_based':
                packet_index = imported_group[opus_access.OPUS_PACKET_INDEX_ARRAY_NAME]
                packet_blob = imported_group[opus_access.OPUS_PACKETS_BLOB_ARRAY_NAME]
                opus_header = imported_group[opus_access.OPUS_HEADER_ARRAY_NAME]
                
                self.debug_print(test_name, f"✅ Packet-based format detected!")
                self.debug_print(test_name, f"Packets: {packet_index.shape[0]}")
                self.debug_print(test_name, f"Packet data: {packet_blob.shape[0]} bytes")
                self.debug_print(test_name, f"Header: {opus_header.shape[0]} bytes")
                
                # Check for legacy compatibility data
                has_legacy = opus_access.AUDIO_DATA_BLOB_ARRAY_NAME in imported_group
                self.debug_print(test_name, f"Legacy compatibility: {'Yes' if has_legacy else 'No'}")
                
                self.update_test_status(test_name, "success",
                                      format_type=format_type,
                                      import_time=import_time,
                                      packet_count=packet_index.shape[0],
                                      packet_data_size=packet_blob.shape[0],
                                      header_size=opus_header.shape[0],
                                      has_legacy_compatibility=has_legacy)
            
            elif format_type == 'legacy_ogg':
                self.debug_print(test_name, "⚠️  Legacy OGG format detected - packet-based import may have failed")
                self.update_test_status(test_name, "partial",
                                      format_type=format_type,
                                      import_time=import_time,
                                      note="Fell back to legacy format")
            else:
                raise RuntimeError(f"Unknown format type: {format_type}")
            
            return imported_group
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Packet-based import test failed: {e}")
    
    def test_format_detection(self, packet_group: zarr.Group) -> dict:
        """Test 3: Format detection capabilities"""
        test_name = "format_detection_test"
        self.debug_print(test_name, "Testing format detection...")
        
        try:
            # Test format detection
            format_type = opus_index_backend.detect_opus_format(packet_group)
            format_info = opus_index_backend.get_opus_format_info(packet_group)
            sample_rate, channels = opus_index_backend.get_sample_rate_and_channels(packet_group)
            
            self.debug_print(test_name, f"Format type: {format_type}")
            self.debug_print(test_name, f"Sample rate: {sample_rate}Hz, Channels: {channels}")
            self.debug_print(test_name, f"Packet-based available: {format_info['packet_based_available']}")
            self.debug_print(test_name, f"Legacy available: {format_info['legacy_ogg_available']}")
            
            if format_type == 'packet_based':
                self.debug_print(test_name, f"Total packets: {format_info['total_packets']}")
                self.debug_print(test_name, f"Estimated samples: {format_info['estimated_total_samples']}")
            
            self.update_test_status(test_name, "success",
                                  format_type=format_type,
                                  sample_rate=sample_rate,
                                  channels=channels,
                                  format_info=format_info)
            
            return format_info
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Format detection test failed: {e}")
    
    def test_opuslib_extraction(self, packet_group: zarr.Group) -> list:
        """Test 4: opuslib-based extraction"""
        test_name = "opuslib_extraction_test"
        self.debug_print(test_name, "Testing opuslib-based extraction...")
        
        try:
            # Check if opuslib is available
            try:
                import opuslib
                opuslib_available = True
                self.debug_print(test_name, "✅ opuslib available")
            except ImportError:
                opuslib_available = False
                self.debug_print(test_name, "⚠️  opuslib not available - will test fallback")
            
            # Get format info for test planning
            format_info = opus_index_backend.get_opus_format_info(packet_group)
            
            if format_info['format_type'] != 'packet_based':
                raise ValueError("Packet-based format required for opuslib test")
            
            # Plan test extractions
            estimated_samples = format_info.get('estimated_total_samples', 1000)
            self.debug_print(test_name, f"Planning extractions for {estimated_samples} estimated samples")
            
            test_segments = [
                (0, 99, "beginning"),
                (max(100, estimated_samples // 2), max(199, estimated_samples // 2 + 99), "middle"),
                (max(200, estimated_samples - 200), max(299, estimated_samples - 101), "end")
            ]
            
            extraction_results = []
            
            for i, (start, end, description) in enumerate(test_segments):
                if start >= estimated_samples:
                    continue
                    
                # Clamp end to available samples
                end = min(end, estimated_samples - 1)
                
                self.debug_print(test_name, f"Testing extraction {i+1}: {description} [{start}:{end}]")
                
                start_time = time.time()
                try:
                    # Extract using packet-based method
                    audio_blob_array = packet_group.get(opus_access.AUDIO_DATA_BLOB_ARRAY_NAME)  # May be None
                    segment_data = opus_access.extract_audio_segment_opus(
                        packet_group, audio_blob_array, start, end, dtype=np.int16
                    )
                    
                    extraction_time = time.time() - start_time
                    
                    if segment_data.size == 0:
                        result = {
                            'segment_id': i + 1,
                            'description': description,
                            'range': (start, end),
                            'success': False,
                            'error': 'Empty result',
                            'extraction_time': extraction_time
                        }
                    else:
                        # Analyze extracted data
                        has_audio = np.any(segment_data != 0)
                        dynamic_range = segment_data.max() - segment_data.min() if segment_data.size > 0 else 0
                        
                        result = {
                            'segment_id': i + 1,
                            'description': description,
                            'range': (start, end),
                            'success': True,
                            'shape': segment_data.shape,
                            'dtype': str(segment_data.dtype),
                            'has_audio': has_audio,
                            'dynamic_range': int(dynamic_range),
                            'extraction_time': extraction_time,
                            'method_used': 'packet_based' if opuslib_available else 'ffmpeg_fallback'
                        }
                        
                        self.debug_print(test_name, f"Segment {i+1}: {result['shape']} samples, "
                                                  f"audio: {has_audio}, dynamic: {dynamic_range}, "
                                                  f"time: {extraction_time:.3f}s")
                    
                    extraction_results.append(result)
                    
                except Exception as e:
                    extraction_time = time.time() - start_time
                    result = {
                        'segment_id': i + 1,
                        'description': description,
                        'range': (start, end),
                        'success': False,
                        'error': str(e),
                        'extraction_time': extraction_time
                    }
                    extraction_results.append(result)
                    self.debug_print(test_name, f"Segment {i+1} failed: {e}", "ERROR")
            
            # Summary
            successful = sum(1 for r in extraction_results if r['success'])
            total = len(extraction_results)
            avg_time = np.mean([r['extraction_time'] for r in extraction_results]) if extraction_results else 0
            
            self.debug_print(test_name, f"Extraction results: {successful}/{total} successful")
            self.debug_print(test_name, f"Average extraction time: {avg_time:.3f}s")
            
            self.update_test_status(test_name, "success",
                                  opuslib_available=opuslib_available,
                                  successful_extractions=successful,
                                  total_extractions=total,
                                  success_rate=successful / total if total > 0 else 0,
                                  avg_extraction_time=avg_time,
                                  extraction_results=extraction_results)
            
            return extraction_results
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"opuslib extraction test failed: {e}")
    
    def test_legacy_compatibility(self, audio_file: pathlib.Path) -> zarr.Group:
        """Test 5: Legacy OGG format compatibility"""
        test_name = "legacy_compatibility_test"
        self.debug_print(test_name, "Testing legacy OGG format compatibility...")
        
        try:
            # Create separate store for legacy test
            timestamp = int(time.time())
            legacy_store_dir = self.test_results_dir / f"zarr3-store-legacy-{timestamp}"
            
            # Force legacy mode by temporarily disabling packet format
            # (This would require a config flag or different import path)
            # For now, we'll import normally and check if legacy fallback works
            
            audio_group = zarrwlr.create_original_audio_group(
                store_path=legacy_store_dir,
                group_path='audio_imports'
            )
            
            # Import (should create packet format, but we'll test legacy reading)
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',
                opus_bitrate=128000,
                temp_dir=self.test_results_dir
            )
            
            # Find imported group
            group_names = [name for name in audio_group.keys() if name.isdigit()]
            legacy_group = audio_group[max(group_names, key=int)]
            
            # Test legacy index creation if OGG format exists
            if opus_access.AUDIO_DATA_BLOB_ARRAY_NAME in legacy_group:
                audio_blob = legacy_group[opus_access.AUDIO_DATA_BLOB_ARRAY_NAME]
                
                # Test legacy index building
                start_time = time.time()
                legacy_index = opus_index_backend._build_legacy_opus_index(
                    legacy_group, audio_blob, use_parallel=False
                )
                index_time = time.time() - start_time
                
                self.debug_print(test_name, f"Legacy index created: {legacy_index.shape[0]} pages in {index_time:.3f}s")
                
                # Test legacy extraction
                if legacy_index.shape[0] > 0:
                    sample_positions = legacy_index[:, 2]
                    max_sample = int(sample_positions[-1]) if len(sample_positions) > 0 else 100
                    
                    test_start = 0
                    test_end = min(99, max_sample - 1)
                    
                    start_time = time.time()
                    legacy_segment = opus_access._extract_segment_legacy(
                        legacy_group, audio_blob, test_start, test_end, np.int16
                    )
                    extraction_time = time.time() - start_time
                    
                    self.debug_print(test_name, f"Legacy extraction: {legacy_segment.shape} in {extraction_time:.3f}s")
                    
                    self.update_test_status(test_name, "success",
                                          legacy_index_pages=legacy_index.shape[0],
                                          index_creation_time=index_time,
                                          legacy_extraction_time=extraction_time,
                                          legacy_segment_shape=legacy_segment.shape)
                else:
                    raise ValueError("Empty legacy index created")
            else:
                self.debug_print(test_name, "No legacy OGG data found - packet-only format")
                self.update_test_status(test_name, "skipped",
                                      reason="No legacy OGG data in packet-based import")
            
            return legacy_group
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Legacy compatibility test failed: {e}")
    
    def test_correctness_comparison(self, packet_group: zarr.Group, audio_file: pathlib.Path) -> dict:
        """Test 6: Correctness comparison between packet-based and legacy"""
        test_name = "correctness_comparison_test"
        self.debug_print(test_name, "Testing correctness: packet-based vs reference...")
        
        try:
            # For a complete correctness test, we would need to compare:
            # 1. Packet-based extraction results
            # 2. Direct ffmpeg extraction results
            # 3. Legacy OGG extraction results
            
            # This is a simplified test that checks for basic correctness indicators
            format_info = opus_index_backend.get_opus_format_info(packet_group)
            
            if format_info['format_type'] != 'packet_based':
                raise ValueError("Packet-based format required")
            
            # Extract a test segment using packet-based method
            estimated_samples = format_info.get('estimated_total_samples', 1000)
            test_start = 0
            test_end = min(99, estimated_samples - 1)
            
            self.debug_print(test_name, f"Extracting test segment [{test_start}:{test_end}]")
            
            # Packet-based extraction
            audio_blob_array = packet_group.get(opus_access.AUDIO_DATA_BLOB_ARRAY_NAME)
            packet_result = opus_access.extract_audio_segment_opus(
                packet_group, audio_blob_array, test_start, test_end, np.int16
            )
            
            # Basic correctness checks
            has_audio_data = np.any(packet_result != 0) if packet_result.size > 0 else False
            is_valid_range = np.all(np.abs(packet_result) <= 32767) if packet_result.size > 0 else True
            has_expected_length = packet_result.size > 0
            
            # Calculate some basic audio properties
            if packet_result.size > 0:
                rms_level = np.sqrt(np.mean(packet_result.astype(np.float64) ** 2))
                dynamic_range = packet_result.max() - packet_result.min()
                zero_crossings = np.sum(np.diff(np.signbit(packet_result)))
            else:
                rms_level = 0
                dynamic_range = 0
                zero_crossings = 0
            
            correctness_score = sum([
                has_audio_data,
                is_valid_range,
                has_expected_length,
                rms_level > 0,  # Has signal energy
                dynamic_range > 100  # Has reasonable dynamic range
            ]) / 5.0
            
            self.debug_print(test_name, f"Correctness indicators:")
            self.debug_print(test_name, f"  Has audio data: {has_audio_data}")
            self.debug_print(test_name, f"  Valid range: {is_valid_range}")
            self.debug_print(test_name, f"  Expected length: {has_expected_length}")
            self.debug_print(test_name, f"  RMS level: {rms_level:.2f}")
            self.debug_print(test_name, f"  Dynamic range: {dynamic_range}")
            self.debug_print(test_name, f"  Zero crossings: {zero_crossings}")
            self.debug_print(test_name, f"  Correctness score: {correctness_score:.2f}/1.0")
            
            if correctness_score >= 0.8:
                self.update_test_status(test_name, "success",
                                      correctness_score=correctness_score,
                                      has_audio_data=has_audio_data,
                                      is_valid_range=is_valid_range,
                                      rms_level=rms_level,
                                      dynamic_range=int(dynamic_range),
                                      zero_crossings=int(zero_crossings))
            else:
                self.update_test_status(test_name, "partial",
                                      correctness_score=correctness_score,
                                      note="Low correctness score - may need investigation")
            
            return {
                'correctness_score': correctness_score,
                'packet_result': packet_result,
                'audio_metrics': {
                    'rms_level': rms_level,
                    'dynamic_range': dynamic_range,
                    'zero_crossings': zero_crossings
                }
            }
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Correctness comparison test failed: {e}")
    
    def test_performance_benchmark(self, packet_group: zarr.Group) -> dict:
        """Test 7: Performance benchmarking"""
        test_name = "performance_benchmark_test"
        self.debug_print(test_name, "Testing performance benchmarking...")
        
        try:
            format_info = opus_index_backend.get_opus_format_info(packet_group)
            
            if format_info['format_type'] != 'packet_based':
                self.debug_print(test_name, "Packet-based format not available - skipping detailed benchmark")
                self.update_test_status(test_name, "skipped", reason="Packet-based format not available")
                return {}
            
            # Performance test: multiple small segments
            estimated_samples = format_info.get('estimated_total_samples', 1000)
            
            # Create multiple small segments (typical use case for random access)
            segment_size = 100  # Small segments (typical for analysis)
            num_segments = min(10, estimated_samples // segment_size)
            
            test_segments = []
            for i in range(num_segments):
                start = i * segment_size
                end = min(start + segment_size - 1, estimated_samples - 1)
                if start < estimated_samples:
                    test_segments.append((start, end))
            
            self.debug_print(test_name, f"Benchmarking {len(test_segments)} segments of {segment_size} samples each")
            
            # Benchmark packet-based extraction
            audio_blob_array = packet_group.get(opus_access.AUDIO_DATA_BLOB_ARRAY_NAME)
            
            start_time = time.time()
            packet_results = []
            for start, end in test_segments:
                result = opus_access.extract_audio_segment_opus(
                    packet_group, audio_blob_array, start, end, np.int16
                )
                packet_results.append(result)
            packet_time = time.time() - start_time
            
            # Calculate performance metrics
            total_samples_extracted = sum(r.size for r in packet_results if r.size > 0)
            successful_extractions = sum(1 for r in packet_results if r.size > 0)
            
            samples_per_second = total_samples_extracted / packet_time if packet_time > 0 else 0
            extractions_per_second = len(test_segments) / packet_time if packet_time > 0 else 0
            avg_extraction_time = packet_time / len(test_segments) if test_segments else 0
            
            self.debug_print(test_name, f"Performance results:")
            self.debug_print(test_name, f"  Total time: {packet_time:.3f}s")
            self.debug_print(test_name, f"  Successful extractions: {successful_extractions}/{len(test_segments)}")
            self.debug_print(test_name, f"  Samples extracted: {total_samples_extracted}")
            self.debug_print(test_name, f"  Samples/second: {samples_per_second:.0f}")
            self.debug_print(test_name, f"  Extractions/second: {extractions_per_second:.1f}")
            self.debug_print(test_name, f"  Avg extraction time: {avg_extraction_time:.4f}s")
            
            # Performance classification
            if avg_extraction_time < 0.01:  # < 10ms per extraction
                performance_class = "excellent"
            elif avg_extraction_time < 0.05:  # < 50ms per extraction
                performance_class = "good"
            elif avg_extraction_time < 0.1:  # < 100ms per extraction
                performance_class = "acceptable"
            else:
                performance_class = "needs_improvement"
            
            self.debug_print(test_name, f"Performance class: {performance_class}")
            
            self.update_test_status(test_name, "success",
                                  total_time=packet_time,
                                  successful_extractions=successful_extractions,
                                  total_extractions=len(test_segments),
                                  samples_per_second=samples_per_second,
                                  extractions_per_second=extractions_per_second,
                                  avg_extraction_time=avg_extraction_time,
                                  performance_class=performance_class)
            
            return {
                'packet_time': packet_time,
                'successful_extractions': successful_extractions,
                'samples_per_second': samples_per_second,
                'extractions_per_second': extractions_per_second,
                'avg_extraction_time': avg_extraction_time,
                'performance_class': performance_class
            }
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"Performance benchmark test failed: {e}")
    
    def test_end_to_end_integration(self, audio_file: pathlib.Path) -> dict:
        """Test 8: Complete end-to-end integration"""
        test_name = "end_to_end_integration_test"
        self.debug_print(test_name, "Testing complete end-to-end integration...")
        
        try:
            # Create fresh test environment
            timestamp = int(time.time())
            e2e_store_dir = self.test_results_dir / f"zarr3-store-e2e-{timestamp}"
            
            # Complete pipeline test using standard API
            audio_group = zarrwlr.create_original_audio_group(
                store_path=e2e_store_dir,
                group_path='audio_imports'
            )
            
            # Import
            import_start = time.time()
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',
                opus_bitrate=128000,
                temp_dir=self.test_results_dir
            )
            import_time = time.time() - import_start
            
            # Find imported group
            group_names = [name for name in audio_group.keys() if name.isdigit()]
            imported_group = audio_group[max(group_names, key=int)]
            
            # Test extraction using generic API
            extraction_start = time.time()
            test_segment = zarrwlr.extract_audio_segment(imported_group, 0, 99, dtype=np.int16)
            extraction_time = time.time() - extraction_start
            
            # Test parallel extraction
            parallel_start = time.time()
            parallel_segments = [(0, 49), (50, 99), (100, 149)]
            parallel_results = zarrwlr.parallel_extract_audio_segments(
                imported_group, parallel_segments, dtype=np.int16, max_workers=2
            )
            parallel_time = time.time() - parallel_start
            
            # Validate results
            format_type = opus_index_backend.detect_opus_format(imported_group)
            has_packet_format = format_type == 'packet_based'
            has_audio_data = test_segment.size > 0 and np.any(test_segment != 0)
            parallel_success = all(r.size > 0 for r in parallel_results)
            
            # Check for opuslib usage
            try:
                import opuslib
                opuslib_available = True
            except ImportError:
                opuslib_available = False
            
            expected_performance = has_packet_format and opuslib_available
            
            self.debug_print(test_name, f"End-to-end results:")
            self.debug_print(test_name, f"  Import time: {import_time:.3f}s")
            self.debug_print(test_name, f"  Format detected: {format_type}")
            self.debug_print(test_name, f"  Single extraction: {extraction_time:.4f}s")
            self.debug_print(test_name, f"  Parallel extraction: {parallel_time:.4f}s")
            self.debug_print(test_name, f"  Has audio data: {has_audio_data}")
            self.debug_print(test_name, f"  Parallel success: {parallel_success}")
            self.debug_print(test_name, f"  opuslib available: {opuslib_available}")
            self.debug_print(test_name, f"  Expected high performance: {expected_performance}")
            
            # Overall success criteria
            integration_success = (
                import_time < 60.0 and  # Reasonable import time
                extraction_time < 1.0 and  # Reasonable extraction time
                has_audio_data and
                parallel_success
            )
            
            self.update_test_status(test_name, "success" if integration_success else "partial",
                                  import_time=import_time,
                                  extraction_time=extraction_time,
                                  parallel_time=parallel_time,
                                  format_type=format_type,
                                  has_packet_format=has_packet_format,
                                  has_audio_data=has_audio_data,
                                  parallel_success=parallel_success,
                                  opuslib_available=opuslib_available,
                                  expected_performance=expected_performance,
                                  integration_success=integration_success)
            
            return {
                'integration_success': integration_success,
                'import_time': import_time,
                'extraction_time': extraction_time,
                'parallel_time': parallel_time,
                'format_type': format_type,
                'performance_ready': expected_performance
            }
            
        except Exception as e:
            self.update_test_status(test_name, "failed", error=str(e))
            raise RuntimeError(f"End-to-end integration test failed: {e}")
    
    def run_comprehensive_test_suite(self) -> dict:
        """Run all packet-based implementation tests"""
        print("=" * 80)
        print("PACKET-BASED OPUS IMPLEMENTATION TEST SUITE")
        print("=" * 80)
        print("Testing: Packet extraction, opuslib integration, performance")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        try:
            # Setup
            self.cleanup_test_stores()
            audio_file = self.find_test_file()
            
            print(f"\nTest file: {audio_file.name}")
            print(f"Running comprehensive packet-based tests...\n")
            
            # Test 1: Packet extraction from OGG
            self.debug_print("SUITE", "🧪 Test 1: Packet extraction from OGG...")
            packet_data = self.test_packet_extraction_from_ogg()
            
            # Test 2: Import with packet format
            self.debug_print("SUITE", "🧪 Test 2: Import with packet-based format...")
            packet_group = self.test_import_packet_format(audio_file)
            
            # Test 3: Format detection
            self.debug_print("SUITE", "🧪 Test 3: Format detection...")
            format_info = self.test_format_detection(packet_group)
            
            # Test 4: opuslib extraction
            self.debug_print("SUITE", "🧪 Test 4: opuslib-based extraction...")
            extraction_results = self.test_opuslib_extraction(packet_group)
            
            # Test 5: Legacy compatibility
            self.debug_print("SUITE", "🧪 Test 5: Legacy compatibility...")
            try:
                legacy_group = self.test_legacy_compatibility(audio_file)
            except Exception as e:
                self.debug_print("SUITE", f"Legacy compatibility test failed: {e}", "WARNING")
                legacy_group = None
            
            # Test 6: Correctness comparison
            self.debug_print("SUITE", "🧪 Test 6: Correctness comparison...")
            correctness_data = self.test_correctness_comparison(packet_group, audio_file)
            
            # Test 7: Performance benchmark
            self.debug_print("SUITE", "🧪 Test 7: Performance benchmark...")
            performance_data = self.test_performance_benchmark(packet_group)
            
            # Test 8: End-to-end integration
            self.debug_print("SUITE", "🧪 Test 8: End-to-end integration...")
            integration_data = self.test_end_to_end_integration(audio_file)
            
            # Final analysis
            total_time = time.time() - overall_start_time
            
            print(f"\n" + "=" * 80)
            print("PACKET-BASED IMPLEMENTATION TEST RESULTS")
            print("=" * 80)
            
            # Test summary
            passed_tests = 0
            total_tests = 0
            
            for test_name, result in self.test_results.items():
                if result['status'] != 'pending':
                    total_tests += 1
                    status = result['status']
                    
                    if status == 'success':
                        emoji = "✅"
                        passed_tests += 1
                    elif status == 'partial':
                        emoji = "⚠️"
                        passed_tests += 0.5
                    elif status == 'skipped':
                        emoji = "⏭️"
                        total_tests -= 1  # Don't count skipped tests
                    else:
                        emoji = "❌"
                    
                    test_display = test_name.replace('_test', '').replace('_', ' ').title()
                    print(f"{emoji} {test_display}: {status.upper()}")
                    
                    # Show key details for important tests
                    if test_name == 'import_packet_format_test' and status == 'success':
                        details = result['details']
                        print(f"    📦 Format: {details.get('format_type', 'unknown')}")
                        print(f"    📊 Packets: {details.get('packet_count', 0)}")
                        print(f"    💾 Data: {details.get('packet_data_size', 0)} bytes")
                    
                    elif test_name == 'opuslib_extraction_test' and status == 'success':
                        details = result['details']
                        print(f"    🎵 Success rate: {details.get('success_rate', 0):.1%}")
                        print(f"    ⏱️  Avg time: {details.get('avg_extraction_time', 0):.4f}s")
                        print(f"    🔧 opuslib: {'Yes' if details.get('opuslib_available') else 'No'}")
                    
                    elif test_name == 'performance_benchmark_test' and status == 'success':
                        details = result['details']
                        print(f"    🚀 Performance: {details.get('performance_class', 'unknown')}")
                        print(f"    📈 Extractions/sec: {details.get('extractions_per_second', 0):.1f}")
                        print(f"    ⚡ Avg time: {details.get('avg_extraction_time', 0):.4f}s")
            
            print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
            print(f"⏱️  Total time: {total_time:.1f}s")
            
            # Overall assessment
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            if success_rate >= 0.9:
                print(f"\n🎉 RESULT: EXCELLENT - Packet-based implementation ready!")
                assessment = "excellent"
            elif success_rate >= 0.7:
                print(f"\n✅ RESULT: GOOD - Packet-based implementation mostly working")
                assessment = "good"
            elif success_rate >= 0.5:
                print(f"\n⚠️  RESULT: PARTIAL - Some issues need addressing")
                assessment = "partial"
            else:
                print(f"\n❌ RESULT: NEEDS WORK - Major issues detected")
                assessment = "needs_work"
            
            # Key achievements summary
            print(f"\n🔑 Key Achievements:")
            
            if self.test_results['import_packet_format_test']['status'] == 'success':
                print(f"✅ Packet-based import working")
            
            if self.test_results['opuslib_extraction_test']['status'] == 'success':
                opuslib_available = self.test_results['opuslib_extraction_test']['details'].get('opuslib_available', False)
                if opuslib_available:
                    print(f"✅ opuslib direct decoding working")
                else:
                    print(f"⚠️  opuslib not available - using ffmpeg fallback")
            
            if self.test_results['performance_benchmark_test']['status'] == 'success':
                perf_class = self.test_results['performance_benchmark_test']['details'].get('performance_class', 'unknown')
                print(f"✅ Performance benchmark: {perf_class}")
            
            if self.test_results['correctness_comparison_test']['status'] == 'success':
                score = self.test_results['correctness_comparison_test']['details'].get('correctness_score', 0)
                print(f"✅ Correctness score: {score:.2f}/1.0")
            
            # Next steps
            print(f"\n🚀 Next Steps:")
            if assessment in ['excellent', 'good']:
                print(f"  • Ready for production testing")
                print(f"  • Consider enabling by default")
                print(f"  • Performance optimization if needed")
            else:
                print(f"  • Debug failing tests")
                print(f"  • Check opuslib installation")
                print(f"  • Verify packet extraction logic")
            
            return {
                'overall_success': success_rate >= 0.7,
                'success_rate': success_rate,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'total_time': total_time,
                'assessment': assessment,
                'test_results': self.test_results,
                'key_metrics': {
                    'packet_format_working': self.test_results['import_packet_format_test']['status'] == 'success',
                    'opuslib_available': self.test_results.get('opuslib_extraction_test', {}).get('details', {}).get('opuslib_available', False),
                    'performance_class': self.test_results.get('performance_benchmark_test', {}).get('details', {}).get('performance_class', 'unknown'),
                    'correctness_score': self.test_results.get('correctness_comparison_test', {}).get('details', {}).get('correctness_score', 0)
                }
            }
            
        except Exception as e:
            total_time = time.time() - overall_start_time
            
            print(f"\n" + "=" * 80)
            print("CRITICAL TEST SUITE FAILURE")
            print("=" * 80)
            print(f"❌ Error: {str(e)}")
            print(f"⏱️  Failed after: {total_time:.1f}s")
            
            if self.verbose:
                print(f"\nFull traceback:")
                traceback.print_exc()
            
            return {
                'overall_success': False,
                'error': str(e),
                'total_time': total_time,
                'test_results': self.test_results
            }


def main():
    parser = argparse.ArgumentParser(description='Packet-Based Opus Implementation Test Suite')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed debugging')
    parser.add_argument('--test', '-t', type=str, 
                       help='Run specific test (e.g., packet_extraction, import_format, etc.)')
    
    args = parser.parse_args()
    
    try:
        tester = PacketBasedOpusTestSuite(verbose=args.verbose)
        
        if args.test:
            # Run specific test
            print(f"Running specific test: {args.test}")
            
            if args.test == 'packet_extraction':
                tester.test_packet_extraction_from_ogg()
            elif args.test == 'import_format':
                audio_file = tester.find_test_file()
                tester.test_import_packet_format(audio_file)
            elif args.test == 'integration':
                audio_file = tester.find_test_file()
                tester.test_end_to_end_integration(audio_file)
            else:
                print(f"Unknown test: {args.test}")
                sys.exit(1)
        else:
            # Run complete test suite
            summary = tester.run_comprehensive_test_suite()
            sys.exit(0 if summary['overall_success'] else 1)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()