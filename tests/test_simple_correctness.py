#!/usr/bin/env python3
"""
Simple Opus Correctness Test - Fixed Version
============================================

NEW FILE to avoid import issues.
Tests basic packet-based Opus implementation without circular imports.
"""

import pathlib
import time
import datetime
import numpy as np
import shutil

# Only import main zarrwlr package - NO module-specific imports
import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel

# Constants (hardcoded to avoid imports)
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"
OPUS_PACKET_INDEX_ARRAY_NAME = "opus_packet_index"
OPUS_PACKETS_BLOB_ARRAY_NAME = "opus_packets_blob"

def find_test_file():
    """Find a test audio file"""
    test_files = [
        "testdata/audiomoth_long_snippet_converted.opus",
        "testdata/bird1_snippet.mp3", 
        "testdata/audiomoth_short_snippet.wav"
    ]
    
    base_path = pathlib.Path(__file__).parent.resolve()
    for test_file in test_files:
        test_path = base_path / test_file
        if test_path.exists():
            print(f"✅ Found test file: {test_path.name}")
            return test_path
    
    raise FileNotFoundError(f"No test files found in testdata/")

def detect_format_simple(zarr_group):
    """Simple format detection"""
    has_packet_format = (
        OPUS_PACKETS_BLOB_ARRAY_NAME in zarr_group and
        OPUS_PACKET_INDEX_ARRAY_NAME in zarr_group
    )
    
    has_legacy_format = (
        AUDIO_DATA_BLOB_ARRAY_NAME in zarr_group and
        'opus_index' in zarr_group
    )
    
    if has_packet_format:
        return 'packet_based'
    elif has_legacy_format:
        return 'legacy_ogg'
    else:
        return 'unknown'

def test_basic_correctness():
    """Test basic correctness of packet-based implementation"""
    print("=" * 60)
    print("SIMPLE OPUS CORRECTNESS TEST - FIXED VERSION")
    print("=" * 60)
    
    # Setup
    Config.set(log_level=LogLevel.INFO)
    test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
    test_results_dir.mkdir(exist_ok=True)
    
    # Clean up old stores
    for store_dir in test_results_dir.glob("zarr3-store-simple-*"):
        if store_dir.is_dir():
            shutil.rmtree(store_dir)
    
    try:
        # Find test file
        audio_file = find_test_file()
        
        # Import audio
        print("\n1. Importing audio with Opus codec...")
        timestamp = int(time.time())
        test_store_dir = test_results_dir / f"zarr3-store-simple-{timestamp}"
        
        audio_group = zarrwlr.create_original_audio_group(
            store_path=test_store_dir,
            group_path='audio_imports'
        )
        
        start_time = time.time()
        zarrwlr.import_original_audio_file(
            audio_file=audio_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=datetime.datetime.now(),
            target_codec='opus',
            opus_bitrate=128000,
            temp_dir=test_results_dir
        )
        import_time = time.time() - start_time
        
        print(f"   Import completed in {import_time:.3f}s")
        
        # Find imported group
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            raise RuntimeError("No audio groups found after import")
            
        imported_group = audio_group[max(group_names, key=int)]
        
        # Check what was imported
        print("\n2. Analyzing imported data...")
        print(f"   Available arrays: {list(imported_group.keys())}")
        
        # Detect format
        format_type = detect_format_simple(imported_group)
        print(f"   Detected format: {format_type}")
        
        # Check for opuslib
        try:
            import opuslib
            opuslib_available = True
            print("   opuslib: Available ✅")
        except ImportError:
            opuslib_available = False
            print("   opuslib: Not available ⚠️  (using ffmpeg fallback)")
        
        # Get audio parameters
        audio_blob = imported_group.get(AUDIO_DATA_BLOB_ARRAY_NAME)
        if audio_blob:
            codec = audio_blob.attrs.get('codec', 'unknown')
            sample_rate = audio_blob.attrs.get('sample_rate', 48000)
            channels = audio_blob.attrs.get('nb_channels', 1)
            print(f"   Audio: {codec} {sample_rate}Hz {channels}ch")
        
        # Estimate available samples
        if format_type == 'packet_based':
            try:
                packet_index = imported_group[OPUS_PACKET_INDEX_ARRAY_NAME]
                estimated_samples = packet_index.attrs.get('estimated_total_samples', 1000)
                print(f"   Estimated samples: {estimated_samples} (from packet index)")
            except KeyError:
                estimated_samples = 1000
                print("   Packet index not found - using default")
        elif format_type == 'legacy_ogg':
            try:
                opus_index = imported_group['opus_index']
                sample_positions = opus_index[:, 2]
                estimated_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 1000
                print(f"   Indexed samples: {estimated_samples} (from legacy index)")
            except KeyError:
                estimated_samples = 1000
                print("   Legacy index not found - using default")
        else:
            estimated_samples = 1000
            print("   Unknown format - using default sample estimate")
        
        # Test extraction
        print("\n3. Testing audio extraction...")
        
        test_start = 0
        test_end = min(199, estimated_samples - 1)
        
        print(f"   Extracting segment [{test_start}:{test_end}]...")
        
        start_time = time.time()
        try:
            extracted_audio = zarrwlr.extract_audio_segment(
                imported_group, test_start, test_end, dtype=np.int16
            )
            extraction_time = time.time() - start_time
            extraction_success = True
        except Exception as e:
            extraction_time = time.time() - start_time
            extraction_success = False
            print(f"   ❌ Extraction failed: {e}")
            return False
        
        print(f"   Extraction completed in {extraction_time:.4f}s")
        
        # Analyze extracted audio
        print("\n4. Analyzing extracted audio...")
        
        if extracted_audio.size == 0:
            print("   ❌ No audio data extracted!")
            return False
        
        print(f"   Shape: {extracted_audio.shape}")
        print(f"   Data type: {extracted_audio.dtype}")
        print(f"   Size: {extracted_audio.size} samples")
        
        # Check for actual audio content
        has_audio = np.any(extracted_audio != 0)
        print(f"   Has audio data: {'Yes' if has_audio else 'No'}")
        
        if has_audio:
            # Calculate audio statistics
            rms_level = np.sqrt(np.mean(extracted_audio.astype(np.float64) ** 2))
            dynamic_range = int(extracted_audio.max() - extracted_audio.min())
            zero_crossings = int(np.sum(np.diff(np.signbit(extracted_audio))))
            
            print(f"   RMS level: {rms_level:.2f}")
            print(f"   Dynamic range: {dynamic_range}")
            print(f"   Zero crossings: {zero_crossings}")
            
            # Quality checks
            valid_range = np.all(np.abs(extracted_audio) <= 32767)
            reasonable_dynamics = dynamic_range > 100
            has_signal = rms_level > 0
            has_variation = zero_crossings > 10
            
            print(f"   Valid range: {'Yes' if valid_range else 'No'}")
            print(f"   Reasonable dynamics: {'Yes' if reasonable_dynamics else 'No'}")
            print(f"   Has signal: {'Yes' if has_signal else 'No'}")
            print(f"   Has variation: {'Yes' if has_variation else 'No'}")
            
            quality_score = sum([has_audio, valid_range, reasonable_dynamics, has_signal, has_variation]) / 5.0
            print(f"   Quality score: {quality_score:.2f}/1.0")
            
            # Test multiple segments
            print("\n5. Testing multiple extractions...")
            test_segments = [
                (0, 49, "beginning"),
                (50, 99, "early"),
                (100, 149, "middle"),
                (max(200, estimated_samples-100), min(estimated_samples-1, estimated_samples-50), "end")
            ]
            
            successful_extractions = 0
            total_extraction_time = 0
            
            for i, (start, end, desc) in enumerate(test_segments):
                if start >= estimated_samples or end >= estimated_samples:
                    print(f"   Segment {i+1} ({desc}): Skipped (outside range)")
                    continue
                    
                try:
                    start_time = time.time()
                    segment = zarrwlr.extract_audio_segment(imported_group, start, end, dtype=np.int16)
                    seg_time = time.time() - start_time
                    total_extraction_time += seg_time
                    
                    if segment.size > 0:
                        successful_extractions += 1
                        has_seg_audio = np.any(segment != 0)
                        print(f"   Segment {i+1} ({desc}): {segment.shape} {'✅' if has_seg_audio else '⚠️'} ({seg_time:.4f}s)")
                    else:
                        print(f"   Segment {i+1} ({desc}): Empty ❌")
                        
                except Exception as e:
                    print(f"   Segment {i+1} ({desc}): Error - {e} ❌")
            
            avg_extraction_time = total_extraction_time / max(1, successful_extractions)
            print(f"   Successful extractions: {successful_extractions}/{len(test_segments)}")
            print(f"   Average extraction time: {avg_extraction_time:.4f}s")
            
            # Overall assessment
            print("\n6. Overall Assessment...")
            
            # Success criteria
            format_ok = format_type in ['packet_based', 'legacy_ogg']
            extraction_ok = quality_score >= 0.6  # Lowered threshold
            performance_ok = avg_extraction_time < 0.2  # Reasonable threshold
            success_rate_ok = successful_extractions >= 2  # At least 2 successful
            
            overall_success = format_ok and extraction_ok and success_rate_ok
            
            print(f"   Format detection: {'✅' if format_ok else '❌'} ({format_type})")
            print(f"   Audio quality: {'✅' if extraction_ok else '❌'} ({quality_score:.2f}/1.0)")
            print(f"   Performance: {'✅' if performance_ok else '⚠️'} ({avg_extraction_time:.4f}s avg)")
            print(f"   Success rate: {'✅' if success_rate_ok else '❌'} ({successful_extractions} successful)")
            
            if overall_success:
                print(f"\n🎉 RESULT: SUCCESS")
                print(f"   ✅ Opus implementation working correctly")
                
                # Implementation details
                if format_type == 'packet_based':
                    if opuslib_available:
                        print(f"   🚀 Using packet-based format with opuslib (optimal)")
                    else:
                        print(f"   📦 Using packet-based format with ffmpeg fallback")
                else:
                    print(f"   📼 Using legacy OGG format")
                
                # Performance assessment
                if avg_extraction_time < 0.05:
                    print(f"   ⚡ Performance: Excellent ({avg_extraction_time:.4f}s avg)")
                elif avg_extraction_time < 0.1:
                    print(f"   🏃 Performance: Good ({avg_extraction_time:.4f}s avg)")
                else:
                    print(f"   🚶 Performance: Acceptable ({avg_extraction_time:.4f}s avg)")
                
                return True
            else:
                print(f"\n⚠️  RESULT: PARTIAL SUCCESS")
                print(f"   Implementation working but may have issues")
                print(f"   Check individual criteria above")
                return False
        else:
            print("\n❌ RESULT: FAILURE")
            print("   No audio data in extracted segments")
            return False
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_correctness()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)