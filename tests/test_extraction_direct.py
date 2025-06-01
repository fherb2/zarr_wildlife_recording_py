#!/usr/bin/env python3
"""
Direct Opus Extraction Test
===========================

Test extraction directly on the most recent import without version issues.
"""

import pathlib
import time
import datetime
import numpy as np
import shutil

import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel

def test_fresh_import_and_extraction():
    """Import fresh and test extraction immediately"""
    print("=" * 60)
    print("FRESH OPUS IMPORT + EXTRACTION TEST")
    print("=" * 60)
    
    Config.set(log_level=LogLevel.INFO)
    test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
    test_results_dir.mkdir(exist_ok=True)
    
    # Clean up old stores to avoid version issues
    for store_dir in test_results_dir.glob("zarr3-store-extraction-*"):
        if store_dir.is_dir():
            shutil.rmtree(store_dir)
    
    try:
        # Find test file
        test_files = [
            "testdata/bird1_snippet.mp3",
            "testdata/audiomoth_short_snippet.wav"
        ]
        
        audio_file = None
        base_path = pathlib.Path(__file__).parent.resolve()
        for test_file in test_files:
            test_path = base_path / test_file
            if test_path.exists():
                audio_file = test_path
                size_mb = test_path.stat().st_size / 1024 / 1024
                print(f"✅ Using test file: {test_path.name} ({size_mb:.2f} MB)")
                break
        
        if not audio_file:
            print("❌ No test files found")
            return False
        
        # Create fresh store
        timestamp = int(time.time())
        test_store_dir = test_results_dir / f"zarr3-store-extraction-{timestamp}"
        
        print(f"\n1. Fresh Import...")
        import_start = time.time()
        
        # Create audio group
        audio_group = zarrwlr.create_original_audio_group(
            store_path=test_store_dir,
            group_path='audio_imports'
        )
        
        # Import with packet-based format
        zarrwlr.import_original_audio_file(
            audio_file=audio_file,
            zarr_original_audio_group=audio_group,
            first_sample_time_stamp=datetime.datetime.now(),
            target_codec='opus',
            opus_bitrate=64000,  # Lower for faster processing
            temp_dir=test_results_dir
        )
        
        import_time = time.time() - import_start
        print(f"   Import completed in {import_time:.3f}s")
        
        # Find imported group
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if not group_names:
            print("❌ No audio groups found after import")
            return False
        
        imported_group = audio_group[max(group_names, key=int)]
        print(f"   Created group: {max(group_names, key=int)}")
        
        # Check what was created
        arrays = list(imported_group.keys())
        print(f"   Arrays: {arrays}")
        
        # Detect format
        has_packet_format = (
            "opus_packets_blob" in imported_group and
            "opus_packet_index" in imported_group
        )
        
        has_legacy_format = (
            "audio_data_blob_array" in imported_group and
            'opus_index' in imported_group
        )
        
        if has_packet_format:
            format_type = 'packet_based'
            print("   ✅ Packet-based format created")
        elif has_legacy_format:
            format_type = 'legacy_ogg'
            print("   📼 Legacy OGG format created")
        else:
            print("   ❓ Unknown format created")
            format_type = 'unknown'
        
        # Get audio parameters
        if format_type == 'packet_based':
            packet_index = imported_group["opus_packet_index"]
            estimated_samples = packet_index.attrs.get('estimated_total_samples', 1000)
            total_packets = packet_index.shape[0]
            print(f"   📊 {estimated_samples} samples, {total_packets} packets")
        elif format_type == 'legacy_ogg':
            opus_index = imported_group['opus_index']
            sample_positions = opus_index[:, 2]
            estimated_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 1000
            total_pages = opus_index.shape[0]
            print(f"   📊 {estimated_samples} samples, {total_pages} pages")
        else:
            estimated_samples = 1000
            print("   📊 Using default sample estimate")
        
        # Check for opuslib
        try:
            import opuslib
            opuslib_available = True
            print("   ✅ opuslib available")
        except ImportError:
            opuslib_available = False
            print("   ⚠️  opuslib not available (ffmpeg fallback)")
        
        # Test 1: Single extraction
        print(f"\n2. Single Extraction Test...")
        
        test_start = 0
        test_end = min(499, estimated_samples - 1)  # 500 samples
        
        print(f"   Extracting [{test_start}:{test_end}]...")
        
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
        
        print(f"   Result: {extracted_audio.shape} in {extraction_time:.4f}s")
        
        # Validate extracted audio
        if extracted_audio.size == 0:
            print("   ❌ Empty result")
            return False
        
        has_audio = np.any(extracted_audio != 0)
        if has_audio:
            rms_level = np.sqrt(np.mean(extracted_audio.astype(np.float64) ** 2))
            dynamic_range = extracted_audio.max() - extracted_audio.min()
            print(f"   ✅ Audio detected: RMS {rms_level:.1f}, range {dynamic_range}")
        else:
            print("   ⚠️  Silent audio (may be normal)")
        
        # Test 2: Multiple extractions
        print(f"\n3. Multiple Extractions Test...")
        
        # Test smaller segments for speed
        segment_size = 100
        num_segments = min(8, estimated_samples // segment_size)
        
        test_segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = min(start + segment_size - 1, estimated_samples - 1)
            test_segments.append((start, end))
        
        print(f"   Testing {len(test_segments)} segments...")
        
        start_time = time.time()
        successful_extractions = 0
        extraction_times = []
        
        for i, (start, end) in enumerate(test_segments):
            try:
                seg_start = time.time()
                segment = zarrwlr.extract_audio_segment(imported_group, start, end, dtype=np.int16)
                seg_time = time.time() - seg_start
                
                if segment.size > 0:
                    successful_extractions += 1
                    extraction_times.append(seg_time)
                    print(f"   Segment {i+1}: {segment.shape} ({seg_time:.4f}s)")
                else:
                    print(f"   Segment {i+1}: Empty")
                    
            except Exception as e:
                print(f"   Segment {i+1}: Error - {e}")
        
        total_time = time.time() - start_time
        avg_time = np.mean(extraction_times) if extraction_times else 0
        
        print(f"\n   Results: {successful_extractions}/{len(test_segments)} successful")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per segment: {avg_time:.4f}s")
        
        # Test 3: Parallel extraction (if enough segments)
        if len(test_segments) >= 3:
            print(f"\n4. Parallel Extraction Test...")
            
            start_time = time.time()
            try:
                parallel_results = zarrwlr.parallel_extract_audio_segments(
                    imported_group, test_segments[:3], dtype=np.int16, max_workers=2
                )
                parallel_time = time.time() - start_time
                
                parallel_successful = sum(1 for r in parallel_results if r.size > 0)
                avg_parallel_time = parallel_time / 3
                
                print(f"   Results: {parallel_successful}/3 successful")
                print(f"   Total time: {parallel_time:.3f}s")
                print(f"   Average per segment: {avg_parallel_time:.4f}s")
                
            except Exception as e:
                print(f"   ❌ Parallel extraction failed: {e}")
                avg_parallel_time = avg_time
        else:
            avg_parallel_time = avg_time
        
        # Performance Analysis
        print(f"\n5. Performance Analysis...")
        
        # Performance thresholds
        if format_type == 'packet_based' and opuslib_available:
            expected_time = 0.02  # 20ms with opuslib
            performance_level = "Optimal"
        elif format_type == 'packet_based':
            expected_time = 0.05  # 50ms with ffmpeg fallback
            performance_level = "Good"
        else:
            expected_time = 0.1   # 100ms with legacy
            performance_level = "Standard"
        
        print(f"   Expected performance: {performance_level}")
        print(f"   Expected time: ~{expected_time:.3f}s per segment")
        
        # Assess performance
        if avg_time <= expected_time:
            perf_assessment = "Excellent"
            perf_emoji = "⚡"
        elif avg_time <= expected_time * 2:
            perf_assessment = "Good"
            perf_emoji = "✅"
        elif avg_time <= expected_time * 5:
            perf_assessment = "Acceptable"
            perf_emoji = "📊"
        else:
            perf_assessment = "Slow"
            perf_emoji = "🐌"
        
        print(f"   Actual performance: {perf_emoji} {perf_assessment} ({avg_time:.4f}s)")
        
        # Success assessment
        extraction_working = successful_extractions >= len(test_segments) * 0.8
        performance_reasonable = avg_time < 0.2  # Less than 200ms
        format_created = format_type in ['packet_based', 'legacy_ogg']
        
        overall_success = extraction_working and performance_reasonable and format_created
        
        print(f"\n6. Overall Assessment...")
        print(f"   Format created: {'✅' if format_created else '❌'} ({format_type})")
        print(f"   Extraction working: {'✅' if extraction_working else '❌'} ({successful_extractions}/{len(test_segments)})")
        print(f"   Performance: {'✅' if performance_reasonable else '❌'} ({avg_time:.4f}s avg)")
        
        if overall_success:
            print(f"\n🎉 SUCCESS: Opus extraction working!")
            
            if format_type == 'packet_based':
                print(f"   🚀 Packet-based format implemented successfully")
                if opuslib_available:
                    print(f"   ⚡ Direct opuslib decoding active")
                else:
                    print(f"   📦 ffmpeg fallback (install opuslib for optimal performance)")
            
            print(f"   📈 Performance: {perf_assessment} ({avg_time:.4f}s per segment)")
            
            if format_type == 'packet_based' and not opuslib_available:
                print(f"\n💡 Performance Tip:")
                print(f"   Install opuslib for 5-20x faster extraction:")
                print(f"   pip install opuslib")
                
        else:
            print(f"\n⚠️  ISSUES DETECTED:")
            if not format_created:
                print(f"   🔧 Format creation failed")
            if not extraction_working:
                print(f"   🔧 Extraction reliability issues")
            if not performance_reasonable:
                print(f"   🔧 Performance too slow")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fresh_import_and_extraction()
    print(f"\nFresh test {'PASSED' if success else 'FAILED'}")
