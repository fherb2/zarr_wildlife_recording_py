#!/usr/bin/env python3
"""
Test Opus Extraction Performance
================================

Test the packet-based extraction to see if we get the expected 5-20x speedup.
"""

import pathlib
import time
import datetime
import numpy as np
import shutil

import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"
OPUS_PACKET_INDEX_ARRAY_NAME = "opus_packet_index"
OPUS_PACKETS_BLOB_ARRAY_NAME = "opus_packets_blob"

def test_extraction_performance():
    """Test extraction performance and correctness"""
    print("=" * 60)
    print("OPUS EXTRACTION PERFORMANCE TEST")
    print("=" * 60)
    
    Config.set(log_level=LogLevel.INFO)
    test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
    
    # Find existing test store (from previous debug run)
    store_dirs = list(test_results_dir.glob("zarr3-store-debug-*"))
    if not store_dirs:
        print("❌ No test stores found. Run debug_opus_performance.py first")
        return False
    
    # Use most recent store
    latest_store = max(store_dirs, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using test store: {latest_store.name}")
    
    try:
        # Open existing store
        store = zarrwlr.create_original_audio_group(
            store_path=latest_store,
            group_path='audio_imports'
        )
        
        # Find imported audio group
        group_names = [name for name in store.keys() if name.isdigit()]
        if not group_names:
            print("❌ No audio groups found in store")
            return False
        
        imported_group = store[max(group_names, key=int)]
        print(f"📦 Found audio group: {max(group_names, key=int)}")
        
        # Detect format
        has_packet_format = (
            OPUS_PACKETS_BLOB_ARRAY_NAME in imported_group and
            OPUS_PACKET_INDEX_ARRAY_NAME in imported_group
        )
        
        has_legacy_format = (
            AUDIO_DATA_BLOB_ARRAY_NAME in imported_group and
            'opus_index' in imported_group
        )
        
        if has_packet_format:
            format_type = 'packet_based'
            print("✅ Packet-based format detected")
        elif has_legacy_format:
            format_type = 'legacy_ogg'
            print("📼 Legacy OGG format detected")
        else:
            print("❌ Unknown format")
            return False
        
        # Check for opuslib
        try:
            import opuslib
            opuslib_available = True
            print("✅ opuslib available for direct decoding")
        except ImportError:
            opuslib_available = False
            print("⚠️  opuslib not available - will use ffmpeg fallback")
        
        # Get audio info
        if format_type == 'packet_based':
            packet_index = imported_group[OPUS_PACKET_INDEX_ARRAY_NAME]
            estimated_samples = packet_index.attrs.get('estimated_total_samples', 1000)
            total_packets = packet_index.shape[0]
            print(f"📊 Audio info: {estimated_samples} samples, {total_packets} packets")
        else:
            opus_index = imported_group['opus_index']
            sample_positions = opus_index[:, 2]
            estimated_samples = int(sample_positions[-1]) if len(sample_positions) > 0 else 1000
            total_pages = opus_index.shape[0]
            print(f"📊 Audio info: {estimated_samples} samples, {total_pages} pages")
        
        # Test 1: Single extraction
        print(f"\n1. Single Extraction Test...")
        test_start = 0
        test_end = min(999, estimated_samples - 1)
        
        print(f"   Extracting segment [{test_start}:{test_end}]...")
        
        start_time = time.time()
        extracted_audio = zarrwlr.extract_audio_segment(
            imported_group, test_start, test_end, dtype=np.int16
        )
        single_extraction_time = time.time() - start_time
        
        print(f"   Result: {extracted_audio.shape} in {single_extraction_time:.4f}s")
        
        if extracted_audio.size == 0:
            print("   ❌ No audio extracted")
            return False
        
        # Check audio quality
        has_audio = np.any(extracted_audio != 0)
        dynamic_range = extracted_audio.max() - extracted_audio.min() if extracted_audio.size > 0 else 0
        
        print(f"   Audio quality: {'✅' if has_audio else '❌'} has_audio, dynamic_range: {dynamic_range}")
        
        # Test 2: Multiple small extractions (key performance test)
        print(f"\n2. Multiple Small Extractions Test...")
        
        segment_size = 100  # Small segments (where we expect speedup)
        num_segments = min(10, estimated_samples // segment_size)
        
        test_segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = min(start + segment_size - 1, estimated_samples - 1)
            test_segments.append((start, end))
        
        print(f"   Testing {len(test_segments)} segments of {segment_size} samples each...")
        
        start_time = time.time()
        successful_extractions = 0
        
        for i, (start, end) in enumerate(test_segments):
            try:
                segment = zarrwlr.extract_audio_segment(imported_group, start, end, dtype=np.int16)
                if segment.size > 0:
                    successful_extractions += 1
            except Exception as e:
                print(f"   Segment {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        avg_time_per_segment = total_time / len(test_segments) if test_segments else 0
        
        print(f"   Results: {successful_extractions}/{len(test_segments)} successful")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per segment: {avg_time_per_segment:.4f}s")
        
        # Test 3: Parallel extraction
        print(f"\n3. Parallel Extraction Test...")
        
        if len(test_segments) >= 4:
            start_time = time.time()
            try:
                parallel_results = zarrwlr.parallel_extract_audio_segments(
                    imported_group, test_segments[:4], dtype=np.int16, max_workers=2
                )
                parallel_time = time.time() - start_time
                
                parallel_successful = sum(1 for r in parallel_results if r.size > 0)
                avg_parallel_time = parallel_time / 4
                
                print(f"   Results: {parallel_successful}/4 successful")
                print(f"   Total time: {parallel_time:.3f}s")
                print(f"   Average per segment: {avg_parallel_time:.4f}s")
                
            except Exception as e:
                print(f"   ❌ Parallel extraction failed: {e}")
                parallel_time = float('inf')
                avg_parallel_time = float('inf')
        else:
            print(f"   Skipped (not enough segments)")
            parallel_time = single_extraction_time
            avg_parallel_time = single_extraction_time
        
        # Performance Analysis
        print(f"\n4. Performance Analysis...")
        
        # Expected performance thresholds
        expected_single_time = 0.1    # 100ms reasonable for single extraction
        expected_batch_time = 0.05    # 50ms for batch with packet-based + opuslib
        
        # Assess single extraction
        if single_extraction_time < expected_single_time:
            single_perf = "Excellent"
            single_emoji = "⚡"
        elif single_extraction_time < 0.2:
            single_perf = "Good"
            single_emoji = "✅"
        else:
            single_perf = "Slow"
            single_emoji = "🐌"
        
        print(f"   Single extraction: {single_emoji} {single_perf} ({single_extraction_time:.4f}s)")
        
        # Assess batch extraction
        if avg_time_per_segment < expected_batch_time:
            batch_perf = "Excellent"
            batch_emoji = "⚡"
        elif avg_time_per_segment < 0.1:
            batch_perf = "Good"
            batch_emoji = "✅"
        else:
            batch_perf = "Slow"
            batch_emoji = "🐌"
        
        print(f"   Batch extraction: {batch_emoji} {batch_perf} ({avg_time_per_segment:.4f}s avg)")
        
        # Implementation assessment
        print(f"\n5. Implementation Assessment...")
        
        if format_type == 'packet_based' and opuslib_available:
            expected_mode = "Optimal (packet-based + opuslib)"
            expected_performance = "High"
        elif format_type == 'packet_based':
            expected_mode = "Good (packet-based + ffmpeg fallback)"
            expected_performance = "Medium"
        else:
            expected_mode = "Legacy (OGG + ffmpeg)"
            expected_performance = "Standard"
        
        print(f"   Mode: {expected_mode}")
        print(f"   Expected performance: {expected_performance}")
        
        # Overall success criteria
        extraction_working = successful_extractions >= len(test_segments) * 0.8  # 80% success rate
        performance_reasonable = avg_time_per_segment < 0.2  # Less than 200ms per segment
        audio_quality_ok = has_audio and dynamic_range > 100
        
        overall_success = extraction_working and performance_reasonable and audio_quality_ok
        
        print(f"\n6. Overall Result...")
        print(f"   Extraction working: {'✅' if extraction_working else '❌'}")
        print(f"   Performance reasonable: {'✅' if performance_reasonable else '❌'}")
        print(f"   Audio quality OK: {'✅' if audio_quality_ok else '❌'}")
        
        if overall_success:
            print(f"\n🎉 SUCCESS: Opus extraction working correctly!")
            
            if format_type == 'packet_based':
                print(f"   🚀 Packet-based implementation active")
                if opuslib_available:
                    print(f"   ⚡ opuslib direct decoding enabled")
                else:
                    print(f"   📦 Using ffmpeg fallback (consider installing opuslib)")
            
            if avg_time_per_segment < 0.05:
                print(f"   🔥 Excellent performance achieved!")
            elif avg_time_per_segment < 0.1:
                print(f"   ✅ Good performance")
            else:
                print(f"   📊 Acceptable performance")
                
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Some issues detected")
            if not extraction_working:
                print(f"   🔧 Fix extraction reliability")
            if not performance_reasonable:
                print(f"   🔧 Improve performance (current: {avg_time_per_segment:.4f}s)")
            if not audio_quality_ok:
                print(f"   🔧 Check audio quality")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_extraction_performance()
    print(f"\nExtraction test {'PASSED' if success else 'FAILED'}")
