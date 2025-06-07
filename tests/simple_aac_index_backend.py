"""
AAC Index Backend Module - Simplified Version for Testing
=========================================================

Simplified implementation for initial testing and development.
This module provides basic AAC frame indexing functionality.
"""

import zarr
import numpy as np
import time
from typing import List, Tuple, Optional, Dict

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("AAC Index Backend module loading (simplified version)...")

# AAC Index constants
AAC_INDEX_DTYPE = np.uint64
AAC_INDEX_COLS = 6  # [byte_offset, frame_size, sample_pos, timestamp_ms, sample_count, frame_flags]
AAC_INDEX_COL_BYTE_OFFSET = 0
AAC_INDEX_COL_FRAME_SIZE = 1  
AAC_INDEX_COL_SAMPLE_POS = 2
AAC_INDEX_COL_TIMESTAMP_MS = 3
AAC_INDEX_COL_SAMPLE_COUNT = 4
AAC_INDEX_COL_FRAME_FLAGS = 5


def build_aac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                   use_parallel: bool = True, max_workers: int = None) -> zarr.Array:
    """
    Create index for AAC frame access (simplified version)
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with AAC audio data
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Number of parallel workers (default: auto-detect)
        
    Returns:
        Created index array
        
    Raises:
        ValueError: If no AAC frames are found
    """
    logger.trace("build_aac_index() requested (simplified version).")
    
    # Extract metadata from array attributes
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 2)
    codec = audio_blob_array.attrs.get('codec', 'aac')
    bitrate = audio_blob_array.attrs.get('aac_bitrate', 160000)
    
    # Validation
    if codec != 'aac':
        raise ValueError(f"Expected AAC codec, but found: {codec}")
    
    logger.trace(f"Creating AAC index for: {sample_rate}Hz, {channels} channels, {bitrate}bps")
    
    start_time = time.time()
    
    # For now, create a dummy index with a few frames
    # This is a placeholder implementation for testing
    logger.warning("Using placeholder AAC index - not analyzing real frames yet")
    
    # Create dummy frame data (this would be replaced with real frame analysis)
    dummy_frames = []
    frame_size = 1024  # Standard AAC frame size
    current_sample = 0
    current_byte = 0
    
    # Create a few dummy frames based on audio size
    audio_size = audio_blob_array.shape[0]
    estimated_frame_count = max(1, audio_size // 1000)  # Rough estimate
    
    for i in range(min(estimated_frame_count, 100)):  # Limit for testing
        dummy_frames.append([
            current_byte,           # byte_offset
            1000,                   # frame_size (dummy)
            current_sample,         # sample_pos
            int(current_sample * 1000 / sample_rate),  # timestamp_ms
            frame_size,             # sample_count
            1                       # frame_flags (keyframe)
        ])
        
        current_byte += 1000
        current_sample += frame_size
        
        if current_byte >= audio_size:
            break
    
    if len(dummy_frames) < 1:
        raise ValueError("Could not create AAC frame index (no frames)")
    
    # Create index array
    logger.trace("Creating index array...")
    index_array = np.array(dummy_frames, dtype=AAC_INDEX_DTYPE)
    
    # Store index in Zarr group
    aac_index = zarr_group.create_array(
        name='aac_index',
        shape=index_array.shape,
        chunks=(min(1000, len(dummy_frames)), AAC_INDEX_COLS),
        dtype=AAC_INDEX_DTYPE
    )
    
    # Write data to the created array
    aac_index[:] = index_array
    
    # Store metadata
    index_attrs = {
        'sample_rate': sample_rate,
        'channels': channels,
        'total_frames': len(dummy_frames),
        'codec': codec,
        'aac_bitrate': bitrate,
        'container_type': 'aac-native',
        'frame_size_samples': frame_size,
        'total_samples': current_sample,
        'duration_ms': dummy_frames[-1][3] if dummy_frames else 0
    }
    
    # Copy additional metadata from audio_blob_array if available
    optional_attrs = [
        'first_sample_time_stamp', 'last_sample_time_stamp',
        'profile', 'compression_type'
    ]
    
    for attr_name in optional_attrs:
        if attr_name in audio_blob_array.attrs:
            index_attrs[attr_name] = audio_blob_array.attrs[attr_name]
    
    aac_index.attrs.update(index_attrs)
    
    total_time = time.time() - start_time
    logger.success(f"AAC index created with {len(dummy_frames)} dummy frames in {total_time:.3f}s")
    return aac_index


def _find_frame_range_for_samples(aac_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Find frame range for sample range using binary search
    
    Args:
        aac_index: AAC index array (shape: n_frames x 6)
        start_sample: First required sample
        end_sample: Last required sample
        
    Returns:
        Tuple (start_frame_idx, end_frame_idx)
    """
    sample_positions = aac_index[:, AAC_INDEX_COL_SAMPLE_POS]
    
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, aac_index.shape[0] - 1)
    
    return start_idx, end_idx


def get_index_statistics(aac_index: zarr.Array) -> Dict[str, any]:
    """
    Get statistics about the AAC index
    
    Args:
        aac_index: AAC index array
        
    Returns:
        Dictionary with index statistics
    """
    total_frames = aac_index.shape[0]
    
    if total_frames == 0:
        return {"total_frames": 0}
    
    frame_sizes = aac_index[:, AAC_INDEX_COL_FRAME_SIZE]
    sample_counts = aac_index[:, AAC_INDEX_COL_SAMPLE_COUNT]
    timestamps = aac_index[:, AAC_INDEX_COL_TIMESTAMP_MS]
    
    stats = {
        "total_frames": total_frames,
        "total_samples": int(aac_index[-1, AAC_INDEX_COL_SAMPLE_POS] + aac_index[-1, AAC_INDEX_COL_SAMPLE_COUNT]),
        "duration_ms": int(timestamps[-1]) if len(timestamps) > 0 else 0,
        "frame_size_stats": {
            "min": int(np.min(frame_sizes)),
            "max": int(np.max(frame_sizes)),
            "mean": float(np.mean(frame_sizes)),
            "std": float(np.std(frame_sizes))
        },
        "samples_per_frame": {
            "min": int(np.min(sample_counts)),
            "max": int(np.max(sample_counts)),
            "mean": float(np.mean(sample_counts))
        },
        "index_size_bytes": aac_index.nbytes,
        "sample_rate": aac_index.attrs.get('sample_rate', 'unknown'),
        "channels": aac_index.attrs.get('channels', 'unknown'),
        "bitrate": aac_index.attrs.get('aac_bitrate', 'unknown')
    }
    
    return stats


def validate_aac_index(aac_index: zarr.Array, audio_blob_array: zarr.Array) -> bool:
    """
    Validate AAC index integrity
    
    Args:
        aac_index: AAC index array to validate
        audio_blob_array: Original audio data array
        
    Returns:
        True if index is valid, False otherwise
    """
    try:
        # Basic structure validation
        if aac_index.shape[1] != AAC_INDEX_COLS:
            logger.error(f"Invalid index structure: expected {AAC_INDEX_COLS} columns, got {aac_index.shape[1]}")
            return False
        
        # Check if sample positions are monotonically increasing
        sample_positions = aac_index[:, AAC_INDEX_COL_SAMPLE_POS]
        if not np.all(sample_positions[1:] >= sample_positions[:-1]):
            logger.error("Sample positions are not monotonically increasing")
            return False
        
        # Check if byte offsets are reasonable
        byte_offsets = aac_index[:, AAC_INDEX_COL_BYTE_OFFSET]
        audio_size = audio_blob_array.shape[0]
        
        if np.any(byte_offsets >= audio_size):
            logger.error("Some byte offsets exceed audio data size")
            return False
        
        # Check frame sizes
        frame_sizes = aac_index[:, AAC_INDEX_COL_FRAME_SIZE]
        if np.any(frame_sizes <= 0) or np.any(frame_sizes > 8192):  # Reasonable AAC frame size limits
            logger.error("Invalid frame sizes detected")
            return False
        
        logger.trace("AAC index validation passed")
        return True
        
    except Exception as e:
        logger.error(f"AAC index validation failed: {e}")
        return False


def benchmark_aac_access(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                        num_extractions: int = 100) -> dict:
    """
    Benchmark AAC random access performance
    
    Args:
        zarr_group: Zarr group with AAC index
        audio_blob_array: Array with AAC audio data
        num_extractions: Number of random extractions to test
        
    Returns:
        Performance benchmark results
    """
    if 'aac_index' not in zarr_group:
        raise ValueError("AAC index not found")
    
    aac_index_array = zarr_group['aac_index']
    total_samples = int(aac_index_array[-1, AAC_INDEX_COL_SAMPLE_POS] + 
                       aac_index_array[-1, AAC_INDEX_COL_SAMPLE_COUNT])
    
    # Generate random extraction ranges
    np.random.seed(42)  # For reproducible results
    segment_length = 4410  # 100ms at 44.1kHz
    
    segments = []
    for _ in range(num_extractions):
        start = np.random.randint(0, max(1, total_samples - segment_length))
        end = min(start + segment_length, total_samples - 1)
        segments.append((start, end))
    
    # Import here to avoid circular imports
    from .aac_access import extract_audio_segment_aac
    
    # Benchmark extraction times
    extraction_times = []
    start_time = time.time()
    
    for start_sample, end_sample in segments:
        extraction_start = time.time()
        try:
            audio_data = extract_audio_segment_aac(
                zarr_group, audio_blob_array, start_sample, end_sample
            )
            extraction_time = time.time() - extraction_start
            extraction_times.append(extraction_time)
            
            if len(audio_data) == 0:
                logger.warning(f"Empty extraction for range [{start_sample}:{end_sample}]")
                
        except Exception as e:
            logger.error(f"Extraction failed for range [{start_sample}:{end_sample}]: {e}")
            extraction_times.append(float('inf'))
    
    total_benchmark_time = time.time() - start_time
    
    # Calculate statistics
    valid_times = [t for t in extraction_times if t != float('inf')]
    
    if not valid_times:
        return {"error": "No successful extractions"}
    
    results = {
        "total_extractions": num_extractions,
        "successful_extractions": len(valid_times),
        "total_time_seconds": total_benchmark_time,
        "extraction_times": {
            "min_ms": min(valid_times) * 1000,
            "max_ms": max(valid_times) * 1000,
            "mean_ms": np.mean(valid_times) * 1000,
            "median_ms": np.median(valid_times) * 1000,
            "std_ms": np.std(valid_times) * 1000
        },
        "performance_metrics": {
            "extractions_per_second": len(valid_times) / total_benchmark_time,
            "average_extraction_ms": np.mean(valid_times) * 1000,
            "success_rate": len(valid_times) / num_extractions
        },
        "index_info": get_index_statistics(aac_index_array)
    }
    
    logger.success(f"AAC benchmark completed: {results['performance_metrics']['average_extraction_ms']:.2f}ms average extraction time")
    return results


def diagnose_aac_data(audio_blob_array: zarr.Array) -> dict:
    """
    Diagnose AAC data for potential issues
    
    Args:
        audio_blob_array: Array with AAC audio data
        
    Returns:
        Diagnostic information
    """
    audio_bytes = bytes(audio_blob_array[()])
    
    diagnosis = {
        'size_bytes': len(audio_bytes),
        'size_mb': len(audio_bytes) / 1024 / 1024,
        'has_adts_headers': False,
        'estimated_frames': 0,
        'sync_patterns_found': 0,
        'issues': []
    }
    
    # Quick ADTS sync pattern count
    sync_count = 0
    pos = 0
    
    while pos < len(audio_bytes) - 2:
        sync_word = int.from_bytes(audio_bytes[pos:pos+2], 'big')
        if (sync_word & 0xFFF0) == 0xFFF0:  # ADTS sync pattern
            sync_count += 1
            diagnosis['has_adts_headers'] = True
            pos += 100  # Skip ahead to avoid false positives
        else:
            pos += 1
    
    diagnosis['sync_patterns_found'] = sync_count
    diagnosis['estimated_frames'] = sync_count
    
    if sync_count == 0:
        diagnosis['issues'].append("No ADTS sync patterns found - may not be ADTS format")
    
    # Check for reasonable AAC file size
    if len(audio_bytes) < 1000:
        diagnosis['issues'].append("File too small to contain meaningful AAC data")
    
    logger.trace(f"AAC diagnosis: {diagnosis}")
    return diagnosis


logger.trace("AAC Index Backend module loaded (simplified version).")