"""
AAC Index Backend - Streamlined for Frame-Stream Direct Codec Access
====================================================================

Optimized AAC frame analysis and index creation for frame-stream direct codec parsing.
Focuses on ADTS frame analysis and efficient byte-range lookups.

OPTIMIZATION FOCUS:
1. 3-Column Structure: Minimal memory footprint (50% space savings)
2. ADTS Frame Analysis: Production-ready real frame parsing
3. Fast Lookups: Optimized for byte-range calculations
4. Frame-Stream Support: Index structure optimized for codec.parse()
5. Thread-Safe Caching: Minimal overhead for repeated access

ARCHITECTURE:
- Real ADTS frame analysis (no synthetic data)
- Calculated values for timestamps and sample counts
- Memory-efficient 3-column index format
- Optimized for Zarr v3 chunking patterns
"""

import zarr
import numpy as np
import time
import threading
from typing import List, Tuple, Optional, Dict

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("AAC Index Backend (Frame-Stream Direct Codec) loading...")

# AAC Index constants - 3-COLUMN OPTIMIZED STRUCTURE
AAC_INDEX_DTYPE = np.uint64
AAC_INDEX_COLS = 3
AAC_INDEX_COL_BYTE_OFFSET = 0
AAC_INDEX_COL_FRAME_SIZE = 1  
AAC_INDEX_COL_SAMPLE_POS = 2

# Constants for calculated values (not stored in index)
AAC_SAMPLES_PER_FRAME = 1024  # ffmpeg always produces 1024-sample frames


class OptimizedIndexCache:
    """
    Lightweight index cache for frame-stream direct codec access patterns
    
    Focuses on fast byte-range lookups with minimal memory overhead.
    """
    
    def __init__(self, cache_size: int = 10):
        self._cache = {}  # index_id -> cached_data
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0}
    
    def get_index_data(self, aac_index: zarr.Array) -> Dict[str, np.ndarray]:
        """
        Get cached index data optimized for frame-stream direct codec byte-range access
        """
        index_id = id(aac_index)
        
        with self._lock:
            # Cache hit
            if index_id in self._cache:
                self._stats['hits'] += 1
                return self._cache[index_id]
            
            # Cache miss - load and optimize for frame-stream direct codec access
            self._stats['misses'] += 1
            data = self._create_optimized_data(aac_index)
            
            # Simple LRU eviction
            if len(self._cache) >= self._cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[index_id] = data
            return data
    
    def _create_optimized_data(self, aac_index: zarr.Array) -> Dict[str, np.ndarray]:
        """Create optimized data structure for frame-stream direct codec access"""
        # Single efficient Zarr read
        full_index = aac_index[:]
        
        # Extract as contiguous arrays for fast access
        byte_offsets = np.ascontiguousarray(full_index[:, AAC_INDEX_COL_BYTE_OFFSET])
        frame_sizes = np.ascontiguousarray(full_index[:, AAC_INDEX_COL_FRAME_SIZE])
        sample_positions = np.ascontiguousarray(full_index[:, AAC_INDEX_COL_SAMPLE_POS])
        
        # Pre-compute frequently used values
        total_frames = len(sample_positions)
        total_samples = int(sample_positions[-1] + AAC_SAMPLES_PER_FRAME) if total_frames > 0 else 0
        sample_rate = aac_index.attrs.get('sample_rate', 48000)
        
        return {
            'byte_offsets': byte_offsets,
            'frame_sizes': frame_sizes,
            'sample_positions': sample_positions,
            'total_frames': total_frames,
            'total_samples': total_samples,
            'sample_rate': sample_rate,
            'metadata': dict(aac_index.attrs)
        }
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._stats = {'hits': 0, 'misses': 0}
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total)
            return {
                'hit_rate': hit_rate,
                'total_requests': total,
                'cached_indices': len(self._cache)
            }


# Global optimized cache
_index_cache = OptimizedIndexCache(cache_size=10)


# ##########################################################
#
# Calculated Value Functions (3-Column Optimization)
# =================================================
#
# ##########################################################

def get_aac_frame_samples() -> int:
    """Return samples per AAC frame - always 1024 for ffmpeg-encoded AAC"""
    return AAC_SAMPLES_PER_FRAME

def calculate_timestamp_ms(sample_pos: int, sample_rate: int) -> int:
    """Calculate timestamp in milliseconds from sample position"""
    return int(sample_pos * 1000 / sample_rate)


# ##########################################################
#
# Real ADTS Frame Analysis (Production Ready)
# ===========================================
#
# ##########################################################

def _analyze_adts_frames(aac_data: bytes, sample_rate: int) -> List[dict]:
    """
    Production-ready ADTS frame analysis optimized for frame-stream direct codec parsing
    
    Args:
        aac_data: Raw ADTS AAC data bytes
        sample_rate: Sample rate for calculations
        
    Returns:
        List of frame dictionaries with 3-column data structure
    """
    frames = []
    pos = 0
    frame_idx = 0
    
    logger.trace("Starting ADTS frame analysis...")
    
    while pos < len(aac_data) - 7:  # ADTS header minimum 7 bytes
        # Look for ADTS sync pattern
        if pos + 1 < len(aac_data):
            sync_word = int.from_bytes(aac_data[pos:pos+2], 'big')
            
            # ADTS sync pattern: 0xFFF (12 bits)
            if (sync_word & 0xFFF0) == 0xFFF0:
                # Parse ADTS header for frame length
                if pos + 6 < len(aac_data):
                    header = aac_data[pos:pos+7]
                    
                    # Extract frame length from ADTS header
                    # Bits 30-43 (13 bits) in ADTS header
                    frame_length = ((header[3] & 0x03) << 11) | \
                                (header[4] << 3) | \
                                ((header[5] & 0xE0) >> 5)
                    
                    # Validate frame size (reasonable range for AAC)
                    if 7 <= frame_length <= 8192:
                        # Store only essential 3-column data
                        frames.append({
                            'byte_offset': pos,
                            'frame_size': frame_length,
                            'sample_pos': frame_idx * AAC_SAMPLES_PER_FRAME
                        })
                        
                        pos += frame_length
                        frame_idx += 1
                        continue
        
        # Move to next byte if no valid frame found
        pos += 1
    
    logger.trace(f"ADTS analysis completed: {len(frames)} frames found")
    return frames


def build_aac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Create optimized 3-column AAC index for frame-stream direct codec access
    
    This function analyzes ADTS frames and creates an index optimized
    for byte-range lookups used by frame-stream direct codec parsing.
    """
    logger.trace("Building AAC index for frame-stream direct codec access...")
    
    # Extract metadata from array attributes
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 2)
    codec = audio_blob_array.attrs.get('codec', 'aac')
    bitrate = audio_blob_array.attrs.get('aac_bitrate', 160000)
    stream_type = audio_blob_array.attrs.get('stream_type', 'unknown')
    
    # Validation
    if codec != 'aac':
        raise ValueError(f"Expected AAC codec, got: {codec}")
    
    logger.trace(f"Index creation: {sample_rate}Hz, {channels}ch, {bitrate}bps, {stream_type}")
    
    start_time = time.time()
    
    # Load AAC bytes for analysis
    audio_bytes = bytes(audio_blob_array[()])
    logger.trace(f"Loaded {len(audio_bytes):,} bytes for frame analysis")
    
    # Real ADTS frame analysis
    logger.trace("Analyzing ADTS frames...")
    frames_info = _analyze_adts_frames(audio_bytes, sample_rate)
    
    if len(frames_info) < 1:
        raise ValueError("No ADTS frames found in audio data")
    
    # Create 3-column optimized index array
    logger.trace("Creating 3-column optimized index array...")
    index_array = np.array([
        [
            frame_dict['byte_offset'],
            frame_dict['frame_size'],
            frame_dict['sample_pos']
        ] 
        for frame_dict in frames_info
    ], dtype=AAC_INDEX_DTYPE)
    
    # Calculate optimal chunking for Zarr storage
    frames_count = len(frames_info)
    chunk_size = min(1000, max(100, frames_count // 10))  # Adaptive chunking
    
    # Store index in Zarr array with optimized chunking
    aac_index = zarr_group.create_array(
        name='aac_index',
        shape=index_array.shape,
        chunks=(chunk_size, AAC_INDEX_COLS),
        dtype=AAC_INDEX_DTYPE,
        overwrite=True
    )
    
    # Write data to Zarr array
    aac_index[:] = index_array
    
    # Calculate comprehensive metadata
    total_samples = frames_info[-1]['sample_pos'] + AAC_SAMPLES_PER_FRAME if frames_info else 0
    duration_ms = calculate_timestamp_ms(total_samples, sample_rate) if total_samples > 0 else 0
    
    # Calculate space savings
    old_size = frames_count * 6 * 8  # 6 columns * 8 bytes
    new_size = frames_count * 3 * 8  # 3 columns * 8 bytes  
    savings_bytes = old_size - new_size
    savings_percent = (savings_bytes / old_size) * 100 if old_size > 0 else 0
    
    # Store comprehensive metadata
    index_attrs = {
        # Core audio metadata
        'sample_rate': sample_rate,
        'channels': channels,
        'total_frames': frames_count,
        'codec': codec,
        'aac_bitrate': bitrate,
        'stream_type': stream_type,
        
        # Frame and timing information
        'frame_size_samples': AAC_SAMPLES_PER_FRAME,
        'total_samples': total_samples,
        'duration_ms': duration_ms,
        
        # Index format and optimization info
        'index_format_version': '3-column-optimized',
        'optimization': 'frame-stream-native',
        'space_savings_bytes': savings_bytes,
        'space_savings_percent': f"{savings_percent:.1f}%",
        
        # Frame-stream specific metadata
        'format': audio_blob_array.attrs.get('format', 'adts'),
        'frame_stream_ready': True,
        'byte_range_optimized': True
    }
    
    # Copy additional metadata from audio array
    optional_attrs = [
        'first_sample_time_stamp', 'last_sample_time_stamp',
        'profile', 'compression_type', 'threading_enabled'
    ]
    
    for attr_name in optional_attrs:
        if attr_name in audio_blob_array.attrs:
            index_attrs[attr_name] = audio_blob_array.attrs[attr_name]
    
    aac_index.attrs.update(index_attrs)
    
    total_time = time.time() - start_time
    
    logger.success(f"AAC index created: {frames_count:,} frames in {total_time:.3f}s")
    logger.success(f"Space optimization: {savings_bytes:,} bytes saved ({savings_percent:.1f}%)")
    logger.success(f"Frame-stream ready: {total_samples:,} samples, {duration_ms/1000:.1f}s duration")
    
    return aac_index


def find_frame_range_for_samples_fast(aac_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Ultra-fast frame range finding optimized for frame-stream direct codec byte-range access
    
    This function is critical for performance - it determines which bytes
    to load from Zarr for frame-stream direct codec parsing.
    """
    index_data = _index_cache.get_index_data(aac_index)
    sample_positions = index_data['sample_positions']
    
    # Bounds checking with cached values
    total_samples = index_data['total_samples']
    if start_sample < 0:
        start_sample = 0
    if end_sample >= total_samples:
        end_sample = total_samples - 1
    
    # Ultra-fast vectorized binary search
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, len(sample_positions) - 1)
    
    # For frame-stream codec: start one frame earlier for proper overlap
    overlap_start_idx = max(0, start_idx - 1)
    
    logger.trace(f"Frame range for samples [{start_sample}:{end_sample}]: frames [{overlap_start_idx}:{end_idx}]")
    return overlap_start_idx, end_idx


def get_index_statistics_fast(aac_index: zarr.Array) -> Dict[str, any]:
    """
    Get comprehensive index statistics using cached data
    """
    index_data = _index_cache.get_index_data(aac_index)
    
    frame_sizes = index_data['frame_sizes']
    sample_rate = index_data['sample_rate']
    total_frames = index_data['total_frames']
    total_samples = index_data['total_samples']
    
    # Fast vectorized statistics
    stats = {
        "index_format": "3-column-optimized-frame-stream",
        "optimization": "frame-stream-native",
        
        # Frame and sample information
        "total_frames": total_frames,
        "total_samples": total_samples,
        "duration_ms": calculate_timestamp_ms(total_samples, sample_rate),
        "samples_per_frame": AAC_SAMPLES_PER_FRAME,
        
        # Frame size statistics
        "frame_size_stats": {
            "min": int(np.min(frame_sizes)) if total_frames > 0 else 0,
            "max": int(np.max(frame_sizes)) if total_frames > 0 else 0,
            "mean": float(np.mean(frame_sizes)) if total_frames > 0 else 0.0,
            "std": float(np.std(frame_sizes)) if total_frames > 0 else 0.0
        },
        
        # Index efficiency
        "index_size_bytes": aac_index.nbytes,
        "space_savings_vs_6col": f"{((6-3)/6)*100:.1f}%",
        "bytes_per_frame": aac_index.nbytes / max(1, total_frames),
        
        # Audio metadata
        "sample_rate": sample_rate,
        "channels": index_data['metadata'].get('channels', 'unknown'),
        "bitrate": index_data['metadata'].get('aac_bitrate', 'unknown'),
        
        # Cache and optimization status
        "cache_status": "cached" if id(aac_index) in _index_cache._cache else "not_cached",
        "frame_stream_ready": index_data['metadata'].get('frame_stream_ready', False)
    }
    
    return stats


def validate_aac_index_fast(aac_index: zarr.Array, audio_blob_array: zarr.Array) -> bool:
    """
    Fast index validation for frame-stream direct codec compatibility
    """
    try:
        # Basic structure validation
        if aac_index.shape[1] != AAC_INDEX_COLS:
            logger.error(f"Invalid index structure: expected {AAC_INDEX_COLS} columns, got {aac_index.shape[1]}")
            return False
        
        # Check format version
        format_version = aac_index.attrs.get('index_format_version', '')
        if '3-column' not in format_version:
            logger.error(f"Invalid format version: {format_version}")
            return False
        
        # Get cached data for efficient validation
        index_data = _index_cache.get_index_data(aac_index)
        sample_positions = index_data['sample_positions']
        byte_offsets = index_data['byte_offsets']
        frame_sizes = index_data['frame_sizes']
        
        # Vectorized monotonic check
        if len(sample_positions) > 1 and not np.all(sample_positions[1:] >= sample_positions[:-1]):
            logger.error("Sample positions not monotonically increasing")
            return False
        
        # Vectorized bounds checking
        audio_size = audio_blob_array.shape[0]
        if len(byte_offsets) > 0:
            if np.any(byte_offsets >= audio_size):
                logger.error("Some byte offsets exceed audio data size")
                return False
            
            if np.any(frame_sizes <= 0) or np.any(frame_sizes > 8192):
                logger.error("Invalid frame sizes detected")
                return False
        
        # Frame-stream codec compatibility check
        stream_type = audio_blob_array.attrs.get('stream_type', '')
        if 'frame-stream' not in stream_type and 'adts' not in stream_type.lower():
            logger.warning(f"Stream type may not be optimal for frame-stream codec: {stream_type}")
        
        logger.trace("AAC index validation passed for frame-stream direct codec access")
        return True
        
    except Exception as e:
        logger.error(f"AAC index validation failed: {e}")
        return False


def benchmark_direct_codec_index(aac_index: zarr.Array, num_lookups: int = 1000) -> dict:
    """
    Benchmark index performance for frame-stream direct codec access patterns
    """
    total_samples = aac_index.attrs.get('total_samples', 0)
    if total_samples == 0:
        return {"error": "Invalid total samples"}
    
    # Generate realistic lookup patterns for frame-stream direct codec
    np.random.seed(42)
    lookup_requests = []
    for _ in range(num_lookups):
        # Typical segment length for frame-stream direct codec extraction
        segment_length = np.random.randint(1000, 5000)  # ~20-100ms
        start = np.random.randint(0, max(1, total_samples - segment_length))
        end = start + segment_length
        lookup_requests.append((start, end))
    
    logger.trace(f"Benchmarking {num_lookups} index lookups for frame-stream direct codec...")
    
    # Cold cache test
    _index_cache.clear()
    cold_times = []
    
    cold_start = time.time()
    for start_sample, end_sample in lookup_requests[:100]:  # Smaller sample for cold
        lookup_start = time.time()
        start_idx, end_idx = find_frame_range_for_samples_fast(aac_index, start_sample, end_sample)
        lookup_time = time.time() - lookup_start
        cold_times.append(lookup_time)
    cold_total = time.time() - cold_start
    
    # Warm cache test
    warm_times = []
    warm_start = time.time()
    for start_sample, end_sample in lookup_requests:
        lookup_start = time.time()
        start_idx, end_idx = find_frame_range_for_samples_fast(aac_index, start_sample, end_sample)
        lookup_time = time.time() - lookup_start
        warm_times.append(lookup_time)
    warm_total = time.time() - warm_start
    
    # Calculate statistics
    cold_mean_us = np.mean(cold_times) * 1000000
    warm_mean_us = np.mean(warm_times) * 1000000
    speedup = cold_mean_us / warm_mean_us if warm_mean_us > 0 else 1
    
    cache_stats = _index_cache.get_stats()
    
    results = {
        "optimization": "frame-stream-direct-codec-index",
        "total_lookups": num_lookups,
        "cold_cache": {
            "mean_microseconds": cold_mean_us,
            "total_time_seconds": cold_total,
            "lookups_tested": len(cold_times)
        },
        "warm_cache": {
            "mean_microseconds": warm_mean_us,
            "total_time_seconds": warm_total,
            "lookups_tested": len(warm_times)
        },
        "performance": {
            "cache_speedup": speedup,
            "lookups_per_second": len(warm_times) / warm_total,
            "cache_hit_rate": cache_stats['hit_rate']
        },
        "target_validation": {
            "target_lookup_microseconds": 5.0,
            "achieved_microseconds": warm_mean_us,
            "target_met": warm_mean_us <= 5.0
        }
    }
    
    logger.trace(f"Index benchmark: {warm_mean_us:.2f}μs mean, {speedup:.2f}x speedup")
    return results


# ##########################################################
#
# Public API Functions (Streamlined)
# =================================
#
# ##########################################################

def get_index_statistics(aac_index: zarr.Array) -> Dict[str, any]:
    """Get index statistics with automatic optimization"""
    return get_index_statistics_fast(aac_index)


def validate_aac_index(aac_index: zarr.Array, audio_blob_array: zarr.Array) -> bool:
    """Validate index with automatic optimization"""
    return validate_aac_index_fast(aac_index, audio_blob_array)


def clear_all_caches():
    """Clear all optimization caches"""
    _index_cache.clear()
    logger.trace("Index cache cleared")


def get_optimization_stats() -> Dict[str, any]:
    """Get optimization statistics for monitoring"""
    cache_stats = _index_cache.get_stats()
    
    return {
        'index_cache': cache_stats,
        'optimization_level': 'frame-stream-native',
        'approach': 'frame-stream-direct'
    }


def configure_optimization(index_cache_size: int = 10) -> None:
    """
    Configure optimization parameters for frame-stream direct codec access
    """
    global _index_cache
    
    if index_cache_size != _index_cache._cache_size:
        old_stats = _index_cache.get_stats()
        _index_cache = OptimizedIndexCache(cache_size=index_cache_size)
        logger.trace(f"Index cache reconfigured: size={index_cache_size}, previous_stats={old_stats}")


logger.trace("AAC Index Backend (Frame-Stream Direct Codec) loaded.")