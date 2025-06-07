"""
AAC Index Backend Module - Optimized 3-Column Frame Index
=========================================================

High-performance AAC frame parsing and index creation for random access.
OPTIMIZED VERSION with 3-column index structure for minimal overhead.

AAC-LC FRAME STRUCTURE:
- Each AAC frame contains exactly 1024 samples (~21.3ms at 48kHz) when encoded with ffmpeg
- Frames are self-contained and can be decoded independently with minimal overlap
- ADTS headers provide frame synchronization and metadata
- Much simpler than Opus packet structure (no complex state management)

OPTIMIZED INDEX STRUCTURE (3 columns):
- byte_offset: Position in AAC stream (uint64)
- frame_size: Size in bytes (uint32) 
- sample_pos: Cumulative sample position (uint64)

CALCULATED VALUES (not stored):
- sample_count: Always 1024 (ffmpeg standard)
- timestamp_ms: Calculated from sample_pos and sample_rate
- frame_flags: Not needed (all frames are effectively keyframes)

PERFORMANCE TARGET:
- Storage: ~8.8MB for 7min audio (vs 20.4MB WAV, 13MB FLAC)
- Index overhead: ~0.14MB (1.6% of compressed size) - 28% reduction vs 6-column
- Random access: ~20ms target (vs ~200ms sequential decode)
"""

import zarr
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict
import hashlib
import av
import tempfile
import pathlib

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("AAC Index Backend module (optimized 3-column) loading...")

# AAC Index constants - OPTIMIZED 3-COLUMN STRUCTURE
AAC_INDEX_DTYPE = np.uint64
AAC_INDEX_COLS = 3  # REDUCED from 6 to 3 columns
AAC_INDEX_COL_BYTE_OFFSET = 0
AAC_INDEX_COL_FRAME_SIZE = 1  
AAC_INDEX_COL_SAMPLE_POS = 2

# Constants for calculated values
AAC_SAMPLES_PER_FRAME = 1024  # ffmpeg always produces 1024-sample frames


# ##########################################################
#
# Memory Monitoring Helper
# ========================
#
# ##########################################################

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class MemoryStats:
    """Helper für Memory-Monitoring (mit psutil Fallback)"""
    
    @staticmethod
    def get_current_memory_mb():
        """Aktueller RAM-Verbrauch in MB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return 0.0  # Fallback


# ##########################################################
#
# Calculated Value Functions (instead of storing)
# ==============================================
#
# ##########################################################

def get_aac_frame_samples() -> int:
    """Return samples per AAC frame - always 1024 for ffmpeg-encoded AAC"""
    return AAC_SAMPLES_PER_FRAME

def calculate_timestamp_ms(sample_pos: int, sample_rate: int) -> int:
    """Calculate timestamp in milliseconds from sample position"""
    return int(sample_pos * 1000 / sample_rate)

def calculate_frame_timestamp_ms(frame_idx: int, sample_rate: int) -> int:
    """Calculate timestamp for a specific frame index"""
    sample_pos = frame_idx * AAC_SAMPLES_PER_FRAME
    return calculate_timestamp_ms(sample_pos, sample_rate)

def get_sample_position_for_frame(frame_idx: int) -> int:
    """Calculate sample position for a given frame index"""
    return frame_idx * AAC_SAMPLES_PER_FRAME


# ##########################################################
#
# AAC Frame Analysis Classes
# ==========================
#
# ##########################################################

class AACFrameInfo:
    """Information about a single AAC frame - optimized structure"""
    def __init__(self, frame_index: int, byte_offset: int, frame_size: int):
        self.frame_index = frame_index
        self.byte_offset = byte_offset
        self.frame_size = frame_size
        # Calculated properties (not stored)
        self._sample_position = None
        
    @property
    def sample_count(self) -> int:
        """Always returns 1024 for ffmpeg-encoded AAC"""
        return get_aac_frame_samples()
    
    @property 
    def sample_position(self) -> int:
        """Calculate sample position from frame index"""
        if self._sample_position is None:
            self._sample_position = get_sample_position_for_frame(self.frame_index)
        return self._sample_position
    
    def timestamp_ms(self, sample_rate: int) -> int:
        """Calculate timestamp for this frame"""
        return calculate_timestamp_ms(self.sample_position, sample_rate)
        
    def __repr__(self):
        return (f"AACFrameInfo(idx={self.frame_index}, offset={self.byte_offset}, "
                f"size={self.frame_size}, samples={self.sample_count}, "
                f"sample_pos={self.sample_position})")


# ##########################################################
#
# Index Creation Functions
# ========================
#
# ##########################################################

def _analyze_real_aac_frames(aac_data: bytes, sample_rate: int) -> List[dict]:
    """Real AAC Frame Analysis - returns minimal data for 3-column index"""
    frames = []
    pos = 0
    frame_idx = 0
    
    while pos < len(aac_data) - 7:  # ADTS header = 7 bytes minimum
        # Suche ADTS Sync Pattern (0xFFF)
        if pos + 1 < len(aac_data):
            sync_word = int.from_bytes(aac_data[pos:pos+2], 'big')
            
            if (sync_word & 0xFFF0) == 0xFFF0:  # ADTS sync
                # Parse ADTS header für Frame-Länge
                if pos + 6 < len(aac_data):
                    header = aac_data[pos:pos+7]
                    frame_length = ((header[3] & 0x03) << 11) | \
                                (header[4] << 3) | \
                                ((header[5] & 0xE0) >> 5)
                    
                    if 7 <= frame_length <= 16384:  # Vernünftige Frame-Größe
                        # OPTIMIZED: Only store essential data for 3-column index
                        frames.append({
                            'byte_offset': pos,
                            'frame_size': frame_length,
                            'sample_pos': frame_idx * AAC_SAMPLES_PER_FRAME  # Calculated
                        })
                        
                        pos += frame_length
                        frame_idx += 1
                        continue
        
        pos += 1
    
    logger.trace(f"Analyzed {len(frames)} AAC frames with 3-column optimization")
    return frames


def build_aac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                   use_parallel: bool = True, max_workers: int = None) -> zarr.Array:
    """
    Create OPTIMIZED 3-column index for AAC frame access
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with AAC audio data
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Number of parallel workers (default: auto-detect)
        
    Returns:
        Created index array with 3 columns: [byte_offset, frame_size, sample_pos]
        
    Raises:
        ValueError: If no AAC frames are found
    """
    logger.trace("build_aac_index() requested with 3-column optimization.")
    
    # Extract metadata from array attributes
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 2)
    codec = audio_blob_array.attrs.get('codec', 'aac')
    bitrate = audio_blob_array.attrs.get('aac_bitrate', 160000)
    
    # Validation
    if codec != 'aac':
        raise ValueError(f"Expected AAC codec, but found: {codec}")
    
    logger.trace(f"Creating OPTIMIZED AAC index for: {sample_rate}Hz, {channels} channels, {bitrate}bps")
    
    start_time = time.time()
    
    # Load audio bytes
    audio_bytes = bytes(audio_blob_array[()])
    
    # Real AAC frame analysis using ADTS parsing
    logger.trace("Analyzing real AAC ADTS frames for 3-column index...")
    frames_info_dicts = _analyze_real_aac_frames(audio_bytes, sample_rate)

    logger.trace(f"Found {len(frames_info_dicts)} real AAC frames")
    
    if len(frames_info_dicts) < 1:
        raise ValueError("Could not find AAC frames in audio data")
    
    # Create OPTIMIZED 3-column index array
    logger.trace("Creating OPTIMIZED 3-column index array...")
    index_array = np.array([
        [
            frame_dict['byte_offset'],
            frame_dict['frame_size'],
            frame_dict['sample_pos']
        ] 
        for frame_dict in frames_info_dicts
    ], dtype=AAC_INDEX_DTYPE)
    
    logger.trace(f"Index array shape: {index_array.shape} (3-column optimization)")
    
    # Store index in Zarr group
    aac_index = zarr_group.create_array(
        name='aac_index',
        shape=index_array.shape,
        chunks=(min(1000, len(frames_info_dicts)), AAC_INDEX_COLS),
        dtype=AAC_INDEX_DTYPE
    )
    
    # Write data to the created array
    aac_index[:] = index_array
    
    # Store metadata - ESSENTIAL: include sample_rate for calculations
    total_samples = frames_info_dicts[-1]['sample_pos'] + AAC_SAMPLES_PER_FRAME if frames_info_dicts else 0
    duration_ms = calculate_timestamp_ms(total_samples, sample_rate) if total_samples > 0 else 0
    
    index_attrs = {
        'sample_rate': sample_rate,  # REQUIRED for timestamp calculations
        'channels': channels,
        'total_frames': len(frames_info_dicts),
        'codec': codec,
        'aac_bitrate': bitrate,
        'container_type': 'aac-native',
        'frame_size_samples': AAC_SAMPLES_PER_FRAME,  # Always 1024
        'total_samples': total_samples,
        'duration_ms': duration_ms,
        'index_format_version': '3-column-optimized'  # Version marker
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
    
    # Calculate space savings
    old_size = len(frames_info_dicts) * 6 * 8  # 6 columns * 8 bytes
    new_size = len(frames_info_dicts) * 3 * 8  # 3 columns * 8 bytes  
    savings_bytes = old_size - new_size
    savings_percent = (savings_bytes / old_size) * 100 if old_size > 0 else 0
    
    logger.success(f"OPTIMIZED AAC index created: {len(frames_info_dicts)} frames in {total_time:.3f}s")
    logger.success(f"Index size reduction: {savings_bytes} bytes ({savings_percent:.1f}% smaller)")
    return aac_index


def _find_frame_range_for_samples(aac_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Find frame range for sample range using binary search - OPTIMIZED for 3-column index
    
    Args:
        aac_index: AAC index array (shape: n_frames x 3)
        start_sample: First required sample
        end_sample: Last required sample
        
    Returns:
        Tuple (start_frame_idx, end_frame_idx) with overlap handling
    """
    sample_positions = aac_index[:, AAC_INDEX_COL_SAMPLE_POS]
    
    # Find frames that contain the requested samples
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, aac_index.shape[0] - 1)
    
    # OVERLAP HANDLING: Start one frame earlier if possible
    overlap_start_idx = max(0, start_idx - 1)
    
    logger.trace(f"Frame range for samples [{start_sample}:{end_sample}]: "
                f"frames [{overlap_start_idx}:{end_idx}] (with overlap)")
    
    return overlap_start_idx, end_idx


def get_frame_info_by_time(aac_index: zarr.Array, timestamp_ms: int) -> Optional[Tuple[int, int, int]]:
    """
    Get frame information by timestamp - OPTIMIZED for 3-column index
    
    Args:
        aac_index: AAC index array
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        Tuple of (frame_index, byte_offset, frame_size) or None if not found
    """
    # Get sample rate from index metadata
    sample_rate = aac_index.attrs.get('sample_rate', 48000)
    
    # Convert timestamp to sample position
    target_sample = int(timestamp_ms * sample_rate / 1000)
    
    # Find frame using sample positions
    sample_positions = aac_index[:, AAC_INDEX_COL_SAMPLE_POS]
    frame_idx = np.searchsorted(sample_positions, target_sample, side='right') - 1
    frame_idx = max(0, min(frame_idx, aac_index.shape[0] - 1))
    
    frame_data = aac_index[frame_idx]
    return (
        frame_idx,
        int(frame_data[AAC_INDEX_COL_BYTE_OFFSET]),
        int(frame_data[AAC_INDEX_COL_FRAME_SIZE])
    )


def get_index_statistics(aac_index: zarr.Array) -> Dict[str, any]:
    """
    Get statistics about the OPTIMIZED AAC index
    
    Args:
        aac_index: AAC index array (3-column format)
        
    Returns:
        Dictionary with index statistics
    """
    total_frames = aac_index.shape[0]
    
    if total_frames == 0:
        return {"total_frames": 0}
    
    frame_sizes = aac_index[:, AAC_INDEX_COL_FRAME_SIZE]
    sample_positions = aac_index[:, AAC_INDEX_COL_SAMPLE_POS]
    
    # Calculate values that were previously stored
    sample_rate = aac_index.attrs.get('sample_rate', 48000)
    total_samples = int(sample_positions[-1] + AAC_SAMPLES_PER_FRAME) if len(sample_positions) > 0 else 0
    duration_ms = calculate_timestamp_ms(total_samples, sample_rate)
    
    stats = {
        "total_frames": total_frames,
        "total_samples": total_samples,
        "duration_ms": duration_ms,
        "frame_size_stats": {
            "min": int(np.min(frame_sizes)),
            "max": int(np.max(frame_sizes)),
            "mean": float(np.mean(frame_sizes)),
            "std": float(np.std(frame_sizes))
        },
        "samples_per_frame": {
            "fixed_value": AAC_SAMPLES_PER_FRAME,
            "description": "Always 1024 for ffmpeg-encoded AAC"
        },
        "index_size_bytes": aac_index.nbytes,
        "index_format": "3-column-optimized",
        "space_savings_vs_6col": f"{((6-3)/6)*100:.1f}%",
        "sample_rate": aac_index.attrs.get('sample_rate', 'unknown'),
        "channels": aac_index.attrs.get('channels', 'unknown'),
        "bitrate": aac_index.attrs.get('aac_bitrate', 'unknown')
    }
    
    return stats


# ##########################################################
#
# Performance Configuration and Optimization
# ===========================================
#
# ##########################################################

def configure_aac_processing(max_workers: int = None, enable_parallel: bool = True,
                            analysis_chunk_size: int = 1000) -> dict:
    """
    Configure AAC processing parameters for 3-column optimization
    
    Args:
        max_workers: Maximum number of worker processes (default: auto-detect)
        enable_parallel: Enable/disable parallel processing globally
        analysis_chunk_size: Chunk size for parallel frame analysis
        
    Returns:
        Configuration dictionary
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # AAC is less CPU intensive than FLAC
    
    config = {
        'max_workers': max_workers,
        'enable_parallel': enable_parallel,
        'analysis_chunk_size': analysis_chunk_size,
        'cpu_count': mp.cpu_count(),
        'psutil_available': PSUTIL_AVAILABLE,
        'expected_frame_size': AAC_SAMPLES_PER_FRAME,
        'index_format': '3-column-optimized',
        'supported_sample_rates': [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000]
    }
    
    logger.trace(f"AAC processing configured for 3-column index: {config}")
    return config


def validate_aac_index(aac_index: zarr.Array, audio_blob_array: zarr.Array) -> bool:
    """
    Validate OPTIMIZED AAC index integrity
    
    Args:
        aac_index: AAC index array to validate (3-column format)
        audio_blob_array: Original audio data array
        
    Returns:
        True if index is valid, False otherwise
    """
    try:
        # Basic structure validation for 3-column format
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
        
        # Validate total samples calculation using 3-column format
        if aac_index.shape[0] > 0:
            calculated_total_samples = int(sample_positions[-1] + AAC_SAMPLES_PER_FRAME)
            expected_samples = aac_index.attrs.get('total_samples', 0)
            
            if expected_samples > 0 and abs(calculated_total_samples - expected_samples) > AAC_SAMPLES_PER_FRAME:
                logger.warning(f"Sample count mismatch: calculated {calculated_total_samples}, expected {expected_samples}")
        
        # Validate that sample positions match frame index * 1024
        for i, sample_pos in enumerate(sample_positions[:10]):  # Check first 10 frames
            expected_pos = i * AAC_SAMPLES_PER_FRAME
            if sample_pos != expected_pos:
                logger.error(f"Frame {i} sample position mismatch: got {sample_pos}, expected {expected_pos}")
                return False
        
        logger.trace("OPTIMIZED AAC index validation passed")
        return True
        
    except Exception as e:
        logger.error(f"AAC index validation failed: {e}")
        return False


# ##########################################################
#
# Benchmark and Diagnostics
# =========================
#
# ##########################################################

def benchmark_aac_access(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                        num_extractions: int = 100) -> dict:
    """
    Benchmark AAC random access performance with OPTIMIZED 3-column index
    
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
    sample_positions = aac_index_array[:, AAC_INDEX_COL_SAMPLE_POS]
    total_samples = int(sample_positions[-1] + AAC_SAMPLES_PER_FRAME) if len(sample_positions) > 0 else 0
    
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
        "index_info": get_index_statistics(aac_index_array),
        "optimization_note": "Using 3-column optimized index format"
    }
    
    logger.success(f"AAC benchmark completed: {results['performance_metrics']['average_extraction_ms']:.2f}ms average extraction time")
    return results


def diagnose_aac_data(audio_blob_array: zarr.Array) -> dict:
    """
    Diagnose AAC data for potential issues - enhanced for 3-column optimization
    
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
        'optimization_format': '3-column-index',
        'expected_samples_per_frame': AAC_SAMPLES_PER_FRAME,
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
    
    # Calculate expected index overhead with 3-column optimization
    if sync_count > 0:
        old_index_size = sync_count * 6 * 8  # 6 columns * 8 bytes
        new_index_size = sync_count * 3 * 8  # 3 columns * 8 bytes
        diagnosis['index_overhead_comparison'] = {
            'old_6col_bytes': old_index_size,
            'new_3col_bytes': new_index_size,
            'savings_bytes': old_index_size - new_index_size,
            'savings_percent': ((old_index_size - new_index_size) / old_index_size) * 100
        }
    
    logger.trace(f"AAC diagnosis (3-column optimized): {diagnosis}")
    return diagnosis


logger.trace("AAC Index Backend module (optimized 3-column) loaded.")