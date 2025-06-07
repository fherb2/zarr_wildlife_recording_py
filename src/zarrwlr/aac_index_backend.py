"""
AAC Index Backend Module - Frame-Level Index Creation
=====================================================

High-performance AAC frame parsing and index creation for random access.

AAC-LC FRAME STRUCTURE:
- Each AAC frame typically contains 1024 samples (~21.3ms at 48kHz)
- Frames are self-contained and can be decoded independently
- ADTS headers provide frame synchronization and metadata
- Much simpler than Opus packet structure (no complex state management)

KEY ADVANTAGES vs OPUS:
- Frame independence: No complex decoder state required
- Simpler frame detection: ADTS sync patterns are reliable
- Better compression: 160kbps vs Opus needing similar bitrates
- Lower overhead: Minimal decoder state vs 18KB for Opus
- Standard format: Universal AAC-LC compatibility

PERFORMANCE TARGET:
- Storage: ~8.8MB for 7min audio (vs 20.4MB WAV, 13MB FLAC)
- Index overhead: ~0.2MB (2.3% of compressed size)
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
logger.trace("AAC Index Backend module loading...")

# AAC Index constants
AAC_INDEX_DTYPE = np.uint64
AAC_INDEX_COLS = 6  # [byte_offset, frame_size, sample_pos, timestamp_ms, sample_count, frame_flags]
AAC_INDEX_COL_BYTE_OFFSET = 0
AAC_INDEX_COL_FRAME_SIZE = 1  
AAC_INDEX_COL_SAMPLE_POS = 2
AAC_INDEX_COL_TIMESTAMP_MS = 3
AAC_INDEX_COL_SAMPLE_COUNT = 4
AAC_INDEX_COL_FRAME_FLAGS = 5


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
# AAC Frame Analysis Classes
# ==========================
#
# ##########################################################

class AACFrameInfo:
    """Information about a single AAC frame"""
    def __init__(self, frame_index: int, byte_offset: int, frame_size: int, 
                 sample_count: int, timestamp_ms: int, is_keyframe: bool = True):
        self.frame_index = frame_index
        self.byte_offset = byte_offset
        self.frame_size = frame_size
        self.sample_count = sample_count  # Typically 1024 for AAC-LC
        self.timestamp_ms = timestamp_ms
        self.is_keyframe = is_keyframe  # AAC frames are generally all keyframes
        self.sample_position = None  # Will be calculated later
        
    def __repr__(self):
        return (f"AACFrameInfo(idx={self.frame_index}, offset={self.byte_offset}, "
                f"size={self.frame_size}, samples={self.sample_count}, "
                f"time={self.timestamp_ms}ms)")



# ##########################################################
#
# Index Creation Functions
# ========================
#
# ##########################################################

def _analyze_real_aac_frames(aac_data: bytes, sample_rate: int) -> List[dict]:
    """Echte AAC Frame-Analyse statt Placeholder"""
    frames = []
    pos = 0
    frame_idx = 0
    current_sample = 0
    
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
                    
                    if 7 <= frame_length <= 8192:  # Vernünftige Frame-Größe
                        frames.append({
                            'byte_offset': pos,
                            'frame_size': frame_length,
                            'sample_pos': current_sample,
                            'timestamp_ms': int(current_sample * 1000 / sample_rate),
                            'sample_count': 1024,  # Standard AAC
                            'frame_flags': 1
                        })
                        
                        pos += frame_length
                        current_sample += 1024
                        frame_idx += 1
                        continue
        
        pos += 1
    
    return frames



def build_aac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                   use_parallel: bool = True, max_workers: int = None) -> zarr.Array:
    """
    Create index for AAC frame access with optional parallelization
    
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
    logger.trace("build_aac_index() requested.")
    
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
    
    # Load audio bytes
    audio_bytes = bytes(audio_blob_array[()])
    
    # Real AAC frame analysis using ADTS parsing
    logger.trace("Analyzing real AAC ADTS frames...")
    frames_info_dicts = _analyze_real_aac_frames(audio_bytes, sample_rate)

    # Convert to AACFrameInfo objects for compatibility
    frame_infos = []
    for frame_dict in frames_info_dicts:
        # Create AACFrameInfo object if the class exists
        try:
            frame_info = AACFrameInfo(
                frame_index=len(frame_infos),
                byte_offset=frame_dict['byte_offset'],
                frame_size=frame_dict['frame_size'],
                sample_count=frame_dict['sample_count'],
                timestamp_ms=frame_dict['timestamp_ms'],
                is_keyframe=True
            )
            frame_infos.append(frame_info)
        except NameError:
            # Fallback if AACFrameInfo class doesn't exist
            class SimpleFrameInfo:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            frame_info = SimpleFrameInfo(
                byte_offset=frame_dict['byte_offset'],
                frame_size=frame_dict['frame_size'],
                sample_count=frame_dict['sample_count'],
                timestamp_ms=frame_dict['timestamp_ms'],
                is_keyframe=True,
                sample_position=0  # Will be set below
            )
            frame_infos.append(frame_info)

    logger.trace(f"Found {len(frame_infos)} real AAC frames")
    
    if len(frame_infos) < 1:
        raise ValueError("Could not find AAC frames in audio data")
    
    # Calculate cumulative sample positions
    logger.trace("Calculating cumulative sample positions...")
    current_sample_position = 0
    for frame_info in frame_infos:
        frame_info.sample_position = current_sample_position
        current_sample_position += frame_info.sample_count
    
    # Create index array
    logger.trace("Creating index array...")
    index_array = np.array([
        [
            frame_info.byte_offset,
            frame_info.frame_size,
            frame_info.sample_position,
            frame_info.timestamp_ms,
            frame_info.sample_count,
            1 if frame_info.is_keyframe else 0  # Frame flags
        ] 
        for frame_info in frame_infos
    ], dtype=AAC_INDEX_DTYPE)
    
    # Store index in Zarr group
    aac_index = zarr_group.create_array(
        name='aac_index',
        shape=index_array.shape,
        chunks=(min(1000, len(frame_infos)), AAC_INDEX_COLS),
        dtype=AAC_INDEX_DTYPE
    )
    
    # Write data to the created array
    aac_index[:] = index_array
    
    # Store metadata
    index_attrs = {
        'sample_rate': sample_rate,
        'channels': channels,
        'total_frames': len(frame_infos),
        'codec': codec,
        'aac_bitrate': bitrate,
        'container_type': 'aac-native',
        'frame_size_samples': frame_infos[0].sample_count if frame_infos else 1024,
        'total_samples': current_sample_position,
        'duration_ms': frame_infos[-1].timestamp_ms if frame_infos else 0
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
    logger.success(f"AAC index created with {len(frame_infos)} frames in {total_time:.3f}s")
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


def get_frame_info_by_time(aac_index: zarr.Array, timestamp_ms: int) -> Optional[Tuple[int, int, int]]:
    """
    Get frame information by timestamp
    
    Args:
        aac_index: AAC index array
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        Tuple of (frame_index, byte_offset, frame_size) or None if not found
    """
    timestamps = aac_index[:, AAC_INDEX_COL_TIMESTAMP_MS]
    
    frame_idx = np.searchsorted(timestamps, timestamp_ms, side='right') - 1
    frame_idx = max(0, min(frame_idx, aac_index.shape[0] - 1))
    
    frame_data = aac_index[frame_idx]
    return (
        frame_idx,
        int(frame_data[AAC_INDEX_COL_BYTE_OFFSET]),
        int(frame_data[AAC_INDEX_COL_FRAME_SIZE])
    )


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


# ##########################################################
#
# Performance Configuration and Optimization
# ===========================================
#
# ##########################################################

def configure_aac_processing(max_workers: int = None, enable_parallel: bool = True,
                            analysis_chunk_size: int = 1000) -> dict:
    """
    Configure AAC processing parameters
    
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
        'expected_frame_size': 1024,  # Standard AAC frame size
        'supported_sample_rates': [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000]
    }
    
    logger.trace(f"AAC processing configured: {config}")
    return config


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
        
        # Validate total samples calculation
        total_samples_calc = aac_index[-1, AAC_INDEX_COL_SAMPLE_POS] + aac_index[-1, AAC_INDEX_COL_SAMPLE_COUNT]
        expected_samples = aac_index.attrs.get('total_samples', 0)
        
        if expected_samples > 0 and abs(total_samples_calc - expected_samples) > 1024:  # Allow small discrepancy
            logger.warning(f"Sample count mismatch: calculated {total_samples_calc}, expected {expected_samples}")
        
        logger.trace("AAC index validation passed")
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


logger.trace("AAC Index Backend module loaded.")
    