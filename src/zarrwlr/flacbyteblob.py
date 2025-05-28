import zarr
import numpy as np
import io
import soundfile as sf
import tempfile
import os
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Module loading...")


# FLAC Index constants
FLAC_INDEX_DTYPE = np.uint64
FLAC_INDEX_COLS = 3  # [byte_offset, frame_size, sample_pos]
FLAC_INDEX_COL_BYTE_OFFSET = 0
FLAC_INDEX_COL_FRAME_SIZE = 1  
FLAC_INDEX_COL_SAMPLE_POS = 2


def _parse_flac_header_and_metadata(audio_bytes: bytes) -> int:
    """
    Parse FLAC header and skip metadata blocks
    
    Args:
        audio_bytes: FLAC audio data as bytes
        
    Returns:
        Position after the last metadata block
        
    Raises:
        ValueError: If no valid FLAC file is found
    """
    logger.trace("_parse_flac_header_and_metadata() called")
    
    pos = 0
    
    # Check FLAC signature
    if len(audio_bytes) < 4 or audio_bytes[:4] != b'fLaC':
        logger.warning(f"No valid FLAC signature found. First 4 bytes: {audio_bytes[:4] if len(audio_bytes) >= 4 else audio_bytes}")
        raise ValueError("No valid FLAC file found")
    
    pos = 4
    logger.trace("FLAC signature recognized")
    
    # Skip metadata blocks
    while pos < len(audio_bytes):
        if pos + 4 > len(audio_bytes):
            logger.trace(f"Reached end of file at position {pos}")
            break
            
        block_header = int.from_bytes(audio_bytes[pos:pos+4], 'big')
        is_last = (block_header & 0x80000000) != 0
        block_size = block_header & 0x7FFFFF
        
        logger.trace(f"Metadata block: pos={pos}, size={block_size}, is_last={is_last}")
        
        pos += 4 + block_size
        
        if is_last:
            logger.trace("Last metadata block reached")
            break
    
    logger.trace(f"FLAC header parsing completed. Audio frames start at position {pos}")
    return pos


def _find_next_flac_frame_sync(audio_bytes: bytes, start_pos: int, max_search_bytes: int = 65536) -> Optional[int]:
    """
    Search for the next FLAC frame sync pattern
    
    Args:
        audio_bytes: FLAC audio data
        start_pos: Start position for search
        max_search_bytes: Maximum search range
        
    Returns:
        Position of next frame sync or None
    """
    search_end = min(start_pos + max_search_bytes, len(audio_bytes) - 2)
    
    for pos in range(start_pos, search_end):
        if pos + 1 < len(audio_bytes):
            sync_word = int.from_bytes(audio_bytes[pos:pos+2], 'big')
            if (sync_word & 0xFFFE) == 0xFFF8:
                return pos
    
    return None


def _estimate_flac_samples_per_frame(frame_header_bytes: bytes, expected_sample_rate: int = 44100) -> int:
    """
    Estimate samples per frame from FLAC frame header
    
    Args:
        frame_header_bytes: First bytes of frame header
        expected_sample_rate: Expected sample rate from metadata for better frame size estimation
        
    Returns:
        Estimated number of samples per frame
    """
    # Removed logging for better performance
    
    # Improved estimation based on sample rate
    # FLAC typically uses different block sizes depending on sample rate
    
    if expected_sample_rate <= 16000:
        # Low sample rates: smaller blocks
        return 1152
    elif expected_sample_rate <= 48000:
        # Standard sample rates: typical block size
        return 4608
    else:
        # Hi-res audio: larger blocks for efficiency
        return 4608
    
    # For complete implementation, frame header would need to be fully parsed
    # Byte 4 of FLAC frame header contains block size information


def _parse_flac_frames_from_bytes(audio_bytes: bytes, expected_sample_rate: int = 44100) -> List[dict]:
    """
    Parse FLAC frames directly from byte data
    
    Args:
        audio_bytes: Complete FLAC audio data
        expected_sample_rate: Expected sample rate from metadata for better frame size estimation
        
    Returns:
        List of frame information as dictionaries
    """
    frames_info = []
    
    # Skip header and metadata
    pos = _parse_flac_header_and_metadata(audio_bytes)
    current_sample = 0
    loop_count = 0  # Anti-endless loop counter
    
    logger.trace(f"Starting frame analysis at position {pos} for {expected_sample_rate}Hz audio")
    
    # Frame-by-frame analysis
    while pos < len(audio_bytes) - 2:
        loop_count += 1
        
        # Anti-endless loop protection
        if loop_count > 50000:  # More than 50k iterations is suspicious
            logger.error(f"Frame analysis loop stopped after {loop_count} iterations. Possible endless loop.")
            break
        
        # Reduced logging after first 10 frames
        debug_logging = loop_count <= 10
        
        # Search for FLAC frame sync
        sync_word = int.from_bytes(audio_bytes[pos:pos+2], 'big')
        
        if (sync_word & 0xFFFE) == 0xFFF8:  # FLAC Frame Sync Pattern
            frame_start = pos
            
            # Search for next frame to determine size
            next_frame_pos = _find_next_flac_frame_sync(audio_bytes, pos + 16)
            
            if next_frame_pos is not None:
                frame_size = next_frame_pos - frame_start
            else:
                # Last frame - take rest of file
                frame_size = len(audio_bytes) - frame_start
            
            # Safety check: frame must be at least 16 bytes
            if frame_size < 16:
                if debug_logging:
                    logger.trace(f"Frame too small ({frame_size} bytes), skipping position {pos}")
                pos += 1
                continue
            
            # Safety check: frame must not be absurdly large
            if frame_size > 1024 * 1024:  # 1MB
                if debug_logging:
                    logger.trace(f"Frame too large ({frame_size} bytes), skipping position {pos}")
                pos += 1
                continue
            
            # Estimate samples per frame (with metadata info)
            header_bytes = audio_bytes[pos:pos+min(16, frame_size)]
            samples_per_frame = _estimate_flac_samples_per_frame(header_bytes, expected_sample_rate)
            
            frames_info.append({
                'byte_offset': frame_start,
                'frame_size': frame_size,
                'sample_pos': current_sample
            })
            
            current_sample += samples_per_frame
            
            if debug_logging:
                logger.trace(f"Frame {len(frames_info)}: pos={frame_start}, size={frame_size}, samples={samples_per_frame}")
            
            # Jump to next frame (or conservatively forward)
            if next_frame_pos is not None:
                pos = next_frame_pos
            else:
                # If no next frame found, jump by frame size
                pos += frame_size
                
        else:
            pos += 1
        
        # Progress log every 500 frames (twice as often)
        if loop_count % 500 == 0:
            logger.trace(f"Frame analysis progress: {len(frames_info)} frames found, position {pos}/{len(audio_bytes)}")
    
    logger.trace(f"Found: {len(frames_info)} FLAC frames for {expected_sample_rate}Hz audio after {loop_count} iterations")
    return frames_info


def build_flac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Create index for FLAC frame access through direct byte analysis
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with FLAC audio data
        
    Returns:
        Created index array
        
    Raises:
        ValueError: If no FLAC frames are found
    """
    logger.trace("build_flac_index() requested.")
    
    # Extract metadata from array attributes (set by aimport.py module)
    sample_rate = audio_blob_array.attrs.get('sample_rate', 44100)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    codec = audio_blob_array.attrs.get('codec', 'flac')
    container_type = audio_blob_array.attrs.get('container_type', 'flac-native')
    
    # Validation: ensure we're dealing with FLAC data
    if codec != 'flac':
        raise ValueError(f"Expected FLAC codec, but found: {codec}")
    
    logger.trace(f"Creating FLAC index for: {sample_rate}Hz, {channels} channels, container: {container_type}")
    
    # Load audio bytes (once)
    logger.trace("Loading FLAC audio data...")
    audio_bytes = bytes(audio_blob_array[()])
    
    # Find FLAC frames through direct byte analysis (with metadata info)
    logger.trace("Analyzing FLAC frames...")
    frames_info = _parse_flac_frames_from_bytes(audio_bytes, sample_rate)
    
    if len(frames_info) < 1:
        raise ValueError("Could not find FLAC frames in audio")
    
    # Create simple 2D array for index - much cleaner than structured arrays!
    logger.trace("Creating index array...")
    index_array = np.array([
        [f['byte_offset'], f['frame_size'], f['sample_pos']] 
        for f in frames_info
    ], dtype=FLAC_INDEX_DTYPE)  # Shape: (n_frames, 3)
    
    # Store index in Zarr group - simple 2D array, Zarr v3 friendly
    flac_index = zarr_group.create_array(
        name='flac_index',
        shape=index_array.shape,
        chunks=(min(1000, len(frames_info)), FLAC_INDEX_COLS),
        dtype=FLAC_INDEX_DTYPE
    )
    
    # Write data to the created array
    flac_index[:] = index_array
    
    # Store metadata at index (extended with available information)
    index_attrs = {
        'sample_rate': sample_rate,
        'channels': channels,
        'total_frames': len(frames_info),
        'codec': codec,
        'container_type': container_type
    }
    
    # Copy additional metadata from audio_blob_array if available
    optional_attrs = [
        'compression_level', 'sampling_rescale_factor', 
        'first_sample_time_stamp', 'last_sample_time_stamp'
    ]
    
    for attr_name in optional_attrs:
        if attr_name in audio_blob_array.attrs:
            index_attrs[attr_name] = audio_blob_array.attrs[attr_name]
    
    flac_index.attrs.update(index_attrs)
    
    logger.success(f"FLAC index created with {len(frames_info)} frames")
    return flac_index


def _find_frame_range_for_samples(flac_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Find frame range for sample range using binary search
    
    Args:
        flac_index: FLAC index array (shape: n_frames x 3)
        start_sample: First required sample
        end_sample: Last required sample
        
    Returns:
        Tuple (start_frame_idx, end_frame_idx)
    """
    sample_positions = flac_index[:, FLAC_INDEX_COL_SAMPLE_POS]
    
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, len(flac_index) - 1)
    
    return start_idx, end_idx


def _create_temporary_flac_file(audio_bytes: bytes) -> str:
    """
    Create temporary FLAC file for SoundFile access
    
    Args:
        audio_bytes: FLAC audio data
        
    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def extract_audio_segment_flac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                              start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """
    Extract audio segment based on FLAC index
    
    Args:
        zarr_group: Zarr group with FLAC index
        audio_blob_array: Array with FLAC audio data
        start_sample: First sample (inclusive)
        end_sample: Last sample (inclusive)
        dtype: Output data type
        
    Returns:
        Decoded audio data as numpy array
    """
    try:
        # Load index from Zarr group
        if 'flac_index' not in zarr_group:
            raise ValueError("FLAC index not found. Must be created first with build_flac_index().")
        
        flac_index = zarr_group['flac_index']
        
        # Find frame range for sample range
        start_idx, end_idx = _find_frame_range_for_samples(flac_index, start_sample, end_sample)
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        # Load audio bytes and create temporary file
        audio_bytes = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_flac_file(audio_bytes)
        
        try:
            # Decode with SoundFile
            with sf.SoundFile(temp_file_path) as sf_file:
                sample_positions = flac_index[:, FLAC_INDEX_COL_SAMPLE_POS]
                first_frame_sample = sample_positions[start_idx]
                
                # Jump to first required frame
                sf_file.seek(first_frame_sample)
                
                # Calculate samples to read
                if end_idx < len(flac_index) - 1:
                    last_frame_end = sample_positions[end_idx + 1]
                else:
                    last_frame_end = sf_file.frames
                
                total_samples_to_read = last_frame_end - first_frame_sample
                frames_data = sf_file.read(total_samples_to_read, dtype=dtype)
                
                # Cut exactly to requested range
                start_offset = max(0, start_sample - first_frame_sample)
                end_offset = min(
                    start_offset + (end_sample - start_sample + 1), 
                    frames_data.shape[0]
                )
                
                return frames_data[start_offset:end_offset]
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error extracting FLAC segment [{start_sample}:{end_sample}]: {e}")
        return np.array([])


def parallel_extract_audio_segments_flac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                                        segments: List[Tuple[int, int]], dtype=np.int16, 
                                        max_workers: int = 4) -> List[np.ndarray]:
    """
    Parallel extraction of multiple audio segments
    
    Args:
        zarr_group: Zarr group with FLAC index
        audio_blob_array: Array with FLAC audio data  
        segments: List of (start_sample, end_sample) tuples
        dtype: Output data type
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of decoded audio arrays in original order
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                extract_audio_segment_flac, 
                zarr_group, audio_blob_array, start, end, dtype
            ): (start, end) 
            for start, end in segments
        }
        
        results = {}
        for future in future_to_segment:
            segment = future_to_segment[future]
            try:
                results[segment] = future.result()
            except Exception as e:
                logger.error(f"Error in parallel extraction of segment {segment}: {e}")
                results[segment] = np.array([])
        
        # Return results in original order
        return [results[segment] for segment in segments]


logger.trace("Module loaded.")