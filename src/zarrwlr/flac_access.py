"""
FLAC Access Public API Module
============================

Public interface for all FLAC operations including:
- FLAC import (ffmpeg conversion + Zarr storage + automatic indexing)
- Audio segment extraction (single and parallel)
- Index management

This module provides the main API that is called by aimport.py and other modules.
"""

import zarr
import numpy as np
import tempfile
import os
import pathlib
import subprocess
import soundfile as sf
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Import backend module
from . import flac_index_backend as flac_index
from .utils import file_size
from .config import Config

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("FLAC Access API module loading...")

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"


def import_flac_to_zarr(zarr_group: zarr.Group, 
                       audio_file: str | pathlib.Path,
                       source_params: dict,
                       first_sample_time_stamp,
                       flac_compression_level: int = 4,
                       temp_dir: str = "/tmp") -> zarr.Array:
    """
    Import audio file to FLAC format in Zarr with automatic index creation
    
    Args:
        zarr_group: Zarr group to store the audio data
        audio_file: Path to source audio file
        source_params: Source audio parameters from aimport.py
        first_sample_time_stamp: Timestamp for first sample
        flac_compression_level: FLAC compression level (0-12)
        temp_dir: Directory for temporary files
        
    Returns:
        Created audio blob array
        
    Raises:
        ValueError: If conversion or indexing fails
    """
    logger.trace(f"import_flac_to_zarr() requested for file '{audio_file}'")
    
    audio_file = pathlib.Path(audio_file)
    
    # Create temporary FLAC file
    logger.trace(f"Creating temporary FLAC file in '{temp_dir}'...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.flac', dir=temp_dir) as tmp_out:
        tmp_file = pathlib.Path(tmp_out.name)
    logger.trace(f"Temporary file created: '{tmp_file.name}'")
    
    try:
        # Prepare ffmpeg command for FLAC conversion
        logger.trace("Preparing ffmpeg command for FLAC conversion...")
        ffmpeg_cmd = ["ffmpeg", "-y"]
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
        ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]  
        ffmpeg_cmd += ["-f", 'flac', str(tmp_file)]
        logger.trace(f"ffmpeg command: {ffmpeg_cmd}")
        
        # Execute ffmpeg conversion
        logger.trace("Starting ffmpeg conversion...")
        subprocess.run(ffmpeg_cmd, check=True)
        logger.trace("ffmpeg conversion completed")
        
        # Get file size for Zarr array creation
        size = file_size(tmp_file)
        logger.trace(f"Converted FLAC file size: {size} bytes")
        
        # Create Zarr array for audio data
        logger.trace("Creating Zarr array for FLAC audio data...")
        audio_blob_array = zarr_group.create_array(
            name=AUDIO_DATA_BLOB_ARRAY_NAME,
            compressor=None,
            shape=(size,),
            chunks=(Config.original_audio_chunk_size,),
            shards=(Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size,),
            dtype=np.uint8,
            overwrite=True,
        )
        logger.trace("Zarr array created")
        
        # Copy FLAC data from temporary file to Zarr array
        logger.trace("Copying FLAC data to Zarr array...")
        offset = 0
        max_buffer_size = int(np.clip(
            Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size, 
            1, 100e6
        ))
        
        with open(tmp_file, "rb") as f:
            for buffer in iter(lambda: f.read(max_buffer_size), b''):
                buffer_array = np.frombuffer(buffer, dtype="u1")
                audio_blob_array[offset:offset + len(buffer_array)] = buffer_array
                offset += len(buffer_array)
        
        logger.trace("FLAC data copied to Zarr array")
        
        # Set audio blob array attributes
        logger.trace("Setting audio blob array attributes...")
        attrs = {
            "codec": "flac",
            "nb_channels": source_params["nb_channels"],
            "sample_rate": source_params["sampling_rate"],
            "sampling_rescale_factor": 1.0,  # No rescaling for FLAC
            "container_type": "flac-native",
            "first_sample_time_stamp": first_sample_time_stamp,
            "compression_level": flac_compression_level
        }
        audio_blob_array.attrs.update(attrs)
        logger.trace("Audio blob array attributes set")
        
        # Create FLAC index automatically
        logger.trace("Creating FLAC index...")
        flac_index.build_flac_index(zarr_group, audio_blob_array)
        logger.trace("FLAC index created")
        
        logger.success(f"FLAC import completed successfully for '{audio_file.name}'")
        return audio_blob_array
        
    finally:
        # Clean up temporary file
        if tmp_file.exists():
            tmp_file.unlink()
            logger.trace("Temporary file cleaned up")


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
        
        flac_index_array = zarr_group['flac_index']
        
        # Find frame range for sample range
        start_idx, end_idx = flac_index._find_frame_range_for_samples(
            flac_index_array, start_sample, end_sample
        )
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        # Load audio bytes and create temporary file
        audio_bytes = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_flac_file(audio_bytes)
        
        try:
            # Decode with SoundFile
            with sf.SoundFile(temp_file_path) as sf_file:
                sample_positions = flac_index_array[:, flac_index.FLAC_INDEX_COL_SAMPLE_POS]
                first_frame_sample = sample_positions[start_idx]
                
                # Jump to first required frame
                sf_file.seek(first_frame_sample)
                
                # Calculate samples to read
                if end_idx < flac_index_array.shape[0] - 1:
                    last_frame_end = sample_positions[end_idx + 1]
                else:
                    last_frame_end = sf_file.frames
                
                total_samples_to_read = last_frame_end - first_frame_sample
                frames_data = sf_file.read(total_samples_to_read, dtype=dtype)
                
                # Ensure correct dtype (SoundFile sometimes ignores dtype parameter)
                if frames_data.dtype != dtype:
                    frames_data = frames_data.astype(dtype)
                
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


# Convenience functions for backward compatibility
def build_flac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Create FLAC index (convenience wrapper)
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with FLAC audio data
        
    Returns:
        Created index array
    """
    return flac_index.build_flac_index(zarr_group, audio_blob_array)


logger.trace("FLAC Access API module loaded.")