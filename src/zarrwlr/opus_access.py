"""
Opus Access Public API Module
=============================

Public interface for all Opus operations including:
- Opus import (ffmpeg conversion + Zarr storage + automatic indexing)
- Audio segment extraction (single and parallel)
- Index management

This module provides the main API that is called by aimport.py and other modules.
Based on the successful FLAC implementation with Opus-specific adaptations:
- 1:1 Opus data copy for Opus sources (no re-encoding)
- Ultrasonic handling (>48kHz sources)
- OGG container structure handling

MIGRATION NOTE: Opus-specific code from aimport.py has been moved here in Step 1.3
for better separation of concerns and maintainability.
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

# Import backend module (will be created in next step)
from . import opus_index_backend as opus_index
from .utils import file_size
from .config import Config

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Opus Access API module loading...")

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"


def import_opus_to_zarr(zarr_group: zarr.Group, 
                       audio_file: str | pathlib.Path,
                       source_params: dict,
                       first_sample_time_stamp,
                       opus_bitrate: int = 160000,
                       temp_dir: str = "/tmp") -> zarr.Array:
    """
    Import audio file to Opus format in Zarr with automatic index creation
    
    Args:
        zarr_group: Zarr group to store the audio data
        audio_file: Path to source audio file
        source_params: Source audio parameters from aimport.py
        first_sample_time_stamp: Timestamp for first sample
        opus_bitrate: Opus bitrate in bits per second (default: 160000)
        temp_dir: Directory for temporary files
        
    Returns:
        Created audio blob array
        
    Raises:
        ValueError: If conversion or indexing fails
    """
    logger.trace(f"import_opus_to_zarr() requested for file '{audio_file}'")
    
    audio_file = pathlib.Path(audio_file)
    
    # Extract source parameters
    source_sample_rate = source_params.get("sampling_rate", 48000)
    source_channels = source_params.get("nb_channels", 1)
    is_opus_source = source_params.get("is_opus", False)
    
    # Ultrasonic detection and handling
    is_ultrasonic = source_sample_rate > 48000
    sampling_rescale_factor = 1.0
    target_sample_rate = source_sample_rate
    
    if is_ultrasonic:
        # Opus limitation: Max 48kHz
        # Trick: Interpret data as 48kHz, store rescale factor
        sampling_rescale_factor = source_sample_rate / 48000.0
        target_sample_rate = 48000
        logger.trace(f"Ultrasonic source detected: {source_sample_rate}Hz -> 48kHz (factor: {sampling_rescale_factor:.3f})")
    
    # Create temporary OGG file
    logger.trace(f"Creating temporary OGG file in '{temp_dir}'...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=temp_dir) as tmp_out:
        tmp_file = pathlib.Path(tmp_out.name)
    logger.trace(f"Temporary file created: '{tmp_file.name}'")
    
    try:
        # Prepare ffmpeg command for Opus conversion
        logger.trace("Preparing ffmpeg command for Opus conversion...")
        
        if is_opus_source and not is_ultrasonic:
            # 1:1 Opus data copy - no re-encoding for perfect quality
            logger.trace("1:1 Opus copy mode (no re-encoding)")
            ffmpeg_cmd = ["ffmpeg", "-y"]
            ffmpeg_cmd += ["-i", str(audio_file)]
            ffmpeg_cmd += ["-c:a", "copy"]  # Copy codec without re-encoding
            ffmpeg_cmd += ["-f", "ogg", str(tmp_file)]
        else:
            # Standard ffmpeg encoding (for non-Opus sources or ultrasonic)
            logger.trace("Standard Opus encoding mode")
            ffmpeg_cmd = ["ffmpeg", "-y"]
            
            if is_ultrasonic:
                # Force input interpretation as 48kHz for Opus compatibility
                ffmpeg_cmd += ["-ar", "48000"]
                logger.trace(f"Ultrasonic mode: forcing input sample rate to 48kHz")
            
            ffmpeg_cmd += ["-i", str(audio_file)]
            ffmpeg_cmd += ["-c:a", "libopus"]
            ffmpeg_cmd += ["-b:a", str(int(opus_bitrate))]
            ffmpeg_cmd += ["-vbr", "off"]  # Constant bitrate for better quality
            ffmpeg_cmd += ["-apply_phase_inv", "false"]  # Phase-correct encoding
            ffmpeg_cmd += ["-f", "ogg", str(tmp_file)]
        
        logger.trace(f"ffmpeg command: {ffmpeg_cmd}")
        
        # Execute ffmpeg conversion
        logger.trace("Starting ffmpeg conversion...")
        subprocess.run(ffmpeg_cmd, check=True)
        logger.trace("ffmpeg conversion completed")
        
        # Get file size for Zarr array creation
        size = file_size(tmp_file)
        logger.trace(f"Converted OGG file size: {size} bytes")
        
        # Create Zarr array for audio data
        logger.trace("Creating Zarr array for Opus audio data...")
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
        
        # Copy OGG data from temporary file to Zarr array
        logger.trace("Copying OGG data to Zarr array...")
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
        
        logger.trace("OGG data copied to Zarr array")
        
        # Set audio blob array attributes
        logger.trace("Setting audio blob array attributes...")
        attrs = {
            "codec": "opus",
            "nb_channels": source_channels,
            "sample_rate": target_sample_rate,
            "sampling_rescale_factor": sampling_rescale_factor,
            "container_type": "ogg",
            "first_sample_time_stamp": first_sample_time_stamp,
            "opus_bitrate": opus_bitrate,
            "is_ultrasonic": is_ultrasonic,
            "original_sample_rate": source_sample_rate if is_ultrasonic else target_sample_rate
        }
        audio_blob_array.attrs.update(attrs)
        logger.trace("Audio blob array attributes set")
        
        # Create Opus index automatically
        logger.trace("Creating Opus index...")
        opus_index.build_opus_index(zarr_group, audio_blob_array)
        logger.trace("Opus index created")
        
        logger.success(f"Opus import completed successfully for '{audio_file.name}'")
        return audio_blob_array
        
    finally:
        # Clean up temporary file
        if tmp_file.exists():
            tmp_file.unlink()
            logger.trace("Temporary file cleaned up")


def _create_temporary_ogg_file(audio_bytes: bytes) -> str:
    """
    Create temporary OGG file for ffmpeg access
    
    Args:
        audio_bytes: OGG audio data
        
    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def extract_audio_segment_opus(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                              start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """
    Extract audio segment based on Opus index
    
    Args:
        zarr_group: Zarr group with Opus index
        audio_blob_array: Array with Opus audio data
        start_sample: First sample (inclusive)
        end_sample: Last sample (inclusive)
        dtype: Output data type
        
    Returns:
        Decoded audio data as numpy array
    """
    try:
        # Load index from Zarr group
        if 'opus_index' not in zarr_group:
            raise ValueError("Opus index not found. Must be created first with build_opus_index().")
        
        opus_index_array = zarr_group['opus_index']
        
        # Get audio parameters
        sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
        channels = audio_blob_array.attrs.get('nb_channels', 1)
        sampling_rescale_factor = audio_blob_array.attrs.get('sampling_rescale_factor', 1.0)
        original_sample_rate = audio_blob_array.attrs.get('original_sample_rate', sample_rate)
        
        # Find page range for sample range
        start_idx, end_idx = opus_index._find_page_range_for_samples(
            opus_index_array, start_sample, end_sample
        )
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        # Determine byte range to extract
        page_start_byte = int(opus_index_array[start_idx, opus_index.OPUS_INDEX_COL_BYTE_OFFSET])
        
        if end_idx < opus_index_array.shape[0] - 1:
            page_end_byte = int(opus_index_array[end_idx + 1, opus_index.OPUS_INDEX_COL_BYTE_OFFSET])
        else:
            page_end_byte = audio_blob_array.shape[0]
        
        # Load complete OGG data and create temporary file for ffmpeg
        # FFmpeg needs a complete OGG stream, not just page fragments
        complete_ogg_data = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_ogg_file(complete_ogg_data)
        
        try:
            # Calculate approximate time positions for ffmpeg seeking
            # This is more reliable than trying to create partial OGG streams
            sample_positions = opus_index_array[:, opus_index.OPUS_INDEX_COL_SAMPLE_POS]
            actual_start_sample = int(sample_positions[start_idx])
            
            # Convert sample positions to time for ffmpeg
            start_time_seconds = actual_start_sample / sample_rate
            duration_samples = end_sample - start_sample + 1
            duration_seconds = duration_samples / sample_rate
            
            # Decode with ffmpeg using time-based seeking (more reliable for OGG)
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time_seconds),  # Seek to start time
                "-t", str(duration_seconds),     # Duration to extract
                "-i", temp_file_path,
                "-ac", str(channels),
                "-ar", str(sample_rate),
                "-f", "s16le" if dtype == np.int16 else "f32le",
                "pipe:1"
            ]
            
            proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pcm_bytes, stderr_output = proc.communicate()
            
            if proc.returncode != 0 or not pcm_bytes:
                error_msg = stderr_output.decode('utf-8', errors='ignore') if stderr_output else "Unknown error"
                raise RuntimeError(f"FFmpeg decoding failed: {error_msg}")
            
            # Convert PCM bytes to numpy array
            samples = np.frombuffer(pcm_bytes, dtype=dtype)
            if samples.size % channels != 0:
                logger.warning(f"Sample count {samples.size} not divisible by channels {channels}")
                # Trim to make divisible
                samples = samples[:samples.size - (samples.size % channels)]
            
            if channels > 1:
                samples = samples.reshape(-1, channels)
            
            # Apply ultrasonic correction to the extracted samples if needed
            if sampling_rescale_factor != 1.0:
                logger.trace(f"Applying ultrasonic correction factor {sampling_rescale_factor}")
                # Note: For time-based extraction, the correction is already applied
                # by using the correct sample_rate in ffmpeg command
            
            return samples
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error extracting Opus segment [{start_sample}:{end_sample}]: {e}")
        return np.array([])


def parallel_extract_audio_segments_opus(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                                        segments: List[Tuple[int, int]], dtype=np.int16, 
                                        max_workers: int = 4) -> List[np.ndarray]:
    """
    Parallel extraction of multiple audio segments
    
    Args:
        zarr_group: Zarr group with Opus index
        audio_blob_array: Array with Opus audio data  
        segments: List of (start_sample, end_sample) tuples
        dtype: Output data type
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of decoded audio arrays in original order
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                extract_audio_segment_opus, 
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
def build_opus_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Create Opus index (convenience wrapper)
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with Opus audio data
        
    Returns:
        Created index array
    """
    return opus_index.build_opus_index(zarr_group, audio_blob_array)


logger.trace("Opus Access API module loaded.")