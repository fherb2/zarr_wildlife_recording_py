"""
AAC Access Public API Module - Updated for 3-Column Index Optimization
======================================================================

Public interface for all AAC operations including:
- AAC import (ffmpeg encoding + Zarr storage + automatic indexing)
- Audio segment extraction (single and parallel) with 3-column index
- Index management

This module provides AAC-LC codec support with OPTIMIZED 3-column indexing
for minimal overhead while maintaining excellent random access performance.

Key Features:
- ffmpeg AAC encoding/conversion (no subprocess overhead for access)
- OPTIMIZED 3-column frame-level indexing for ~21ms granularity random access
- 160kbps target bitrate for optimal quality/size balance
- Zarr v3 storage with optimized chunking
- Parallel segment extraction support
- Overlap handling for accurate random access
"""

import zarr
import numpy as np
import tempfile
import pathlib
import subprocess
import av
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Import backend module with 3-column optimization
from . import aac_index_backend as aac_index
from .utils import file_size
from .config import Config

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("AAC Access API module (3-column optimized) loading...")

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"


def import_aac_to_zarr(zarr_group: zarr.Group, 
                      audio_file: str | pathlib.Path,
                      source_params: dict,
                      first_sample_time_stamp,
                      aac_bitrate: int = 160000,
                      temp_dir: str = "/tmp") -> zarr.Array:
    """Import audio file to AAC-LC format in Zarr using ffmpeg (design-compliant)"""
    
    logger.trace(f"import_aac_to_zarr() requested for file '{audio_file}'")
    
    audio_file = pathlib.Path(audio_file)
    
    # Create temporary AAC file
    logger.trace(f"Creating temporary AAC file in '{temp_dir}'...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.aac', dir=temp_dir) as tmp_out:
        tmp_file = pathlib.Path(tmp_out.name)
    logger.trace(f"Temporary file created: '{tmp_file.name}'")
    
    try:
        # Use ffmpeg for import (design-compliant)
        logger.trace("Starting AAC conversion with ffmpeg...")
        _convert_to_aac_ffmpeg(audio_file, tmp_file, aac_bitrate, source_params)
        logger.trace("ffmpeg AAC conversion completed")
        
        # Get file size for Zarr array creation
        size = file_size(tmp_file)
        logger.trace(f"Converted AAC file size: {size} bytes")
        
        # Create Zarr array for audio data
        logger.trace("Creating Zarr array for AAC audio data...")
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
        
        # Copy AAC data from temporary file to Zarr array
        logger.trace("Copying AAC data to Zarr array...")
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
        
        logger.trace("AAC data copied to Zarr array")
        
        # Set audio blob array attributes - IMPORTANT: Include sample_rate for calculations
        logger.trace("Setting audio blob array attributes...")
        attrs = {
            "codec": "aac",
            "nb_channels": source_params["nb_channels"],
            "sample_rate": source_params["sampling_rate"],  # ESSENTIAL for 3-column index calculations
            "sampling_rescale_factor": 1.0,  # No rescaling for AAC
            "container_type": "aac-native",
            "first_sample_time_stamp": first_sample_time_stamp,
            "aac_bitrate": aac_bitrate,
            "profile": "aac_low"  # AAC-LC profile
        }
        audio_blob_array.attrs.update(attrs)
        logger.trace("Audio blob array attributes set")
        
        # Create AAC index automatically with 3-column optimization
        logger.trace("Creating OPTIMIZED 3-column AAC index...")
        from . import aac_index_backend as aac_index
        aac_index.build_aac_index(zarr_group, audio_blob_array)
        logger.trace("OPTIMIZED AAC index created")
        
        logger.success(f"AAC import completed successfully for '{audio_file.name}' with 3-column index optimization")
        return audio_blob_array
        
    finally:
        # Clean up temporary file
        if tmp_file.exists():
            tmp_file.unlink()
            logger.trace("Temporary file cleaned up")


def _convert_to_aac_ffmpeg(input_file: pathlib.Path, 
                          output_file: pathlib.Path,
                          bitrate: int,
                          source_params: dict):
    """Convert audio file to AAC using ffmpeg (design-compliant)"""
    
    logger.trace(f"Using ffmpeg for AAC conversion at {bitrate} bps...")
    
    # Prepare ffmpeg command for AAC conversion
    ffmpeg_cmd = ["ffmpeg", "-y"]
    
    # Input
    ffmpeg_cmd += ["-i", str(input_file)]
    
    # AAC encoding parameters
    ffmpeg_cmd += ["-c:a", "aac"]
    ffmpeg_cmd += ["-profile:a", "aac_low"]  # AAC-LC profile
    ffmpeg_cmd += ["-b:a", str(bitrate)]
    
    # Quality parameters
    ffmpeg_cmd += ["-ar", str(source_params.get("sampling_rate", 48000))]  # Sample rate
    ffmpeg_cmd += ["-ac", str(source_params.get("nb_channels", 2))]       # Channels
    
    # ADTS output format for raw AAC with headers
    ffmpeg_cmd += ["-f", "adts"]
    
    # Additional quality settings
    ffmpeg_cmd += ["-cutoff", "20000"]  # Frequency cutoff for quality
    
    # Output file
    ffmpeg_cmd += [str(output_file)]
    
    logger.trace(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")
    
    # Execute ffmpeg conversion
    try:
        result = subprocess.run(
            ffmpeg_cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        logger.trace("ffmpeg conversion completed successfully")
        
        # Log any warnings from ffmpeg
        if result.stderr:
            logger.trace(f"ffmpeg stderr: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        raise ValueError(f"AAC conversion failed: {e}")
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please ensure ffmpeg is installed and in PATH.")
        raise ValueError("ffmpeg not found. Please install ffmpeg.")
    
    # Verify output file was created
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise ValueError(f"ffmpeg did not create valid output file: {output_file}")
    
    logger.trace(f"AAC file created: {output_file.stat().st_size} bytes")

def _create_temporary_aac_file(audio_bytes: bytes) -> str:
    """
    Create temporary AAC file for PyAV access
    
    Args:
        audio_bytes: AAC audio data
        
    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".aac")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def extract_audio_segment_aac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                             start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """Extract audio segment based on OPTIMIZED 3-column AAC index with overlap handling"""
    
    logger.trace(f"AAC extraction requested: samples {start_sample}-{end_sample} (3-column index)")
    
    try:
        # Load index from Zarr group
        if 'aac_index' not in zarr_group:
            raise ValueError("AAC index not found. Must be created first with build_aac_index().")
        
        aac_index_array = zarr_group['aac_index']
        
        # Verify 3-column format
        if aac_index_array.shape[1] != 3:
            raise ValueError(f"Expected 3-column index, got {aac_index_array.shape[1]} columns")
        
        # Find frame range for sample range WITH OVERLAP HANDLING
        start_idx, end_idx = aac_index._find_frame_range_for_samples(
            aac_index_array, start_sample, end_sample
        )
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        logger.trace(f"Using frames [{start_idx}:{end_idx}] for sample range [{start_sample}:{end_sample}]")
        
        # Load audio bytes and create temporary file
        audio_bytes = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_aac_file(audio_bytes)
        
        try:
            # Decode with PyAV
            container = av.open(temp_file_path)
            audio_stream = container.streams.audio[0]
            
            # Calculate frame positions from 3-column index
            sample_positions = aac_index_array[:, aac_index.AAC_INDEX_COL_SAMPLE_POS]
            first_frame_sample = sample_positions[start_idx]
            
            # Seek to approximate position
            seek_time = first_frame_sample / audio_stream.sample_rate
            container.seek(int(seek_time * 1000000))  # Convert to microseconds
            
            # Decode frames and collect samples
            decoded_samples = []
            current_sample = 0
            frames_decoded = 0
            target_frames = end_idx - start_idx + 1
            
            logger.trace(f"Starting PyAV decode from frame {start_idx}, need {target_frames} frames")
            
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    # FIXED: Use PyAV to_ndarray() without format parameter
                    try:
                        frame_array = frame.to_ndarray()  # ✅ CORRECT API
                    except Exception as e:
                        logger.error(f"PyAV to_ndarray failed: {e}")
                        continue
                    
                    # Convert to target dtype manually if needed
                    if frame_array.dtype != dtype:
                        if dtype == np.int16:
                            if frame_array.dtype.kind == 'f':  # floating point
                                frame_array = (frame_array * 32767).astype(np.int16)
                            else:
                                frame_array = frame_array.astype(np.int16)
                        elif dtype == np.int32:
                            if frame_array.dtype.kind == 'f':  # floating point
                                frame_array = (frame_array * 2147483647).astype(np.int32)
                            else:
                                frame_array = frame_array.astype(np.int32)
                        elif dtype == np.float32:
                            if frame_array.dtype.kind == 'f':
                                frame_array = frame_array.astype(np.float32)
                            else:
                                frame_array = (frame_array.astype(np.float32) / 32767.0)
                        else:
                            frame_array = frame_array.astype(dtype)
                    
                    decoded_samples.append(frame_array)
                    current_sample += frame_array.shape[0]
                    frames_decoded += 1
                    
                    # Stop when we have enough samples (with some buffer)
                    if current_sample >= (end_sample - start_sample + 2048):  # Extra buffer for overlap
                        break
                
                if current_sample >= (end_sample - start_sample + 2048):
                    break
            
            # Concatenate all decoded samples
            if decoded_samples:
                full_audio = np.concatenate(decoded_samples, axis=0)
                
                # Cut exactly to requested range, accounting for overlap
                # The overlap handling in _find_frame_range_for_samples already started us earlier
                actual_start_sample = sample_positions[start_idx]
                start_offset = max(0, start_sample - actual_start_sample)
                end_offset = min(
                    start_offset + (end_sample - start_sample + 1), 
                    full_audio.shape[0]
                )
                
                result = full_audio[start_offset:end_offset]
                logger.trace(f"Extracted {result.shape[0]} samples from {len(decoded_samples)} frames")
                return result
            else:
                logger.warning("No samples decoded - PyAV decoding failed")
                return np.array([], dtype=dtype)
                
        finally:
            # Clean up temporary file
            if pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
                
    except Exception as e:
        logger.error(f"Error extracting AAC segment [{start_sample}:{end_sample}]: {e}")
        return np.array([], dtype=dtype)
    

def parallel_extract_audio_segments_aac(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                                        segments: List[Tuple[int, int]], dtype=np.int16, 
                                        max_workers: int = 4) -> List[np.ndarray]:
    """
    Parallel extraction of multiple audio segments using OPTIMIZED 3-column index
    
    Args:
        zarr_group: Zarr group with AAC index (3-column format)
        audio_blob_array: Array with AAC audio data  
        segments: List of (start_sample, end_sample) tuples
        dtype: Output data type
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of decoded audio arrays in original order
    """
    logger.trace(f"Parallel AAC extraction requested: {len(segments)} segments, {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                extract_audio_segment_aac, 
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
                results[segment] = np.array([], dtype=dtype)
        
        # Return results in original order
        return [results[segment] for segment in segments]


# Convenience functions for backward compatibility
def build_aac_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """
    Create OPTIMIZED 3-column AAC index (convenience wrapper)
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with AAC audio data
        
    Returns:
        Created index array (3-column format)
    """
    return aac_index.build_aac_index(zarr_group, audio_blob_array)


logger.trace("AAC Access API module (3-column optimized) loaded.")