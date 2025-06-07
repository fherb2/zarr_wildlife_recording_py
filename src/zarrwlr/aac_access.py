"""
AAC Access Public API Module
============================

Public interface for all AAC operations including:
- AAC import (PyAV encoding + Zarr storage + automatic indexing)
- Audio segment extraction (single and parallel)
- Index management

This module provides AAC-LC codec support similar to the FLAC implementation
but with superior compression (160kbps vs ~650kbps FLAC) while maintaining
excellent random access performance through frame-level indexing.

Key Features:
- PyAV native AAC encoding/decoding (no subprocess overhead)
- Frame-level indexing for ~21ms granularity random access
- 160kbps target bitrate for optimal quality/size balance
- Zarr v3 storage with optimized chunking
- Parallel segment extraction support
"""

import zarr
import numpy as np
import tempfile
import pathlib
import subprocess
import av
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Import backend module
from . import aac_index_backend as aac_index
from .utils import file_size
from .config import Config

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("AAC Access API module loading...")

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"


def import_aac_to_zarr(zarr_group: zarr.Group, 
                      audio_file: str | pathlib.Path,
                      source_params: dict,
                      first_sample_time_stamp,
                      aac_bitrate: int = 160000,
                      temp_dir: str = "/tmp") -> zarr.Array:
    """
    Import audio file to AAC-LC format in Zarr with automatic index creation
    
    Args:
        zarr_group: Zarr group to store the audio data
        audio_file: Path to source audio file
        source_params: Source audio parameters from aimport.py
        first_sample_time_stamp: Timestamp for first sample
        aac_bitrate: AAC bitrate in bits/second (default: 160000)
        temp_dir: Directory for temporary files
        
    Returns:
        Created audio blob array
        
    Raises:
        ValueError: If conversion or indexing fails
    """
    logger.trace(f"import_aac_to_zarr() requested for file '{audio_file}'")
    
    audio_file = pathlib.Path(audio_file)
    
    # Create temporary AAC file
    logger.trace(f"Creating temporary AAC file in '{temp_dir}'...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.aac', dir=temp_dir) as tmp_out:
        tmp_file = pathlib.Path(tmp_out.name)
    logger.trace(f"Temporary file created: '{tmp_file.name}'")
    
    try:
        # Convert using PyAV for better control and consistency
        logger.trace("Starting PyAV AAC conversion...")
        _convert_to_aac_pyav(audio_file, tmp_file, aac_bitrate, source_params)
        logger.trace("PyAV AAC conversion completed")
        
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
        
        # Set audio blob array attributes
        logger.trace("Setting audio blob array attributes...")
        attrs = {
            "codec": "aac",
            "profile": "aac_low",  # AAC-LC profile
            "nb_channels": source_params["nb_channels"],
            "sample_rate": source_params["sampling_rate"],
            "sampling_rescale_factor": 1.0,  # No rescaling for AAC
            "container_type": "aac-native",
            "first_sample_time_stamp": first_sample_time_stamp,
            "aac_bitrate": aac_bitrate,
            "compression_type": "lossy"
        }
        audio_blob_array.attrs.update(attrs)
        logger.trace("Audio blob array attributes set")
        
        # Create AAC index automatically
        logger.trace("Creating AAC index...")
        aac_index.build_aac_index(zarr_group, audio_blob_array)
        logger.trace("AAC index created")
        
        logger.success(f"AAC import completed successfully for '{audio_file.name}'")
        return audio_blob_array
        
    finally:
        # Clean up temporary file
        if tmp_file.exists():
            tmp_file.unlink()
            logger.trace("Temporary file cleaned up")


def _convert_to_aac_pyav(input_file: pathlib.Path, 
                        output_file: pathlib.Path,
                        bitrate: int,
                        source_params: dict):
    """
    Convert audio file to AAC using PyAV for native processing
    
    Args:
        input_file: Source audio file
        output_file: Target AAC file
        bitrate: Target bitrate in bits/second
        source_params: Source audio parameters
    """
    logger.trace(f"Converting {input_file.name} to AAC at {bitrate} bps using PyAV...")
    
    try:
        # Open input container
        input_container = av.open(str(input_file))
        input_stream = input_container.streams.audio[0]
        
        # Open output container with AAC format
        output_container = av.open(str(output_file), 'w', format='adts')  # ADTS format for raw AAC
        
        # Create AAC output stream
        output_stream = output_container.add_stream('aac')
        output_stream.bit_rate = bitrate
        output_stream.sample_rate = source_params["sampling_rate"]
        output_stream.channels = source_params["nb_channels"]
        
        # Set AAC-LC profile explicitly
        output_stream.codec_context.profile = "aac_low"
        
        logger.trace(f"AAC stream configured: {output_stream.sample_rate}Hz, {output_stream.channels}ch, {output_stream.bit_rate}bps")
        
        # Process audio frames
        frame_count = 0
        for packet in input_container.demux(input_stream):
            for frame in packet.decode():
                # Ensure frame has correct format for AAC encoding
                frame.pts = None  # Let encoder set PTS
                
                # Encode frame to AAC
                for output_packet in output_stream.encode(frame):
                    output_container.mux(output_packet)
                
                frame_count += 1
                if frame_count % 1000 == 0:
                    logger.trace(f"Processed {frame_count} frames...")
        
        # Flush encoder
        for output_packet in output_stream.encode():
            output_container.mux(output_packet)
        
        # Close containers
        output_container.close()
        input_container.close()
        
        logger.trace(f"AAC conversion completed. Processed {frame_count} frames.")
        
    except Exception as e:
        logger.error(f"PyAV AAC conversion failed: {e}")
        # Fallback to ffmpeg if PyAV fails
        logger.trace("Falling back to ffmpeg for AAC conversion...")
        _convert_to_aac_ffmpeg_fallback(input_file, output_file, bitrate, source_params)


def _convert_to_aac_ffmpeg_fallback(input_file: pathlib.Path, 
                                   output_file: pathlib.Path,
                                   bitrate: int,
                                   source_params: dict):
    """
    Fallback AAC conversion using ffmpeg subprocess
    
    Args:
        input_file: Source audio file
        output_file: Target AAC file  
        bitrate: Target bitrate in bits/second
        source_params: Source audio parameters
    """
    logger.trace(f"Using ffmpeg fallback for AAC conversion at {bitrate} bps...")
    
    # Prepare ffmpeg command for AAC conversion
    ffmpeg_cmd = ["ffmpeg", "-y"]
    ffmpeg_cmd += ["-i", str(input_file)]
    ffmpeg_cmd += ["-c:a", "aac"]
    ffmpeg_cmd += ["-profile:a", "aac_low"]  # AAC-LC profile
    ffmpeg_cmd += ["-b:a", str(bitrate)]
    ffmpeg_cmd += ["-f", "adts"]  # ADTS container for raw AAC
    ffmpeg_cmd += [str(output_file)]
    
    logger.trace(f"ffmpeg fallback command: {ffmpeg_cmd}")
    
    # Execute ffmpeg conversion
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.trace("ffmpeg fallback conversion completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg fallback conversion failed: {e}")
        raise ValueError(f"Both PyAV and ffmpeg AAC conversion failed")


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
    """
    Extract audio segment based on AAC index
    
    Args:
        zarr_group: Zarr group with AAC index
        audio_blob_array: Array with AAC audio data
        start_sample: First sample (inclusive)
        end_sample: Last sample (inclusive)
        dtype: Output data type
        
    Returns:
        Decoded audio data as numpy array
    """
    try:
        # Load index from Zarr group
        if 'aac_index' not in zarr_group:
            raise ValueError("AAC index not found. Must be created first with build_aac_index().")
        
        aac_index_array = zarr_group['aac_index']
        
        # Find frame range for sample range
        start_idx, end_idx = aac_index._find_frame_range_for_samples(
            aac_index_array, start_sample, end_sample
        )
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        # Load audio bytes and create temporary file
        audio_bytes = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_aac_file(audio_bytes)
        
        try:
            # Decode with PyAV
            container = av.open(temp_file_path)
            audio_stream = container.streams.audio[0]
            
            # Calculate frame positions from index
            sample_positions = aac_index_array[:, aac_index.AAC_INDEX_COL_SAMPLE_POS]
            first_frame_sample = sample_positions[start_idx]
            
            # Seek to approximate position
            seek_time = first_frame_sample / audio_stream.sample_rate
            container.seek(int(seek_time * 1000000))  # Convert to microseconds
            
            # Decode frames and collect samples
            decoded_samples = []
            current_sample = 0
            
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    frame_array = frame.to_ndarray(dtype=np.float32)
                    
                    # Convert to target dtype if needed
                    if dtype != np.float32:
                        if dtype == np.int16:
                            frame_array = (frame_array * 32767).astype(np.int16)
                        elif dtype == np.int32:
                            frame_array = (frame_array * 2147483647).astype(np.int32)
                        else:
                            frame_array = frame_array.astype(dtype)
                    
                    decoded_samples.append(frame_array)
                    current_sample += frame_array.shape[0]
                    
                    # Stop when we have enough samples
                    if current_sample >= (end_sample - start_sample + 1000):  # Add some buffer
                        break
                
                if current_sample >= (end_sample - start_sample + 1000):
                    break
            
            # Concatenate all decoded samples
            if decoded_samples:
                full_audio = np.concatenate(decoded_samples, axis=0)
                
                # Cut exactly to requested range
                start_offset = max(0, start_sample - first_frame_sample)
                end_offset = min(
                    start_offset + (end_sample - start_sample + 1), 
                    full_audio.shape[0]
                )
                
                return full_audio[start_offset:end_offset]
            else:
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
    Parallel extraction of multiple audio segments
    
    Args:
        zarr_group: Zarr group with AAC index
        audio_blob_array: Array with AAC audio data  
        segments: List of (start_sample, end_sample) tuples
        dtype: Output data type
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of decoded audio arrays in original order
    """
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
    Create AAC index (convenience wrapper)
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with AAC audio data
        
    Returns:
        Created index array
    """
    return aac_index.build_aac_index(zarr_group, audio_blob_array)


logger.trace("AAC Access API module loaded.")
