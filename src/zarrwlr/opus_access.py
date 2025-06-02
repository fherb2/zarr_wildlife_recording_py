"""
Opus Access Public API Module - Packet-Based Implementation
===========================================================

KEY IMPROVEMENTS:
- Packet-based storage instead of OGG container
- opuslib for direct decoding (no ffmpeg for extraction)
- Sample-accurate random access
- Eliminates ffmpeg startup overhead for small segments

Essential Functions for aimport.py compatibility:
- import_opus_to_zarr()
- extract_audio_segment_opus() 
- parallel_extract_audio_segments_opus()
"""

import zarr
import numpy as np
import tempfile
import os
import pathlib
import subprocess
import time
import struct
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import struct 
from typing import Dict, Optional 

# Import backend module - LAZY IMPORT to avoid circular imports
opus_index = None
from .utils import file_size
from .config import Config

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)

# Try to import opuslib for direct decoding
try:
    import opuslib
    OPUSLIB_AVAILABLE = True
    logger.trace("opuslib available for direct Opus decoding")
except ImportError:
    OPUSLIB_AVAILABLE = False
    logger.warning("opuslib not available - falling back to ffmpeg for extraction")

# Constants
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"  # Legacy OGG container
OPUS_PACKETS_BLOB_ARRAY_NAME = "opus_packets_blob"   # NEW: Raw packets
OPUS_PACKET_INDEX_ARRAY_NAME = "opus_packet_index"   # NEW: Packet index
OPUS_HEADER_ARRAY_NAME = "opus_header"               # NEW: OpusHead

# OGG/Opus constants for parsing
OGG_PAGE_HEADER_SIZE = 27
OGG_SYNC_PATTERN = b'OggS'
OPUS_HEAD_MAGIC = b'OpusHead'
OPUS_TAGS_MAGIC = b'OpusTags'


class OpusPacketExtractor:
    """Extract raw Opus packets from OGG container"""
    
    def __init__(self, ogg_data: bytes):
        self.ogg_data = ogg_data
        self.pos = 0
        self.packets = []
        self.opus_header = None
        self.total_samples = 0
        
    def extract_all_packets(self) -> Tuple[List[bytes], bytes, int]:
        """
        Extract all Opus packets from OGG container
        
        Returns:
            Tuple of (packets_list, opus_header, total_samples)
        """
        logger.trace("Extracting Opus packets from OGG container...")
        
        while self.pos < len(self.ogg_data) - OGG_PAGE_HEADER_SIZE:
            if self._find_next_page():
                self._process_page()
        
        logger.trace(f"Extracted {len(self.packets)} Opus packets, total samples: {self.total_samples}")
        return self.packets, self.opus_header, self.total_samples
    
    def _find_next_page(self) -> bool:
        """Find next OGG page"""
        while self.pos < len(self.ogg_data) - 4:
            if self.ogg_data[self.pos:self.pos+4] == OGG_SYNC_PATTERN:
                return True
            self.pos += 1
        return False
    
    def _process_page(self):
        """Process a single OGG page and extract packets"""
        if self.pos + OGG_PAGE_HEADER_SIZE > len(self.ogg_data):
            return
        
        # Parse OGG page header
        header = struct.unpack('<4sBBQIIIB', self.ogg_data[self.pos:self.pos + OGG_PAGE_HEADER_SIZE])
        granule_position = header[3]
        segment_count = header[7]
        
        # Read segment table
        seg_table_start = self.pos + OGG_PAGE_HEADER_SIZE
        if seg_table_start + segment_count > len(self.ogg_data):
            return
        
        segment_table = self.ogg_data[seg_table_start:seg_table_start + segment_count]
        
        # Process segments (Opus packets)
        packet_start = seg_table_start + segment_count
        for segment_size in segment_table:
            if packet_start + segment_size > len(self.ogg_data):
                break
            
            packet_data = self.ogg_data[packet_start:packet_start + segment_size]
            self._process_packet(packet_data, granule_position)
            packet_start += segment_size
        
        # Update position to next page
        total_page_size = OGG_PAGE_HEADER_SIZE + segment_count + sum(segment_table)
        self.pos += total_page_size
    
    def _process_packet(self, packet_data: bytes, granule_position: int):
        """Process individual packet"""
        if not packet_data:
            return
        
        # Check if this is OpusHead header
        if packet_data.startswith(OPUS_HEAD_MAGIC):
            self.opus_header = packet_data
            logger.trace("Found OpusHead header packet")
            return
        
        # Skip OpusTags
        if packet_data.startswith(OPUS_TAGS_MAGIC):
            logger.trace("Skipping OpusTags packet")
            return
        
        # Regular Opus audio packet
        self.packets.append(packet_data)
        
        # Update total samples from granule position
        if granule_position != 0xFFFFFFFFFFFFFFFF:
            self.total_samples = granule_position


def _extract_opus_packets_from_ogg(ogg_data: bytes) -> Tuple[List[bytes], bytes, int]:
    """Extract Opus packets from OGG container"""
    extractor = OpusPacketExtractor(ogg_data)
    return extractor.extract_all_packets()


def _create_packet_based_zarr_structure(zarr_group: zarr.Group, packets: List[bytes], 
                                       opus_header: bytes, source_params: dict,
                                       first_sample_time_stamp, opus_bitrate: int) -> zarr.Array:
    """
    Create packet-based Zarr structure for efficient random access
    
    Args:
        zarr_group: Zarr group to store data
        packets: List of raw Opus packets
        opus_header: OpusHead header packet
        source_params: Source audio parameters
        first_sample_time_stamp: First sample timestamp
        opus_bitrate: Opus bitrate
        
    Returns:
        Packet index array for compatibility
    """
    logger.trace("Creating packet-based Zarr structure...")
    
    # 1. Store raw packets as concatenated blob
    total_packet_size = sum(len(packet) for packet in packets)
    packet_blob = zarr_group.create_array(
        name=OPUS_PACKETS_BLOB_ARRAY_NAME,
        compressor=None,
        shape=(total_packet_size,),
        chunks=(Config.original_audio_chunk_size,),
        shards=(Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size,),
        dtype=np.uint8,
        overwrite=True,
    )
    
    # 2. Create packet index: [offset, size, samples_per_packet, cumulative_samples]
    packet_index_data = []
    offset = 0
    cumulative_samples = 0
    
    for packet in packets:
        packet_size = len(packet)
        
        # Estimate samples per packet (typical Opus frame: 960 samples at 48kHz)
        # TODO: Could parse packet header for exact sample count
        samples_per_packet = 960  # Standard 20ms frame at 48kHz
        
        packet_index_data.append([offset, packet_size, samples_per_packet, cumulative_samples])
        
        offset += packet_size
        cumulative_samples += samples_per_packet
    
    packet_index = zarr_group.create_array(
        name=OPUS_PACKET_INDEX_ARRAY_NAME,
        shape=(len(packets), 4),
        chunks=(min(1000, len(packets)), 4),
        dtype=np.uint64,
        overwrite=True,
    )
    
    # 3. Store OpusHead header
    if opus_header:
        header_array = zarr_group.create_array(
            name=OPUS_HEADER_ARRAY_NAME,
            shape=(len(opus_header),),
            chunks=(len(opus_header),),
            dtype=np.uint8,
            overwrite=True,
        )
        header_array[:] = np.frombuffer(opus_header, dtype=np.uint8)
        logger.trace("Stored OpusHead header for decoder initialization")
    
    # 4. Write packet data
    logger.trace("Writing packet data to Zarr arrays...")
    
    # Write concatenated packets
    offset = 0
    for packet in packets:
        packet_array = np.frombuffer(packet, dtype=np.uint8)
        packet_blob[offset:offset + len(packet_array)] = packet_array
        offset += len(packet_array)
    
    # Write packet index
    packet_index[:] = np.array(packet_index_data, dtype=np.uint64)
    
    # 5. Set metadata
    source_sample_rate = source_params.get("sampling_rate", 48000)
    source_channels = source_params.get("nb_channels", 1)
    is_ultrasonic = source_sample_rate > 48000
    sampling_rescale_factor = source_sample_rate / 48000.0 if is_ultrasonic else 1.0
    target_sample_rate = 48000 if is_ultrasonic else source_sample_rate
    
    attrs = {
        "codec": "opus",
        "nb_channels": source_channels,
        "sample_rate": target_sample_rate,
        "sampling_rescale_factor": sampling_rescale_factor,
        "container_type": "packet_based",  # NEW: Not OGG anymore
        "first_sample_time_stamp": first_sample_time_stamp,
        "opus_bitrate": opus_bitrate,
        "is_ultrasonic": is_ultrasonic,
        "original_sample_rate": source_sample_rate,
        "total_packets": len(packets),
        "packet_based_format": True,  # NEW: Flag for packet-based format
        "estimated_total_samples": cumulative_samples
    }
    
    # Apply metadata to all arrays
    packet_blob.attrs.update(attrs)
    packet_index.attrs.update(attrs)
    
    logger.trace(f"Created packet-based structure: {len(packets)} packets, {total_packet_size} bytes")
    return packet_index


def _create_temporary_ogg_file(audio_bytes: bytes) -> str:
    """Create temporary OGG file for ffmpeg access (legacy support)"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def _get_packet_data(zarr_group: zarr.Group, packet_idx: int) -> bytes:
    """Get raw packet data by index"""
    packet_index = zarr_group[OPUS_PACKET_INDEX_ARRAY_NAME]
    packet_blob = zarr_group[OPUS_PACKETS_BLOB_ARRAY_NAME]
    
    offset, size, _, _ = packet_index[packet_idx]
    return bytes(packet_blob[offset:offset + size])


def _find_packets_for_sample_range(zarr_group: zarr.Group, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """Find packet range that covers the requested sample range"""
    packet_index = zarr_group[OPUS_PACKET_INDEX_ARRAY_NAME]
    cumulative_samples = packet_index[:, 3]  # Column 3: cumulative samples
    
    # Find start packet
    start_packet_idx = np.searchsorted(cumulative_samples, start_sample, side='right') - 1
    start_packet_idx = max(0, start_packet_idx)
    
    # Find end packet
    end_packet_idx = np.searchsorted(cumulative_samples, end_sample, side='right')
    end_packet_idx = min(end_packet_idx, packet_index.shape[0] - 1)
    
    return int(start_packet_idx), int(end_packet_idx)


def extract_audio_segment_opus(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                              start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """
    Extract single audio segment using packet-based access or ffmpeg fallback
    
    Args:
        zarr_group: Zarr group with audio data
        audio_blob_array: Audio blob array (for legacy compatibility)
        start_sample: First sample to extract
        end_sample: Last sample to extract
        dtype: Output data type
        
    Returns:
        Extracted audio segment as numpy array
    """
    # Check if packet-based format is available
    if (OPUS_PACKET_INDEX_ARRAY_NAME in zarr_group and 
        OPUS_PACKETS_BLOB_ARRAY_NAME in zarr_group and 
        OPUSLIB_AVAILABLE):
        
        return _extract_segment_packet_based(zarr_group, start_sample, end_sample, dtype)
    else:
        # Fallback to legacy OGG container method
        logger.trace("Using legacy OGG container extraction")
        return _extract_segment_legacy(zarr_group, audio_blob_array, start_sample, end_sample, dtype)


def _extract_segment_packet_based(zarr_group: zarr.Group, start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """Extract segment using packet-based structure with opuslib"""
    try:
        # Get audio parameters
        packet_index = zarr_group[OPUS_PACKET_INDEX_ARRAY_NAME]
        sample_rate = packet_index.attrs.get('sample_rate', 48000)
        channels = packet_index.attrs.get('nb_channels', 1)
        
        # Initialize Opus decoder
        decoder = opuslib.Decoder(sample_rate, channels)
        
        # Find relevant packets
        start_packet_idx, end_packet_idx = _find_packets_for_sample_range(zarr_group, start_sample, end_sample)
        
        logger.trace(f"Extracting samples {start_sample}-{end_sample} using packets {start_packet_idx}-{end_packet_idx}")
        
        # Decode relevant packets
        pcm_data = []
        current_sample = 0
        
        for packet_idx in range(start_packet_idx, end_packet_idx + 1):
            packet_data = _get_packet_data(zarr_group, packet_idx)
            
            # Decode packet
            frame_samples = decoder.decode(packet_data)
            
            # Convert to requested dtype
            if dtype == np.int16:
                if frame_samples.dtype != np.int16:
                    frame_samples = (frame_samples * 32767).astype(np.int16)
            elif dtype == np.float32:
                if frame_samples.dtype != np.float32:
                    frame_samples = frame_samples.astype(np.float32) / 32767.0
            
            pcm_data.append(frame_samples)
            current_sample += frame_samples.shape[0] // channels
        
        # Concatenate all frames
        if not pcm_data:
            return np.array([])
        
        full_audio = np.concatenate(pcm_data, axis=0)
        
        # Trim to exact sample range
        samples_per_channel = full_audio.shape[0] // channels if channels > 1 else full_audio.shape[0]
        start_offset = max(0, start_sample - (start_packet_idx * 960))  # Approximate packet start
        end_offset = min(samples_per_channel, start_offset + (end_sample - start_sample + 1))
        
        if channels > 1:
            trimmed = full_audio[start_offset * channels:end_offset * channels]
            return trimmed.reshape(-1, channels)
        else:
            return full_audio[start_offset:end_offset]
            
    except Exception as e:
        logger.error(f"Error in packet-based extraction: {e}")
        return np.array([])


def _extract_segment_legacy(zarr_group: zarr.Group, audio_blob_array: zarr.Array,
                           start_sample: int, end_sample: int, dtype=np.int16) -> np.ndarray:
    """Legacy extraction using OGG container and ffmpeg"""
    try:
        # Load index from Zarr group
        if 'opus_index' not in zarr_group:
            raise ValueError("Opus index not found. Must be created first with build_opus_index().")
        
        opus_index_array = zarr_group['opus_index']
        
        # Get audio parameters
        sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
        channels = audio_blob_array.attrs.get('nb_channels', 1)
        
        # Find page range for sample range
        opus_backend = _get_opus_index_backend()
        start_idx, end_idx = opus_backend._find_page_range_for_samples(
            opus_index_array, start_sample, end_sample
        )
        
        if start_idx > end_idx:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")
        
        # Load complete OGG data and create temporary file for ffmpeg
        complete_ogg_data = bytes(audio_blob_array[()])
        temp_file_path = _create_temporary_ogg_file(complete_ogg_data)
        
        try:
            # Calculate time positions for ffmpeg seeking
            sample_positions = opus_index_array[:, 2]  # OPUS_INDEX_COL_SAMPLE_POS from backend
            actual_start_sample = int(sample_positions[start_idx])
            
            # Convert sample positions to time for ffmpeg
            start_time_seconds = actual_start_sample / sample_rate
            duration_samples = end_sample - start_sample + 1
            duration_seconds = duration_samples / sample_rate
            
            # Decode with ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time_seconds),
                "-t", str(duration_seconds),
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
                samples = samples[:samples.size - (samples.size % channels)]
            
            if channels > 1:
                samples = samples.reshape(-1, channels)
            
            return samples
                
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error extracting Opus segment [{start_sample}:{end_sample}]: {e}")
        return np.array([])


def parallel_extract_audio_segments_opus(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                                        segments: List[Tuple[int, int]], dtype=np.int16, 
                                        max_workers: int = 4) -> List[np.ndarray]:
    """Parallel extraction using packet-based access or ffmpeg fallback"""
    
    # AUTO-DETECT FORMAT
    has_packet_format = (
        OPUS_PACKET_INDEX_ARRAY_NAME in zarr_group and 
        OPUS_PACKETS_BLOB_ARRAY_NAME in zarr_group
    )
    
    has_legacy_format = (
        audio_blob_array is not None and 
        'opus_index' in zarr_group
    )
    
    if has_packet_format:
        logger.trace(f"Using packet-based parallel extraction for {len(segments)} segments")
        # Use packet-based extraction for all segments
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
                    logger.error(f"Error in packet-based parallel extraction of segment {segment}: {e}")
                    results[segment] = np.array([])
            
            # Return results in original order
            return [results[segment] for segment in segments]
    
    elif has_legacy_format:
        logger.trace(f"Using legacy parallel extraction for {len(segments)} segments")
        # Use legacy method
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {
                executor.submit(
                    _extract_segment_legacy, 
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
                    logger.error(f"Error in legacy parallel extraction of segment {segment}: {e}")
                    results[segment] = np.array([])
            
            # Return results in original order
            return [results[segment] for segment in segments]
    
    else:
        raise ValueError("No supported Opus format found for parallel extraction")


def import_opus_to_zarr(zarr_group: zarr.Group, 
                       audio_file: str | pathlib.Path,
                       source_params: dict,
                       first_sample_time_stamp,
                       opus_bitrate: int = 160000,
                       temp_dir: str = "/tmp") -> zarr.Array:
    """
    FIXED: Dynamic timeout calculation + optimized ffmpeg parameters
    
    Timeout Strategy:
    - Base timeout: 60 seconds
    - Dynamic scaling: +10 seconds per 100MB
    - Opus sources: Faster (copy mode)
    - Transcoding: Slower (requires encoding)
    """
    import time
    start_time = time.time()
    
    print(f"🚀 DEBUG: import_opus_to_zarr() started at {time.time():.3f}")
    logger.trace(f"import_opus_to_zarr() unified pipeline for '{audio_file}'")
    
    audio_file = pathlib.Path(audio_file)
    print(f"📁 DEBUG: Processing file: {audio_file}")
    
    # Get file size for DYNAMIC TIMEOUT calculation
    try:
        file_size_mb = audio_file.stat().st_size / 1024 / 1024
        print(f"📏 DEBUG: Input file size: {file_size_mb:.1f} MB")
    except Exception:
        file_size_mb = 100  # Conservative fallback
        print("⚠️ DEBUG: Could not get file size, using 100MB estimate")
    
    # Extract source parameters
    source_sample_rate = source_params.get("sampling_rate", 48000)
    source_channels = source_params.get("nb_channels", 1)
    is_opus_source = source_params.get("is_opus", False)
    
    print(f"📊 DEBUG: Source params - Rate: {source_sample_rate}Hz, Channels: {source_channels}, Opus: {is_opus_source}")
    
    # ULTRASONIC DETECTION AND HANDLING
    is_ultrasonic = source_sample_rate > 48000
    sampling_rescale_factor = 1.0
    target_sample_rate = source_sample_rate
    
    if is_ultrasonic:
        sampling_rescale_factor = float(source_sample_rate) / 48000.0
        target_sample_rate = 48000
        print(f"🔊 DEBUG: Ultrasonic detected: {source_sample_rate}Hz -> 48kHz (factor: {sampling_rescale_factor:.3f})")
    else:
        print(f"🔊 DEBUG: Normal sample rate: {source_sample_rate}Hz")
    
    # DYNAMIC TIMEOUT CALCULATION (based on file size and processing type)
    base_timeout = 60  # Base 60 seconds
    
    if is_opus_source and not is_ultrasonic:
        # Opus copy mode: Very fast (mostly I/O bound)
        timeout_per_mb = 2  # 2 seconds per MB
        processing_mode = "copy"
    else:
        # Transcoding mode: CPU intensive
        timeout_per_mb = 8  # 8 seconds per MB for transcoding
        processing_mode = "transcode"
    
    # Calculate dynamic timeout with reasonable limits  
    dynamic_timeout = int(base_timeout + (file_size_mb * timeout_per_mb))
    dynamic_timeout = max(60, min(dynamic_timeout, 600))  # Between 60s and 10min
    
    print(f"⏱️ DEBUG: Timeout calculation - Mode: {processing_mode}, File: {file_size_mb:.1f}MB")
    print(f"⏱️ DEBUG: Dynamic timeout: {dynamic_timeout:.0f}s ({dynamic_timeout/60:.1f}min)")
    
    # Create temporary raw Opus file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.opus', dir=temp_dir) as tmp_out:
        tmp_file = pathlib.Path(tmp_out.name)
    
    print(f"📝 DEBUG: Temporary file created: {tmp_file}")
    
    try:
        print(f"⚙️ DEBUG: Starting ffmpeg conversion at {time.time() - start_time:.3f}s")
        
        # OPTIMIZED FFMPEG COMMAND (based on search results)
        ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"]
        ffmpeg_cmd += ["-i", str(audio_file)]
        
        # Handle ultrasonic sources (resample to 48kHz)
        if is_ultrasonic:
            ffmpeg_cmd += ["-ar", "48000"]
            print("🔊 DEBUG: Added ultrasonic resampling")
        
        # Smart codec handling with PERFORMANCE OPTIMIZATIONS
        if is_opus_source and not is_ultrasonic:
            # Opus source without ultrasonic: EXTRACT without recompression
            ffmpeg_cmd += ["-c:a", "copy"]
            print("📋 DEBUG: Using codec copy for Opus source (no recompression)")
        else:
            # Non-Opus source OR ultrasonic Opus: TRANSCODE to Opus
            # PERFORMANCE OPTIMIZATION: Use faster preset and threading
            ffmpeg_cmd += [
                "-c:a", "libopus", 
                "-b:a", str(int(opus_bitrate)),
                "-application", "audio",  # Optimize for audio (not voice)
                "-frame_duration", "20",  # 20ms frames (default, good balance)
                "-packet_loss", "0"       # No packet loss expected
            ]
            
            # Add threading for faster encoding (CPU cores)
            import os
            cpu_count = os.cpu_count() or 4
            thread_count = min(cpu_count, 8)  # Use max 8 threads
            # Note: Opus encoding is not heavily threaded, but this helps with demuxing
            
            if is_opus_source:
                print("🔄 DEBUG: Transcoding Opus source due to ultrasonic resampling")
            else:
                print(f"🔄 DEBUG: Transcoding {source_params.get('codec_name', 'unknown')} source to Opus")
        
        # Output format: RAW OPUS (no container)
        ffmpeg_cmd += ["-f", "opus", str(tmp_file)]
        
        print(f"🎬 DEBUG: ffmpeg command: {' '.join(ffmpeg_cmd)}")
        
        # IMPROVED PROCESS EXECUTION with proper timeout handling
        print(f"▶️ DEBUG: Starting ffmpeg with {dynamic_timeout:.0f}s timeout at {time.time() - start_time:.3f}s")
        
        try:
            # Use Popen with timeout (Python 3.3+)
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                universal_newlines=True
            )
            
            # Monitor output with timeout
            output_lines = []
            progress_count = 0
            last_progress_time = time.time()
            
            while True:
                try:
                    # Check for new output (non-blocking with short timeout)
                    import select
                    import sys
                    
                    if sys.platform != 'win32':
                        # Unix-like systems: use select
                        ready, _, _ = select.select([process.stdout], [], [], 1.0)
                        if ready:
                            line = process.stdout.readline()
                        else:
                            line = ""
                    else:
                        # Windows: fallback to blocking read with timeout via threading
                        import threading
                        import queue
                        
                        def read_line(proc, q):
                            try:
                                line = proc.stdout.readline()
                                q.put(line)
                            except:
                                q.put("")
                        
                        q = queue.Queue()
                        t = threading.Thread(target=read_line, args=(process, q))
                        t.daemon = True
                        t.start()
                        
                        try:
                            line = q.get(timeout=1.0)
                        except queue.Empty:
                            line = ""
                    
                    if line:
                        line = line.strip()
                        output_lines.append(line)
                        last_progress_time = time.time()  # Reset timeout on activity
                        
                        # Show progress periodically
                        if 'time=' in line and 'speed=' in line:
                            progress_count += 1
                            if progress_count % 20 == 1 or file_size_mb < 50:  # Show less for large files
                                elapsed = time.time() - start_time
                                print(f"📈 DEBUG: [{elapsed:.0f}s] {line}")
                        elif 'error' in line.lower() or 'failed' in line.lower():
                            print(f"❌ DEBUG: {line}")
                        elif any(keyword in line for keyword in ['Input #0', 'Output #0', 'Stream mapping']):
                            print(f"ℹ️ DEBUG: {line}")
                    
                    # Check if process finished
                    if process.poll() is not None:
                        break
                    
                    # Check for timeout (both total and inactivity)
                    current_time = time.time()
                    total_elapsed = current_time - start_time
                    inactivity_time = current_time - last_progress_time
                    
                    if total_elapsed > dynamic_timeout:
                        print(f"⏰ DEBUG: TOTAL TIMEOUT after {total_elapsed:.0f}s (limit: {dynamic_timeout:.0f}s)")
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        raise TimeoutError(f"ffmpeg total timeout after {total_elapsed:.0f}s")
                    
                    if inactivity_time > 60:  # 60 seconds without any output
                        print(f"⏰ DEBUG: INACTIVITY TIMEOUT after {inactivity_time:.0f}s without output")
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        raise TimeoutError(f"ffmpeg inactivity timeout after {inactivity_time:.0f}s")
                
                except Exception as e:
                    if "timeout" in str(e).lower():
                        raise
                    else:
                        print(f"⚠️ DEBUG: Exception in output monitoring: {e}")
                        continue
            
            # Get final return code
            return_code = process.returncode
            conversion_time = time.time() - start_time
            
            print(f"✅ DEBUG: ffmpeg completed with code {return_code} in {conversion_time:.1f}s")
            
            if return_code != 0:
                # Show last few lines for error diagnosis
                error_lines = output_lines[-5:] if output_lines else ['No output captured']
                error_msg = '\n'.join(error_lines)
                print(f"❌ DEBUG: ffmpeg error output (last 5 lines):\n{error_msg}")
                raise subprocess.CalledProcessError(return_code, ffmpeg_cmd, output=error_msg)
            
        except TimeoutError as e:
            print(f"⏰ DEBUG: ffmpeg TIMEOUT: {e}")
            raise RuntimeError(f"ffmpeg conversion timed out: {e}")
        except Exception as e:
            print(f"❌ DEBUG: ffmpeg subprocess failed: {e}")
            raise RuntimeError(f"ffmpeg conversion failed: {e}")
        
        # Read and validate raw Opus data
        print(f"📖 DEBUG: Reading raw Opus data at {time.time() - start_time:.3f}s")
        
        if not tmp_file.exists():
            print("❌ DEBUG: Temporary file does not exist!")
            raise RuntimeError("ffmpeg did not create output file")
        
        output_file_size = tmp_file.stat().st_size
        print(f"📏 DEBUG: Output file size: {output_file_size} bytes ({output_file_size/1024/1024:.1f}MB)")
        
        if output_file_size == 0:
            print("❌ DEBUG: ffmpeg created empty file!")
            raise RuntimeError("ffmpeg created empty output file")
        
        with open(tmp_file, "rb") as f:
            raw_opus_data = f.read()
        
        print(f"💾 DEBUG: Raw Opus data loaded: {len(raw_opus_data)} bytes at {time.time() - start_time:.3f}s")
        
        # Validate Opus data
        if len(raw_opus_data) < 100:
            print(f"⚠️ DEBUG: Suspiciously small Opus output: {len(raw_opus_data)} bytes")
            raise ValueError("ffmpeg produced unusually small Opus output")
        
        if b'OpusHead' not in raw_opus_data[:2000]:
            print("⚠️ DEBUG: OpusHead not found in first 2000 bytes")
            if b'OpusHead' not in raw_opus_data:
                print("❌ DEBUG: No OpusHead found anywhere in data!")
                raise ValueError("No valid Opus data found - ffmpeg may have failed silently")
            else:
                opus_head_pos = raw_opus_data.find(b'OpusHead')
                print(f"🎯 DEBUG: OpusHead found at position {opus_head_pos}")
        else:
            print("✅ DEBUG: OpusHead found in first 2000 bytes")
        
        print(f"🔍 DEBUG: Starting packet parsing at {time.time() - start_time:.3f}s")
        
        # PARSE RAW OPUS (with timeout protection)
        try:
            print("🏗️ DEBUG: Creating RawOpusParser...")
            parser = RawOpusParser(raw_opus_data)
            print("🏗️ DEBUG: RawOpusParser created, calling extract_packets()...")
            
            # Reasonable timeout for packet extraction
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Packet extraction timed out")
            
            # Use smaller timeout for packet parsing (should be fast)
            parsing_timeout = min(120, dynamic_timeout // 4)  # Max 2min for parsing
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(parsing_timeout)
            
            try:
                packets, opus_header, total_samples = parser.extract_packets()
                signal.alarm(0)  # Cancel timeout
                print(f"✅ DEBUG: extract_packets() completed at {time.time() - start_time:.3f}s")
            except TimeoutError:
                print(f"⏰ DEBUG: Packet extraction TIMEOUT after {parsing_timeout}s!")
                signal.alarm(0)
                raise RuntimeError(f"Packet extraction timed out after {parsing_timeout}s - possible infinite loop in RawOpusParser")
            
            print(f"📦 DEBUG: Packets extracted: {len(packets) if packets else 0}")
            print(f"📋 DEBUG: Opus header: {'Found' if opus_header else 'Missing'}")
            print(f"🎵 DEBUG: Total samples: {total_samples}")
            
            if not packets:
                print("❌ DEBUG: No packets extracted!")
                raise ValueError("No Opus packets extracted from raw stream")
                
            if not opus_header:
                print("❌ DEBUG: No OpusHead header found!")
                raise ValueError("No OpusHead header found in raw stream")
            
            # Basic packet count validation
            if len(packets) > 100000:
                print(f"⚠️ DEBUG: Very large packet count: {len(packets)} - possible parsing issue")
                logger.warning(f"Large number of packets extracted: {len(packets)} for {file_size_mb:.1f}MB file")
            elif len(packets) < 10:
                print(f"⚠️ DEBUG: Small packet count: {len(packets)} - very short audio?")
            else:
                print(f"✅ DEBUG: Reasonable packet count: {len(packets)} for {file_size_mb:.1f}MB file")
            
        except Exception as e:
            print(f"❌ DEBUG: Packet parsing failed: {e}")
            raise RuntimeError(f"Opus packet parsing failed: {e}")
        
        # CREATE PACKET-BASED ZARR STRUCTURE (following FLAC pattern exactly)
        try:
            print(f"🏗️ DEBUG: Creating Zarr structure at {time.time() - start_time:.3f}s")
            
            # Follow FLAC pattern: get file size first
            opus_data_size = len(raw_opus_data)
            print(f"🏗️ DEBUG: Opus data size for Zarr: {opus_data_size} bytes")
            
            # Create audio blob array (EXACTLY like FLAC)
            audio_blob_array = zarr_group.create_array(
                name=AUDIO_DATA_BLOB_ARRAY_NAME,  # Use same name as FLAC!
                compressor=None,
                shape=(opus_data_size,),
                chunks=(Config.original_audio_chunk_size,),
                shards=(Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size,),
                dtype=np.uint8,
                overwrite=True,
            )
            print(f"🏗️ DEBUG: Zarr audio blob array created")
            
            # Write raw opus data (EXACTLY like FLAC writes FLAC data)
            print(f"🏗️ DEBUG: Writing {opus_data_size} bytes to Zarr...")
            
            # Use FLAC's proven buffered write pattern
            offset = 0
            buffer_calculation = Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size
            max_buffer_size = max(1, min(int(buffer_calculation), 100_000_000))
            
            # Write in chunks like FLAC does
            pos = 0
            while pos < len(raw_opus_data):
                chunk_size = min(max_buffer_size, len(raw_opus_data) - pos)
                chunk_data = raw_opus_data[pos:pos + chunk_size]
                buffer_array = np.frombuffer(chunk_data, dtype="u1")
                audio_blob_array[offset:offset + len(buffer_array)] = buffer_array
                offset += len(buffer_array)
                pos += chunk_size
                
                # Progress for large files
                if opus_data_size > 50*1024*1024:  # Show progress for >50MB
                    progress = (pos * 100) // len(raw_opus_data)
                    print(f"🏗️ DEBUG: Writing progress: {progress:.1f}%")
            
            print(f"🏗️ DEBUG: Zarr data written successfully")
            
            # Set attributes (follow FLAC pattern exactly)
            source_sample_rate = source_params.get("sampling_rate", 48000)
            source_channels = source_params.get("nb_channels", 1)
            is_ultrasonic = source_sample_rate > 48000
            sampling_rescale_factor = source_sample_rate / 48000.0 if is_ultrasonic else 1.0
            target_sample_rate = 48000 if is_ultrasonic else source_sample_rate
            
            attrs = {
                "codec": "opus",  # Different from FLAC
                "nb_channels": source_channels,
                "sample_rate": target_sample_rate,
                "sampling_rescale_factor": sampling_rescale_factor,
                "container_type": "opus-native",  # Like FLAC's "flac-native"
                "first_sample_time_stamp": first_sample_time_stamp,
                "opus_bitrate": opus_bitrate,  # Like FLAC's compression_level
                "is_ultrasonic": is_ultrasonic,
                "original_sample_rate": source_sample_rate,
            }
            
            audio_blob_array.attrs.update(attrs)
            print(f"🏗️ DEBUG: Zarr attributes set")
            
            # Create index automatically (like FLAC does)  
            print("🔍 DEBUG: Creating Opus index...")
            try:
                # Use relative import to avoid __init__.py dependency
                from . import opus_index_backend
                opus_index_backend.build_opus_index(zarr_group, audio_blob_array)
                print("✅ DEBUG: Opus index created successfully")
            except ImportError as import_err:
                print(f"⚠️ DEBUG: Index module import failed: {import_err} - continuing without index")
            except Exception as e:
                print(f"⚠️ DEBUG: Index creation failed: {e} - continuing without index")
            
            print(f"✅ DEBUG: Zarr structure created at {time.time() - start_time:.3f}s")
            return audio_blob_array  # Return audio blob like FLAC does
            
        except Exception as e:
            print(f"❌ DEBUG: Zarr structure creation failed: {e}")
            raise
        
        # Skip automatic index creation for very large files
        if len(raw_opus_data) > 50*1024*1024:  # 50MB threshold
            print("⏩ DEBUG: Skipping automatic index creation for large file (can be done separately)")
        
        total_time = time.time() - start_time
        print(f"🏁 DEBUG: Import completed successfully in {total_time:.1f}s")
        if file_size_mb > 0:
            print(f"📊 DEBUG: Performance: {file_size_mb/total_time:.1f} MB/s")
        
        return audio_blob_array  # Return the audio blob array like FLAC
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"💥 DEBUG: Import failed after {total_time:.1f}s: {e}")
        raise
        
    finally:
        # Clean up temporary file
        if tmp_file.exists():
            tmp_file.unlink()
            print("🧹 DEBUG: Temporary file cleaned up")           
            

def parallel_extract_audio_segments_opus_optimized(
    zarr_group: zarr.Group, audio_blob_array: zarr.Array,
    segments: List[Tuple[int, int]], 
    dtype=np.int16, max_workers: int = 4,
    max_batch_duration_seconds: float = 30.0,
    max_segments_per_batch: int = 50) -> List[np.ndarray]:
    """
    Batch-optimized extraction using packet-based structure
    
    NEW: Uses packet-based access for maximum efficiency
    """
    if (OPUS_PACKET_INDEX_ARRAY_NAME in zarr_group and 
        OPUS_PACKETS_BLOB_ARRAY_NAME in zarr_group and 
        OPUSLIB_AVAILABLE):
        
        logger.trace(f"Using packet-based batch extraction for {len(segments)} segments")
        return parallel_extract_audio_segments_opus(zarr_group, audio_blob_array, segments, dtype, max_workers)
    else:
        logger.trace(f"Using legacy extraction for {len(segments)} segments")
        return parallel_extract_audio_segments_opus(zarr_group, audio_blob_array, segments, dtype, max_workers)


def _get_opus_index_backend():
    """Lazy import opus_index_backend to avoid circular imports"""
    global opus_index
    if opus_index is None:
        # DIRECT import to avoid __init__.py circular dependency
        from . import opus_index_backend as backend
        opus_index = backend
    return opus_index


def build_opus_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array) -> zarr.Array:
    """Create Opus index (convenience wrapper for legacy compatibility)"""
    opus_backend = _get_opus_index_backend()
    return opus_backend.build_opus_index(zarr_group, audio_blob_array)


# Configuration flag for backward compatibility
if not hasattr(Config, 'keep_legacy_ogg_blob'):
    Config.keep_legacy_ogg_blob = False  # Default: no legacy blob


logger.trace("Opus Access API module loaded (packet-based implementation with opuslib support).")


# opus_access.py - STEP 1.1 ADDITIONS
# Add these components to the existing opus_access.py file

import struct
from typing import Dict, Optional

# Add after existing imports and constants

class OpusContainerDetector:
    """Smart detection and extraction of Opus streams from various containers"""
    
    @staticmethod
    def detect_format(audio_file: pathlib.Path) -> str:
        """
        Detect container format for Opus audio
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            'raw_opus', 'ogg', 'webm', 'mkv', 'mp4', 'other'
        """
        logger.trace(f"Detecting container format for {audio_file.name}")
        
        suffix = audio_file.suffix.lower()
        
        # File extension based detection (fast)
        extension_map = {
            '.opus': 'raw_opus',
            '.ogg': 'ogg', 
            '.oga': 'ogg',
            '.webm': 'webm',
            '.mkv': 'mkv',
            '.mp4': 'mp4',
            '.m4a': 'mp4',
        }
        
        if suffix in extension_map:
            detected = extension_map[suffix]
            logger.trace(f"Format detected by extension: {detected}")
            return detected
        
        # Fallback: File header analysis
        return OpusContainerDetector._detect_by_header(audio_file)
    
    @staticmethod
    def _detect_by_header(audio_file: pathlib.Path) -> str:
        """Detect format by file header magic bytes"""
        try:
            with open(audio_file, 'rb') as f:
                header = f.read(32)
            
            if header.startswith(b'OggS'):
                return 'ogg'
            elif header[4:8] == b'ftyp' or header.startswith(b'\x00\x00\x00\x20ftyp'):
                return 'mp4'
            elif header.startswith(b'\x1a\x45\xdf\xa3'):  # EBML header
                return 'webm'  # or mkv, but both use same extraction
            elif header.startswith(b'OpusHead'):
                return 'raw_opus'
            else:
                logger.warning(f"Unknown container format for {audio_file.name}")
                return 'other'
                
        except Exception as e:
            logger.error(f"Error detecting format for {audio_file.name}: {e}")
            return 'other'
    
    @staticmethod
    def can_extract_directly(audio_file: pathlib.Path, source_params: dict) -> bool:
        """
        Check if we can extract Opus directly without ffmpeg transcoding
        
        Args:
            audio_file: Path to audio file
            source_params: Source parameters from ffprobe
            
        Returns:
            True if direct extraction possible
        """
        is_opus = source_params.get('is_opus', False)
        if not is_opus:
            return False
            
        container_format = OpusContainerDetector.detect_format(audio_file)
        
        # We can extract directly from these formats
        supported_formats = ['raw_opus', 'ogg', 'webm', 'mkv']
        
        can_extract = container_format in supported_formats
        logger.trace(f"Direct extraction possible for {audio_file.name}: {can_extract} (format: {container_format})")
        
        return can_extract
    
    @staticmethod  
    def extract_raw_opus_stream(audio_file: pathlib.Path) -> bytes:
        """
        Extract raw Opus stream from any supported container
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Raw Opus stream data
            
        Raises:
            ValueError: If container format not supported for direct extraction
        """
        container_format = OpusContainerDetector.detect_format(audio_file)
        logger.trace(f"Extracting raw Opus from {container_format} container: {audio_file.name}")
        
        if container_format == 'raw_opus':
            return OpusContainerDetector._extract_from_raw_opus(audio_file)
        elif container_format == 'ogg':
            return OpusContainerDetector._extract_from_ogg(audio_file)
        elif container_format in ['webm', 'mkv']:
            return OpusContainerDetector._extract_from_matroska(audio_file)
        else:
            raise ValueError(f"Direct extraction not supported for {container_format} format")
    
    @staticmethod
    def _extract_from_raw_opus(audio_file: pathlib.Path) -> bytes:
        """Extract from raw .opus file (already raw Opus stream)"""
        logger.trace(f"Reading raw Opus file: {audio_file.name}")
        with open(audio_file, 'rb') as f:
            return f.read()
    
    @staticmethod
    def _extract_from_ogg(audio_file: pathlib.Path) -> bytes:
        """Extract Opus stream from OGG container using ffmpeg"""
        logger.trace(f"Extracting Opus from OGG container: {audio_file.name}")
        
        # Use ffmpeg to extract raw Opus from OGG
        with tempfile.NamedTemporaryFile(suffix='.opus') as tmp_opus:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(audio_file),
                '-c:a', 'copy', '-f', 'opus', tmp_opus.name
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                with open(tmp_opus.name, 'rb') as f:
                    return f.read()
                    
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to extract Opus from OGG: {e}")
    
    @staticmethod
    def _extract_from_matroska(audio_file: pathlib.Path) -> bytes:
        """Extract Opus stream from WebM/MKV container using ffmpeg"""
        logger.trace(f"Extracting Opus from Matroska container: {audio_file.name}")
        
        # Use ffmpeg to extract raw Opus from WebM/MKV
        with tempfile.NamedTemporaryFile(suffix='.opus') as tmp_opus:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(audio_file),
                '-c:a', 'copy', '-f', 'opus', tmp_opus.name
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                with open(tmp_opus.name, 'rb') as f:
                    return f.read()
                    
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to extract Opus from Matroska: {e}")



class RawOpusParser:
    """Parser for raw Opus streams (no container)"""
    
    def __init__(self, opus_data: bytes):
        """Initialize parser with raw Opus stream data"""
        self.opus_data = opus_data
        self.pos = 0
        logger.trace(f"RawOpusParser initialized with {len(opus_data)} bytes")
    
    def extract_packets(self) -> Tuple[List[bytes], bytes, int]:
        """Extract Opus packets from raw stream"""
        logger.trace("Extracting packets from raw Opus stream...")
        
        packets = []
        opus_header = None
        total_samples = 0
        
        try:
            # Parse OpusHead header
            opus_header = self._parse_opus_head()
            if not opus_header:
                raise ValueError("No OpusHead found in raw Opus stream")
            
            # Skip OpusTags (if present)
            self._skip_opus_tags()
            
            # Extract audio packets
            packets = self._extract_audio_packets()
            
            # Estimate total samples (each packet ≈ 960 samples at 48kHz)
            total_samples = len(packets) * 960
            
            logger.trace(f"Extracted {len(packets)} packets from raw Opus stream")
            return packets, opus_header, total_samples
            
        except Exception as e:
            logger.error(f"Error in extract_packets(): {e}")
            # Print detailed debug info
            logger.error(f"Parser state: pos={self.pos}, data_len={len(self.opus_data)}")
            raise
    
    def _extract_audio_packets(self) -> List[bytes]:
        """Extract audio packets from remaining stream - FIXED VERSION"""
        packets = []
        
        # Simple approach: Split remaining data into reasonable packet sizes
        remaining_data = self.opus_data[self.pos:]
        logger.trace(f"Extracting packets from {len(remaining_data)} bytes remaining data")
        
        if len(remaining_data) == 0:
            logger.warning("No remaining data for packet extraction")
            return packets
        
        # FIXED: More robust packet splitting with proper integer handling
        pos = 0
        packet_count = 0
        
        while pos < len(remaining_data) and packet_count < 100000:  # Safety limit
            # FIXED: Ensure all calculations use integers
            remaining_bytes = len(remaining_data) - pos
            
            if remaining_bytes <= 0:
                break
            
            # Estimate packet size (20-200 bytes typical for Opus)
            # FIXED: Force integer conversion and bounds checking
            base_packet_size = min(200, remaining_bytes)
            packet_size = int(base_packet_size)  # EXPLICIT int conversion
            
            # FIXED: Ensure packet_size is valid
            if packet_size <= 0:
                logger.warning(f"Invalid packet size {packet_size} at position {pos}")
                break
            
            # FIXED: Bounds check before slicing
            end_pos = min(pos + packet_size, len(remaining_data))
            
            if end_pos <= pos:
                logger.warning(f"Invalid end position {end_pos} <= {pos}")
                break
            
            # Extract packet
            packet = remaining_data[pos:end_pos]
            
            if len(packet) > 0:
                packets.append(packet)
                packet_count += 1
            
            # FIXED: Ensure pos increment is integer
            pos = int(end_pos)
            
            # Safety check: avoid infinite loops
            if packet_count % 10000 == 0:
                logger.trace(f"Processed {packet_count} packets, pos={pos}/{len(remaining_data)}")
        
        logger.trace(f"Extracted {len(packets)} audio packets from raw stream")
        return packets
    
    def _parse_opus_head(self) -> Optional[bytes]:
        """Parse OpusHead header from raw stream"""
        # Look for OpusHead magic
        opus_head_pos = self.opus_data.find(OPUS_HEAD_MAGIC)
        if opus_head_pos == -1:
            logger.error("OpusHead not found in raw Opus stream")
            return None
        
        self.pos = opus_head_pos
        
        # FIXED: Explicit integer conversion for header parsing
        header_min_size = 19
        if self.pos + header_min_size > len(self.opus_data):
            logger.error("Incomplete OpusHead in raw Opus stream")
            return None
        
        # Read basic header (19 bytes minimum)
        header_data = self.opus_data[self.pos:self.pos + header_min_size]
        
        # Parse channel mapping to determine total header size
        channel_mapping = header_data[18]
        header_size = header_min_size  # FIXED: Start with known integer
        
        if channel_mapping > 0:
            # Extended header with channel mapping table
            extended_header_size = 21
            if self.pos + extended_header_size > len(self.opus_data):
                logger.error("Incomplete OpusHead channel mapping")
                return None
            
            channel_count = header_data[9]  # Get channel count
            # FIXED: Explicit integer arithmetic
            header_size = int(extended_header_size + channel_count)
        
        # FIXED: Final bounds check with integer conversion
        if self.pos + header_size > len(self.opus_data):
            logger.error(f"Incomplete OpusHead header: need {header_size}, have {len(self.opus_data) - self.pos}")
            return None
        
        opus_header = self.opus_data[self.pos:self.pos + header_size]
        self.pos += header_size  # FIXED: Ensure this is integer arithmetic
        
        logger.trace(f"Parsed OpusHead: {header_size} bytes")
        return opus_header
    
    def _skip_opus_tags(self):
        """Skip OpusTags if present - FIXED VERSION"""
        if self.pos + len(OPUS_TAGS_MAGIC) <= len(self.opus_data):
            if self.opus_data[self.pos:self.pos + len(OPUS_TAGS_MAGIC)] == OPUS_TAGS_MAGIC:
                # Find end of OpusTags
                tags_start = self.pos
                
                # Skip magic
                self.pos += len(OPUS_TAGS_MAGIC)
                
                # FIXED: Safer OpusTags parsing with bounds checking
                try:
                    # Read vendor string length
                    if self.pos + 4 > len(self.opus_data):
                        return
                    
                    vendor_length_bytes = self.opus_data[self.pos:self.pos + 4]
                    vendor_length = struct.unpack('<I', vendor_length_bytes)[0]
                    # FIXED: Explicit integer conversion
                    vendor_length = int(vendor_length)
                    
                    self.pos += 4 + vendor_length
                    
                    # Read comment count  
                    if self.pos + 4 > len(self.opus_data):
                        return
                    
                    comment_count_bytes = self.opus_data[self.pos:self.pos + 4]
                    comment_count = struct.unpack('<I', comment_count_bytes)[0]
                    # FIXED: Explicit integer conversion
                    comment_count = int(comment_count)
                    
                    self.pos += 4
                    
                    # Skip comments with bounds checking
                    for i in range(comment_count):
                        if self.pos + 4 > len(self.opus_data):
                            break
                        comment_length_bytes = self.opus_data[self.pos:self.pos + 4]
                        comment_length = struct.unpack('<I', comment_length_bytes)[0]
                        # FIXED: Explicit integer conversion and bounds check
                        comment_length = int(comment_length)
                        
                        if self.pos + 4 + comment_length > len(self.opus_data):
                            break
                            
                        self.pos += 4 + comment_length
                    
                    total_skipped = self.pos - tags_start
                    logger.trace(f"Skipped OpusTags: {total_skipped} bytes")
                    
                except (struct.error, ValueError, OverflowError) as e:
                    logger.error(f"Error parsing OpusTags: {e}")
                    # Skip the problematic tags section
                    self.pos = tags_start + len(OPUS_TAGS_MAGIC)

# Helper function for integration
def _extract_raw_opus_stream(audio_file: pathlib.Path, source_params: dict) -> bytes:
    """
    Extract raw Opus stream from file if possible
    
    Args:
        audio_file: Path to audio file
        source_params: Source parameters from ffprobe
        
    Returns:
        Raw Opus stream data
        
    Raises:
        ValueError: If direct extraction not possible
    """
    if not OpusContainerDetector.can_extract_directly(audio_file, source_params):
        raise ValueError(f"Cannot extract Opus directly from {audio_file.name}")
    
    return OpusContainerDetector.extract_raw_opus_stream(audio_file)


def debug_opus_data_structure(opus_data: bytes, max_bytes: int = 1000):
    """Debug function to analyze Opus data structure"""
    print(f"🔍 OPUS DATA ANALYSIS:")
    print(f"   Total size: {len(opus_data)} bytes")
    
    # Look for OpusHead
    opus_head_pos = opus_data.find(b'OpusHead')
    if opus_head_pos >= 0:
        print(f"   OpusHead found at position: {opus_head_pos}")
        
        # Show first few bytes around OpusHead
        start = max(0, opus_head_pos - 10)
        end = min(len(opus_data), opus_head_pos + 50)
        header_context = opus_data[start:end]
        print(f"   Header context: {header_context[:30]}...")
    else:
        print("   ❌ OpusHead NOT found!")
    
    # Look for OpusTags
    opus_tags_pos = opus_data.find(b'OpusTags')
    if opus_tags_pos >= 0:
        print(f"   OpusTags found at position: {opus_tags_pos}")
    
    # Show first bytes of data
    first_bytes = opus_data[:min(max_bytes, len(opus_data))]
    print(f"   First {len(first_bytes)} bytes structure:")
    
    # Look for patterns that might indicate packet boundaries
    for i in range(0, min(200, len(first_bytes)), 20):
        chunk = first_bytes[i:i+20]
        print(f"     {i:3d}: {chunk}")
        
    return {
        'opus_head_pos': opus_head_pos,
        'opus_tags_pos': opus_tags_pos,
        'total_size': len(opus_data)
    }