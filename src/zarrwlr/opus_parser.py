"""
Enhanced RawOpusParser - opuslib-Based Implementation
====================================================

Clean opuslib-based implementation for robust packet detection:
- opuslib for packet validation and sample counting
- Position tracking for random access
- Sample-accurate extraction for scientific applications
"""

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple
import opuslib

from .logsetup import get_module_logger
logger = get_module_logger(__file__)

# Opus constants
OPUS_HEAD_MAGIC = b'OpusHead'
OPUS_TAGS_MAGIC = b'OpusTags'

@dataclass
class OpusStreamInfo:
    """Stream information from OpusHead header"""
    original_sample_rate: int
    channel_count: int
    pre_skip: int
    output_gain: int
    channel_mapping: int
    opus_header: bytes

@dataclass
class OpusPacketInfo:
    """Single Opus packet information"""
    data: bytes
    byte_position: int
    sample_count: int
    cumulative_samples: int


class OpusParser:
    """opuslib-based Opus parser with position tracking"""
    
    def __init__(self, opus_data: bytes):
        self.opus_data = opus_data
        self.stream_info: Optional[OpusStreamInfo] = None
        self.pos = 0
        self.decoder_state = None
        logger.trace(f"OpusParser initialized with {len(opus_data)} bytes")
    
    def extract_packets_with_positions(self) -> List[OpusPacketInfo]:
        """
        Extract all Opus packets using opuslib for validation and sample counting
        
        Returns:
            List of OpusPacketInfo with position and sample information
        """
        logger.trace("Starting opuslib-based packet extraction")
        
        try:
            self.pos = 0
            
            # Parse OpusHead header
            self.stream_info = self._parse_opus_head()
            if not self.stream_info:
                raise ValueError("No valid OpusHead found in Opus stream")
            
            # Initialize opuslib decoder
            self._initialize_opuslib_decoder()
            
            # Skip OpusTags if present  
            self._skip_opus_tags()
            
            # Extract audio packets
            packet_infos = self._extract_audio_packets_with_opuslib()
            
            logger.trace(f"Extraction completed: {len(packet_infos)} packets")
            return packet_infos
            
        except Exception as e:
            logger.error(f"Error in packet extraction: {e}")
            raise
        finally:
            self.decoder_state = None
    
    def _initialize_opuslib_decoder(self):
        """Initialize opuslib decoder for packet validation and sample counting"""
        try:
            sample_rate = self.stream_info.original_sample_rate
            channels = self.stream_info.channel_count
            
            # For ultrasonic sources, decode to 48kHz
            if sample_rate > 48000:
                decoder_sample_rate = 48000
            else:
                decoder_sample_rate = sample_rate
            
            self.decoder_state = opuslib.Decoder(decoder_sample_rate, channels)
            logger.trace(f"opuslib decoder initialized: {decoder_sample_rate}Hz, {channels}ch")
            
        except Exception as e:
            logger.error(f"Failed to initialize opuslib decoder: {e}")
            raise RuntimeError(f"opuslib decoder initialization failed: {e}")
    
    def _parse_opus_head(self) -> Optional[OpusStreamInfo]:
        """Parse OpusHead header (same as before - this works correctly)"""
        # Find OpusHead magic
        opus_head_pos = self.opus_data.find(OPUS_HEAD_MAGIC)
        if opus_head_pos == -1:
            logger.error("OpusHead magic not found in stream")
            return None
        
        self.pos = opus_head_pos
        logger.trace(f"Found OpusHead at position {opus_head_pos}")
        
        # Minimum OpusHead size is 19 bytes
        if self.pos + 19 > len(self.opus_data):
            logger.error("Incomplete OpusHead header")
            return None
        
        # Parse OpusHead structure (RFC 7845 Section 5.1)
        try:
            header_data = self.opus_data[self.pos:self.pos + 19]
            
            magic = header_data[0:8]
            version = header_data[8]
            channel_count = header_data[9]
            pre_skip = struct.unpack('<H', header_data[10:12])[0]
            original_sample_rate = struct.unpack('<I', header_data[12:16])[0]
            output_gain = struct.unpack('<h', header_data[16:18])[0]  # signed
            channel_mapping = header_data[18]
            
            # Validate OpusHead
            if magic != OPUS_HEAD_MAGIC:
                logger.error(f"Invalid OpusHead magic: {magic}")
                return None
            
            if channel_count == 0 or channel_count > 255:
                logger.error(f"Invalid channel count: {channel_count}")
                return None
            
            # Determine header size (channel mapping may extend header)
            header_size = 19
            if channel_mapping > 0:
                # Extended header with channel mapping table
                if self.pos + 21 > len(self.opus_data):
                    logger.error("Incomplete OpusHead channel mapping")
                    return None
                
                stream_count = self.opus_data[self.pos + 19]
                coupled_count = self.opus_data[self.pos + 20]
                header_size = 21 + channel_count  # Include channel mapping table
                
                if self.pos + header_size > len(self.opus_data):
                    logger.error("Incomplete OpusHead channel mapping table")
                    return None
            
            # Extract complete header
            opus_header = self.opus_data[self.pos:self.pos + header_size]
            self.pos += header_size
            
            stream_info = OpusStreamInfo(
                original_sample_rate=original_sample_rate,
                channel_count=channel_count,
                pre_skip=pre_skip,
                output_gain=output_gain,
                channel_mapping=channel_mapping,
                opus_header=opus_header
            )
            
            logger.trace(f"Parsed OpusHead: {channel_count}ch, {original_sample_rate}Hz, "
                        f"pre_skip={pre_skip}, gain={output_gain}")
            
            return stream_info
            
        except (struct.error, IndexError) as e:
            logger.error(f"Error parsing OpusHead: {e}")
            return None
    
    def _skip_opus_tags(self):
        """Skip OpusTags header if present"""
        if self.pos + len(OPUS_TAGS_MAGIC) > len(self.opus_data):
            return
        
        if self.opus_data[self.pos:self.pos + len(OPUS_TAGS_MAGIC)] != OPUS_TAGS_MAGIC:
            return
        
        logger.trace(f"Found OpusTags at position {self.pos}")
        tags_start = self.pos
        
        try:
            self.pos += len(OPUS_TAGS_MAGIC)
            
            # Read vendor string length
            if self.pos + 4 > len(self.opus_data):
                return
            
            vendor_length = struct.unpack('<I', self.opus_data[self.pos:self.pos + 4])[0]
            self.pos += 4 + vendor_length
            
            # Read comment count
            if self.pos + 4 > len(self.opus_data):
                return
            
            comment_count = struct.unpack('<I', self.opus_data[self.pos:self.pos + 4])[0]
            self.pos += 4
            
            # Skip comments
            for _ in range(comment_count):
                if self.pos + 4 > len(self.opus_data):
                    break
                
                comment_length = struct.unpack('<I', self.opus_data[self.pos:self.pos + 4])[0]
                self.pos += 4 + comment_length
                
                if self.pos > len(self.opus_data):
                    break
            
            skipped_bytes = self.pos - tags_start
            logger.trace(f"Skipped OpusTags: {skipped_bytes} bytes")
            
        except (struct.error, OverflowError) as e:
            logger.warning(f"Error parsing OpusTags: {e}")
            self.pos = tags_start + len(OPUS_TAGS_MAGIC)
    
    def _extract_audio_packets_with_opuslib(self) -> List[OpusPacketInfo]:
        """Extract audio packets using opuslib for validation and sample counting"""
        packet_infos = []
        cumulative_samples = 0
        
        logger.trace(f"Starting audio packet extraction from position {self.pos}")
        
        while self.pos < len(self.opus_data):
            packet_start_pos = self.pos
            
            # Find next valid packet using opuslib validation
            packet_data, packet_length = self._find_next_valid_packet()
            
            if not packet_data:
                logger.warning(f"No more valid packets found at position {self.pos}")
                break
            
            try:
                # Use opuslib to get exact sample count
                sample_count = self._get_packet_sample_count(packet_data)
                
                packet_info = OpusPacketInfo(
                    data=packet_data,
                    byte_position=packet_start_pos,
                    sample_count=sample_count,
                    cumulative_samples=cumulative_samples
                )
                
                packet_infos.append(packet_info)
                cumulative_samples += sample_count
                self.pos += packet_length
                
                logger.trace(f"Packet {len(packet_infos)}: {packet_length} bytes, "
                           f"{sample_count} samples")
                
            except Exception as e:
                logger.error(f"Error processing packet at position {packet_start_pos}: {e}")
                break
        
        logger.trace(f"Extracted {len(packet_infos)} packets, "
                    f"total samples: {cumulative_samples}")
        
        return packet_infos
    
    def _find_next_valid_packet(self) -> Tuple[bytes, int]:
        """Optimized packet detection for Raw Opus streams with padding"""
        max_search = 100  # Search window for padding
        
        for offset in range(max_search):
            test_pos = self.pos + offset
            if test_pos >= len(self.opus_data) - 20:
                return None, 0
            
            # Focus on most common sizes (10 bytes primary)
            for size in [10, 11, 12, 9, 13, 14, 15, 8, 16, 20]:
                if test_pos + size > len(self.opus_data):
                    continue
                    
                candidate_data = self.opus_data[test_pos:test_pos + size]
                
                if self._is_valid_opus_packet(candidate_data):
                    # Update position to skip any padding
                    self.pos = test_pos
                    return candidate_data, size
        
        return None, 0
    
    def _is_valid_opus_packet(self, packet_data: bytes) -> bool:
        """Validate if packet is a valid Opus audio packet using opuslib"""
        if not packet_data or len(packet_data) < 1:
            return False
        
        try:
            sample_count = self._get_packet_sample_count(packet_data)
            # Valid sample counts for Opus (typical ranges)
            return 0 < sample_count <= 5760  # Max 120ms at 48kHz
        except Exception:
            return False
    
    def _get_packet_sample_count(self, packet_data: bytes) -> int:
        """Get exact sample count for packet using opuslib"""
        if not self.decoder_state:
            raise RuntimeError("opuslib decoder not initialized")
        
        try:
            # Use opuslib API to get exact sample count
            sample_count = opuslib.api.decoder.get_nb_samples(
                self.decoder_state.decoder_state,  # Access internal decoder state
                packet_data,
                len(packet_data)
            )
            
            return sample_count
            
        except Exception as e:
            raise RuntimeError(f"opuslib sample count failed: {e}")
    
    def get_stream_info(self) -> Optional[OpusStreamInfo]:
        """Get stream information"""
        return self.stream_info