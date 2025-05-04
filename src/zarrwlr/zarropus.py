

from typing import TypedDict, TypeAlias, Dict
from collections import defaultdict
from io import BytesIO
from dataclasses import dataclass, asdict
import os
import zarr
import numpy as np
from typing import Sequence
from pathlib import Path

def is_opus_file(file_path: Path) -> bool:
    with open(file_path, 'rb') as f:
        data = f.read(36)  # ausreichend für ersten Page-Header + OpusHead
        return b'OpusHead' in data and b'OggS' in data








# Opus files (ogg container) and Opus data have always the head:
@dataclass
class OpusHead:
    """OpusHead Opus file mandatory meta data.
    
    This data build the header data of opus files
    and alle data are mandatory for opus.
    """
    version: int
    channels: int
    pre_skip: int
    sample_rate: int
    output_gain: int
    mapping_family: int

    def as_dict(self) -> dict:
        """Opus header data as dictionary."""
        return asdict(self)

# ... and some other not standardizes key-value pairs.
# We combine this to:
class OpusMetadata(TypedDict, total=False):
    """Opus file meta data
    
    ...consisting of mandatory opus head data and additional
    and optional tags.
    """

    opus_head: OpusHead   # static tags of each opus-blob,
                          # readable also as dictionary
    tags: Dict[str, str]  # dynamic tags as dictionary

def extract_opus_metadata(blob: bytes) -> OpusMetadata:
    """extract_opus_metadata Extract mandatory header tags and
    additional tags from opus byte blobs."""

    metadata: OpusMetadata = {}

    # search the opus head page (identifier: 'OpusHead')
    head_start = blob.find(b'OpusHead')
    if head_start == -1:
        raise ValueError("OpusHead not found")

    # value offsets relative to 'OpusHead' (RFC 7845)
    version = blob[head_start + 8]
    channels = blob[head_start + 9]
    pre_skip = int.from_bytes(blob[head_start + 10:12], "little")
    sample_rate = int.from_bytes(blob[head_start + 12:16], "little")
    output_gain = int.from_bytes(blob[head_start + 16:18], "little", signed=True)
    mapping_family = blob[head_start + 18]

    head = OpusHead(
        version=version,
        channels=channels,
        pre_skip=pre_skip,
        sample_rate=sample_rate,
        output_gain=output_gain,
        mapping_family=mapping_family
    )
    metadata['opus_head'] = head.as_dict()

    # look for tags
    tags_start = blob.find(b'OpusTags')
    if tags_start != -1:
        # Anzahl der Kommentare als little-endian uint32
        vendor_len = int.from_bytes(blob[tags_start + 8:12], "little")
        offset = tags_start + 12 + vendor_len

        user_comment_list_length = int.from_bytes(blob[offset:offset + 4], "little")
        offset += 4
        tags: Dict[str, str] = {}

        for _ in range(user_comment_list_length):
            length = int.from_bytes(blob[offset:offset + 4], "little")
            offset += 4
            raw = blob[offset:offset + length]
            offset += length
            try:
                key_value = raw.decode("utf-8")
                if '=' in key_value:
                    key, value = key_value.split("=", 1)
                    tags[key.upper()] = value
            except UnicodeDecodeError:
                continue  # überspringen, falls fehlerhafte Tags

        metadata['tags'] = tags
    return metadata

class PacketEntry(TypedDict):
    offset: int        # Byte-Offset im Blob, wo das Paket beginnt
    size: int          # Paketgröße in Bytes
    time: float        # Zeitstempel in Sekunden (auf Samplebasis)
    sample: int        # Sample-Position im Stream (Start-Sample)
    granule: int       # granule_position der Page, in der das Paket endet
    serial: int        # Bitstream-ID

PacketIndex: TypeAlias = dict[int, list[PacketEntry]]  # serial → Liste von Paketen

def build_packet_index(opus_blob: bytes, sample_rate: int = 48000) -> PacketIndex:
    """
    Build a package-index from an Ogg/Opus-Byte-Blob.
    Get back a dict: { serial → list of PacketEntry }
    """

    stream = BytesIO(opus_blob)
    index: PacketIndex = defaultdict(list)

    page_num = 0
    page_sequence_by_serial = {}
    last_granule_by_serial = {}

    while True:
        start = stream.tell()
        header = stream.read(27)
        if len(header) < 27:
            break  # EOF

        if header[:4] != b'OggS':
            raise ValueError(f"Invalid Ogg page at position {start}")

        version = header[4]
        if version != 0:
            raise ValueError(f"Unsupported Ogg stream version: {version}")
        header_type = header[5]
        granule = int.from_bytes(header[6:14], 'little')
        serial = int.from_bytes(header[14:18], 'little')
        page_seq = int.from_bytes(header[18:22], 'little')
        seg_count = header[26]

        # plausibility check of the page sequence
        expected_seq = page_sequence_by_serial.get(serial, page_seq)
        if page_seq != expected_seq:
            print(f"Warnung: Page-Sequence mismatch for serial {serial} at page {page_num} (expected {expected_seq}, got {page_seq})")
        page_sequence_by_serial[serial] = page_seq + 1

        # segment table and read data
        seg_table = stream.read(seg_count)
        seg_sizes = list(seg_table)
        page_data_size = sum(seg_sizes)
        page_data = stream.read(page_data_size)

        # extract package
        offset = 0
        current_packet = b''
        current_start = start + 27 + seg_count  # Start des Page-Data-Bereichs
        # continuation = header_type & 0x01 -> Thats a special signal what we don't need here.
        #                                      It says, that this page is a continuation of
        #                                      the last page from the package before.

        # go throughout the segments of this package...
        for i, seg_size in enumerate(seg_sizes):
            segment = page_data[offset:offset + seg_size]
            offset += seg_size
            current_packet += segment

            if seg_size < 255:
                # Package end reached.
                # The test of segment size == 255 is save, since value 255 is a special
                # signal of the coding algorithm. The only moment where a segment is exactly
                # 255 in length is, if it is the last segment of a package.
                sample_end = granule
                last_sample = last_granule_by_serial.get(serial, 0)
                packet_samples = sample_end - last_sample
                sample_start = sample_end - packet_samples
                last_granule_by_serial[serial] = sample_end

                time_sec = sample_start / sample_rate
                packet_entry: PacketEntry = {
                    "offset": current_start + offset - len(current_packet),
                    "size": len(current_packet),
                    "sample": sample_start,
                    "time": time_sec,
                    "granule": sample_end,
                    "serial": serial
                }
                index[serial].append(packet_entry)
                current_packet = b''

        page_num += 1

    return dict(index)