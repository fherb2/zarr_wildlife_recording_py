"""
Opus Index Backend Module - Sequential + Parallel Implementation
================================================================

High-performance OGG-Opus page parsing and index creation.
Based on the successful FLAC 3-phase parallelization architecture.

PERFORMANCE TARGET:
- Sequential: Baseline implementation (fallback)
- Parallel: 3-5x speedup with 3-phase architecture
- Memory efficient: Zarr-reference system for large files

3-PHASE ARCHITECTURE (planned):
===============================

Phase 1: Parallel OGG-Page Search
- Search for "OggS" signatures in parallel chunks
- Memory-efficient through Zarr-references
- Output: Sorted list of all page positions

Phase 2: Parallel Page-Detail Calculation  
- Page size, granule position, hash calculated in parallel
- All page boundaries known from Phase 1
- Output: Complete page details (without sample positions)

Phase 3: Sequential Sample-Position Accumulation
- Accumulative calculation based on granule positions
- Handle ultrasonic sample rate corrections
- Output: Final Opus index ready for Zarr

OGG-SPECIFIC ADAPTATIONS:
========================
- Variable page sizes (vs. FLAC's relatively constant frame sizes)
- Segment tables for complex page structure
- Granule position interpolation for missing values
- Ultrasonic sample rate correction integration
"""

import zarr
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import hashlib
import struct

# import and initialize logging
from zarrwlr.logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Opus Index Backend module loading...")

# OGG/Opus Index constants
OPUS_INDEX_DTYPE = np.uint64
OPUS_INDEX_COLS = 3  # [byte_offset, page_size, sample_pos]
OPUS_INDEX_COL_BYTE_OFFSET = 0
OPUS_INDEX_COL_PAGE_SIZE = 1  
OPUS_INDEX_COL_SAMPLE_POS = 2

# OGG Container constants
OGG_PAGE_HEADER_SIZE = 27
OGG_SYNC_PATTERN = b'OggS'
OGG_MAX_PAGE_SIZE = 65536


# ##########################################################
#
# Memory Monitoring Helper (shared with FLAC)
# ============================================
#
# ##########################################################

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class MemoryStats:
    """Helper for Memory-Monitoring (with psutil fallback)"""
    
    @staticmethod
    def get_current_memory_mb():
        """Current RAM usage in MB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return 0.0  # Fallback


# ##########################################################
#
# Sequential OGG-Page Parsing (Fallback Implementation)
# =====================================================
#
# ##########################################################

def _parse_ogg_page_header(header_bytes: bytes) -> dict:
    """
    Parse OGG page header (27 bytes + segment table)
    
    Args:
        header_bytes: First bytes of OGG page (at least 27 bytes)
        
    Returns:
        dict: Parsed header information
        
    Raises:
        ValueError: If header is invalid
    """
    if len(header_bytes) < OGG_PAGE_HEADER_SIZE:
        raise ValueError(f"Header too short: {len(header_bytes)} < {OGG_PAGE_HEADER_SIZE}")
    
    if header_bytes[:4] != OGG_SYNC_PATTERN:
        raise ValueError(f"Invalid OGG sync pattern: {header_bytes[:4]}")
    
    # Parse header fields (little-endian)
    header = struct.unpack('<4sBBQIIIB', header_bytes[:OGG_PAGE_HEADER_SIZE])
    
    return {
        'sync_pattern': header[0],  # b'OggS'
        'version': header[1],       # Should be 0
        'header_type': header[2],   # Flags (first page, last page, etc.)
        'granule_position': header[3],  # 64-bit sample position
        'serial_number': header[4], # Stream serial number
        'page_sequence': header[5], # Page sequence number
        'checksum': header[6],      # CRC32 checksum
        'segment_count': header[7]  # Number of segments in page
    }


def _calculate_ogg_page_size(audio_bytes: bytes, page_start: int) -> Tuple[int, dict]:
    """
    Calculate total OGG page size including header, segment table, and body
    
    Args:
        audio_bytes: Complete audio data
        page_start: Byte position of page start
        
    Returns:
        Tuple of (page_size, page_info)
        
    Raises:
        ValueError: If page is incomplete or invalid
    """
    if page_start + OGG_PAGE_HEADER_SIZE > len(audio_bytes):
        raise ValueError("Incomplete page header")
    
    # Parse header
    header_bytes = audio_bytes[page_start:page_start + OGG_PAGE_HEADER_SIZE]
    page_info = _parse_ogg_page_header(header_bytes)
    
    segment_count = page_info['segment_count']
    
    # Read segment table
    seg_table_start = page_start + OGG_PAGE_HEADER_SIZE
    seg_table_end = seg_table_start + segment_count
    
    if seg_table_end > len(audio_bytes):
        raise ValueError("Incomplete segment table")
    
    segment_table = audio_bytes[seg_table_start:seg_table_end]
    
    # Calculate body size (sum of all segment sizes)
    page_body_size = sum(segment_table)
    
    # Total page size
    total_page_size = OGG_PAGE_HEADER_SIZE + segment_count + page_body_size
    
    if page_start + total_page_size > len(audio_bytes):
        raise ValueError("Incomplete page body")
    
    page_info['segment_table'] = segment_table
    page_info['page_body_size'] = page_body_size
    page_info['total_page_size'] = total_page_size
    
    return total_page_size, page_info


def _find_next_ogg_page(audio_bytes: bytes, start_pos: int, max_search_bytes: int = 65536) -> Optional[int]:
    """
    Search for the next OGG page sync pattern
    
    Args:
        audio_bytes: OGG audio data
        start_pos: Start position for search
        max_search_bytes: Maximum search range
        
    Returns:
        Position of next page sync or None
    """
    search_end = min(start_pos + max_search_bytes, len(audio_bytes) - 4)
    
    for pos in range(start_pos, search_end):
        if audio_bytes[pos:pos+4] == OGG_SYNC_PATTERN:
            return pos
    
    return None


def _parse_ogg_pages_sequential(audio_bytes: bytes, expected_sample_rate: int = 48000) -> List[dict]:
    """
    Sequential OGG page parsing (fallback implementation)
    Ported from opusbyteblob.py but structured like FLAC backend
    
    Args:
        audio_bytes: Complete OGG audio data
        expected_sample_rate: Expected sample rate for corrections
        
    Returns:
        List of page information as dictionaries
    """
    pages_info = []
    pos = 0
    current_sample = 0
    loop_count = 0
    
    logger.trace(f"Starting sequential OGG page analysis for {expected_sample_rate}Hz audio")
    
    # Page-by-page analysis
    while pos < len(audio_bytes) - OGG_PAGE_HEADER_SIZE:
        loop_count += 1
        
        # Anti-endless loop protection
        if loop_count > 100000:
            logger.error(f"Page analysis loop stopped after {loop_count} iterations. Possible endless loop.")
            break
        
        # Search for OGG page sync
        if audio_bytes[pos:pos+4] == OGG_SYNC_PATTERN:
            page_start = pos
            
            try:
                # Calculate page size and get page info
                page_size, page_info = _calculate_ogg_page_size(audio_bytes, page_start)
                
                # Extract granule position for sample calculation
                granule_position = page_info['granule_position']
                
                # Sample position handling
                if granule_position != 0xFFFFFFFFFFFFFFFF:  # Valid granule position
                    # Use granule position as absolute sample position
                    sample_position = granule_position
                else:
                    # Missing granule position - interpolate based on previous
                    if pages_info:
                        # Estimate based on typical Opus frame size (960 samples at 48kHz)
                        estimated_samples_per_page = 960
                        sample_position = pages_info[-1]['sample_pos'] + estimated_samples_per_page
                    else:
                        sample_position = 0
                
                pages_info.append({
                    'byte_offset': page_start,
                    'page_size': page_size,
                    'sample_pos': sample_position,
                    'granule_position': granule_position,
                    'page_sequence': page_info['page_sequence'],
                    'segment_count': page_info['segment_count']
                })
                
                # Move to next page
                pos = page_start + page_size
                
            except ValueError as e:
                logger.warning(f"Invalid OGG page at position {pos}: {e}")
                pos += 1
                continue
        else:
            pos += 1
        
        # Progress log every 1000 pages
        if loop_count % 1000 == 0:
            logger.trace(f"Sequential page analysis progress: {len(pages_info)} pages, position {pos}/{len(audio_bytes)}")
    
    logger.trace(f"Sequential analysis: {len(pages_info)} pages found after {loop_count} iterations")
    
    # Post-process: Ensure monotonic sample positions
    _ensure_monotonic_sample_positions(pages_info, expected_sample_rate)
    
    return pages_info


def _parse_ogg_pages_from_positions(audio_bytes: bytes, page_positions: List[int], 
                                  expected_sample_rate: int = 48000) -> List[dict]:
    """
    Parse OGG pages from known positions (optimized version of sequential parsing)
    Used after parallel page position finding to create page details
    
    Args:
        audio_bytes: Complete OGG audio data
        page_positions: Pre-found page positions from parallel search
        expected_sample_rate: Expected sample rate for corrections
        
    Returns:
        List of page information as dictionaries
    """
    pages_info = []
    
    logger.trace(f"Creating page details from {len(page_positions)} known positions")
    
    for i, page_start in enumerate(page_positions):
        try:
            # Determine page end
            if i + 1 < len(page_positions):
                # Next page starts at next position
                max_page_end = page_positions[i + 1]
            else:
                # Last page - search for actual end or use file end
                max_page_end = len(audio_bytes)
            
            # Calculate actual page size and get page info
            page_size, page_info = _calculate_ogg_page_size(audio_bytes, page_start)
            
            # Validate page doesn't exceed expected bounds
            if page_start + page_size > max_page_end:
                logger.warning(f"Page {i} size calculation exceeded expected bounds, truncating")
                page_size = max_page_end - page_start
            
            # Extract granule position for sample calculation
            granule_position = page_info['granule_position']
            
            # Sample position handling (same logic as sequential)
            if granule_position != 0xFFFFFFFFFFFFFFFF:  # Valid granule position
                sample_position = granule_position
            else:
                # Missing granule position - interpolate
                if pages_info:
                    estimated_samples_per_page = 960  # Typical Opus frame size
                    sample_position = pages_info[-1]['sample_pos'] + estimated_samples_per_page
                else:
                    sample_position = 0
            
            pages_info.append({
                'byte_offset': page_start,
                'page_size': page_size,
                'sample_pos': sample_position,
                'granule_position': granule_position,
                'page_sequence': page_info['page_sequence'],
                'segment_count': page_info['segment_count']
            })
            
        except ValueError as e:
            logger.warning(f"Invalid OGG page at position {page_start}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing page {i} at position {page_start}: {e}")
            continue
    
    logger.trace(f"Created details for {len(pages_info)} pages from positions")
    
    # Post-process: Ensure monotonic sample positions (same as sequential)
    _ensure_monotonic_sample_positions(pages_info, expected_sample_rate)
    
    return pages_info
    """
    Ensure sample positions are monotonically increasing
    Interpolate missing or invalid granule positions
    
    Args:
        pages_info: List of page information (modified in-place)
        expected_sample_rate: Sample rate for interpolation
    """
    if not pages_info:
        return
    
    # Typical Opus frame size at 48kHz
    default_samples_per_page = 960 if expected_sample_rate == 48000 else int(960 * expected_sample_rate / 48000)
    
    # First pass: fix obviously invalid positions
    last_valid_sample = 0
    for i, page in enumerate(pages_info):
        if page['granule_position'] == 0xFFFFFFFFFFFFFFFF or page['sample_pos'] < last_valid_sample:
            # Invalid or decreasing - interpolate
            if i == 0:
                page['sample_pos'] = 0
            else:
                page['sample_pos'] = pages_info[i-1]['sample_pos'] + default_samples_per_page
        
        last_valid_sample = page['sample_pos']
    
    logger.trace(f"Sample position correction completed for {len(pages_info)} pages")


# ##########################################################
#
# Phase 1: Parallel OGG-Page Search (PRODUCTION-READY)
# ====================================================
#
# ##########################################################

class OggChunkReference:
    """Referenz auf einen Chunk im Zarr-Array für OGG-Page-Suche (keine Daten-Kopie)"""
    def __init__(self, zarr_store_path: str, group_path: str, array_name: str,
                 start_byte: int, end_byte: int, chunk_id: int):
        self.zarr_store_path = zarr_store_path
        self.group_path = group_path
        self.array_name = array_name
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.chunk_id = chunk_id


class OggPageSearchResult:
    """Ergebnis der OGG-Page-Suche für einen Chunk"""
    def __init__(self, chunk_start: int, chunk_end: int, page_positions: List[int]):
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.page_positions = page_positions
        self.processing_time = 0.0
        self.error = None


def _find_ogg_pages_in_chunk(chunk_ref: OggChunkReference) -> OggPageSearchResult:
    """
    Memory-effiziente OGG-Page-Suche mit Zarr-Referenz
    Worker lädt Daten on-demand aus Zarr-Store (analog FLAC-Implementation)
    
    Args:
        chunk_ref: Referenz auf Chunk im Zarr-Array
        
    Returns:
        OggPageSearchResult mit gefundenen Page-Positionen
    """
    start_time = time.time()
    
    try:
        # Worker öffnet Zarr-Store selbst (memory-efficient)
        store = zarr.storage.LocalStore(chunk_ref.zarr_store_path)
        root = zarr.open_group(store, mode='r')
        audio_array = root[chunk_ref.group_path][chunk_ref.array_name]
        
        # Lade NUR den benötigten Chunk (on-demand)
        chunk_data = bytes(audio_array[chunk_ref.start_byte:chunk_ref.end_byte])
        
        page_positions = []
        pos = 0
        
        # Suche nach OGG-Page-Sync-Pattern (b'OggS')
        while pos < len(chunk_data) - 4:  # Need at least 4 bytes for sync pattern
            if chunk_data[pos:pos+4] == OGG_SYNC_PATTERN:
                absolute_pos = chunk_ref.start_byte + pos
                page_positions.append(absolute_pos)
                
                # Skip ahead efficiently - OGG pages are typically several KB
                # This prevents finding partial patterns within page data
                pos += max(64, len(chunk_data) // 1000)  # Adaptive skip
            else:
                pos += 1
        
        result = OggPageSearchResult(chunk_ref.start_byte, chunk_ref.end_byte, page_positions)
        result.processing_time = time.time() - start_time
        return result
        
    except Exception as e:
        result = OggPageSearchResult(chunk_ref.start_byte, chunk_ref.end_byte, [])
        result.processing_time = time.time() - start_time
        result.error = str(e)
        return result


def _create_ogg_chunk_references(zarr_store_path: str, group_path: str, array_name: str,
                               total_size: int, chunk_size_mb: int = 4) -> List[OggChunkReference]:
    """
    Erstelle Chunk-Referenzen für parallele OGG-Page-Suche ohne Daten zu laden
    
    Args:
        zarr_store_path: Pfad zum Zarr-Store
        group_path: Pfad zur Audio-Gruppe
        array_name: Name des Audio-Arrays
        total_size: Gesamtgröße der Audio-Daten
        chunk_size_mb: Chunk-Größe in MB
        
    Returns:
        Liste von OggChunkReference-Objekten
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_refs = []
    
    # Start from beginning (OGG headers are typically at start)
    chunk_id = 0
    chunk_start = 0
    
    while chunk_start < total_size:
        chunk_end = min(chunk_start + chunk_size_bytes, total_size)
        
        # Ensure we don't cut in the middle of potential OGG page headers
        # Add small overlap to catch pages that span chunk boundaries
        overlap = 1024  # 1KB overlap should catch any page header
        if chunk_end < total_size:
            chunk_end = min(chunk_end + overlap, total_size)
        
        chunk_ref = OggChunkReference(
            zarr_store_path=zarr_store_path,
            group_path=group_path,
            array_name=array_name,
            start_byte=chunk_start,
            end_byte=chunk_end,
            chunk_id=chunk_id
        )
        
        chunk_refs.append(chunk_ref)
        chunk_start = chunk_end - overlap  # Account for overlap in next chunk
        chunk_id += 1
    
    return chunk_refs


def _find_ogg_pages_parallel(zarr_store_path: str, group_path: str, array_name: str,
                           max_workers: int = None, chunk_size_mb: int = 4) -> List[int]:
    """
    Phase 1: Memory-effiziente parallele OGG-Page-Suche mit Zarr-Referenzen
    
    Args:
        zarr_store_path: Pfad zum Zarr-Store  
        group_path: Pfad zur Audio-Gruppe
        array_name: Name des Audio-Arrays
        max_workers: Anzahl paralleler Worker
        chunk_size_mb: Chunk-Größe in MB
        
    Returns:
        Sortierte Liste aller OGG-Page-Positionen
    """
    start_time = time.time()
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 6)
    
    # Hole Array-Größe ohne Daten zu laden
    store = zarr.storage.LocalStore(zarr_store_path)
    root = zarr.open_group(store, mode='r')
    audio_array = root[group_path][array_name]
    total_size = audio_array.shape[0]
    
    logger.trace(f"Phase 1: Starte parallele OGG-Page-Suche mit {max_workers} Workern")
    logger.trace(f"Audio-Daten: {total_size} bytes, Chunk-Größe: {chunk_size_mb} MB")
    
    # Erstelle Chunk-Referenzen (KEINE Daten geladen!)
    chunk_refs = _create_ogg_chunk_references(
        zarr_store_path, group_path, array_name, total_size, chunk_size_mb
    )
    
    logger.trace(f"Erstellt {len(chunk_refs)} Chunks für parallele Verarbeitung")
    
    # Parallel processing mit Chunk-Referenzen
    all_page_positions = []
    total_processing_time = 0.0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_find_ogg_pages_in_chunk, chunk_ref)
            for chunk_ref in chunk_refs
        ]
        
        for i, future in enumerate(futures):
            result = future.result()
            total_processing_time += result.processing_time
            
            if result.error:
                logger.warning(f"Chunk {i} processing error: {result.error}")
                error_count += 1
            else:
                all_page_positions.extend(result.page_positions)
                logger.trace(f"Chunk {i}: {len(result.page_positions)} pages gefunden")
    
    # Remove duplicates that may occur at chunk boundaries and sort
    unique_positions = sorted(set(all_page_positions))
    
    processing_time = time.time() - start_time
    avg_worker_time = total_processing_time / len(chunk_refs) if chunk_refs else 0
    
    logger.trace(f"Phase 1: {len(unique_positions)} eindeutige Page-Positionen gefunden")
    logger.trace(f"Verarbeitungszeit: {processing_time:.3f}s total, {avg_worker_time:.3f}s avg/worker")
    logger.trace(f"Parallel efficiency: {total_processing_time/processing_time:.1f}x bei {max_workers} workers")
    
    if error_count > 0:
        logger.warning(f"Phase 1: {error_count} Chunk-Verarbeitungsfehler aufgetreten")
    
    return unique_positions


def _process_ogg_pages_parallel(zarr_store_path: str, group_path: str, array_name: str,
                              page_positions: List[int], max_workers: int = None) -> List[dict]:
    """
    Phase 2: Parallel OGG page detail calculation (PLACEHOLDER)
    
    Args:
        zarr_store_path: Path to Zarr store
        group_path: Path to audio group  
        array_name: Name of audio array
        page_positions: Page positions from Phase 1
        max_workers: Number of parallel workers
        
    Returns:
        List of page details (without sample positions)
    """
    logger.warning("Parallel OGG page processing not yet implemented - falling back to sequential")
    raise NotImplementedError("Parallel processing will be implemented in Phase 2")


def _accumulate_sample_positions_opus(page_details: List[dict], sampling_rescale_factor: float = 1.0) -> List[dict]:
    """
    Phase 3: Sequential sample position accumulation with Opus-specific corrections
    
    Args:
        page_details: Page details from Phase 2 (without sample positions)
        sampling_rescale_factor: Ultrasonic correction factor
        
    Returns:
        Page details with accumulated sample positions
    """
    logger.trace(f"Phase 3: Accumulating sample positions with rescale factor {sampling_rescale_factor}")
    
    # Apply ultrasonic correction if needed
    if sampling_rescale_factor != 1.0:
        logger.trace(f"Applying ultrasonic sample rate correction: factor = {sampling_rescale_factor}")
        for page in page_details:
            if page['sample_pos'] != 0xFFFFFFFFFFFFFFFF:
                page['sample_pos'] = int(page['sample_pos'] * sampling_rescale_factor)
    
    return page_details


# ##########################################################
#
# Path Utilities (shared with FLAC)
# =================================
#
# ##########################################################

def _get_zarr_store_path(zarr_group: zarr.Group) -> str:
    """
    Robust extraction of Zarr store path for different Zarr versions
    (Identical to FLAC implementation)
    """
    try:
        store = zarr_group.store
        
        # Method 1: Direct path attribute (Zarr v3)
        if hasattr(store, 'path'):
            return str(store.path)
        
        # Method 2: dir_path method (some Zarr versions)
        if hasattr(store, 'dir_path'):
            return str(store.dir_path())
        
        # Method 3: root attribute (LocalStore)
        if hasattr(store, 'root'):
            return str(store.root)
        
        # Method 4: Parse from string representation
        store_str = str(store)
        if 'file://' in store_str:
            import re
            match = re.search(r'file://([^\']+)', store_str)
            if match:
                return match.group(1)
        
        # Method 5: Check for map attribute (MemoryStore compatibility)
        if hasattr(store, 'map') and hasattr(store.map, 'root'):
            return str(store.map.root)
        
        raise ValueError(f"Cannot determine store path from {type(store)}: {store_str}")
        
    except Exception as e:
        raise ValueError(f"Failed to extract store path: {e}")


def _get_zarr_array_path_components(zarr_group: zarr.Group, array: zarr.Array) -> Tuple[str, str, str]:
    """
    Get path components for Zarr array access in parallel workers
    (Identical to FLAC implementation)
    """
    store_path = _get_zarr_store_path(zarr_group)
    group_path = zarr_group.path if zarr_group.path else ""
    
    # Extract just the array name (not the full path)
    if hasattr(array, 'name'):
        array_name = array.name
        # If name contains path separators, take only the last part
        if '/' in array_name:
            array_name = array_name.split('/')[-1]
    else:
        # Fallback: try to get name from array's path or basename
        array_name = "audio_data_blob_array"  # Default fallback
    
    return store_path, group_path, array_name


def _find_page_range_for_samples(opus_index: zarr.Array, start_sample: int, end_sample: int) -> Tuple[int, int]:
    """
    Find OGG page range for sample range using binary search
    
    Args:
        opus_index: Opus index array (shape: n_pages x 3)
        start_sample: First required sample
        end_sample: Last required sample
        
    Returns:
        Tuple (start_page_idx, end_page_idx)
    """
    sample_positions = opus_index[:, OPUS_INDEX_COL_SAMPLE_POS]
    
    start_idx = np.searchsorted(sample_positions, start_sample, side='right') - 1
    start_idx = max(0, start_idx)
    
    end_idx = np.searchsorted(sample_positions, end_sample, side='right')
    end_idx = min(end_idx, opus_index.shape[0] - 1)
    
    return start_idx, end_idx


# ##########################################################
#
# Main Public API
# ===============
#
# ##########################################################

def build_opus_index(zarr_group: zarr.Group, audio_blob_array: zarr.Array, 
                    use_parallel: bool = True, max_workers: int = None) -> zarr.Array:
    """
    Create index for OGG-Opus page access with optional parallelization
    
    Args:
        zarr_group: Zarr group for index storage
        audio_blob_array: Array with OGG-Opus audio data
        use_parallel: Whether to use parallel processing (default: True, falls back to sequential)
        max_workers: Number of parallel workers (default: auto-detect)
        
    Returns:
        Created index array
        
    Raises:
        ValueError: If no OGG pages are found
    """
    logger.trace("build_opus_index() requested.")
    
    # Extract metadata from array attributes
    sample_rate = audio_blob_array.attrs.get('sample_rate', 48000)
    channels = audio_blob_array.attrs.get('nb_channels', 1)
    codec = audio_blob_array.attrs.get('codec', 'opus')
    container_type = audio_blob_array.attrs.get('container_type', 'ogg')
    sampling_rescale_factor = audio_blob_array.attrs.get('sampling_rescale_factor', 1.0)
    
    # Validation
    if codec != 'opus':
        raise ValueError(f"Expected Opus codec, but found: {codec}")
    
    if container_type != 'ogg':
        raise ValueError(f"Expected OGG container, but found: {container_type}")
    
    logger.trace(f"Creating Opus index for: {sample_rate}Hz, {channels} channels, "
                f"container: {container_type}, rescale_factor: {sampling_rescale_factor}")
    
    pages_info = []
    
    if use_parallel:
        # Use 3-phase parallel processing (Phase 1 implemented!)
        logger.trace("Attempting 3-phase parallel OGG index creation")
        
        try:
            # Determine Zarr paths for parallel access
            zarr_store_path, group_path, array_name = _get_zarr_array_path_components(zarr_group, audio_blob_array)
            
            logger.trace(f"Parallel processing paths: store={zarr_store_path}, group={group_path}, array={array_name}")
            
            total_start_time = time.time()
            
            # Phase 1: Parallel OGG-page search (NEW - IMPLEMENTED!)
            page_positions = _find_ogg_pages_parallel(
                zarr_store_path, group_path, array_name, max_workers, chunk_size_mb=4
            )
            
            if len(page_positions) < 1:
                raise ValueError("Could not find OGG pages in audio (parallel)")
            
            logger.trace(f"Phase 1 completed: {len(page_positions)} pages found")
            
            # Phase 2: Parallel page-detail calculation (TODO: Next step)
            logger.warning("Phase 2 (page detail calculation) not yet implemented - falling back to sequential for final processing")
            
            # TEMPORARY: Use sequential processing for page details until Phase 2 is implemented
            audio_bytes = bytes(audio_blob_array[()])
            pages_info = _parse_ogg_pages_from_positions(audio_bytes, page_positions, sample_rate)
            
            # Phase 3: Sequential sample-position accumulation
            if sampling_rescale_factor != 1.0:
                pages_info = _accumulate_sample_positions_opus(pages_info, sampling_rescale_factor)
            
            total_time = time.time() - total_start_time
            logger.success(f"Hybrid parallel/sequential index creation: {len(pages_info)} pages in {total_time:.3f}s")
            
            # Convert to final format
            pages_info = [
                {
                    'byte_offset': p['byte_offset'],
                    'page_size': p['page_size'],
                    'sample_pos': p['sample_pos']
                }
                for p in pages_info
            ]
            
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            use_parallel = False
    
    if not use_parallel:
        # Sequential processing (current implementation)
        logger.trace("Using sequential Opus index creation")
        
        # Load audio bytes
        audio_bytes = bytes(audio_blob_array[()])
        
        # Sequential page parsing
        pages_info = _parse_ogg_pages_sequential(audio_bytes, sample_rate)
        
        # Apply ultrasonic correction
        if sampling_rescale_factor != 1.0:
            pages_info = _accumulate_sample_positions_opus(pages_info, sampling_rescale_factor)
        
        if len(pages_info) < 1:
            raise ValueError("Could not find OGG pages in audio (sequential)")
    
    # Create index array (same format for both parallel and sequential)
    logger.trace("Creating index array...")
    index_array = np.array([
        [p['byte_offset'], p['page_size'], p['sample_pos']] 
        for p in pages_info
    ], dtype=OPUS_INDEX_DTYPE)
    
    # Store index in Zarr group
    opus_index = zarr_group.create_array(
        name='opus_index',
        shape=index_array.shape,
        chunks=(min(1000, len(pages_info)), OPUS_INDEX_COLS),
        dtype=OPUS_INDEX_DTYPE
    )
    
    # Write data to the created array
    opus_index[:] = index_array
    
    # Store metadata
    index_attrs = {
        'sample_rate': sample_rate,
        'channels': channels,
        'total_pages': len(pages_info),
        'codec': codec,
        'container_type': container_type,
        'sampling_rescale_factor': sampling_rescale_factor,
        'parallel_processing_used': use_parallel
    }
    
    # Copy additional metadata from audio_blob_array if available
    optional_attrs = [
        'opus_bitrate', 'is_ultrasonic', 'original_sample_rate',
        'first_sample_time_stamp', 'last_sample_time_stamp'
    ]
    
    for attr_name in optional_attrs:
        if attr_name in audio_blob_array.attrs:
            index_attrs[attr_name] = audio_blob_array.attrs[attr_name]
    
    opus_index.attrs.update(index_attrs)
    
    processing_method = "parallel (3-phase)" if use_parallel else "sequential (fallback)"
    logger.success(f"Opus index created with {len(pages_info)} pages using {processing_method}")
    return opus_index


# ##########################################################
#
# Performance Configuration and Diagnostics
# ==========================================
#
# ##########################################################

def configure_parallel_processing(max_workers: int = None, chunk_size_mb: int = 4, 
                                enable_parallel: bool = True) -> dict:
    """
    Configure parallel processing parameters (identical to FLAC)
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 6)
    
    config = {
        'max_workers': max_workers,
        'chunk_size_mb': chunk_size_mb,
        'enable_parallel': enable_parallel,
        'cpu_count': mp.cpu_count(),
        'psutil_available': PSUTIL_AVAILABLE
    }
    
    logger.trace(f"Parallel processing configured: {config}")
    return config


def diagnose_opus_data(audio_blob_array: zarr.Array) -> dict:
    """
    Diagnose OGG-Opus data for potential issues
    
    Args:
        audio_blob_array: Array with OGG-Opus audio data
        
    Returns:
        Diagnostic information
    """
    audio_bytes = bytes(audio_blob_array[()])
    
    diagnosis = {
        'size_bytes': len(audio_bytes),
        'size_mb': len(audio_bytes) / 1024 / 1024,
        'has_ogg_signature': audio_bytes[:4] == OGG_SYNC_PATTERN,
        'ogg_pages_found': 0,
        'estimated_duration_seconds': 0.0,
        'sample_rate': audio_blob_array.attrs.get('sample_rate', 48000),
        'issues': []
    }
    
    # Check OGG signature
    if not diagnosis['has_ogg_signature']:
        diagnosis['issues'].append("Missing OGG signature (OggS)")
        return diagnosis
    
    # Quick page count
    pos = 0
    page_count = 0
    last_granule = 0
    
    while pos < len(audio_bytes) - 4:
        if audio_bytes[pos:pos+4] == OGG_SYNC_PATTERN:
            page_count += 1
            
            # Try to extract granule position for duration estimate
            if pos + 14 < len(audio_bytes):
                try:
                    granule = struct.unpack('<Q', audio_bytes[pos+6:pos+14])[0]
                    if granule != 0xFFFFFFFFFFFFFFFF:
                        last_granule = granule
                except:
                    pass
            
            pos += OGG_PAGE_HEADER_SIZE  # Skip ahead
        else:
            pos += 1
    
    diagnosis['ogg_pages_found'] = page_count
    
    if page_count == 0:
        diagnosis['issues'].append("No OGG pages found")
    
    # Estimate duration
    if last_granule > 0:
        diagnosis['estimated_duration_seconds'] = last_granule / diagnosis['sample_rate']
    
    logger.trace(f"Opus data diagnosis: {diagnosis}")
    return diagnosis


logger.trace("Opus Index Backend module loaded.")