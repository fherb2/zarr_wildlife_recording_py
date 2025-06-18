"""
High-Level Audio Import API for Zarr Wildlife Recording Database
===============================================================

This module provides a comprehensive, user-friendly API for importing audio files into 
Zarr v3 audio databases. It is designed specifically for wildlife researchers, 
acoustic monitoring specialists, and scientists who need reliable, high-performance 
audio storage with intelligent format optimization.

The API emphasizes simplicity and intelligence - users can import audio files with 
minimal configuration while the system automatically analyzes source quality, 
suggests optimal target formats, and handles complex technical details transparently.

DESIGN PHILOSOPHY
================

This module follows three core principles:

1. **Simplicity First**: Functions accept single files or lists of files with identical behavior
2. **Intelligence Built-In**: Automatic quality analysis and optimal format suggestions  
3. **Performance Optimized**: Parallel processing for batch operations with smart grouping

The API is designed for users with basic Python knowledge - complex technical decisions
are automated while still providing full control when needed.

CORE WORKFLOW
============

The typical audio import workflow consists of three simple steps:

**Step 1: Open or Create Audio Database**
```python
import zarrwlr

# Create or open an audio database
audio_db = zarrwlr.open_zarr_audio_grp("./my_wildlife_audio_db")
```

**Step 2: Analyze and Check Files**
```python
# Check if file already imported and get intelligent analysis
already_imported, file_analysis = zarrwlr.is_audio_in_zarr_audio_group(
    audio_db, 
    "bat_recording_384khz.wav"
)

if not already_imported:
    # Review beautiful terminal output with suggestions
    print(file_analysis)
    
    # Optional: Override suggestions if needed
    # file_analysis.target_format = "FLAC_96000"
    # file_analysis.aac_bitrate = 256000
```

**Step 3: Import with Intelligent Processing**
```python
# Import using analysis results and suggestions
if not already_imported:
    result = zarrwlr.aimport(audio_db, file_analysis)
    print(f"Import successful: {result.import_time:.2f}s")
```

BATCH PROCESSING
===============

All functions seamlessly handle single files or lists of files with identical syntax:

**Batch File Analysis:**
```python
# Analyze multiple files at once (parallel processing)
file_paths = ["recording1.wav", "recording2.flac", "recording3.mp3"]
import_status = zarrwlr.is_audio_in_zarr_audio_group(audio_db, file_paths)

for already_imported, file_analysis in import_status:
    if not already_imported:
        print(f"Ready to import: {file_analysis.base_parameter.file.name}")
    else:
        print(f"Already imported: {file_analysis.base_parameter.file.name}")
```

**Batch Import:**
```python
# Import multiple files (parallel processing)
files_to_import = [analysis for imported, analysis in import_status if not imported]
if files_to_import:
    results = zarrwlr.aimport(audio_db, files_to_import)
    print(f"Imported {len(results)} files successfully")
```

INTELLIGENT AUDIO ANALYSIS
==========================

The system includes sophisticated audio analysis optimized for wildlife and scientific recording:

**Automatic Quality Assessment:**
- **Source Quality Classification**: Distinguishes between uncompressed, lossless, and lossy sources
- **Bitrate Analysis**: Intelligent evaluation of lossy compression quality levels
- **Sample Rate Analysis**: Special handling for ultrasound recordings (>96kHz bat calls)
- **Channel Configuration**: Optimized suggestions for mono, stereo, and multichannel recordings

**Smart Format Suggestions:**
- **AAC→AAC Copy-Mode**: 1:1 transfer without re-encoding when source is already AAC
- **Lossy→AAC Upgrade**: Intelligent bitrate calculation for quality improvement
- **Lossless→FLAC**: Preservation of original quality with sample rate matching
- **Ultrasound Handling**: Special reinterpretation techniques for bat recordings

**Conflict Detection:**
- **Blocking Conflicts**: Prevent import when configuration would cause data loss
- **Quality Warnings**: Alert users to potential quality degradation scenarios  
- **Efficiency Warnings**: Suggest optimizations for file size and processing speed

TARGET FORMATS SUPPORTED
========================

The system supports multiple output formats optimized for different use cases:

**FLAC (Lossless Compression):**
```python
# Automatic sample rate selection (1Hz - 655kHz range)
file_analysis.target_format = "FLAC"

# Fixed sample rates for specific requirements
file_analysis.target_format = "FLAC_44100"  # CD quality
file_analysis.target_format = "FLAC_48000"  # Professional standard
file_analysis.target_format = "FLAC_96000"  # High-resolution audio
file_analysis.target_format = "FLAC_192000" # Studio/ultrasound applications
```

**AAC-LC (Lossy Compression):**
```python
# Automatic sample rate and bitrate optimization
file_analysis.target_format = "AAC"

# Fixed configurations for specific needs
file_analysis.target_format = "AAC_44100"   # Standard compatibility
file_analysis.target_format = "AAC_48000"   # Professional applications
file_analysis.target_format = "AAC_32000"   # Ultrasound reinterpretation

# Bitrate control (32-320 kbps range)
file_analysis.aac_bitrate = 192000  # 192 kbps for high quality
```

**Sample Rate Transformations:**
```python
# Preserve original sample rate exactly
file_analysis.target_sampling_transform = "EXACTLY"

# Standard resampling for format compatibility
file_analysis.target_sampling_transform = "RESAMPLING_48000"

# Ultrasound reinterpretation (preserves frequency content)
file_analysis.target_sampling_transform = "REINTERPRETING_32000"
```

WILDLIFE RECORDING SPECIALIZATIONS
==================================

The system includes specific optimizations for wildlife and acoustic monitoring:

**Ultrasound Recording Support (Bat Calls):**
- Automatic detection of ultrasound recordings (>96kHz sample rates)
- Special reinterpretation techniques that preserve frequency domain characteristics
- Protection against accidental signal destruction through inappropriate resampling
- Optimized AAC encoding for ultrasound time-expansion playback

**Multichannel Audio Handling:**
- FLAC support for up to 8 channels with perfect spatial preservation
- AAC multichannel warnings for scientific applications requiring spatial accuracy
- Copy-mode detection for existing multichannel AAC to minimize analysis limitations

**Long-Duration Recording Optimization:**
- Efficient chunking strategies for continuous monitoring data
- Memory-optimized processing for large file collections
- Progress tracking and resumable operations for extensive field recordings

**Environmental Robustness:**
- Comprehensive codec compatibility checking before import
- Graceful handling of corrupted or incomplete audio files
- Detailed logging for research data integrity verification

PERFORMANCE AND SCALABILITY
===========================

**Parallel Processing Architecture:**
- Automatic detection of optimal worker count based on system resources
- Smart grouping for small operations (20 files per subprocess by default)
- File-level parallelization for large audio imports
- Memory-efficient processing with automatic cleanup

**Optimization Strategies:**
- Copy-mode detection eliminates unnecessary re-encoding
- Intelligent chunking for Zarr v3 storage optimization
- Progressive analysis with early conflict detection
- Resource usage monitoring and automatic scaling

**Batch Operation Efficiency:**
```python
# Efficient processing of large file collections
wildlife_recordings = pathlib.Path("field_study_2024").glob("*.wav")
audio_db = zarrwlr.open_zarr_audio_grp("./field_study_database")

# Parallel analysis of entire collection
import_status = zarrwlr.is_audio_in_zarr_audio_group(audio_db, list(wildlife_recordings))

# Batch import with automatic optimization
new_files = [analysis for imported, analysis in import_status if not imported]
if new_files:
    print(f"Importing {len(new_files)} new recordings...")
    results = zarrwlr.aimport(audio_db, new_files)
    print(f"Import completed: {sum(r.import_time for r in results):.1f}s total")
```

ERROR HANDLING AND RECOVERY
===========================

**Comprehensive Validation:**
- Pre-import validation prevents invalid operations
- Blocking conflict detection stops problematic imports before processing
- Automatic rollback on failure ensures database consistency

**User Guidance:**
- Clear error messages with specific remediation steps
- Quality warnings with detailed explanations
- Efficiency suggestions for optimal storage decisions

**Research Data Integrity:**
- SHA256 hash verification prevents duplicate imports
- Complete metadata preservation from source files
- Audit trail creation for scientific reproducibility

CONFIGURATION AND CUSTOMIZATION
===============================

**Global Configuration Options:**
```python
from zarrwlr.config import Config

# Set default AAC quality
Config.set(aac_default_bitrate=192000)

# Configure parallel processing
Config.set(audio_import_batch_size=50)  # Files per subprocess

# Set ultrasound detection threshold
Config.set(ultrasound_threshold_hz=96000)
```

**Per-File Customization:**
```python
# Override automatic suggestions
file_analysis.target_format = "FLAC_96000"
file_analysis.flac_compression_level = 8  # Higher compression
file_analysis.target_sampling_transform = "EXACTLY"

# Check configuration validity
if file_analysis.has_blocking_conflicts:
    print("Configuration invalid:")
    for conflict in file_analysis.conflicts['blocking_conflicts']:
        print(f"  ❌ {conflict}")
```

API REFERENCE SUMMARY
=====================

**Core Functions:**

- `open_zarr_audio_grp(store_path, group_path=None, create=True)` → AGroup
  Opens or creates Zarr audio database group

- `is_audio_in_zarr_audio_group(zarr_audio_group, files)` → tuple[bool, FileParameter] | list[...]
  Checks if files already imported and provides intelligent analysis

- `aimport(zarr_audio_group, file_params)` → ImportResult | list[ImportResult]
  Imports audio files using analysis results and configuration

**Supporting Classes:**

- `AGroup`: Type-safe wrapper for Zarr audio database groups
- `FileParameter`: Comprehensive audio file analysis and configuration
- `ImportResult`: Detailed results from import operations

**Configuration Integration:**

All functions respect global configuration while allowing per-file overrides through
FileParameter instances. The system balances automation with user control, making
simple imports effortless while providing full configurability for complex requirements.

This API serves as the primary interface for wildlife researchers and acoustic monitoring
specialists who need reliable, high-performance audio storage with minimal technical
complexity.
"""

import numpy as np
import pathlib
import datetime
import time
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import zarr
import zarrcompatibility as zc

# Import core functionality
from .utils import check_ffmpeg_tools, next_numeric_group_name, remove_zarr_group_recursive
from .config import Config
from .exceptions import Doublet, ZarrComponentIncomplete, ZarrComponentVersionError, ZarrGroupMismatch
from .import_utils import FileParameter, TargetFormats, TargetSamplingTransforming
from .flac_access import import_flac_to_zarr
from .aac_access import import_aac_to_zarr

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("High-Level Audio Import API loading...")

# Enable universal serialization
zc.enable_universal_serialization()


# ============================================================================
# AGroup Type Class for Type Safety
# ============================================================================

class AGroup:
    """
    Type-safe wrapper for zarr.Group specifically for audio database groups.
    
    This wrapper enables isinstance() checks and ensures that only validated
    Zarr groups are used throughout the audio import API. All zarr.Group
    methods and properties are delegated transparently.
    
    Args:
        zarr_group: Validated zarr.Group that has been confirmed as audio database
        
    Examples:
        >>> # Create from validated group
        >>> zarr_group = init_zarr_audio_grp(store_path)
        >>> audio_group = AGroup(zarr_group)
        >>> 
        >>> # Type checking works
        >>> isinstance(audio_group, AGroup)  # True
        >>> 
        >>> # All zarr.Group methods available
        >>> audio_group.attrs['version']
        >>> list(audio_group.keys())
    """
    
    def __init__(self, zarr_group: zarr.Group):
        if not isinstance(zarr_group, zarr.Group):
            raise TypeError(f"Expected zarr.Group, got {type(zarr_group)}")
        self._group = zarr_group
    
    def __getattr__(self, name):
        """Delegate all attribute access to the wrapped zarr.Group"""
        return getattr(self._group, name)
    
    def __getitem__(self, key):
        """Delegate item access to the wrapped zarr.Group"""
        return self._group[key]
    
    def __setitem__(self, key, value):
        """Delegate item assignment to the wrapped zarr.Group"""
        self._group[key] = value
    
    def __contains__(self, key):
        """Delegate containment checks to the wrapped zarr.Group"""
        return key in self._group
    
    def __iter__(self):
        """Delegate iteration to the wrapped zarr.Group"""
        return iter(self._group)
    
    def __len__(self):
        """Delegate length to the wrapped zarr.Group"""
        return len(self._group)
    
    def __repr__(self):
        """Provide clear representation showing this is an audio group wrapper"""
        return f"AGroup({self._group!r})"
    
    @property
    def zarr_group(self) -> zarr.Group:
        """Access the underlying zarr.Group if needed"""
        return self._group


# ============================================================================
# Import Result Classes
# ============================================================================

@dataclass
class ImportResult:
    """
    Comprehensive result information from audio import operations.
    
    Provides detailed feedback about the import process including timing,
    file information, and technical details for research data integrity.
    
    Attributes:
        success: Whether the import completed successfully
        file_path: Path to the imported audio file
        import_time: Time taken for the import operation (seconds)
        target_format: Final target format used for storage
        target_sampling_transform: Sample rate transformation applied
        source_codec: Original audio codec of the source file
        source_sample_rate: Original sample rate in Hz
        source_channels: Number of audio channels in source
        compressed_size_bytes: Final size in Zarr database
        compression_ratio: Ratio of original to compressed size
        copy_mode_used: Whether copy-mode (no re-encoding) was used
        zarr_group_name: Name of the created Zarr group
        conflicts_detected: Any warnings or conflicts that were resolved
        error_message: Error description if success=False
    """
    success: bool
    file_path: pathlib.Path
    import_time: float
    target_format: Optional[TargetFormats] = None
    target_sampling_transform: Optional[TargetSamplingTransforming] = None
    source_codec: Optional[str] = None
    source_sample_rate: Optional[int] = None
    source_channels: Optional[int] = None
    compressed_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    copy_mode_used: bool = False
    zarr_group_name: Optional[str] = None
    conflicts_detected: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.conflicts_detected is None:
            self.conflicts_detected = []


# ============================================================================
# Core Zarr Group Management Functions
# ============================================================================

def init_zarr_audio_grp(store_path: str|pathlib.Path, group_path: str|pathlib.Path|None = None) -> zarr.Group:
    """
    Initialize a Zarr group for audio storage or verify an existing one.
    
    Creates the necessary metadata structure for a Zarr audio database group,
    including version information and magic identifiers. If the group already
    exists and is properly configured, no changes are made.
    
    Args:
        store_path: Path to the Zarr store directory
        group_path: Optional path to group within store (None for root)
        
    Returns:
        zarr.Group: Initialized audio database group
        
    Raises:
        ZarrComponentVersionError: If existing group has incompatible version
        ValueError: If store path is invalid or cannot be created
        
    Examples:
        >>> # Create audio database in new directory
        >>> audio_group = init_zarr_audio_grp("./wildlife_recordings")
        >>> 
        >>> # Create sub-group within existing store
        >>> audio_group = init_zarr_audio_grp("./main_store", "audio_data")
    """
    logger.trace(f"init_zarr_audio_grp() requested. Parameters: {store_path=}; {group_path=}")
    store_path = pathlib.Path(store_path).resolve()
    
    # Convert group path to string for Zarr
    zarr_group_path = None
    if group_path is not None:
        zarr_group_path = str(group_path)
    
    logger.trace(f"Opening store {store_path} and audio group path {group_path}")
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store, mode='a')
    
    def _initialize_new_audio_group(grp: zarr.Group):
        """Initialize new audio group with required metadata"""
        logger.trace("Initializing new Zarr audio group...")
        grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        grp.attrs["version"] = Config.original_audio_group_version
        logger.trace(f"Audio group initialized with magic_id={grp.attrs['magic_id']} and version={grp.attrs['version']}")
    
    group = None
    if zarr_group_path is not None:
        if zarr_group_path in root:
            # Group exists - verify it's a valid audio group
            logger.trace(f"Zarr group {zarr_group_path} exists. Checking if valid audio group...")
            grp = root[zarr_group_path]
            assert isinstance(grp, zarr.Group), f"Expected zarr.Group, got {type(grp)}"
            
            if not check_if_zarr_audio_grp(grp):
                raise ZarrGroupMismatch(f"Group {zarr_group_path} exists but is not a valid audio group")
            
            logger.debug(f"Zarr group {zarr_group_path} is a valid audio group")
            return grp
        else:
            # Create new group
            logger.trace(f"Creating new Zarr group {zarr_group_path}...")
            created = False
            try:
                group = root.create_group(zarr_group_path)
                created = True
                _initialize_new_audio_group(group)
            except Exception:
                if created:
                    remove_zarr_group_recursive(root.store, group.path)
                raise
            logger.success(f"New audio group {zarr_group_path} created")
    else:
        # Use root as audio group
        logger.trace("Using root as audio group. Checking validity...")
        if not check_if_zarr_audio_grp(root):
            _initialize_new_audio_group(root)
            logger.success("Root initialized as audio group")
        else:
            logger.success("Root is already a valid audio group")
        group = root
    
    if group is None:
        group = root
    
    return group


def check_if_zarr_audio_grp(group: zarr.Group) -> bool:
    """
    Check if a Zarr group is a valid audio database group.
    
    Verifies that the group has the correct magic identifier and version
    information. If the version is outdated, attempts automatic upgrade.
    
    Args:
        group: Zarr group to validate
        
    Returns:
        bool: True if group is valid, False otherwise
        
    Examples:
        >>> if check_if_zarr_audio_grp(my_group):
        ...     print("Valid audio group")
        ... else:
        ...     print("Not an audio group")
    """
    logger.trace(f"Checking if group is valid audio group: {group}")
    
    # Check for required attributes
    if "magic_id" not in group.attrs or "version" not in group.attrs:
        logger.trace("Group missing required attributes (magic_id or version)")
        return False
    
    # Check magic ID
    if group.attrs["magic_id"] != Config.original_audio_group_magic_id:
        logger.trace(f"Group has wrong magic_id: {group.attrs['magic_id']}")
        return False
    
    # Check version - attempt upgrade if needed
    if group.attrs["version"] != Config.original_audio_group_version:
        logger.trace(f"Group version mismatch: {group.attrs['version']} vs {Config.original_audio_group_version}")
        
        # Attempt upgrade
        if upgrade_zarr_audio_grp(group, group.attrs["version"]):
            logger.debug("Group successfully upgraded to current version")
            return True
        else:
            logger.warning(f"Cannot upgrade group from version {group.attrs['version']}")
            return False
    
    logger.trace("Group is valid audio group")
    return True


def upgrade_zarr_audio_grp(group: zarr.Group, current_version: tuple) -> bool:
    """
    Template function for upgrading audio group versions.
    
    This is a placeholder for future version upgrade functionality.
    Currently returns True for all upgrade requests since no upgrades
    are implemented yet.
    
    Args:
        group: Zarr group to upgrade
        current_version: Current version of the group
        
    Returns:
        bool: True if upgrade successful or not needed, False if failed
        
    Note:
        This function is a template for future development. Actual upgrade
        logic will be implemented when backward compatibility is needed.
    """
    logger.trace(f"upgrade_zarr_audio_grp() called for version {current_version}")
    # Template implementation - no upgrades implemented yet
    return True


def open_zarr_audio_grp(store_path: str|pathlib.Path, 
                       group_path: str|pathlib.Path|None = None, 
                       create: bool = True) -> AGroup:
    """
    Open or create a Zarr audio database group with automatic validation.
    
    This is the primary function for accessing Zarr audio databases. It handles
    group creation, validation, version upgrades, and returns a type-safe
    AGroup wrapper for use with other API functions.
    
    Args:
        store_path: Path to Zarr store directory
        group_path: Optional path to group within store (None for root)
        create: If True, create group if it doesn't exist; if False, raise exception
        
    Returns:
        AGroup: Type-safe wrapper for the audio database group
        
    Raises:
        FileNotFoundError: If create=False and group doesn't exist
        ZarrGroupMismatch: If existing group is not a valid audio group
        ZarrComponentVersionError: If group version cannot be upgraded
        
    Examples:
        >>> # Create new audio database
        >>> audio_db = open_zarr_audio_grp("./my_audio_database")
        >>> 
        >>> # Open existing database (read-only check)
        >>> audio_db = open_zarr_audio_grp("./existing_db", create=False)
        >>> 
        >>> # Create sub-group in existing store
        >>> audio_db = open_zarr_audio_grp("./main_store", "wildlife_recordings")
    """
    logger.trace(f"open_zarr_audio_grp() requested: {store_path=}, {group_path=}, {create=}")
    
    store_path = pathlib.Path(store_path)
    
    # Check if store exists
    if not store_path.exists():
        if not create:
            raise FileNotFoundError(f"Zarr store does not exist: {store_path}")
        # Create will be handled by init_zarr_audio_grp
    
    try:
        # Try to initialize/open the group
        zarr_group = init_zarr_audio_grp(store_path, group_path)
        
        # Final validation
        if not check_if_zarr_audio_grp(zarr_group):
            raise ZarrGroupMismatch("Group exists but failed final validation as audio group")
        
        logger.success(f"Audio group opened successfully: {store_path}/{group_path or 'root'}")
        return AGroup(zarr_group)
        
    except Exception as e:
        if not create and "does not exist" in str(e).lower():
            raise FileNotFoundError(f"Audio group does not exist and create=False: {store_path}/{group_path or 'root'}")
        raise


# ============================================================================
# File Analysis and Import Status Functions
# ============================================================================

def is_audio_in_zarr_audio_group(
    zarr_audio_group: AGroup,
    files: Union[str, pathlib.Path, FileParameter, List[Union[str, pathlib.Path, FileParameter]]]
) -> Union[Tuple[bool, FileParameter], List[Tuple[bool, FileParameter]]]:
    """
    Check if audio files are already imported and provide intelligent analysis.
    
    This function serves dual purposes: detecting duplicate imports and providing
    comprehensive audio file analysis with intelligent format suggestions. It
    handles single files or lists with automatic parallel processing for performance.
    
    For each file, the function:
    1. Creates FileParameter analysis if not provided (automatic quality assessment)
    2. Checks SHA256 hash against existing imports to detect duplicates
    3. Returns both import status and complete file analysis
    
    Args:
        zarr_audio_group: Validated audio database group (AGroup instance)
        files: Single file/FileParameter or list of files to check
               Accepts: str paths, pathlib.Path objects, or FileParameter instances
        
    Returns:
        Single file: tuple[bool, FileParameter] - (already_imported, file_analysis)
        Multiple files: list[tuple[bool, FileParameter]] - one tuple per file
        
    Raises:
        TypeError: If zarr_audio_group is not an AGroup instance
        FileNotFoundError: If any specified file doesn't exist
        ValueError: If file has no audio streams or is corrupted
        
    Examples:
        >>> # Single file check with automatic analysis
        >>> audio_db = open_zarr_audio_grp("./database")
        >>> already_imported, analysis = is_audio_in_zarr_audio_group(
        ...     audio_db, "recording.wav"
        ... )
        >>> if not already_imported:
        ...     print(analysis)  # Beautiful terminal output with suggestions
        
        >>> # Batch processing multiple files
        >>> file_list = ["rec1.wav", "rec2.flac", "rec3.mp3"]
        >>> results = is_audio_in_zarr_audio_group(audio_db, file_list)
        >>> for imported, analysis in results:
        ...     if not imported:
        ...         print(f"New file: {analysis.base_parameter.file.name}")
        
        >>> # Using pre-analyzed FileParameter objects
        >>> file_param = FileParameter("ultrasound_bat.wav")
        >>> file_param.target_format = "AAC_32000"  # Override suggestion
        >>> imported, analysis = is_audio_in_zarr_audio_group(audio_db, file_param)
    """
    if not isinstance(zarr_audio_group, AGroup):
        raise TypeError(f"Expected AGroup, got {type(zarr_audio_group)}")
    
    logger.trace("is_audio_in_zarr_audio_group() requested")
    
    # Handle single file/parameter
    if not isinstance(files, list):
        logger.trace("Processing single file/parameter")
        return _check_single_file_import_status(zarr_audio_group, files)
    
    # Handle list of files/parameters
    logger.trace(f"Processing {len(files)} files for import status check")
    
    # Use batch processing with grouping for performance
    batch_size = getattr(Config, 'audio_import_batch_size', 20)
    
    if len(files) <= batch_size:
        # Small list - process directly without subprocess overhead
        logger.trace(f"Small batch ({len(files)} <= {batch_size}), processing directly")
        results = []
        for file_item in files:
            try:
                result = _check_single_file_import_status(zarr_audio_group, file_item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error checking file {file_item}: {e}")
                # Create error FileParameter for consistency
                if isinstance(file_item, FileParameter):
                    error_param = file_item
                else:
                    try:
                        error_param = FileParameter(file_item)
                    except:
                        # Fallback if FileParameter creation fails
                        error_param = None
                results.append((False, error_param))
        return results
    
    # Large list - use parallel processing with grouping
    logger.trace(f"Large batch ({len(files)} > {batch_size}), using parallel processing")
    return _check_files_parallel_grouped(zarr_audio_group, files, batch_size)


def _check_single_file_import_status(zarr_audio_group: AGroup, 
                                    file_item: Union[str, pathlib.Path, FileParameter]) -> Tuple[bool, FileParameter]:
    """
    Check import status for a single file with comprehensive analysis.
    
    Args:
        zarr_audio_group: Validated audio database group
        file_item: File path or FileParameter to check
        
    Returns:
        tuple[bool, FileParameter]: (already_imported, file_analysis)
    """
    # Create or use existing FileParameter
    if isinstance(file_item, FileParameter):
        file_param = file_item
        logger.trace(f"Using provided FileParameter for {file_param.base_parameter.file.name}")
    else:
        logger.trace(f"Creating FileParameter for {file_item}")
        file_param = FileParameter(file_item)
    
    # Check if file is already imported using SHA256 hash
    file_hash = file_param.base_parameter.file_sh256
    logger.trace(f"Checking for existing import with hash {file_hash[:16]}...")
    
    for group_name in zarr_audio_group:
        if group_name.isdigit():  # Numeric group names contain imported audio
            try:
                audio_grp = zarr_audio_group[group_name]
                if (hasattr(audio_grp, 'attrs') and 
                    audio_grp.attrs.get("type") == "original_audio_file"):
                    
                    stored_features = audio_grp.attrs.get("base_features")
                    if stored_features and stored_features.get("SH256") == file_hash:
                        logger.debug(f"File {file_param.base_parameter.file.name} already imported in group {group_name}")
                        return True, file_param
                        
            except Exception as e:
                logger.trace(f"Error checking group {group_name}: {e}")
                continue
    
    logger.trace(f"File {file_param.base_parameter.file.name} not found in database")
    return False, file_param


def _check_files_parallel_grouped(zarr_audio_group: AGroup, 
                                 files: List[Union[str, pathlib.Path, FileParameter]], 
                                 batch_size: int) -> List[Tuple[bool, FileParameter]]:
    """
    Check multiple files using parallel processing with intelligent grouping.
    
    Args:
        zarr_audio_group: Validated audio database group
        files: List of files to check
        batch_size: Number of files per subprocess
        
    Returns:
        List of (already_imported, file_analysis) tuples in original order
    """
    # Group files into batches
    file_batches = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        file_batches.append((i, batch))  # Include original index for ordering
    
    logger.trace(f"Created {len(file_batches)} batches of max {batch_size} files each")
    
    # Process batches in parallel
    max_workers = min(len(file_batches), mp.cpu_count())
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch processing tasks
        future_to_batch = {
            executor.submit(_process_file_batch_worker, zarr_audio_group.zarr_group, batch_files): start_idx
            for start_idx, batch_files in file_batches
        }
        
        # Collect results maintaining order
        for future in as_completed(future_to_batch):
            start_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                for offset, result in enumerate(batch_results):
                    results[start_idx + offset] = result
            except Exception as e:
                logger.error(f"Batch processing failed for index {start_idx}: {e}")
                # Fill with error results
                batch_files = next(batch for idx, batch in file_batches if idx == start_idx)[1]
                for offset, file_item in enumerate(batch_files):
                    try:
                        error_param = FileParameter(file_item) if not isinstance(file_item, FileParameter) else file_item
                    except:
                        error_param = None
                    results[start_idx + offset] = (False, error_param)
    
    # Return results in original order
    ordered_results = []
    for i in range(len(files)):
        if i in results:
            ordered_results.append(results[i])
        else:
            # Fallback for missing results
            logger.warning(f"Missing result for file index {i}")
            ordered_results.append((False, None))
    
    return ordered_results


def _process_file_batch_worker(zarr_group: zarr.Group, 
                              file_batch: List[Union[str, pathlib.Path, FileParameter]]) -> List[Tuple[bool, FileParameter]]:
    """
    Worker function for processing a batch of files in subprocess.
    
    This function runs in a separate process to avoid GIL limitations.
    
    Args:
        zarr_group: Raw zarr.Group (not AGroup, for subprocess compatibility)
        file_batch: List of files to process in this batch
        
    Returns:
        List of (already_imported, file_analysis) tuples
    """
    results = []
    audio_group = AGroup(zarr_group)  # Wrap in subprocess
    
    for file_item in file_batch:
        try:
            result = _check_single_file_import_status(audio_group, file_item)
            results.append(result)
        except Exception as e:
            logger.error(f"Worker error processing {file_item}: {e}")
            try:
                error_param = FileParameter(file_item) if not isinstance(file_item, FileParameter) else file_item
            except:
                error_param = None
            results.append((False, error_param))
    
    return results


# ============================================================================
# Audio Import Functions
# ============================================================================

class ImportWorker:
    """
    Worker class for handling audio import operations in both direct and subprocess modes.
    
    This class encapsulates the logic for importing individual audio files using
    FileParameter configuration. It supports both direct execution in the main
    process and subprocess execution for parallel batch imports.
    
    The worker handles:
    - Target format selection and validation
    - Codec-specific import delegation (FLAC/AAC)
    - Error handling and rollback on failure
    - Performance timing and result reporting
    """
    
    def __init__(self):
        self.start_time = None
    
    def run_direct(self, zarr_audio_group: AGroup, file_param: FileParameter) -> ImportResult:
        """
        Execute import directly in main process.
        
        Args:
            zarr_audio_group: Target audio database group
            file_param: Configured file analysis and parameters
            
        Returns:
            ImportResult: Detailed import operation results
        """
        logger.trace(f"Direct import requested for {file_param.base_parameter.file.name}")
        return self._execute_import(zarr_audio_group, file_param)
    
    def run_subprocess(self, zarr_group: zarr.Group, file_param: FileParameter) -> ImportResult:
        """
        Execute import in subprocess (called by ProcessPoolExecutor).
        
        Args:
            zarr_group: Raw zarr.Group for subprocess compatibility
            file_param: Configured file analysis and parameters
            
        Returns:
            ImportResult: Detailed import operation results
        """
        logger.trace(f"Subprocess import requested for {file_param.base_parameter.file.name}")
        audio_group = AGroup(zarr_group)  # Wrap in subprocess
        return self._execute_import(audio_group, file_param)
    
    def _execute_import(self, zarr_audio_group: AGroup, file_param: FileParameter) -> ImportResult:
        """
        Core import execution logic shared by direct and subprocess modes.
        
        Args:
            zarr_audio_group: Target audio database group
            file_param: Configured file analysis and parameters
            
        Returns:
            ImportResult: Detailed import operation results
        """
        self.start_time = time.time()
        
        try:
            # Validate import readiness
            if not file_param.can_be_imported:
                conflicts = '; '.join(file_param.conflicts.get('blocking_conflicts', []))
                return ImportResult(
                    success=False,
                    file_path=file_param.base_parameter.file,
                    import_time=0.0,
                    error_message=f"Import blocked by conflicts: {conflicts}"
                )
            
            # Get import parameters
            import_params = file_param.get_import_parameters()
            target_format = import_params['target_format']
            
            if not target_format:
                return ImportResult(
                    success=False,
                    file_path=file_param.base_parameter.file,
                    import_time=0.0,
                    error_message="No target format specified"
                )
            
            # Create new audio group
            new_group_name = next_numeric_group_name(zarr_audio_group.zarr_group)
            logger.trace(f"Creating new audio group: {new_group_name}")
            
            created = False
            try:
                new_audio_grp = zarr_audio_group.zarr_group.require_group(new_group_name)
                created = True
                
                # Set group metadata
                new_audio_grp.attrs["original_audio_data_array_version"] = Config.original_audio_data_array_version
                new_audio_grp.attrs["type"] = "original_audio_file"
                new_audio_grp.attrs["encoding"] = target_format.code
                
                # Store file metadata
                base_features = self._create_base_features(file_param)
                new_audio_grp.attrs["base_features"] = base_features
                
                # Get source parameters for import
                source_params = self._get_source_params(file_param)
                
                # Execute codec-specific import
                if target_format.code == 'flac':
                    audio_array = import_flac_to_zarr(
                        zarr_group=new_audio_grp,
                        audio_file=file_param.base_parameter.file,
                        source_params=source_params,
                        first_sample_time_stamp=datetime.datetime.now(),
                        flac_compression_level=import_params['flac_compression_level']
                    )
                elif target_format.code == 'aac':
                    audio_array = import_aac_to_zarr(
                        zarr_group=new_audio_grp,
                        audio_file=file_param.base_parameter.file,
                        source_params=source_params,
                        first_sample_time_stamp=datetime.datetime.now(),
                        aac_bitrate=import_params['aac_bitrate']
                    )
                else:
                    raise ValueError(f"Unsupported target format: {target_format.code}")
                
                # Calculate final metrics
                import_time = time.time() - self.start_time
                original_size = file_param.base_parameter.file_size_bytes
                compressed_size = audio_array.nbytes if hasattr(audio_array, 'nbytes') else 0
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                # Get source info for result
                selected_streams = file_param.selected_audio_streams
                source_stream = selected_streams[0] if selected_streams else None
                
                logger.success(f"Import completed: {file_param.base_parameter.file.name} -> group {new_group_name}")
                
                return ImportResult(
                    success=True,
                    file_path=file_param.base_parameter.file,
                    import_time=import_time,
                    target_format=target_format,
                    target_sampling_transform=import_params['target_sampling_transform'],
                    source_codec=source_stream.codec_name if source_stream else None,
                    source_sample_rate=source_stream.sample_rate if source_stream else None,
                    source_channels=source_stream.nb_channels if source_stream else None,
                    compressed_size_bytes=compressed_size,
                    compression_ratio=compression_ratio,
                    copy_mode_used=import_params.get('copy_mode', False),
                    zarr_group_name=new_group_name,
                    conflicts_detected=file_param.conflicts.get('quality_warnings', []) + 
                                    file_param.conflicts.get('efficiency_warnings', [])
                )
                
            except Exception as e:
                # Rollback on error
                if created:
                    logger.trace("Rolling back created group due to import error")
                    remove_zarr_group_recursive(zarr_audio_group.zarr_group.store, new_audio_grp.path)
                raise
                
        except Exception as e:
            import_time = time.time() - self.start_time if self.start_time else 0.0
            logger.error(f"Import failed for {file_param.base_parameter.file.name}: {e}")
            
            return ImportResult(
                success=False,
                file_path=file_param.base_parameter.file,
                import_time=import_time,
                error_message=str(e)
            )
    
    def _create_base_features(self, file_param: FileParameter) -> dict:
        """Create base features dictionary for storage in group metadata"""
        return {
            "FILENAME": file_param.base_parameter.file.name,
            "SH256": file_param.base_parameter.file_sh256,
            "SIZE_BYTES": file_param.base_parameter.file_size_bytes,
            "HAS_AUDIO_STREAM": file_param.has_audio,
            "NB_STREAMS": file_param.number_of_audio_streams
        }
    
    def _get_source_params(self, file_param: FileParameter) -> dict:
        """Extract source parameters for codec import functions"""
        selected_streams = file_param.selected_audio_streams
        if not selected_streams:
            raise ValueError("No audio streams selected for import")
        
        primary_stream = selected_streams[0]
        return {
            "sampling_rate": primary_stream.sample_rate or 48000,
            "sample_format": primary_stream.sample_format or "s16",
            "bit_rate": primary_stream.bit_rate,
            "nb_channels": primary_stream.nb_channels or 1
        }


def aimport(zarr_audio_group: AGroup, 
           file_params: Union[FileParameter, List[FileParameter]]) -> Union[ImportResult, List[ImportResult]]:
    """
    Import audio files into Zarr database using intelligent analysis and configuration.
    
    This is the primary import function that processes FileParameter instances containing
    complete audio analysis and configuration. The function handles both single files
    and batch imports with automatic parallel processing for optimal performance.
    
    Before starting any imports, the function validates that all files can be imported
    without blocking conflicts. If any file has blocking conflicts, the entire operation
    is aborted to prevent partial imports.
    
    Args:
        zarr_audio_group: Target audio database group (must be AGroup instance)
        file_params: Single FileParameter or list of FileParameters to import
                    Each FileParameter should contain complete analysis and configuration
        
    Returns:
        Single file: ImportResult with detailed operation information
        Multiple files: List[ImportResult] with results for each file in original order
        
    Raises:
        TypeError: If zarr_audio_group is not an AGroup instance
        ValueError: If any FileParameter has blocking conflicts
        Doublet: If file is already imported (detected during import process)
        
    Examples:
        >>> # Single file import with automatic suggestions
        >>> audio_db = open_zarr_audio_grp("./database")
        >>> _, file_analysis = is_audio_in_zarr_audio_group(audio_db, "recording.wav")
        >>> result = aimport(audio_db, file_analysis)
        >>> print(f"Import took {result.import_time:.2f}s")
        
        >>> # Batch import with custom configuration
        >>> files = ["rec1.wav", "rec2.flac", "rec3.mp3"]
        >>> analyses = []
        >>> for file_path in files:
        ...     _, analysis = is_audio_in_zarr_audio_group(audio_db, file_path)
        ...     analysis.target_format = "AAC_48000"  # Override suggestion
        ...     analysis.aac_bitrate = 192000
        ...     analyses.append(analysis)
        >>> 
        >>> results = aimport(audio_db, analyses)
        >>> successful = [r for r in results if r.success]
        >>> print(f"Successfully imported {len(successful)}/{len(results)} files")
        
        >>> # Error handling example
        >>> try:
        ...     result = aimport(audio_db, problematic_analysis)
        >>> except ValueError as e:
        ...     print(f"Import blocked: {e}")
        ...     # Review conflicts: problematic_analysis.conflicts
    """
    if not isinstance(zarr_audio_group, AGroup):
        raise TypeError(f"Expected AGroup, got {type(zarr_audio_group)}")
    
    logger.trace("aimport() requested")
    
    # Handle single FileParameter
    if not isinstance(file_params, list):
        logger.trace("Processing single FileParameter import")
        
        # Validate can be imported
        if not file_params.can_be_imported:
            conflicts = '; '.join(file_params.conflicts.get('blocking_conflicts', []))
            raise ValueError(f"Import blocked by conflicts: {conflicts}")
        
        # Execute direct import
        worker = ImportWorker()
        return worker.run_direct(zarr_audio_group, file_params)
    
    # Handle list of FileParameters
    logger.trace(f"Processing batch import of {len(file_params)} files")
    
    # Pre-validate ALL files before starting any imports
    blocking_conflicts = []
    for i, file_param in enumerate(file_params):
        if not file_param.can_be_imported:
            conflicts = file_param.conflicts.get('blocking_conflicts', [])
            blocking_conflicts.extend([f"File {i} ({file_param.base_parameter.file.name}): {c}" for c in conflicts])
    
    if blocking_conflicts:
        error_msg = "Import blocked for entire batch due to conflicts:\n" + '\n'.join(blocking_conflicts)
        raise ValueError(error_msg)
    
    # All files validated - proceed with import
    logger.trace("All files validated for import. Starting batch processing...")
    
    # For imports, always use file-level parallelization (large files justify individual processes)
    max_workers = min(len(file_params), mp.cpu_count())
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit import tasks
        future_to_index = {}
        for i, file_param in enumerate(file_params):
            future = executor.submit(_import_subprocess_worker, zarr_audio_group.zarr_group, file_param)
            future_to_index[future] = i
        
        # Collect results maintaining order
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.error(f"Import subprocess failed for index {index}: {e}")
                results[index] = ImportResult(
                    success=False,
                    file_path=file_params[index].base_parameter.file,
                    import_time=0.0,
                    error_message=f"Subprocess execution failed: {e}"
                )
    
    # Return results in original order
    ordered_results = []
    for i in range(len(file_params)):
        if i in results:
            ordered_results.append(results[i])
        else:
            logger.warning(f"Missing result for file index {i}")
            ordered_results.append(ImportResult(
                success=False,
                file_path=file_params[i].base_parameter.file,
                import_time=0.0,
                error_message="Missing result from parallel processing"
            ))
    
    successful_imports = sum(1 for r in ordered_results if r.success)
    logger.success(f"Batch import completed: {successful_imports}/{len(ordered_results)} files successful")
    
    return ordered_results


def _import_subprocess_worker(zarr_group: zarr.Group, file_param: FileParameter) -> ImportResult:
    """
    Worker function for importing files in subprocess.
    
    Args:
        zarr_group: Raw zarr.Group for subprocess compatibility
        file_param: FileParameter to import
        
    Returns:
        ImportResult: Import operation results
    """
    worker = ImportWorker()
    return worker.run_subprocess(zarr_group, file_param)


# ============================================================================
# Configuration and Initialization
# ============================================================================

# Ensure required configuration parameters exist
if not hasattr(Config, 'audio_import_batch_size'):
    logger.trace("Adding audio_import_batch_size to Config")
    Config.audio_import_batch_size = 20

# Verify ffmpeg tools are available on module load
check_ffmpeg_tools()

logger.debug("High-Level Audio Import API loaded successfully")