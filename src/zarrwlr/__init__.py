"""zarrwlr package initialization."""

from .logsetup import get_module_logger

# Get logger for this module
logger = get_module_logger(__file__)

# Hauptfunktionen für Audio-Import und -Verarbeitung
from .aimport import (
    create_original_audio_group,
    import_original_audio_file,
    extract_audio_segment,
    parallel_extract_audio_segments,
    base_features_from_audio_file,
    is_audio_in_original_audio_group,
    check_if_original_audio_group,
    audio_codec_compression
)

# Konfiguration und Ausnahmen
from .config import Config
from .exceptions import (
    Doublet,
    ZarrComponentIncomplete,
    ZarrComponentVersionError,
    ZarrGroupMismatch,
    OggImportError
)

# Datentypen und Enums
from .packagetypes import (
    AudioFileBaseFeatures,
    AudioCompression,
    AudioSampleFormatMap
)

# FLAC-spezifische Funktionen (für erweiterte Nutzung)
from .flac_access import (
    import_flac_to_zarr,
    extract_audio_segment_flac,
    parallel_extract_audio_segments_flac,
    build_flac_index
)

# FLAC Index Backend (für erweiterte Nutzung)
from . import flac_index_backend

# Opus-spezifische Funktionen (für erweiterte Nutzung) - LAZY IMPORT to avoid circular import
def _import_opus_functions():
    """Lazy import of opus functions to avoid circular import issues"""
    try:
        from .opus_access import (
            import_opus_to_zarr,
            extract_audio_segment_opus,
            parallel_extract_audio_segments_opus,
            build_opus_index
        )
        # FIXED: Also import opus_index_backend
        from . import opus_index_backend
        
        return {
            'import_opus_to_zarr': import_opus_to_zarr,
            'extract_audio_segment_opus': extract_audio_segment_opus,
            'parallel_extract_audio_segments_opus': parallel_extract_audio_segments_opus,
            'build_opus_index': build_opus_index,
            'opus_index_backend': opus_index_backend  # NEW: Export backend module
        }
    except ImportError as e:
        logger.warning(f"Could not import opus functions: {e}")
        return {}

# Import opus functions at module level but with lazy loading
_opus_funcs = _import_opus_functions()
if _opus_funcs:
    import_opus_to_zarr = _opus_funcs['import_opus_to_zarr']
    extract_audio_segment_opus = _opus_funcs['extract_audio_segment_opus']
    parallel_extract_audio_segments_opus = _opus_funcs['parallel_extract_audio_segments_opus']
    build_opus_index = _opus_funcs['build_opus_index']
    opus_index_backend = _opus_funcs['opus_index_backend']  # NEW: Make backend available

# Utilities 
from .utils import (
    file_size,
    next_numeric_group_name,
    remove_zarr_group_recursive,
    safe_int_conversion
)

# Logging-Typen
from .logsetup import LogLevel

# Was bei "from zarrwlr import *" geladen wird
__all__ = [
    # Hauptfunktionen
    "create_original_audio_group",
    "import_original_audio_file", 
    "extract_audio_segment",
    "parallel_extract_audio_segments",
    "base_features_from_audio_file",
    "is_audio_in_original_audio_group",
    "check_if_original_audio_group",
    "audio_codec_compression",
    
    # Konfiguration
    "Config",
    
    # Ausnahmen
    "Doublet",
    "ZarrComponentIncomplete", 
    "ZarrComponentVersionError",
    "ZarrGroupMismatch",
    "OggImportError",
    
    # Datentypen
    "AudioFileBaseFeatures",
    "AudioCompression",
    "AudioSampleFormatMap",
    "LogLevel",
    
    # FLAC-Funktionen
    "import_flac_to_zarr",
    "extract_audio_segment_flac", 
    "parallel_extract_audio_segments_flac",
    "build_flac_index",
    "flac_index_backend",  # FLAC backend module
    
    # Utilities
    "file_size",
    "next_numeric_group_name",
    "remove_zarr_group_recursive",
    "safe_int_conversion"
]

# Only add opus functions to __all__ if they were successfully imported
if _opus_funcs:
    __all__.extend([
        "import_opus_to_zarr",
        "extract_audio_segment_opus", 
        "parallel_extract_audio_segments_opus",
        "build_opus_index",
        "opus_index_backend"  # NEW: Export opus backend module
    ])

logger.info("zarrwlr package loaded successfully.")