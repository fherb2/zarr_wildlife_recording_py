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

# Opus-spezifische Funktionen (für erweiterte Nutzung)
from .opus_access import (
    import_opus_to_zarr,
    extract_audio_segment_opus,
    parallel_extract_audio_segments_opus,
    build_opus_index
)

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
    
    # Utilities
    "file_size",
    "next_numeric_group_name",
    "remove_zarr_group_recursive",
    "safe_int_conversion"
]

logger.info("zarrwlr package loaded successfully.")