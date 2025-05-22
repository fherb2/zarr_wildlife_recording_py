"""Package configuration constants and variables"""

from .types import LogLevel
from typing import get_type_hints
import pathlib

# Since this gives initializing for the logger, we can not call the logger machine here!


class Config:
    """Package wide configuration variables"""

    # configurable values
    # -------------------
    _CONFIGURABLE_KEYS = [  "log_level",
                            "debug",
                            "original_audio_chunk_size",
                            "original_audio_chunks_per_shard",
                         ]

    log_level: LogLevel = LogLevel.DEBUG    # Standard logging level
    log_filepath: pathlib.Path|None = None  # set a logging path if you want to write logging outputs
                                            # eg. pathlibPath("./mylogfile.log")
    original_audio_chunk_size: int       = int(2**20) # 1 MByte (don't use small chunks: too slowly!)
    original_audio_chunks_per_shard: int = int(50*2**20 / original_audio_chunk_size) # 50 MByte is a good value for access for both: local resources and network

    # immutable keys  
    # --------------  
    version = (1,0)                       # Package Version; tuple of (Major, Minor, Patch) ; Patch is optional
    original_audio_group_version = (1,0)  # tuple of (Major, Minor, Patch) ; Patch is optional
    original_audio_group_magic_id = "zarrwlr_original_audio_group_fherb2_23091969052025"
    original_audio_data_array_version = (1,0)  # tuple of (Major, Minor, Patch) ; Patch is optional
    
    # Internal tracking for logging system
    # ------------------------------------
    # these are not configuration values
    _logging_configured = False
    _last_log_level = None
    _last_log_filepath = None

    @classmethod
    def set(cls, **kwargs):
        """Set / overwrite package wide configuration variables"""
        from .logsetup import LoggingManager  # Import here to avoid circular imports; Sometimes it makes sense to
                                              # write imports not at the binning of a Python file.
        
        old_log_level = cls.log_level
        old_log_filepath = cls.log_filepath
        
        for key, value in kwargs.items():
            if hasattr(cls, key):
                if key in cls._CONFIGURABLE_KEYS:
                    expected_type = get_type_hints(cls).get(key)
                    if expected_type and not isinstance(value, expected_type):
                        raise TypeError(f"Expected {key} to be of type {expected_type.__name__}, got {type(value).__name__}")
                    setattr(cls, key, value)
                else:
                    raise AttributeError(f"Sorry, value of key '{key}' is immutable.")
            else:
                raise AttributeError(f"Invalid config key: {key}. No such key.")
            
        # Check if logging configuration changed
        if (cls.log_level != old_log_level or cls.log_filepath != old_log_filepath):
            LoggingManager.reconfigure()
