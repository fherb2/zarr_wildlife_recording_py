"""Package configuration constants and variables"""

import logging
from .types import LogLevel
from typing import get_type_hints

# get the logger for package   
logger = logging.getLogger(__name__)


class Config:
    """Package wide configuration variables"""

    _CONFIGURABLE_KEYS = [  "log_level",
                            "debug",
                            "original_audio_chunk_size",
                            "original_audio_chunks_per_shard",
                         ]

    log_level: LogLevel = LogLevel.NOTSET
    debug: bool = False # special debugging mode for additional deep inspecting via print/log messages
    original_audio_chunk_size: int       = int(2**20) # 1 MByte (don't use small chunks: too slowly!)
    original_audio_chunks_per_shard: int = int(50*2**20 / original_audio_chunk_size) # 50 MByte is a good value for access for both: local resources and network

    _IMMUTABLE_KEYS = [ "version",
                        "original_audio_group_version",
                      ]
    
    version = (1,0)                       # Package Version; tuple of (Major, Minor, Patch) ; Patch is optional
    original_audio_group_version = (1,0)  # tuple of (Major, Minor, Patch) ; Patch is optional

    @classmethod
    def set(cls, **kwargs):
        """Set / overwrite package wide configuration variables"""
        for key, value in kwargs.items():
            if key in cls._IMMUTABLE_KEYS:
                raise AttributeError(f"Invalid config key: {key}. Value is immutable.")
            if hasattr(cls, key) and key in cls._CONFIGURABLE_KEYS:
                expected_type = get_type_hints(cls).get(key)
                if expected_type and not isinstance(value, expected_type):
                    raise TypeError(f"Expected {key} to be of type {expected_type.__name__}, got {type(value).__name__}")
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Invalid config key: {key}. No such key.")
            
    @classmethod
    def set_logging(cls, log_level:LogLevel, debug:bool=False) -> None:
        cls.set(log_level=log_level)
        cls.set(debug=debug)
