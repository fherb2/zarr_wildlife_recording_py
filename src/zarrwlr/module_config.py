

import logging
from zarrwlr.logging_setup import LogLevel

# get the module logger   
logger = logging.getLogger(__name__)

class ModuleStaticConfig:
    """Immutable configuration values for the module"""
    versions = { "library": (1,0),                     # tuple of (Major, Minor, Patch) ; Patch is optional
                 "file_blob_group_version": (1,0)                                     
               }

class ModuleConfig:
    """Module wide configuration variables"""
    log_level: LogLevel = "NOTSET"
    debug: bool = False # special debugging mode for additional deep inspecting via print/log messages
    new_file_blob_shard_size: int = 8*1024*1024 # 8 MByte is a good value for access for both: local resources and network
    new_file_blob_chunk_size: int = 4096 # eg. a bit more than size of 3 opus pages

    @classmethod
    def set(cls, **kwargs):
        """Set / overwrite module wide configuration variables"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                # special checks and exceptions
                if key == "log_level":
                    if isinstance(value, str):
                        try:
                            value = LogLevel(value.upper())
                        except ValueError:
                            raise ValueError(
                                f"Invalid log_level: '{value}'. "
                                f"Must be one of: {[lvl.value for lvl in LogLevel]}"
                            )
                    elif not isinstance(value, (LogLevel, type(None))):
                        raise TypeError("log_level must be a str or LogLevel")
                    
                # key and value are excepted
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Invalid config key: {key}")
            
    @classmethod
    def set_logging(cls, log_level:LogLevel, debug:bool) -> None:
        cls.set(log_level=log_level)
        cls.set(debug=debug)
