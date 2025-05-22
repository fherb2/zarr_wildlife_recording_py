"""Package wide logging system based on Python logging module."""

import logging
import logging.handlers
import pathlib
import sys
from typing import Dict, Optional
from .types import LogLevel


class LoggingManager:
    """Manages the logging configuration for the entire package"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _file_handler: Optional[logging.Handler] = None
    _console_handler: Optional[logging.Handler] = None
    _current_log_level: Optional[LogLevel] = None
    _current_log_filepath: Optional[pathlib.Path] = None
    _initialized = False
    
    @classmethod
    def _get_log_level_value(cls, log_level: LogLevel) -> int:
        """Convert LogLevel enum to logging level integer"""
        level_mapping = {
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.NOTSET: logging.NOTSET
        }
        return level_mapping[log_level]
    
    @classmethod
    def _create_formatter(cls) -> logging.Formatter:
        """Create a consistent log formatter"""
        return logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    @classmethod
    def _setup_file_handler(cls, log_filepath: pathlib.Path) -> Optional[logging.Handler]:
        """Setup file handler for logging"""
        try:
            # Ensure directory exists
            log_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_filepath,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(cls._create_formatter())
            return file_handler
            
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not setup file logging to {log_filepath}: {e}", file=sys.stderr)
            return None
    
    @classmethod
    def _setup_console_handler(cls) -> logging.Handler:
        """Setup console handler for logging"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(cls._create_formatter())
        return console_handler
    
    @classmethod
    def _validate_log_filepath(cls, log_filepath: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
        """Validate and normalize log filepath"""
        if log_filepath is None:
            return None
        
        try:
            # Convert to Path if string
            if isinstance(log_filepath, str):
                log_filepath = pathlib.Path(log_filepath)
            
            # If it's a directory, add default filename
            if log_filepath.is_dir():
                log_filepath = log_filepath / "zarrwlr.log"
            
            # If parent directory doesn't exist, try to create it
            if not log_filepath.parent.exists():
                log_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if parent directory is valid
            if not log_filepath.parent.is_dir():
                return None
            
            return log_filepath
            
        except (OSError, ValueError):
            return None
    
    @classmethod
    def configure(cls, force_reconfigure: bool = False):
        """Configure the logging system based on current config"""
        from .config import Config  # Import here to avoid circular imports
        
        # Check if reconfiguration is needed
        if (not force_reconfigure and 
            cls._initialized and 
            cls._current_log_level == Config.log_level and 
            cls._current_log_filepath == Config.log_filepath):
            return
        
        # Remove existing handlers from all loggers
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        
        # Close existing handlers
        if cls._file_handler:
            cls._file_handler.close()
            cls._file_handler = None
        
        if cls._console_handler:
            cls._console_handler.close()
            cls._console_handler = None
        
        # Update current settings
        cls._current_log_level = Config.log_level
        cls._current_log_filepath = Config.log_filepath
        
        # If logging is disabled, disable all loggers
        if Config.log_level == LogLevel.NOTSET:
            for logger in cls._loggers.values():
                logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
            cls._initialized = True
            return
        
        # Get logging level
        log_level = cls._get_log_level_value(Config.log_level)
        
        # Setup file handler if filepath is provided and valid
        validated_filepath = cls._validate_log_filepath(Config.log_filepath)
        if validated_filepath:
            cls._file_handler = cls._setup_file_handler(validated_filepath)
            if cls._file_handler:
                cls._file_handler.setLevel(log_level)
        
        # Setup console handler
        cls._console_handler = cls._setup_console_handler()
        cls._console_handler.setLevel(log_level)
        
        # Configure all existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(log_level)
            
            # Add handlers
            if cls._file_handler:
                logger.addHandler(cls._file_handler)
            logger.addHandler(cls._console_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
        
        cls._initialized = True
    
    @classmethod
    def reconfigure(cls):
        """Reconfigure the logging system (called when config changes)"""
        cls.configure(force_reconfigure=True)
    
    @classmethod
    def get_logger(cls, module_name: str) -> logging.Logger:
        """Get a logger for a specific module"""
        if module_name not in cls._loggers:
            # Create new logger
            logger = logging.getLogger(f"zarrwlr.{module_name}")
            cls._loggers[module_name] = logger
            
            # Configure if not already done
            if not cls._initialized:
                cls.configure()
            
            # Apply current configuration to new logger
            if cls._current_log_level != LogLevel.NOTSET:
                log_level = cls._get_log_level_value(cls._current_log_level)
                logger.setLevel(log_level)
                
                if cls._file_handler:
                    logger.addHandler(cls._file_handler)
                if cls._console_handler:
                    logger.addHandler(cls._console_handler)
                
                logger.propagate = False
            else:
                logger.setLevel(logging.CRITICAL + 1)  # Disable logging
        
        return cls._loggers[module_name]


def get_module_logger(module_file: str) -> logging.Logger:
    """
    Convenience function to get a logger for a module.
    
    Usage in any module:
        from .logsetup import get_module_logger
        logger = get_module_logger(__file__)
    
    Args:
        module_file: Usually __file__ from the calling module
        
    Returns:
        Logger instance for the module
    """
    module_name = pathlib.Path(module_file).stem
    return LoggingManager.get_logger(module_name)
