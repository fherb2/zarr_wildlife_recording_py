"""Package wide logging system based on Python logging module."""

import logging
import logging.handlers
import pathlib
import sys
import textwrap
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
    def _create_formatter(cls, for_console: bool = True) -> logging.Formatter:
        """Create formatter - colored for console, plain for files"""
        if for_console:
            return ColoredDotMillisecondFormatter(
                fmt='%(asctime)s\n%(levelname)s (%(name)s): %(message)s',
                use_colors=True,
                use_symbols=True,
                width=120
            )
        else:
            return DotMillisecondFormatter(
                fmt='%(asctime)s\n%(levelname)s (%(name)s): %(message)s',
                width=120
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
            file_handler.setFormatter(cls._create_formatter(for_console=False))
            return file_handler
            
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not setup file logging to {log_filepath}: {e}", file=sys.stderr)
            return None
    
    @classmethod
    def _setup_console_handler(cls) -> logging.Handler:
        """Setup console handler for logging"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(cls._create_formatter(for_console=True))
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
    
class DotMillisecondFormatter(logging.Formatter):
    INDENT_SIZE = 2
    DEFAULT_WIDTH = 120
    
    def __init__(self, *args, width=DEFAULT_WIDTH, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = ' ' * self.INDENT_SIZE
        self.width = width


    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, datefmt).replace(',', '.')
    
    def format(self, record):
        formatted = super().format(record)
        lines = formatted.split('\n')
        
        if len(lines) > 1:
            # Erste Zeile bleibt unverändert
            result_lines = [lines[0]]
            
            # Alle Message-Zeilen verarbeiten (ab Index 1)
            for line in lines[1:]:
                # Bereits vorhandene Einrückung entfernen falls vorhanden
                clean_line = line.lstrip()
                
                # Verfügbare Breite berechnen (Gesamtbreite minus Einrückung)
                available_width = self.width - self.INDENT_SIZE
                
                if len(clean_line) <= available_width:
                    # Zeile passt, einfach einrücken
                    result_lines.append(self.indent + clean_line)
                else:
                    # Zeile umbrechen mit textwrap
                    wrapped_lines = textwrap.wrap(
                        clean_line, 
                        width=available_width,
                        break_long_words=True,
                        break_on_hyphens=True
                    )
                    # Alle umgebrochenen Zeilen einrücken
                    for wrapped_line in wrapped_lines:
                        result_lines.append(self.indent + wrapped_line)
            
            formatted = '\n'.join(result_lines)
        
        return formatted



class ColoredDotMillisecondFormatter(logging.Formatter):
    """Enhanced formatter with color support and improved readability"""
    
    INDENT_SIZE = 2
    DEFAULT_WIDTH = 120
    
    # ANSI Color Codes
    class Colors:
        # Standard colors
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        
        # Bright colors
        BRIGHT_BLACK = '\033[90m'    # Gray
        BRIGHT_RED = '\033[91m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_WHITE = '\033[97m'
        
        # Text styles
        BOLD = '\033[1m'
        DIM = '\033[2m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'
        BLINK = '\033[5m'
        REVERSE = '\033[7m'
        STRIKETHROUGH = '\033[9m'
        
        # Reset
        RESET = '\033[0m'
        
        # Background colors
        BG_RED = '\033[41m'
        BG_GREEN = '\033[42m'
        BG_YELLOW = '\033[43m'
        BG_BLUE = '\033[44m'
    
    # Level-specific color mapping
    LEVEL_COLORS: Dict[str, str] = {
        'DEBUG': Colors.BRIGHT_BLACK,      # Gray
        'INFO': Colors.BRIGHT_BLUE,        # Bright Blue
        'WARNING': Colors.BRIGHT_YELLOW,   # Bright Yellow
        'ERROR': Colors.BRIGHT_RED,        # Bright Red
        'CRITICAL': Colors.BOLD + Colors.RED + Colors.BG_YELLOW,  # Bold Red on Yellow
    }
    
    # Unicode symbols for better visual distinction
    LEVEL_SYMBOLS: Dict[str, str] = {
        'DEBUG': '🔍',      # Magnifying glass
        'INFO': 'ℹ️ ',       # Information
        'WARNING': '⚠️ ',    # Warning sign
        'ERROR': '❌',      # Cross mark
        'CRITICAL': '🚨',   # Rotating light
    }
    
    def __init__(self, *args, width=DEFAULT_WIDTH, use_colors=None, use_symbols=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = ' ' * self.INDENT_SIZE
        self.width = width
        self.use_symbols = use_symbols
        
        # Auto-detect color support if not specified
        if use_colors is None:
            self.use_colors = self._should_use_colors()
        else:
            self.use_colors = use_colors
    
    def _should_use_colors(self) -> bool:
        """Auto-detect if colors should be used based on terminal capabilities"""
        # Check if output is going to a terminal
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check environment variables
        import os
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        
        # Common terminals that support colors
        if any(x in term for x in ['color', 'ansi', 'xterm', 'screen']):
            return True
        if colorterm in ['truecolor', '24bit']:
            return True
        
        # Windows Terminal and modern terminals
        if os.name == 'nt':
            # Windows 10+ supports ANSI colors
            try:
                import platform
                version = platform.version()
                if version and int(version.split('.')[0]) >= 10:
                    return True
            except:
                pass
        
        return False

    def formatTime(self, record, datefmt=None):
        """Format time with milliseconds using dot separator"""
        return super().formatTime(record, datefmt).replace(',', '.')
    
    def _colorize_level(self, levelname: str) -> str:
        """Apply color to level name"""
        if not self.use_colors:
            return levelname
        
        color = self.LEVEL_COLORS.get(levelname, '')
        symbol = self.LEVEL_SYMBOLS.get(levelname, '') if self.use_symbols else ''
        
        if color:
            return f"{color}{symbol}{levelname}{self.Colors.RESET}"
        return f"{symbol}{levelname}"
    
    # def _add_visual_separators(self, formatted: str) -> str:
    #     """Add visual separators for better readability"""
    #     if not self.use_colors:
    #         return formatted
        
    #     # Add subtle separator line for multi-line messages
    #     lines = formatted.split('\n')
    #     if len(lines) > 2:  # Only for messages with actual content lines
    #         # Add a subtle line after the header
    #         lines.insert(1, f"{self.Colors.DIM}{'─' * min(40, self.width // 3)}{self.Colors.RESET}")
        
    #     return '\n'.join(lines)
    
    def format(self, record):
        """Format the log record with colors and proper indentation"""
        # Store original levelname
        original_levelname = record.levelname
        
        # Apply colors to levelname
        record.levelname = self._colorize_level(original_levelname)
        
        # Get standard formatting
        formatted = super().format(record)
        
        # Restore original levelname (important for other handlers)
        record.levelname = original_levelname
        
        # Process multi-line formatting
        lines = formatted.split('\n')
        
        if len(lines) > 1:
            result_lines = [lines[0]]  # First line (timestamp + level)
            
            # Process message lines (starting from index 1)
            for line in lines[1:]:
                clean_line = line.lstrip()
                available_width = self.width - self.INDENT_SIZE
                
                if len(clean_line) <= available_width:
                    result_lines.append(self.indent + clean_line)
                else:
                    # Word wrap long lines
                    wrapped_lines = textwrap.wrap(
                        clean_line, 
                        width=available_width,
                        break_long_words=True,
                        break_on_hyphens=True
                    )
                    for wrapped_line in wrapped_lines:
                        result_lines.append(self.indent + wrapped_line)
            
            formatted = '\n'.join(result_lines)
        
        # Add visual separators for complex messages
        # formatted = self._add_visual_separators(formatted)
        
        return formatted


# Alternative: Simple color-only formatter (ohne Symbols)
class SimpleColorFormatter(ColoredDotMillisecondFormatter):
    """Simplified version with just colors, no symbols"""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_symbols'] = False
        super().__init__(*args, **kwargs)






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
