"""Advanced logging system based on loguru with performance optimization and flexible configuration.

This module provides a sophisticated logging system built on top of loguru, designed for
Python packages that require high-performance logging with flexible configuration options.
The system offers several key features that make it suitable for production applications:

Key Features
------------
- **Performance Optimization**: Disabled log levels use null methods to eliminate runtime overhead
- **Module-specific Log Levels**: Different modules can have different log levels while respecting global limits
- **Multiple Output Destinations**: Console, file, and network logging with independent configurations
- **Terminal Line Wrapping**: Intelligent line wrapping for console output with proper indentation
- **Dynamic Reconfiguration**: Log settings can be changed at runtime without restarting the application
- **Loguru Integration**: Leverages loguru's powerful features while adding custom functionality

Architecture Overview
--------------------
The system consists of three main components:

1. **OptimizedLogger**: A wrapper around loguru that implements null methods for disabled log levels,
   providing zero-overhead logging calls when log levels are disabled.

2. **LoggingManager**: A singleton-style class that manages the global logging configuration,
   handles multiple output destinations, and coordinates logger instances.

3. **Module Logger Factory**: The `get_module_logger()` function that creates optimized logger
   instances for individual modules with proper context binding.

Log Level Management
-------------------
The system implements a hierarchical log level system:
- Global log level acts as an upper bound (most restrictive)
- Module-specific levels can be more restrictive than global but not less
- NOTSET level completely disables logging for performance
- Level changes trigger automatic reconfiguration of all loggers

Output Destinations
------------------
Multiple output destinations are supported simultaneously:

- **Console**: Colored output with optional line wrapping and intelligent indentation
- **File**: Plain text logging with rotation, retention, and compression
- **Network**: TCP/UDP socket logging for centralized log collection

Performance Considerations
-------------------------
The system is designed for high-performance applications:
- Disabled log levels use null methods (no string formatting or function calls)
- Module context is bound once per logger, not per log call
- Handler configuration is cached and only updated when necessary
- Loguru's efficient formatting and filtering mechanisms are preserved

Usage Examples
--------------
Basic module logging:
    ```python
    from .logsetup import get_module_logger
    logger = get_module_logger(__file__)
    logger.info("This is an info message")
    ```

Runtime configuration changes:
    ```python
    from .config import Config
    from .packagetypes import LogLevel
    
    # Change global log level
    Config.set(log_level=LogLevel.WARNING)
    
    # Set module-specific level
    Config.set_module_log_level("my_module", LogLevel.DEBUG)
    
    # Enable file logging with line wrapping
    Config.set(
        log_filepath=pathlib.Path("app.log"),
        terminal_log_max_line_length=120
    )
    ```

Integration with mkdocs
----------------------
This module is documented using numpy-style docstrings compatible with mkdocs
and sphinx documentation generators. Each public class and method includes
comprehensive documentation with parameters, return values, and usage examples.
"""

import pathlib
import sys
import socket
import textwrap
from typing import Dict, Optional, Any, Callable
from loguru import logger as loguru_logger
from .packagetypes import LogLevel
from .config import NetworkLogConfig


class OptimizedLogger:
    """High-performance logger wrapper with null methods for disabled levels.
    
    This class wraps loguru's logger functionality and provides performance optimization
    by replacing disabled log level methods with null operations. This eliminates the
    overhead of method calls, string formatting, and condition checking for log levels
    that are not active.
    
    The logger automatically binds module context to all log records and supports
    dynamic reconfiguration when log levels change at runtime.
    
    Parameters
    ----------
    module_name : str
        Name of the module this logger represents, used for context binding
    effective_level : LogLevel
        The effective log level for this logger instance
        
    Attributes
    ----------
    module_name : str
        The module name bound to this logger
    effective_level : LogLevel
        Current effective log level for this logger
    """
    
    def __init__(self, module_name: str, effective_level: LogLevel):
        """Initialize the optimized logger with module context and effective level.
        
        Sets up the logger instance with the specified module name and configures
        all logging methods based on the effective log level. Disabled levels are
        replaced with null methods for optimal performance.
        
        Parameters
        ----------
        module_name : str
            Name of the module this logger represents
        effective_level : LogLevel
            The effective log level that determines which methods are active
        """
        self.module_name = module_name
        self.effective_level = effective_level
        self._setup_methods()
    
    def _null_method(self, *args, **kwargs):
        """No-operation method for performance optimization of disabled log levels.
        
        This method does absolutely nothing and is used to replace logging methods
        for disabled log levels. This approach eliminates the overhead of method
        calls, argument processing, and conditional checks when logging is disabled.
        
        Parameters
        ----------
        *args : tuple
            Ignored positional arguments
        **kwargs : dict
            Ignored keyword arguments
        """
        pass
    
    def _get_level_priority(self, level: LogLevel) -> int:
        """Convert log level enum to numeric priority for comparison.
        
        Assigns numeric priorities to log levels where higher numbers represent
        more restrictive (higher priority) levels. This enables easy comparison
        of log levels to determine if a particular level should be logged.
        
        Parameters
        ----------
        level : LogLevel
            The log level to convert to numeric priority
            
        Returns
        -------
        int
            Numeric priority where higher values are more restrictive
        """
        priority_mapping = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1, 
            LogLevel.INFO: 2,
            LogLevel.SUCCESS: 3,
            LogLevel.WARNING: 4,
            LogLevel.ERROR: 5,
            LogLevel.CRITICAL: 6,
            LogLevel.NOTSET: 999
        }
        return priority_mapping.get(level, 999)
    
    def _is_level_enabled(self, level: LogLevel) -> bool:
        """Determine if a specific log level is enabled for this logger.
        
        Compares the requested log level against the effective level to determine
        if log messages at the specified level should be processed or ignored.
        NOTSET level disables all logging.
        
        Parameters
        ----------
        level : LogLevel
            The log level to check for enablement
            
        Returns
        -------
        bool
            True if the level is enabled and should be logged, False otherwise
        """
        if self.effective_level == LogLevel.NOTSET:
            return False
        
        effective_priority = self._get_level_priority(self.effective_level)
        level_priority = self._get_level_priority(level)
        
        return level_priority >= effective_priority
    
    def _setup_methods(self):
        """Configure all logging methods based on the current effective level.
        
        Dynamically assigns either functional logging methods or null methods to
        each log level (trace, debug, info, success, warning, error, critical)
        based on whether that level is enabled. This setup is performed once
        per level change to optimize runtime performance.
        """
        level_methods = {
            'trace': LogLevel.TRACE,
            'debug': LogLevel.DEBUG,
            'info': LogLevel.INFO,
            'success': LogLevel.SUCCESS,
            'warning': LogLevel.WARNING,
            'error': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL
        }
        
        for method_name, level in level_methods.items():
            if self._is_level_enabled(level):
                # Create bound method that includes module context
                method = self._create_log_method(method_name)
                setattr(self, method_name, method)
            else:
                # Use null method for disabled levels
                setattr(self, method_name, self._null_method)
    
    def _create_log_method(self, level_name: str) -> Callable:
        """Create a bound logging method for a specific log level.
        
        Generates a logging method that automatically binds the module context
        and forwards calls to the appropriate loguru method. The method includes
        proper call depth handling to ensure accurate source code location
        information in log records.
        
        Parameters
        ----------
        level_name : str
            Name of the log level method to create (e.g., 'info', 'debug')
            
        Returns
        -------
        Callable
            A bound method that can be called with message and arguments
        """
        def log_method(message, *args, **kwargs):
            # Add module context to the log record
            extra_context = kwargs.pop('extra', {})
            extra_context['module'] = self.module_name

            # Get the actual loguru method and use bind() to set extra context properly
            loguru_method = getattr(loguru_logger.bind(module=self.module_name).opt(depth=1), level_name)
            return loguru_method(message, *args, **kwargs)
        
        return log_method
    
    def exception(self, message, *args, **kwargs):
        """Log an exception with full traceback information.
        
        Special logging method for exceptions that includes the full stack trace
        and exception details. Only active when ERROR level or lower is enabled.
        Uses the same module binding and performance optimization as other methods.
        
        Parameters
        ----------
        message : str
            Description message for the exception
        *args : tuple
            Additional positional arguments for message formatting
        **kwargs : dict
            Additional keyword arguments for loguru
        """
        if self._is_level_enabled(LogLevel.ERROR):
            return loguru_logger.bind(module=self.module_name).opt(depth=1).exception(message, *args, **kwargs)
        else:
            self._null_method(message, *args, **kwargs)
    
    def update_level(self, new_effective_level: LogLevel):
        """Update the effective log level and reconfigure all methods.
        
        Changes the effective log level for this logger instance and rebuilds
        all logging methods accordingly. This allows for dynamic reconfiguration
        of log levels at runtime without creating new logger instances.
        
        Parameters
        ----------
        new_effective_level : LogLevel
            The new effective log level to apply to this logger
        """
        self.effective_level = new_effective_level
        self._setup_methods()


class LoggingManager:
    """Centralized manager for the entire logging system configuration and state.
    
    This class manages the global logging configuration, coordinates multiple output
    destinations (console, file, network), and maintains a registry of all logger
    instances. It handles dynamic reconfiguration and ensures consistency across
    all loggers when settings change.
    
    The manager implements a singleton-like pattern where all operations are
    class-level, ensuring a single point of configuration for the entire system.
    
    Attributes
    ----------
    _loggers : Dict[str, OptimizedLogger]
        Registry of all active logger instances by module name
    _handler_ids : Dict[str, int]
        Mapping of handler names to their loguru handler IDs
    _current_config : Dict[str, Any]
        Cached copy of current configuration to detect changes
    _initialized : bool
        Flag indicating whether the logging system has been initialized
    """
    
    _loggers: Dict[str, OptimizedLogger] = {}
    _handler_ids: Dict[str, int] = {}  # Track loguru handler IDs
    _current_config: Dict[str, Any] = {}
    _initialized = False
    
    @classmethod
    def _loguru_level_name(cls, log_level: LogLevel) -> str:
        """Convert custom LogLevel enum to loguru-compatible level name.
        
        Maps the package's LogLevel enumeration to the corresponding string
        names that loguru expects. Handles the special case of NOTSET by
        mapping it to CRITICAL level for effective disabling.
        
        Parameters
        ----------
        log_level : LogLevel
            The custom log level to convert
            
        Returns
        -------
        str
            Loguru-compatible level name string
        """
        level_mapping = {
            LogLevel.CRITICAL: "CRITICAL",
            LogLevel.ERROR: "ERROR", 
            LogLevel.WARNING: "WARNING",
            LogLevel.SUCCESS: "SUCCESS",
            LogLevel.INFO: "INFO",
            LogLevel.DEBUG: "DEBUG",
            LogLevel.TRACE: "TRACE",
            LogLevel.NOTSET: "CRITICAL"  # Effectively disable
        }
        return level_mapping[log_level]
    
    @classmethod
    def _escape_braces_filter(cls, record):
        """Escaped geschweifte Klammern in Log-Messages um Format-Konflikte zu vermeiden."""
        message = record.get('message')
        if isinstance(message, str) and '{' in message:
            # Escapen um loguru Format-Konflikte zu vermeiden
            record['message'] = message.replace('{', '{{').replace('}', '}}')
        return True

    @classmethod
    def _create_console_format(cls) -> str:
        """Create format string for console output with colors and symbols"""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]: <15}</cyan> | "
            "<level>{message}</level>"
        )
    
    @classmethod
    def _wrap_console_message(cls, record) -> str:
        """Apply intelligent line wrapping to console messages based on configuration.
        
        Wraps long log messages for console output while preserving single-line
        format for file and network outputs. Calculates proper indentation to
        align continuation lines with the message content, not the timestamp.
        
        Parameters
        ----------
        record : loguru.Record
            The log record containing the message to potentially wrap
            
        Returns
        -------
        str
            The original message or wrapped message with proper indentation
        """
        from .config import Config
        
        if Config.terminal_log_max_line_length is None:
            # No wrapping, return original message
            return record["message"]
        
        # Calculate the prefix length (timestamp + level + module + separators)
        # Example: "2025-05-24 17:17:54.210 | INFO     | example_module1 | "
        prefix_length = (
            23 +  # timestamp: "YYYY-MM-DD HH:mm:ss.SSS"
            3 +   # " | "
            8 +   # level padded to 8 chars
            3 +   # " | " 
            15 +  # module name padded to 15 chars
            3     # " | "
        )  # Total: 55 characters
        
        max_message_length = Config.terminal_log_max_line_length - prefix_length
        
        if len(record["message"]) <= max_message_length:
            return record["message"]
        
        # Wrap the message
        indent = " " * prefix_length
        wrapped_lines = textwrap.wrap(
            record["message"], 
            width=max_message_length,
            subsequent_indent=""
        )
        
        if len(wrapped_lines) <= 1:
            return record["message"]
        
        # Join with newline and indent for continuation lines
        return wrapped_lines[0] + "\n" + "\n".join(indent + line for line in wrapped_lines[1:])
    
    @classmethod
    def _console_format_function(cls, record):
        """Custom format function for console output with line wrapping support.
        
        Applies the console-specific formatting including colors, proper spacing,
        and intelligent line wrapping. Returns the formatted message with proper
        newline handling for multi-line wrapped content.
        
        Parameters
        ----------
        record : loguru.Record
            The log record to format for console display
            
        Returns
        -------
        str
            Formatted string ready for console output with colors and wrapping
        """
        # Get the wrapped message
        wrapped_message = cls._wrap_console_message(record)
        
        # Apply the standard format with wrapped message
        formatted = (
            f"<green>{record['time']:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            f"<level>{record['level'].name: <8}</level> | "
            f"<cyan>{record['extra'].get('module', 'unknown'): <15}</cyan> | "
            f"<level>{wrapped_message}</level>\n"
        )
        
        return formatted
    
    @classmethod
    def _create_file_format(cls) -> str:
        """Create format string for file output without colors or special formatting.
        
        Generates a clean, parseable format string for file logging that excludes
        color codes and uses consistent spacing. This format is designed to be
        easily parsed by log analysis tools and maintains single-line entries.
        
        Returns
        -------
        str
            Format string suitable for file output
        """
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[module]: <15} | "
            "{message}"
        )
    
    @classmethod
    def _create_network_format(cls) -> str:
        """Create format string for network transmission to remote log collectors.
        
        Generates a compact, structured format suitable for network transmission
        to centralized logging systems. Excludes colors and minimizes bandwidth
        usage while maintaining essential log information.
        
        Returns
        -------
        str
            Format string optimized for network transmission
        """
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level} | "
            "{extra[module]} | "
            "{message}"
        )
    
    @classmethod
    def _setup_console_handler(cls, log_level: LogLevel):
        """Configure and register the console output handler.
        
        Sets up loguru handler for console output with colored formatting,
        custom line wrapping, and appropriate log level filtering. Uses
        a custom sink function to properly handle newlines in wrapped messages.
        
        Parameters
        ----------
        log_level : LogLevel
            Minimum log level for console output
        """
        def console_sink(message):
            """Custom console sink that properly handles newlines."""
            import sys
            print(message, end='', file=sys.stdout, flush=True)
        
        handler_id = loguru_logger.add(
                console_sink,
                format=cls._console_format_function,
                level=cls._loguru_level_name(log_level),
                filter=cls._escape_braces_filter,  # Filter hinzugefügt
                colorize=True
            )
        cls._handler_ids["console"] = handler_id
    
    @classmethod
    def _setup_file_handler(cls, log_filepath: pathlib.Path, log_level: LogLevel):
        """Configure and register the file output handler with rotation and compression.
        
        Sets up loguru handler for file output with automatic rotation, retention
        policies, and compression. Creates necessary directories and handles
        permission errors gracefully with fallback error reporting.
        
        Parameters
        ----------
        log_filepath : pathlib.Path
            Path where log files should be written
        log_level : LogLevel
            Minimum log level for file output
        """
        try:
            # Ensure directory exists
            log_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            handler_id = loguru_logger.add(
                str(log_filepath),
                format=cls._create_file_format(),
                level=cls._loguru_level_name(log_level),
                filter=cls._escape_braces_filter,  # Filter hinzugefügt
                rotation="10 MB",
                retention="1 week",
                compression="zip",
                encoding="utf-8"
            )
            cls._handler_ids["file"] = handler_id
            
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not setup file logging to {log_filepath}: {e}", file=sys.stderr)
    
    @classmethod
    def _setup_network_handler(cls, network_config: NetworkLogConfig, log_level: LogLevel):
        """Configure and register network output handler for remote log collection.
        
        Sets up loguru handler for network output supporting both TCP and UDP
        protocols. Implements connection timeouts and graceful error handling
        with fallback to stderr for network failures.
        
        Parameters
        ----------
        network_config : NetworkLogConfig
            Network configuration including host, port, and protocol
        log_level : LogLevel
            Minimum log level for network output
        """
        try:
            if network_config.protocol == 'TCP':
                # TCP socket handler
                def tcp_sink(message):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.settimeout(5.0)  # 5 second timeout
                            sock.connect((network_config.host, network_config.port))
                            sock.send(message.encode('utf-8') + b'\n')
                    except Exception as e:
                        # Fallback to stderr if network fails
                        print(f"Network logging failed: {e}", file=sys.stderr)
                
                handler_id = loguru_logger.add(
                    tcp_sink,
                    format=cls._create_network_format(),
                    level=cls._loguru_level_name(log_level),
                    filter=cls._escape_braces_filter  # Filter hinzugefügt
                )
                
            elif network_config.protocol == 'UDP':
                # UDP socket handler
                def udp_sink(message):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                            sock.settimeout(5.0)
                            sock.sendto(message.encode('utf-8'), (network_config.host, network_config.port))
                    except Exception as e:
                        print(f"Network logging failed: {e}", file=sys.stderr)
                
                handler_id = loguru_logger.add(
                    udp_sink,
                    format=cls._create_network_format(),
                    level=cls._loguru_level_name(log_level),
                    filter=cls._escape_braces_filter  # Filter hinzugefügt
                )
            else:
                raise ValueError(f"Unsupported network protocol: {network_config.protocol}")
            
            cls._handler_ids["network"] = handler_id
            
        except Exception as e:
            print(f"Warning: Could not setup network logging: {e}", file=sys.stderr)
    
    @classmethod
    def _remove_handlers(cls):
        """Remove all currently registered loguru handlers.
        
        Safely removes all handlers that have been registered by this logging
        system, cleaning up resources and preparing for reconfiguration.
        Handles cases where handlers may have already been removed externally.
        """
        for handler_name, handler_id in cls._handler_ids.items():
            try:
                loguru_logger.remove(handler_id)
            except ValueError:
                # Handler already removed
                pass
        cls._handler_ids.clear()
    
    @classmethod
    def configure(cls, force_reconfigure: bool = False):
        """Configure or reconfigure the entire logging system based on current settings.
        
        Performs complete setup of the logging system including handler registration,
        logger instance updates, and configuration caching. Detects configuration
        changes to avoid unnecessary reconfiguration overhead.
        
        Parameters
        ----------
        force_reconfigure : bool, optional
            Force reconfiguration even if settings appear unchanged, by default False
        """
        from .config import Config  # Import here to avoid circular imports
        
        # Create current config snapshot
        current_config = {
            'log_level': Config.log_level,
            'log_filepath': Config.log_filepath,
            'network_log_config': Config.network_log_config,
            'module_log_levels': Config.module_log_levels.copy(),
            'terminal_log_max_line_length': Config.terminal_log_max_line_length
        }
        
        # Check if reconfiguration is needed
        if not force_reconfigure and cls._initialized and cls._current_config == current_config:
            return
        
        # Remove existing handlers (but not the default one initially)
        if cls._initialized:
            cls._remove_handlers()
        else:
            # On first initialization, remove loguru's default handler
            loguru_logger.remove()
        
        # Store current config
        cls._current_config = current_config.copy()
        
        # If logging is completely disabled, set up a null handler
        if Config.log_level == LogLevel.NOTSET:
            # Add a handler that filters out everything
            handler_id = loguru_logger.add(
                sys.stdout,
                format="",
                level="CRITICAL",
                filter=lambda record: False  # Filter out everything
            )
            cls._handler_ids["null"] = handler_id
            
            # Update all existing logger instances
            for logger_instance in cls._loggers.values():
                logger_instance.update_level(LogLevel.NOTSET)
            cls._initialized = True
            return
        
        # Setup console handler (always active when not NOTSET)
        cls._setup_console_handler(Config.log_level)
        
        # Setup file handler if filepath is provided
        if Config.log_filepath:
            validated_filepath = cls._validate_log_filepath(Config.log_filepath)
            if validated_filepath:
                cls._setup_file_handler(validated_filepath, Config.log_level)
        
        # Setup network handler if config is provided
        if Config.network_log_config:
            cls._setup_network_handler(Config.network_log_config, Config.log_level)
        
        # Update all existing logger instances
        for module_name, logger_instance in cls._loggers.items():
            effective_level = Config.get_effective_log_level(module_name)
            logger_instance.update_level(effective_level)
        
        cls._initialized = True
    
    @classmethod
    def _validate_log_filepath(cls, log_filepath: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
        """Validate and normalize the log file path with directory creation.
        
        Ensures the log file path is valid, creates necessary directories,
        and handles various edge cases like directory-only paths and
        permission issues. Returns None if the path cannot be used.
        
        Parameters
        ----------
        log_filepath : Optional[pathlib.Path]
            The log file path to validate and normalize
            
        Returns
        -------
        Optional[pathlib.Path]
            Validated and normalized path, or None if invalid
        """
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
    def reconfigure(cls):
        """Trigger immediate reconfiguration of the logging system.
        
        Forces a complete reconfiguration regardless of detected changes.
        This method is typically called automatically when configuration
        changes are made, but can be called manually if needed.
        """
        cls.configure(force_reconfigure=True)
    
    @classmethod
    def get_logger(cls, module_name: str) -> OptimizedLogger:
        """Get or create an optimized logger instance for a specific module.
        
        Returns an OptimizedLogger instance configured with the current effective
        log level for the specified module. Handles both creation of new loggers
        and updating of existing ones when configuration changes occur.
        
        Parameters
        ----------
        module_name : str
            Name of the module requesting a logger instance
            
        Returns
        -------
        OptimizedLogger
            Configured logger instance for the specified module
        """
        # Configure if not already done
        if not cls._initialized:
            cls.configure()
        
        # Get effective level for this module
        from .config import Config
        effective_level = Config.get_effective_log_level(module_name)
        
        # Always create/update the logger with current effective level
        if module_name in cls._loggers:
            # Update existing logger
            cls._loggers[module_name].update_level(effective_level)
        else:
            # Create new logger instance
            logger_instance = OptimizedLogger(module_name, effective_level)
            cls._loggers[module_name] = logger_instance
        
        return cls._loggers[module_name]


def get_module_logger(module_file: str) -> OptimizedLogger:
    """Create an optimized logger instance for a module using its file path.
    
    Convenience function that extracts the module name from a file path
    (typically __file__) and returns a properly configured logger instance.
    This is the primary entry point for modules to obtain their loggers.
    
    The function automatically handles module name extraction, logger creation,
    and configuration updates, providing a simple interface for module-level
    logging setup.
    
    Parameters
    ----------
    module_file : str
        File path of the calling module, typically __file__
        
    Returns
    -------
    OptimizedLogger
        Configured logger instance for the calling module
        
    Examples
    --------
    Basic usage in a module:
        >>> from .logsetup import get_module_logger
        >>> logger = get_module_logger(__file__)
        >>> logger.info("Module initialized")
        
    The logger will automatically use the filename (without extension) as
    the module name for context binding in log records.
    """
    module_name = pathlib.Path(module_file).stem
    # Always return the current logger (will be updated if config changed)
    return LoggingManager.get_logger(module_name)