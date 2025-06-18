"""Advanced configuration management system with YAML import/export capabilities.

This module provides a sophisticated configuration management system designed for Python
packages that require flexible, runtime-configurable settings with persistent storage
capabilities. The system combines static class-based configuration for optimal IDE
support with dynamic YAML-based serialization for deployment and sharing.

Key Features
------------
- **Static Class Configuration**: All configuration parameters are defined as class
  variables, enabling full IDE IntelliSense support and type checking
- **Runtime Modification**: Configuration can be changed at runtime with automatic
  validation and system reconfiguration
- **YAML Serialization**: Export and import configuration to/from YAML files with
  comment preservation and human-readable formatting
- **Type Safety**: Comprehensive type validation for all configuration parameters
- **Immutable Protection**: Critical system parameters are protected from modification
- **Automatic Integration**: Changes trigger automatic reconfiguration of dependent
  systems (like logging)

Architecture Overview
--------------------
The configuration system is built around a single `Config` class that serves as both
a static configuration container and a dynamic configuration manager:

1. **Class Variables**: All configuration parameters are defined as typed class variables,
   providing excellent IDE support and static analysis capabilities.

2. **Validation System**: The `set()` method provides runtime validation using Python's
   type hints and custom validation logic for complex types.

3. **Serialization Engine**: Custom serialization handlers for complex objects enable
   round-trip YAML export/import while maintaining object integrity.

4. **Change Detection**: Automatic detection of configuration changes triggers
   reconfiguration of dependent systems without manual intervention.

Configuration Categories
-----------------------
The system distinguishes between different types of configuration parameters:

- **Configurable Parameters**: Listed in `_CONFIGURABLE_KEYS`, these can be modified
  at runtime and are included in YAML export/import operations.
- **Immutable Parameters**: System constants that cannot be changed after initial
  definition, protecting critical system behavior.
- **Internal Tracking**: Private variables used for system state management, excluded
  from serialization and user modification.

YAML Integration
---------------
The YAML integration preserves the human-readable nature of configuration files:

- **Comment Preservation**: Inline comments from the source code are automatically
  included in exported YAML files for documentation.
- **Type Conversion**: Complex Python objects are intelligently serialized to
  YAML-compatible structures and restored on import.
- **Validation**: All imported values pass through the same validation pipeline
  as runtime modifications.

Supported Complex Types
----------------------
The serialization system handles various complex types commonly used in configuration:

- **pathlib.Path**: Serialized as string paths, restored as Path objects
- **Enum Types**: Serialized as string values, validated on import
- **Custom Classes**: Objects with `__dict__` attributes are serialized as nested
  dictionaries and reconstructed on import
- **Collections**: Dictionaries and lists are handled natively by YAML

Usage Examples
--------------
Basic configuration modification:
    ```python
    from .config import Config
    from .types import LogLevel
    
    # Modify configuration at runtime
    Config.set(
        log_level=LogLevel.INFO,
        terminal_log_max_line_length=100
    )
    ```

YAML export and import:
    ```python
    import pathlib
    
    # Export current configuration
    Config.export_to_yaml(pathlib.Path("config.yaml"))
    
    # Import configuration from file
    Config.import_from_yaml(pathlib.Path("config.yaml"))
    ```

Module-specific configuration:
    ```python
    # Set different log levels for different modules
    Config.set_module_log_level("database", LogLevel.DEBUG)
    Config.set_module_log_level("api", LogLevel.WARNING)
    ```

Template Usage
--------------
This configuration system is designed as a template for Python packages. Developers
should extend the `Config` class by:

1. Adding new configuration parameters as typed class variables
2. Including new parameter names in `_CONFIGURABLE_KEYS`
3. Adding custom validation logic in the `set()` method if needed
4. The system will automatically handle YAML serialization for standard types

Integration with Development Tools
---------------------------------
The class-based approach ensures excellent integration with development tools:

- **IDE Support**: Full IntelliSense and autocomplete for all configuration parameters
- **Type Checking**: Static type checkers can validate configuration usage
- **Documentation**: Type hints and docstrings provide comprehensive API documentation
- **Refactoring**: IDE refactoring tools can safely rename and reorganize configuration
"""

from __future__ import annotations # This has to be the first line of code.
import pathlib
import inspect
import re
import yaml
from datetime import datetime
from typing import get_type_hints, Dict, Optional, Any
from .packagetypes import LogLevel


class Config:
    """Advanced configuration management system with YAML serialization support.
    
    Provides a centralized configuration system that combines static class-based
    parameter definition with dynamic runtime modification and persistent YAML
    storage. All configuration parameters are defined as class variables to
    ensure optimal IDE support and type safety.
    
    The class distinguishes between configurable parameters (modifiable at runtime),
    immutable parameters (system constants), and internal tracking variables
    (excluded from serialization).
    
    Attributes
    ----------
    log_level : LogLevel
        Global logging level that acts as upper bound for all loggers
    log_filepath : pathlib.Path or None
        File path for log output, None disables file logging
    network_log_config : NetworkLogConfig or None
        Network logging configuration, None disables network logging
    module_log_levels : Dict[str, LogLevel]
        Module-specific log level overrides
    terminal_log_max_line_length : int or None
        Maximum line length for terminal output, None disables wrapping
    your_changeable_parameter : int
        Example configurable parameter for template customization
    """
    
    # configurable values
    # -------------------
    _CONFIGURABLE_KEYS = [  "log_level",
                            "log_filepath", 
                            "network_log_config",
                            "module_log_levels",
                            "terminal_log_max_line_length",
                            "audio_import_batch_size",
                            "original_audio_chunk_size",
                            "original_audio_chunks_per_shard",
                            "aac_default_bitrate",
                            "aac_enable_pyav_native", 
                            "aac_fallback_to_ffmpeg",
                            "aac_frame_analysis_method",
                            "aac_index_chunk_size",
                            "aac_index_target_chunk_kb",
                            "aac_index_max_chunks",
                            "aac_index_min_chunk_frames",
                            "aac_max_worker_core_percent",
                            "aac_enable_parallel_analysis",
                            "aac_memory_limit_mb",
                            "aac_quality_preset"
                         ]
    
    log_level: LogLevel = LogLevel.ERROR    # Global logging level (acts as upper bound)
    log_filepath: pathlib.Path|None = "zarrwlr_test.log" #None  # set a logging path if you want to write logging outputs
                                            # eg. pathlibPath("./mylogfile.log")
    network_log_config: NetworkLogConfig|None = None  # Network logging configuration
    module_log_levels: Dict[str, LogLevel] = {}  # Module-specific log levels
    terminal_log_max_line_length: int|None = 120  # Maximum line length for terminal output (None = no wrapping)
    audio_import_batch_size: int = 20  # Files per subprocess for batch operations
    original_audio_chunk_size = 1024 * 1024  # 1MB
    original_audio_chunks_per_shard = 4      # 4 chunks per shard
    aac_default_bitrate: int = 160000              # Default AAC bitrate in bits/second (160 kbps)
    aac_enable_pyav_native: bool = True             # Use PyAV for native AAC processing (faster)
    aac_fallback_to_ffmpeg: bool = True             # Fall back to ffmpeg if PyAV fails
    aac_frame_analysis_method: str = "pyav"         # Frame analysis method: "pyav" or "manual"
    aac_index_chunk_size: int = 20000              # Frames per index chunk in Zarr
    aac_index_target_chunk_kb: int = 512          # Target chunk size in KB
    aac_index_max_chunks: int = 50                # Maximum number of chunks
    aac_index_min_chunk_frames: int = 5000        # Minimum frames per chunk
    aac_max_worker_core_percent: int = 80          # Percentage of CPU cores to use for AAC processing
    aac_enable_parallel_analysis: bool = True      # Enable parallel frame analysis
    aac_memory_limit_mb: int = 500                 # Memory limit for AAC processing in MB
    aac_quality_preset: str = "balanced"           # Quality preset: "fast", "balanced", "quality"

    
    # immutable keys  (can not be changed during runtime; only changeable at this position)
    # --------------  
    version = (1,0)                       # example of a package version tuple; (Major, Minor, Patch)
    original_audio_group_version = (1,0)  # tuple of (Major, Minor, Patch) ; Patch is optional
    original_audio_group_magic_id = "zarrwlr_original_audio_group_fherb2_23091969052025"
    original_audio_data_array_version = (1,0)  # tuple of (Major, Minor, Patch) ; Patch is optional
    
    # Internal tracking for logging system
    # ------------------------------------
    # these are not configuration values
    _logging_configured = False
    _last_log_level = None
    _last_log_filepath = None
    _last_network_config = None
    _last_module_log_levels = None
    _last_terminal_max_line_length = None
    
    @classmethod
    def set_module_log_level(cls, module_name: str, log_level: LogLevel):
        """Set log level for a specific module with automatic system reconfiguration.
        
        Configures a module-specific log level that overrides the global setting
        for the specified module. The module level cannot be less restrictive than
        the global level. Automatically triggers logging system reconfiguration.
        
        Parameters
        ----------
        module_name : str
            Name of the module to configure
        log_level : LogLevel
            Log level to apply to the specified module
        """
        cls.module_log_levels[module_name] = log_level
        # Trigger reconfiguration
        from .logsetup import LoggingManager
        LoggingManager.reconfigure()
    
    @classmethod
    def get_effective_log_level(cls, module_name: str) -> LogLevel:
        """Calculate the effective log level for a module considering global limits.
        
        Determines the actual log level that should be used for a specific module,
        taking into account both module-specific settings and the global log level
        limit. The global level acts as an upper bound (more restrictive limit).
        
        Parameters
        ----------
        module_name : str
            Name of the module to query
            
        Returns
        -------
        LogLevel
            The effective log level for the specified module
        """
        module_level = cls.module_log_levels.get(module_name, cls.log_level)
        
        # Global level acts as upper bound (lower priority levels are blocked)
        global_priority = cls._get_log_level_priority(cls.log_level)
        module_priority = cls._get_log_level_priority(module_level)
        
        # Return the more restrictive level (higher priority number)
        if global_priority > module_priority:
            return cls.log_level
        return module_level
    
    @classmethod
    def _get_log_level_priority(cls, log_level: LogLevel) -> int:
        """Convert log level to numeric priority for comparison operations.
        
        Assigns numeric priorities to log levels where higher numbers represent
        more restrictive levels. This enables easy comparison and determination
        of effective log levels when combining global and module-specific settings.
        
        Parameters
        ----------
        log_level : LogLevel
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
            LogLevel.NOTSET: 999  # Most restrictive
        }
        return priority_mapping.get(log_level, 999)
    
    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        """Convert complex Python objects to YAML-serializable representations.
        
        Handles the conversion of complex Python objects that cannot be directly
        serialized to YAML into appropriate dictionary or primitive representations.
        This method provides a generalized approach for handling custom classes
        and complex types in configuration serialization.
        
        Parameters
        ----------
        value : Any
            The value to serialize for YAML export
            
        Returns
        -------
        Any
            YAML-serializable representation of the input value
        """
        if value is None:
            return None
        elif isinstance(value, pathlib.Path):
            return str(value)
        elif isinstance(value, LogLevel):
            return value.value
        elif hasattr(value, '__dict__'):
            # Handle custom classes by converting to dict
            result = {'_class_name': value.__class__.__name__}
            result.update(value.__dict__)
            return result
        elif isinstance(value, dict):
            # Recursively serialize dictionary values
            return {k: cls._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively serialize sequence values
            return [cls._serialize_value(item) for item in value]
        else:
            return value
    
    @classmethod
    def _deserialize_value(cls, value: Any, expected_type: type) -> Any:
        """Convert YAML-loaded values back to appropriate Python objects.
        
        Handles the restoration of complex Python objects from their YAML
        representations, including custom classes and special types like
        pathlib.Path and enum values. Provides type validation and error
        handling for invalid configurations.
        
        Parameters
        ----------
        value : Any
            The YAML-loaded value to deserialize
        expected_type : type
            The expected Python type for this value
            
        Returns
        -------
        Any
            Properly typed Python object
            
        Raises
        ------
        ValueError
            If the value cannot be converted to the expected type
        TypeError
            If the value type is incompatible with expectations
        """
        if value is None:
            return None
        
        # Handle pathlib.Path
        if expected_type == pathlib.Path or expected_type == Optional[pathlib.Path]:
            if isinstance(value, str):
                return pathlib.Path(value)
            return value
        
        # Handle LogLevel enum
        if expected_type == LogLevel:
            if isinstance(value, str):
                try:
                    return LogLevel(value)
                except ValueError:
                    raise ValueError(f"Invalid LogLevel value: {value}")
            return value
        
        # Handle custom classes (like NetworkLogConfig)
        if isinstance(value, dict) and '_class_name' in value:
            class_name = value.pop('_class_name')
            if class_name == 'NetworkLogConfig':
                return NetworkLogConfig(**{k: v for k, v in value.items() if k != '_class_name'})
            else:
                raise ValueError(f"Unknown class in YAML: {class_name}")
        
        # Handle dictionaries with nested deserialization
        if isinstance(value, dict):
            # For module_log_levels, deserialize the LogLevel values
            if expected_type == Dict[str, LogLevel]:
                return {k: cls._deserialize_value(v, LogLevel) for k, v in value.items()}
            return value
        
        return value
    
    @classmethod
    def _extract_inline_comments(cls) -> Dict[str, str]:
        """Extract inline comments from the Config class source code.
        
        Parses the source code of the Config class to extract inline comments
        that appear after variable assignments. These comments are preserved
        in YAML export to maintain documentation and human readability.
        
        Returns
        -------
        Dict[str, str]
            Mapping of variable names to their inline comments
        """
        try:
            source = inspect.getsource(cls)
            comments = {}
            
            # Look for lines with variable assignments and inline comments
            pattern = r'^\s*(\w+)\s*[:=].*?#\s*(.+)$'
            
            for line in source.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    var_name, comment = match.groups()
                    if var_name in cls._CONFIGURABLE_KEYS:
                        comments[var_name] = comment.strip()
            
            return comments
        except Exception:
            # If source extraction fails, return empty dict
            return {}
    
    @classmethod
    def export_to_yaml(cls, filepath: pathlib.Path, include_metadata: bool = True):
        """Export current configuration to a YAML file with comments and metadata.
        
        Creates a human-readable YAML file containing all configurable parameters
        with their current values, inline comments from the source code, and
        optional metadata about the export operation. Only parameters listed in
        _CONFIGURABLE_KEYS are included in the export.
        
        Parameters
        ----------
        filepath : pathlib.Path
            Path where the YAML file should be written
        include_metadata : bool, optional
            Whether to include export metadata in the file, by default True
            
        Raises
        ------
        OSError
            If the file cannot be written due to permissions or path issues
        """
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current values for configurable keys
        config_data = {}
        comments = cls._extract_inline_comments()
        
        for key in cls._CONFIGURABLE_KEYS:
            if hasattr(cls, key):
                value = getattr(cls, key)
                config_data[key] = cls._serialize_value(value)
        
        # Create YAML content with comments
        yaml_lines = []
        
        if include_metadata:
            yaml_lines.append(f"# Configuration exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            yaml_lines.append("# This file was automatically generated from the Config class")
            yaml_lines.append("")
        
        # Convert to YAML and add comments
        yaml_content = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
        
        for line in yaml_content.split('\n'):
            if line.strip():
                # Check if this line contains a key we have a comment for
                key_match = re.match(r'^(\w+):', line)
                if key_match:
                    key = key_match.group(1)
                    if key in comments:
                        line = f"{line}  # {comments[key]}"
                yaml_lines.append(line)
            else:
                yaml_lines.append(line)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yaml_lines))
    
    @classmethod
    def import_from_yaml(cls, filepath: pathlib.Path, validate: bool = True):
        """Import configuration from a YAML file with validation and type conversion.
        
        Loads configuration parameters from a YAML file, converting values to
        appropriate Python types and optionally validating them through the
        standard configuration validation pipeline. Only parameters that exist
        in _CONFIGURABLE_KEYS are imported.
        
        Parameters
        ----------
        filepath : pathlib.Path
            Path to the YAML file to import
        validate : bool, optional
            Whether to use the validation pipeline during import, by default True
            
        Raises
        ------
        FileNotFoundError
            If the specified YAML file does not exist
        yaml.YAMLError
            If the YAML file is malformed or cannot be parsed
        ValueError
            If validation fails for any configuration parameter
        TypeError
            If type conversion fails for any parameter
        """
        # Load YAML file
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        if not yaml_data:
            return
        
        # Get type hints for validation
        type_hints = get_type_hints(cls)
        
        # Process each configurable key
        config_updates = {}
        for key, value in yaml_data.items():
            if key in cls._CONFIGURABLE_KEYS:
                expected_type = type_hints.get(key)
                if expected_type:
                    try:
                        converted_value = cls._deserialize_value(value, expected_type)
                        config_updates[key] = converted_value
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Failed to convert '{key}' from YAML: {e}")
                else:
                    config_updates[key] = value
        
        # Apply configuration updates
        if validate:
            cls.set(**config_updates)
        else:
            # Direct assignment without validation (for internal use)
            for key, value in config_updates.items():
                setattr(cls, key, value)
    
    @classmethod
    def set(cls, **kwargs):
        """Set multiple configuration parameters with validation and system integration.
        
        Updates one or more configuration parameters with comprehensive validation,
        type checking, and automatic triggering of dependent system reconfiguration.
        Only parameters listed in _CONFIGURABLE_KEYS can be modified.
        
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to update as keyword arguments
            
        Raises
        ------
        AttributeError
            If a parameter name is invalid or immutable
        TypeError
            If a parameter value has an incorrect type
        ValueError
            If a parameter value is invalid (e.g., negative numbers where positive required)
            
        Examples
        --------
        Update multiple configuration parameters:
            >>> Config.set(
            ...     log_level=LogLevel.INFO,
            ...     terminal_log_max_line_length=100,
            ...     log_filepath=pathlib.Path("app.log")
            ... )
        """
        from .logsetup import LoggingManager  # Import here to avoid circular imports
        
        old_log_level = cls.log_level
        old_log_filepath = cls.log_filepath
        old_network_config = cls.network_log_config
        old_module_log_levels = cls.module_log_levels.copy()
        old_terminal_max_line_length = cls.terminal_log_max_line_length
        
        for key, value in kwargs.items():
            if hasattr(cls, key):
                if key in cls._CONFIGURABLE_KEYS:
                    # Special handling for certain types
                    if key == "network_log_config":
                        if value is not None and not isinstance(value, NetworkLogConfig):
                            raise TypeError(f"Expected {key} to be NetworkLogConfig or None, got {type(value).__name__}")
                    elif key == "module_log_levels":
                        if not isinstance(value, dict):
                            raise TypeError(f"Expected {key} to be dict, got {type(value).__name__}")
                        # Validate all values are LogLevel
                        for mod_name, mod_level in value.items():
                            if not isinstance(mod_level, LogLevel):
                                raise TypeError(f"Module log level for '{mod_name}' must be LogLevel, got {type(mod_level).__name__}")
                    elif key == "terminal_log_max_line_length":
                        if value is not None and not isinstance(value, int):
                            raise TypeError(f"Expected {key} to be int or None, got {type(value).__name__}")
                        if value is not None and value <= 0:
                            raise ValueError(f"Expected {key} to be positive integer or None, got {value}")

                    elif key.startswith("aac_"):
                        if key == "aac_default_bitrate":
                            if not isinstance(value, int) or value < 32000 or value > 320000:
                                raise ValueError(f"AAC bitrate must be between 32000 and 320000 bps, got {value}")
                        elif key == "aac_frame_analysis_method":
                            if value not in ["pyav", "manual"]:
                                raise ValueError(f"AAC frame analysis method must be 'pyav' or 'manual', got {value}")
                        elif key == "aac_quality_preset":
                            if value not in ["fast", "balanced", "quality"]:
                                raise ValueError(f"AAC quality preset must be 'fast', 'balanced', or 'quality', got {value}")
                        elif key == "aac_memory_limit_mb":
                            if not isinstance(value, int) or value < 100:
                                raise ValueError(f"AAC memory limit must be at least 100 MB, got {value}")
                        elif key == "aac_max_workers":
                            if not isinstance(value, int) or value < 1 or value > 16:
                                raise ValueError(f"AAC max workers must be between 1 and 16, got {value}")

                    else:
                        expected_type = get_type_hints(cls).get(key)
                        if expected_type and not isinstance(value, expected_type):
                            raise TypeError(f"Expected {key} to be of type {expected_type.__name__}, got {type(value).__name__}")
                    
                    setattr(cls, key, value)
                else:
                    raise AttributeError(f"Sorry, value of key '{key}' is immutable.")
            else:
                raise AttributeError(f"Invalid config key: {key}. No such key.")
        
        # Check if logging configuration changed
        if (cls.log_level != old_log_level or 
            cls.log_filepath != old_log_filepath or
            cls.network_log_config != old_network_config or
            cls.module_log_levels != old_module_log_levels or
            cls.terminal_log_max_line_length != old_terminal_max_line_length):
            LoggingManager.reconfigure()

class NetworkLogConfig:
    """Configuration container for network logging parameters.
    
    Encapsulates the network connection details required for remote logging,
    including host, port, and protocol specification. This class serves as
    an example of how complex configuration objects are handled by the
    serialization system.
    
    Parameters
    ----------
    host : str
        Hostname or IP address of the log server
    port : int
        Port number for the log server connection
    protocol : str, optional
        Network protocol to use ('TCP' or 'UDP'), by default 'TCP'
        
    Attributes
    ----------
    host : str
        The configured hostname or IP address
    port : int
        The configured port number
    protocol : str
        The configured protocol (always uppercase)
    """
    def __init__(self, host: str, port: int, protocol: str = 'TCP'):
        """Initialize network logging configuration with connection parameters.
        
        Validates and stores the network connection parameters for use by
        the logging system. Protocol names are automatically normalized
        to uppercase for consistency.
        
        Parameters
        ----------
        host : str
            Hostname or IP address of the log server
        port : int
            Port number for the log server connection
        protocol : str, optional
            Network protocol ('TCP' or 'UDP'), by default 'TCP'
        """
        self.host = host
        self.port = port
        self.protocol = protocol.upper()
        
    def __repr__(self):
        """Return a detailed string representation of the network configuration.
        
        Returns
        -------
        str
            String representation showing all configuration parameters
        """
        return f"NetworkLogConfig(host='{self.host}', port={self.port}, protocol='{self.protocol}')"
    
    def __eq__(self, other):
        """Compare two NetworkLogConfig instances for equality.
        
        Parameters
        ----------
        other : NetworkLogConfig
            Another instance to compare against
            
        Returns
        -------
        bool
            True if all parameters match, False otherwise
        """
        if not isinstance(other, NetworkLogConfig):
            return False
        return (self.host == other.host and 
                self.port == other.port and 
                self.protocol == other.protocol)
