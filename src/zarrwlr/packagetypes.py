"""Types and Enum definitions"""

import json
import yaml
from enum import Enum, auto
from collections.abc import MutableMapping
import numpy as np

class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"
    NOTSET = "NOTSET"

class RestrictedDict(MutableMapping):
    """Base class for dictionaries with fixed and restricted keys.
    
    Example:

        class AudioFileInfo(RestrictedDict):
            # Key specifications:

            key_specs = [ 
                            # as: (key-name, data-type, default-value)
                            ("filename", str, None), 
                            ("size_bytes", int, 0),
                            ("codec", str, "unknown"),
                            ("container", str, "unknown"),
                            ("sha256", str, None),
                        ]

    """

    key_specs: list[tuple[str, type, object]] = []  # Muss von Subklassen überschrieben werden

    def __init__(self, **kwargs):
        if not self.key_specs:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty `key_specs` list"
            )

        # Strukturierte Key-Spezifikation
        self._specs = {
            key: {"key": key, "type": typ, "default": default}
            for key, typ, default in self.key_specs
        }

        # Default-Werte übernehmen
        self._data = {
            key: spec["default"] for key, spec in self._specs.items()
        }

        # Initialwerte einfügen mit Typprüfung
        for key, value in kwargs.items():
            self[key] = value  # geht durch __setitem__

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._specs:
            raise KeyError(f"Invalid key: {key}")
        expected_type = self._specs[key]["type"]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Expected type {expected_type.__name__} for key '{key}', got {type(value).__name__}"
            )
        self._data[key] = value

    def __delitem__(self, key):
        raise NotImplementedError("Deletion is not allowed")

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    # Export
    def to_dict(self) -> dict:
        return dict(self._data)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self._data, **kwargs)

    def to_yaml(self, **kwargs) -> str:
        return yaml.dump(self._data, **kwargs, sort_keys=False)

    # Import
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class JSONEnumMeta(type(Enum)):
    """Metaclass die automatisch JSON-Serialisierung zu Enums hinzufügt."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Automatisch JSON-Methoden hinzufügen falls nicht vorhanden
        if not hasattr(cls, '__json__'):
            cls.__json__ = lambda self: self.value
        
        if not hasattr(cls, 'to_json'):
            cls.to_json = lambda self: self.value
        
        if not hasattr(cls, 'from_json'):
            @classmethod
            def from_json(cls, value):
                try:
                    return cls(value)
                except ValueError:
                    raise ValueError(f"'{value}' is not a valid {cls.__name__}")
            cls.from_json = from_json
        
        return cls

class AudioCompression(Enum, metaclass=JSONEnumMeta):
    """Shows the principle kind of compression (lossy, lossless, uncompressed)."""
    def _generate_next_value_(name, start, count, last_values):
        # is used by 'auto()'
        return name  # -> set value automated to "UNCOMPRESSED", "LOSSLESS", ...

    UNCOMPRESSED = auto()
    LOSSLESS_COMPRESSED = auto()
    LOSSY_COMPRESSED = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.value


    
class OriginalAudioBlobFeatures(RestrictedDict):
    """Parameters of the file-blob of the original audio data."""

    CONTAINER_FORMAT = "container_format"
    SAMPLING_RATE_OF_COMPRESSION = "sampling_rate_of_compression"
    SAMPLING_RESCALE_FACTOR = "sampling_rescale_factor"
    SAMPLE_FORMAT_AS_DTYPE = "sample_format_as_dtype"
    SAMPLE_FORMAT_IS_PLANAR = "sample_format_is_planar"
    CHANNELS = "channels"

    key_specs = [ 
                # as: (key-name, data-type, default-value)
                (CONTAINER_FORMAT, str, None),
                (SAMPLING_RATE_OF_COMPRESSION, int, None),
                (SAMPLING_RESCALE_FACTOR, float, 1.0),
                (SAMPLE_FORMAT_AS_DTYPE, np.dtype, np.int16),
                (SAMPLE_FORMAT_IS_PLANAR, bool, False),
                (CHANNELS, int, 1)
            ]