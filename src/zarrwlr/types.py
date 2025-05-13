"""Types and Enum definitions"""

from enum import Enum
import json
import yaml
from collections.abc import MutableMapping

class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
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