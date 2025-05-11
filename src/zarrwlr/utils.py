"""Mixed general helpers of Zarr Wildlife Recording library"""

import zarr
from pathlib import Path
from io import BufferedReader
import os
from types import MappingProxyType
from collections.abc import Mapping, MutableMapping
from enum import Enum
import logging
import json
import yaml

# get the module logger
logger = logging.getLogger(__name__)

class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"

def next_numeric_group_name(zarr_group: zarr.Group) -> str:
    """next_numeric_group_name Return next free number for numbered groups.

    Eg. audio Files are imported into numbered Zarr groups. In order
    to add a new file, we have to find the next free number after the highest
    existing group number. Gaps in consecutive numbering are not filled!
    """

    existing = [int(k) for k in zarr_group.group_keys() if k.isdigit()]
    next_index = max(existing, default=-1) + 1
    return str(next_index)

def file_size(file: str | Path | BufferedReader) -> int:
    """Get file size in byte of a file or a file handle (BufferedReader)"""
    if isinstance(file, str):
        file = Path(file)

    if isinstance(file, BufferedReader):
        current_pos = file.tell() # we are save in case some code comes in front of this
        file.seek(0, 2)  # 2 = SEEK_END
        file_size = file.tell()
        file.seek(current_pos) 
    elif isinstance(file, Path):
        file_size = os.stat(file).st_size        
    else:
        raise ValueError("file must be of type: str, Path (both as a valid file path) or BufferedReader (as already opened file).")

    return file_size


def make_immutable(obj):
    """Makes object immutable recursively"""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: make_immutable(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(make_immutable(item) for item in obj)
    else:
        return obj  # primitive types or immutable objects

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