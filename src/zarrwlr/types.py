"""Types and Enum definitions"""

import json
import yaml
from enum import Enum, auto
from collections.abc import MutableMapping
import numpy as np

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


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


class AudioCompression(Enum):
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


class AudioFileBaseFeatures(RestrictedDict):
    """Basic information about an audio file."""

    FILENAME = "filename"
    SIZE_BYTES = "size_bytes"
    SH256 = "sha256"
    HAS_AUDIO_STREAM = "has_audio_stream"
    CONTAINER_FORMAT = "container_format"
    NB_STREAMS = "nb_streams"
    CODEC_PER_STREAM = "codec_per_stream"
    SAMPLING_RATE_PER_STREAM = "sampling_rate_per_stream"
    SAMPLE_FORMAT_PER_STREAM_AS_DTYPE = "sample_format_per_stream_as_dtype"
    SAMPLE_FORMAT_PER_STREAM_IS_PLANAR = "sample_format_per_stream_is_planar"
    CHANNELS_PER_STREAM = "channels_per_stream"
    CODEC_COMPRESSION_KIND_PER_STREAM = "codec_compression_kind_per_stream"
    
    key_specs = [ 
                # as: (key-name, data-type, default-value)
                (FILENAME, str, None), 
                (SIZE_BYTES, int, None),
                (SH256, str, None),
                (HAS_AUDIO_STREAM, bool, False),
                (CONTAINER_FORMAT, str, None),
                (NB_STREAMS, int, 0),
                (CODEC_PER_STREAM, list, []),
                (SAMPLING_RATE_PER_STREAM, list[int], []),
                (SAMPLE_FORMAT_PER_STREAM_AS_DTYPE, list, []),
                (SAMPLE_FORMAT_PER_STREAM_IS_PLANAR, list[bool], []),
                (CHANNELS_PER_STREAM, list[int], []),
                (CODEC_COMPRESSION_KIND_PER_STREAM, list[AudioCompression], [])
            ]

AudioSampleFormatMap = {
        "u8": np.uint8,
        "s16": np.int16,
        "s32": np.int32,
        "flt": np.float32,
        "fltp": np.float32,  # Achtung: planar → evtl. reshapen!
        "dbl": np.float64,
        "dblp": np.float64,  # planar
}
    
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