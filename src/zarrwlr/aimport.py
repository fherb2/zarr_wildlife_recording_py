

import subprocess
import tempfile
import logging
from pathlib import Path
import hashlib
import json
from enum import Enum, auto
import yaml
from zarrwlr.utils import RestrictedDict, next_numeric_group_name
import zarr
from zarrwlr.module_config import ModuleStaticConfig

# get the module logger   
logger = logging.getLogger(__name__)

def _check_ffmpeg_tools():
    """Check if ffmpeg and ffprobe are installed and callable."""
    tools = ["ffmpeg", "ffprobe"]

    for tool in tools:
        try:
            subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"Missing Command line tool {tool}. Please install ist.")
            exit(1)

# We do this check during import:
_check_ffmpeg_tools()

class AudioCompression(Enum):
    """Shows the principle kind of compression (lossy, lossless, uncompressed)."""
    def _generate_next_value_(name, start, count, last_values):
        return name  # -> set value automated to "UNCOMPRESSED", "LOSSLESS", ...

    UNCOMPRESSED = auto()
    LOSSLESS_COMPRESSED = auto()
    LOSSY_COMPRESSED = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.value

def audio_codec_compression(codec_name: str) -> AudioCompression:
    """Recognize principle AudioCompression from codec name."""
    if codec_name.startswith("pcm_"):
        return AudioCompression.UNCOMPRESSED
    
    lossless_codecs = {"flac", "alac", "wavpack", "ape", "tak"}
    if codec_name in lossless_codecs:
        return AudioCompression.LOSSLESS_COMPRESSED
    
    lossy_codecs = {"mp3", "aac", "opus", "ogg", "ac3", "eac3", "wma"}
    if codec_name in lossy_codecs:
        return AudioCompression.LOSSY_COMPRESSED
    
    return AudioCompression.UNKNOWN

class AudioFileBaseFeatures(RestrictedDict):
    """Basic information about an audio file."""

    FILENAME = "filename"
    SIZE_BYTES = "size_bytes"
    SH256 = "sha256"
    HAS_AUDIO_STREAM = "has_audio_stream"
    CONTAINER_FORMAT = "container_format"
    NB_STREAMS = "nb_streams"
    CODEC_PER_STREAM = "codec_per_stream"
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
                (CODEC_COMPRESSION_KIND_PER_STREAM, list, [])
            ]

def base_features_from_audio_file(file: str|Path) -> AudioFileBaseFeatures:
    """Get basic information about an audio file."""

    file_path = Path(file)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # initialize return value
    base_features = AudioFileBaseFeatures()

    # name and size
    base_features[base_features.FILENAME]   = file_path.name
    base_features[base_features.SIZE_BYTES] = file_path.stat().st_size

    # Container und Codec mit ffprobe (aus ffmpeg)
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "format=format_name:stream=codec_name",
            "-of", "json",
            str(file_path)
        ]
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                text=True)
        metadata = json.loads(result.stdout)
        print(f"{metadata=}\n\n")


        base_features[base_features.CONTAINER_FORMAT] = \
                                        metadata.get("format", {}).get("format_name", "unknown")
        streams                                       = metadata.get("streams", [])
        base_features[base_features.NB_STREAMS]       = len(streams)
        base_features[base_features.HAS_AUDIO_STREAM] = len(streams) > 0
        base_features[base_features.CODEC_PER_STREAM] = list({stream.get("codec_name", "unknown") for stream in streams})

        base_features[base_features.CODEC_COMPRESSION_KIND_PER_STREAM] = []
        for codec in base_features[base_features.CODEC_PER_STREAM]:
            base_features[base_features.CODEC_COMPRESSION_KIND_PER_STREAM].append(
                                        audio_codec_compression(codec)
                                        )

        # calc hash (SHA256)
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        base_features[base_features.SH256] = hasher.hexdigest()

    except Exception as e:
        # so reset to default
        base_features = AudioFileBaseFeatures()

    return base_features



class FileMeta:
    """Full audio relevant meta information of a file."""
    def __init__(self, file_path: str|Path, audio_only:bool=True):
        file_path = Path(file_path)
        # Befehl für ffprobe, um alle relevanten Metadaten zu extrahieren
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(file_path)
        ]
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                text=True)
        self.meta = self._typing_numbers(json.loads(result.stdout))

        # remove non-audio streams (but doesn't changes the index number of the audio
        # streams if other streams are sorted out before)
        self.meta["streams"] = [
                                stream for stream in self.meta.get("streams", [])
                                if stream.get("codec_type") == "audio"
                            ]
        
    def __str__(self):
        return self.as_yaml()
    
    def as_yaml(self):
        return yaml.dump(self.meta, sort_keys=False, allow_unicode=True)
    
    def as_dict(self):
        return self.meta
    
    @classmethod
    def _typing_numbers(cls, obj):
        """Recursive typing if objects are int or float"""
        if isinstance(obj, dict):
            return {k: cls._typing_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._typing_numbers(i) for i in obj]
        elif isinstance(obj, str):
            # Konvertiere Strings, die wie Zahlen aussehen
            if obj.isdigit():
                return int(obj)
            try:
                return float(obj)
            except ValueError:
                return obj
        else:
            return obj
        
def convert_audio_to_ogg(input_path, output_dir, codec='flac', opus_bitrate='160k', ultrasound=False):
    """
    Konvertiert eine Audiodatei in einen Ogg-Container mit FLAC oder Opus.
    - Unterstützt Ultraschallmodus: PCM-Daten bleiben, Zeitbasis wird manipuliert
    - Rückgabe: Pfad zur Ogg-Datei, Originalrate (nur im Ultraschallmodus)
    """
    assert codec in ('flac', 'opus')

    #with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=output_dir) as tmp_out:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=output_dir) as tmp_out:
        output_path = tmp_out.name

    print(f"{output_path}")
    
    exit(0)
    
    
    original_rate = None

    if codec == 'opus' and input_path.endswith('.opus') and not ultrasound:
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-c", "copy", "-f", "ogg", output_path
            ], check=True)
            return output_path, None
        except subprocess.CalledProcessError:
            print("Direct copy failed, falling back to re-encode.")

    ffmpeg_cmd = ["ffmpeg", "-y"]

    if codec == 'opus':
        if ultrasound:
            # Ermittele ursprüngliche Samplingrate
  #          original_rate = get_sampling_rate(input_path)
            ffmpeg_cmd += ["-sample_rate", "48000"]
        ffmpeg_cmd += ["-i", input_path, "-c:a", "libopus", "-b:a", opus_bitrate]

    elif codec == 'flac':
        ffmpeg_cmd += ["-i", input_path, "-c:a", "flac"]

    ffmpeg_cmd += ["-f", "ogg", output_path]

    subprocess.run(ffmpeg_cmd, check=True)

    return output_path, original_rate

def import_audio_file(file: str|Path, zarr_original_audio_group: zarr.Group):
    # Analyze File-/Audio-/Codec-Type: we need 
    #   original file name
    #   original file size
    #   original file type
    #   original codec name
    #   original file byte-Hash
    # Is 'original file name' and 'original size' known in database?
    #   If yes: Is 'original hash' the same as in database?
    #               If yes: get warning and raise Exception
    #               If no: get warning only and continue
    # Look for all other meta data inside the file.
    # Calculate the next free 'array-number'.
    # Create array, decode file and import depending on source and target codec
    # Create Index and put it into array
    # Put all meta data as attributes.
    # Set attribute 'import_finalized' with true. Thats the marker for a completed import. 

    # 1) Analyze File-/Audio-/Codec-Type
    base_features = base_features_from_audio_file(file)
    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{file}' has no audio stream")

    # 2) Is 'original file name' and 'original size' known in database?

    # 3) Get all other meta data inside the file.
    all_file_meta = str(FileMeta(file))
    print("\n\n--> all_file_meta:")
    print(all_file_meta)
    print("---End---\n")

    # 4) Calculate the next free 'group-number'.
    new_audio_group_name = next_numeric_group_name(zarr_group=zarr_original_audio_group)

    # 5) Create array, decode/encode file and import byte blob
    #    Use the right import strategy (to opus, to flac, byte-copy, transform sample-rate...)
    new_original_audio_grp = zarr_original_audio_group.require_group(new_audio_group_name)
    # add a version atrtribute to this group
    new_original_audio_grp.attrs["original_audio_group_version"] = ModuleStaticConfig.versions["file_blob_group_version"]

    convert_audio_to_ogg(file, "/tmp", codec='flac', opus_bitrate='160k', ultrasound=False)