
import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path
import hashlib
import json
from enum import Enum, auto
import yaml
import mutagen
from mutagen import File as MutagenFile
import zarr
from .utils import RestrictedDict, next_numeric_group_name, file_size
from .config import Config
from .exceptions import Doublet

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

AUDIO_CHUNK_SIZE:int            = int(2**20) # 8 kByte
AUDIO_CHUNKS_PER_SHARD:int      =int(40*1024*1024 / AUDIO_CHUNK_SIZE)
AUDIO_SHARDED_ARRAY_SIZE:int    =AUDIO_CHUNKS_PER_SHARD * AUDIO_CHUNK_SIZE

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


def mutagen_info(filepath: str | Path) -> dict:

    audio = MutagenFile(Path(filepath), easy=False)
    if not audio:
        raise ValueError(f"Unsupported or unreadable file for 'mutagen': {filepath}")

    info = {}
    if audio.info:
        info['technical'] = {
            'length': getattr(audio.info, 'length', None),
            'bitrate': getattr(audio.info, 'bitrate', None),
            'sample_rate': getattr(audio.info, 'sample_rate', None),
            'channels': getattr(audio.info, 'channels', None),
            'codec': audio.__class__.__name__,
        }

    info['tags'] = {}
    if audio.tags:
        for key, value in audio.tags.items():
            try:
                # Bei manchen Formaten sind die Werte Listen, bei anderen nicht
                info['tags'][key] = value.text if hasattr(value, 'text') else str(value)
            except Exception:
                info['tags'][key] = str(value)

    return info



def is_audio_in_original_audio_group( zarr_original_audio_group: zarr.Group,
                                      base_features: AudioFileBaseFeatures,
                                      sh246_check_only: bool = False
                                    ) -> bool:
    for group_name in zarr_original_audio_group:
        # we need check this group only the name is only a number
        if group_name.isdigit():
            # this should be a original audio file database group...
            zarr_audio_database_grp = zarr_original_audio_group[group_name]
            if "type" in zarr_audio_database_grp.attrs:
                if zarr_audio_database_grp.attrs["type"] == "original_audio_file":
                    # Ok: Now we are shure, that this is a group of an imported audio file
                    # Check if the base_features are the same:
                    bf = zarr_audio_database_grp.attrs["base_features"]
                    if (base_features.SH256) == bf["SH256"]:
                        if sh246_check_only:
                            return True
                        elif    (base_features.FILENAME   == bf["FILENAME"]) \
                            and (base_features.SIZE_BYTES == bf["SIZE_BYTES"]):
                            return True
    return False

class FileMeta:
    """Full audio relevant meta information of a file by ffprobe and mutagen."""
    def __init__(self, file: str|Path, audio_only:bool=True):
        file = Path(file)

        # from ffprobe
        # ------------
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(file)
        ]
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                text=True)
        self.meta = {"ffprobe": self._typing_numbers(json.loads(result.stdout)) }

        # remove non-audio streams (but doesn't changes the index number of the audio
        # streams if other streams are sorted out before)
        self.meta["ffprobe"]["streams"] = [
                                stream for stream in self.meta.get("streams", [])
                                if stream.get("codec_type") == "audio"
                            ]
        
        # via mutagen
        # -----------
        self.meta["mutagen"] = mutagen_info(file)
        
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
        
def convert_audio_to_ogg(audio_file: str|Path, 
                         target_codec:str = 'flac', # "flac" or "opus"
                         flac_compression_level: int = 4, # 0...12; 4 is a really good value: fast and low
                                                          # data; higher values does not really reduce 
                                                          # data size but need more time and energy. 
                                                          # Note:
                                                          # The highest values can produce more data
                                                          # than lower values. '12' as maximum must not
                                                          # be the best compression. Check ist: More than
                                                          # 4 is not really less memory consumption but
                                                          # wasted time and energy.
                         opus_bitrate:str = '160k',
                         temp_dir="/tmp") -> tuple[Path, float, str]:
    """
    Konvertiert eine Audiodatei in einen Ogg-Container mit FLAC oder Opus.
    - Unterstützt Ultraschallmodus: PCM-Daten bleiben, Zeitbasis wird manipuliert
    - Rückgabe: Pfad zur Ogg-Datei, Faktor Original-Samplingrate / gespeicherter-samplingrate
    """

    def get_source_params(input_file: Path) -> tuple[int, bool]:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate:stream=codec_name",
            "-of", "json", str(input_file)
        ]
        out = subprocess.check_output(args=cmd)
        info = json.loads(out)
        sampling_rate = int(info['streams'][0]['sample_rate'])
        is_opus = info['streams'][0]['codec_name'] == "opus"
        return sampling_rate, is_opus

    assert target_codec in ('flac', 'opus')
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12)

    audio_file = Path(audio_file)

    original_rate, is_opus = get_source_params(audio_file)

    # in order to avoid downsampling, we rescale the time-base and sign this
    # as ultrasonic
    sampling_rescale = 1.0
    is_ultrasonic = (target_codec == "opus") and (original_rate > 48000)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg', dir=temp_dir) as tmp_out:
        tmp_file = Path(tmp_out.name)

    if target_codec == 'opus' and is_opus and not is_ultrasonic:
        # copy opus encoded data directly into ogg.opus 
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_file),
            "-c", "copy", "-f", "ogg", str(tmp_file)
        ], check=True)
        return tmp_file, sampling_rescale, target_codec

    ffmpeg_cmd = ["ffmpeg", "-y"]   

    if target_codec == 'opus':
        if is_ultrasonic:
            # we interprete sampling rate as "48000" to can use opus for ultrasonic
            # (for use late, it must be re-intrepreted)
            ffmpeg_cmd += ["-sample_rate", "48000"]
            sampling_rescale = float(original_rate) / 48000.0
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "libopus", "-b:a", opus_bitrate]
        ffmpeg_cmd += ["-vbr", "off"] # constant bitrate is a bit better in quality than VRB=On
        ffmpeg_cmd += ["-apply_phase_inv", "false"] # Phasenrichtige Kodierung: keine Tricks!

    elif target_codec == 'flac':
        ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
        ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]

    ffmpeg_cmd += ["-f", "ogg", str(tmp_file)]

    subprocess.run(ffmpeg_cmd, check=True)

    return tmp_file, sampling_rescale, target_codec

def import_original_audio_file(file: str|Path, 
                      zarr_original_audio_group: zarr.Group,
                      target_codec:str = 'flac'
                      ):
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

    # 0) Create Group or/and check if the original audio file group has the right version

    # 1) Analyze File-/Audio-/Codec-Type
    base_features = base_features_from_audio_file(file)
    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{file}' has no audio stream")

    # 2) Is file yet present in database?
    if is_audio_in_original_audio_group(zarr_original_audio_group, base_features):
        raise Doublet("Same file seems yet in database. Same hash, same file name same size.")

    # 3) Get all other meta data inside the file.
    all_file_meta = str(FileMeta(file))

    # 4) Calculate the next free 'group-number'.
    new_audio_group_name = next_numeric_group_name(zarr_group=zarr_original_audio_group)

    # 5) Create array, decode/encode file and import byte blob
    #    Use the right import strategy (to opus, to flac, byte-copy, transform sample-rate...)
    new_original_audio_grp = zarr_original_audio_group.require_group(new_audio_group_name)
    # add a version atrtribute to this group
    new_original_audio_grp.attrs["original_audio_group_version"] = Config.original_audio_group_version
    # do conversation to ogg.flac or ogg.opus; scale sampling in case of opus and >48kS/s
    tmp_file, sampling_base_scaling, target_codec = convert_audio_to_ogg(file, target_codec=target_codec)

    tmp_file_byte_size = file_size(tmp_file)

    ogg_file_blob_array = new_original_audio_grp.create_array(
        name            = "ogg_file_blob",
        shape           = tmp_file_byte_size,
        chunks          = (AUDIO_CHUNK_SIZE,),
        shards          = (AUDIO_SHARDED_ARRAY_SIZE,),
        dtype           = np.uint8,
        overwrite       = True,
    )

    with open(tmp_file, "rb") as f:
        for offset in range(0, tmp_file_byte_size, AUDIO_CHUNK_SIZE):
            buffer = f.read(AUDIO_CHUNK_SIZE)
            ogg_file_blob_array[offset : offset + len(buffer)] = np.frombuffer(buffer, dtype="u1")

    # Save Meta data to the group
    new_original_audio_grp.attrs["type"]                    = "original_audio_file"
    new_original_audio_grp.attrs["encoding"]                = target_codec
    new_original_audio_grp.attrs["sampling_base_scaling"]   = sampling_base_scaling
    new_original_audio_grp.attrs["base_features"]           = base_features
    new_original_audio_grp.attrs["ffprobe_meta_data_structure"] = all_file_meta

    # 6) Create and save index inside the group (as array, not attribute since the size of structured data)


    # 7) We can finally save the attribute "import_fimalzed" with True
    new_original_audio_grp.attrs["import_finalized"] = True
