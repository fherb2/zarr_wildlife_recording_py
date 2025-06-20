
import numpy as np
import subprocess
import logging
from pathlib import Path
import hashlib
import json
import yaml
from mutagen import File as MutagenFile
import zarr
from .utils import next_numeric_group_name, remove_zarr_group_recursive
from .config import Config
from .exceptions import Doublet, ZarrComponentIncomplete, ZarrComponentVersionError, ZarrGroupMismatch
from .types import AudioFileBaseFeatures, AudioCompressionBaseType, AudioSampleFormatMap
from .oggfileblob import import_audio_to_blob as ogg_import_audio_to_blob
from .oggfileblob import create_index as ogg_create_index

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

AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"

def create_original_audio_group(store_path: str|Path, group_path:zarr.Group|None = None):

    store = zarr.DirectoryStore(store_path)
    root = zarr.open_group(store, mode='a')

    def initialize_group(grp:zarr.Group):
        grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        grp.attrs["version"]  = Config.original_audio_group_version
        grp.attrs["finally_created"] = True # Marker that the creation was completely finalized.

    if group_path:
        if group_path in root:
            # group exist; check, if the right type
            check_if_original_audio_group(group = root[group_path])
            return
        else:
            # create this group
            group = root.create_group(group_path)
            initialize_group(group)
    else:
        # root is this group
        group = root
        check_if_original_audio_group(group = root[group_path])

def check_if_original_audio_group(group:zarr.Group) -> bool:
    if "finally_created" not in group.attrs:
        raise ZarrComponentIncomplete(f"Incomplete initialized or foreign group: {group}. Either given group is not an 'original audio group' or group initialization was broken.")
    grp_ok =     ("magic_id" in group.attrs) \
             and (group.attrs["magic_id"] == Config.original_audio_group_magic_id) \
             and ("version" in group.attrs)
    if grp_ok and not (group.attrs["version"] == Config.original_audio_group_version):
        raise ZarrComponentVersionError(f"Original audio group has version {group.attrs['version']} but current zarrwlr needs version {Config.original_audio_group_version} for this group. Please, upgrade group to get access.")
    elif grp_ok:
        return True
    raise ZarrGroupMismatch(f"The group '{group}' is not an original audio group.")

def audio_codec_compression(codec_name: str) -> AudioCompressionBaseType:
    """Recognize principle AudioCompressionBaseType from codec name."""
    if codec_name.startswith("pcm_"):
        return AudioCompressionBaseType.UNCOMPRESSED
    
    lossless_codecs = {"flac", "alac", "wavpack", "ape", "tak"}
    if codec_name in lossless_codecs:
        return AudioCompressionBaseType.LOSSLESS_COMPRESSED
    
    lossy_codecs = {"mp3", "aac", "opus", "ogg", "ac3", "eac3", "wma"}
    if codec_name in lossy_codecs:
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    return AudioCompressionBaseType.UNKNOWN

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
        base_features[base_features.SAMPLING_RATE_PER_STREAM] = list({int(stream.get("sample_rate", None)) for stream in streams})
        base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE] = list({AudioSampleFormatMap.get(stream.get("sample_fmt", "s16")) for stream in streams})
        base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR] = list({stream.get("sample_fmt", "s16").endswith("p") for stream in streams})
        base_features[base_features.CHANNELS_PER_STREAM] = list({int(stream.get("channels", None)) for stream in streams})

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

    except Exception:
        # so reset to default
        base_features = AudioFileBaseFeatures()

    return base_features

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
                    bf:AudioFileBaseFeatures = zarr_audio_database_grp.attrs["base_features"]
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
        self.meta = {"ffprobe": self._ffprobe_info(file)}
        # via mutagen
        # -----------
        self.meta["mutagen"] = self._mutagen_info(file)

    @classmethod
    def _ffprobe_info(cls, file: Path) -> dict:
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
        info = {"ffprobe": cls._typing_numbers(json.loads(result.stdout)) }
        # remove non-audio streams (but doesn't changes the index number of the audio
        # streams if other streams are sorted out before)
        info["ffprobe"]["streams"] = [
                                stream for stream in info["ffprobe"].get("streams", [])
                                if stream.get("codec_type") == "audio"
                            ]
        return info

    @staticmethod
    def _mutagen_info(file: Path) -> dict:
        audio = MutagenFile(file, easy=False)
        if not audio:
            raise ValueError(f"Unsupported or unreadable file for 'mutagen': {file}")
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
        


def import_original_audio_file(
                    file: str|Path, 
                    zarr_original_audio_group: zarr.Group,
                    target_codec:str = 'flac',
                    flac_compression_level: int = 4, # 0...12; 4 is a really good value: fast and low
                                                        # data; higher values does not really reduce 
                                                        # data size but need more time and energy. 
                                                        # Note:
                                                        # The highest values can produce more data
                                                        # than lower values. '12' as maximum must not
                                                        # be the best compression. Check it: More than
                                                        # 4 is not really less memory consumption but
                                                        # wasted time and energy.
                    opus_bitrate:int = 160_000, # bit per second
                    temp_dir="/tmp",
                    chunk_size:int = Config.original_audio_chunk_size
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

    # 0) Check if group is original audio file group and has the right version
    check_if_original_audio_group(zarr_original_audio_group)

    # 1) Analyze File-/Audio-/Codec-Type
    base_features = base_features_from_audio_file(file)
    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{file}' has no audio stream")
    if base_features.NB_STREAMS > 1:
        raise NotImplementedError("Audio file has more than one audio stream. Import of more than one audio streams (with any number of audio channels) is not yet supported.")

    # 2) Is file yet present in database?
    if is_audio_in_original_audio_group(zarr_original_audio_group, base_features):
        raise Doublet("Same file seems yet in database. Same hash, same file name same size.")

    # 3) Get the complete meta data inside the file.
    all_file_meta = str(FileMeta(file))

    # 4) Calculate the next free 'group-number'.
    new_audio_group_name = next_numeric_group_name(zarr_group=zarr_original_audio_group)

    created = False
    try:
        new_original_audio_grp = zarr_original_audio_group.require_group(new_audio_group_name)
        created = True

        # add a version attribute to this group
        new_original_audio_grp.attrs["original_audio_data_array_version"] = Config.original_audio_data_array_version

        # Save Original-Audio-Meta data to the group
        # These data specify the source and not the following file blob!
        new_original_audio_grp.attrs["type"]                    = "original_audio_file"
        new_original_audio_grp.attrs["encoding"]                = target_codec
        new_original_audio_grp.attrs["base_features"]           = base_features
        new_original_audio_grp.attrs["meta_data_structure"]     = all_file_meta

        # 5) Create array, decode/encode file and import byte blob
        #    Use the right import strategy (to opus, to flac, byte-copy, transform sample-rate...)
        audio_blob_array = new_original_audio_grp.create_array(
                                    name            = AUDIO_DATA_BLOB_ARRAY_NAME,
                                    shape           = (0,), # we append data step by step
                                    chunks          = (Config.original_audio_chunk_size,),
                                    shards          = (Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size,),
                                    dtype           = np.uint8,
                                    overwrite       = True,
                                )
        ogg_import_audio_to_blob(   file,
                                    audio_blob_array,
                                    target_codec,
                                    flac_compression_level,
                                    opus_bitrate,
                                    temp_dir,
                                    chunk_size
                                    )
        
        # 6) Create and save index inside the group (as array, not attribute since the size of structured data)
        ogg_create_index(audio_blob_array, new_original_audio_grp)

        # 7) We can finally save the attribute "finally_created" with True
        new_original_audio_grp.attrs["finally_created"] = True # Marker that the creation was completely finalized.

    except Exception:
        if created:
            remove_zarr_group_recursive(zarr_original_audio_group.store, new_original_audio_grp.path)
        raise  # raise original exception
