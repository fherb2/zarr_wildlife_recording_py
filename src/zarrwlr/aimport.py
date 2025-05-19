
import numpy as np
import subprocess
import logging
import pathlib
import hashlib
import json
import datetime
import yaml
import tempfile
from mutagen import File as MutagenFile
import zarr
from .utils import next_numeric_group_name, remove_zarr_group_recursive
from .config import Config
from .exceptions import Doublet, ZarrComponentIncomplete, ZarrComponentVersionError, ZarrGroupMismatch
from .types import AudioFileBaseFeatures, AudioCompression, AudioSampleFormatMap
from .oggfileblob import import_audio_to_blob as ogg_import_audio_to_blob
from .oggfileblob import create_index as ogg_create_index
from .flacbyteblob import FLACIndexer
from .opusbyteblob import OggOpusIndexer

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

STD_TEMP_DIR = "/tmp"
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array"

def create_original_audio_group(store_path: str|Path, group_path:zarr.Group|None = None):

    store = zarr.DirectoryStore(store_path)
    root = zarr.open_group(store, mode='a')

    def initialize_group(grp:zarr.Group):
        grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        grp.attrs["version"]  = Config.original_audio_group_version

    if group_path:
        if group_path in root:
            # group exist; check, if the right type
            check_if_original_audio_group(group = root[group_path])
            return
        else:
            # create this group
            created = False
            try:
                group = root.create_group(group_path)
                created = True
                initialize_group(group)
            except Exception:
                if created:
                    remove_zarr_group_recursive(root.store, group.path)
                raise  # raise original exception
    else:
        # root is this group
        group = root
        check_if_original_audio_group(group = root[group_path])

def check_if_original_audio_group(group:zarr.Group):
    grp_ok =     ("magic_id" in group.attrs) \
             and (group.attrs["magic_id"] == Config.original_audio_group_magic_id) \
             and ("version" in group.attrs)
    if grp_ok and not (group.attrs["version"] == Config.original_audio_group_version):
        raise ZarrComponentVersionError(f"Original audio group has version {group.attrs['version']} but current zarrwlr needs version {Config.original_audio_group_version} for this group. Please, upgrade group to get access.")
    elif grp_ok:
        return
    raise ZarrGroupMismatch(f"The group '{group}' is not an original audio group.")

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

def base_features_from_audio_file(file: str|pathlib.Path) -> AudioFileBaseFeatures:
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
                    audio_file: str|pathlib.Path, 
                    zarr_original_audio_group: zarr.Group,
                    first_sample_time_stamp: datetime.datetime|None,
                    last_sample_time_stamp: datetime.datetime|None = None,
                    target_codec:str = 'flac', # 'flac' or 'opus'
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
    #
    # In case, the target codec is 'opus', a lossy compress format:
    # 
    #   In contrast to 'flac', 'opus' codec can only use sampling frequencies of 48kS/s (or lower)! 
    #   If the source has a higher sampling frequency, we would loose information. This is important
    #   insofar as the source could be specialized for ultrasonic frequencies! In order to can compress
    #   such high frequencies, we use a trick: For the compressing algorithm, we say 'source is 48kS/s'.
    #   but we remember the factor of the real sampling frequency to this 48kS/s. If we uncompress
    #   the data later for processing or export, we reinterprete the sampling fequency to the original 
    #   value.
    #   There's only one drawback: If these audio has also very deep frequencies, so the opus comprression
    #   algorithm can interprete these as inaudible and remove all of them. To avoid this, use 'flac'
    #   or downsample the audio source to 48kS/s before.
    #   



    assert target_codec in ('flac', 'opus') 
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12)

    audio_file = pathlib.Path(audio_file)

    # 0) Check if group is original audio file group and has the right version
    check_if_original_audio_group(zarr_original_audio_group)

    # 1) Analyze File-/Audio-/Codec-Type
    base_features = base_features_from_audio_file(audio_file)
    source_params = _get_source_params(audio_file)
    # target_sample_format = _get_ffmpeg_sample_fmt(source_params["sample_format"], 
    #                                               target_codec)

    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{audio_file}' has no audio stream")
    if base_features.NB_STREAMS > 1:
        raise NotImplementedError("Audio file has more than one audio stream. Import of more than one audio streams (with any number of audio channels) is not yet supported.")




    # 2) Is file yet present in database?
    if is_audio_in_original_audio_group(zarr_original_audio_group, base_features):
        raise Doublet("Same file seems yet in database. Same hash, same file name same size.")

    # 3) Get the complete meta data inside the file.
    all_file_meta = str(FileMeta(audio_file))

    # 4) Calculate the next free 'group-number'.
    new_audio_group_name = next_numeric_group_name(zarr_group=zarr_original_audio_group)

    # 5) Create numeric group for the new audio
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

        # 6) Create array, decode/encode file and import byte blob
        #    Use the right import strategy (to opus, to flac, byte-copy, transform sample-rate...)
        audio_blob_array = new_original_audio_grp.create_array(
                                    name            = AUDIO_DATA_BLOB_ARRAY_NAME,
                                    shape           = (0,), # we append data step by step
                                    chunks          = (Config.original_audio_chunk_size,),
                                    shards          = (Config.original_audio_chunks_per_shard * Config.original_audio_chunk_size,),
                                    dtype           = np.uint8,
                                    overwrite       = True,
                                )
            
        # create temp file but hold it not open; we use its name following
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp', dir=temp_dir) as tmp_out:
            tmp_file = pathlib.Path(tmp_out.name)

        opus_bitrate_str = str(int(opus_bitrate/1000))+'k'
        sampling_rescale_factor = 1.0
        target_sample_rate = base_features[base_features.SAMPLING_RATE_PER_STREAM[0]]
        is_ultrasonic = (target_codec == "opus") and (source_params["sampling_rate"] > 48000)

        if target_codec == 'opus' and source_params["is_opus"] and not is_ultrasonic:
            # copy opus encoded data directly into ogg.opus 
            ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(audio_file),
                            "-c:a", "copy",
                            "-f", "ogg", str(tmp_file)
                            ]

        elif target_codec == 'opus':
            ffmpeg_cmd = ["ffmpeg", "-y"]
            if is_ultrasonic:
                # we interprete sampling rate as "48000" to can use opus for ultrasonic
                ffmpeg_cmd += ["-sample_rate", "48000"]
                target_sample_rate = 48000
                sampling_rescale_factor = float(source_params["bit_rate"]) / 48000.0
            ffmpeg_cmd += ["-i", str(audio_file)]
            ffmpeg_cmd += ["-c:a", "libopus", "-b:a", opus_bitrate_str]
            ffmpeg_cmd += ["-vbr", "off"] # constant bitrate is a bit better in quality than VRB=On
            ffmpeg_cmd += ["-apply_phase_inv", "false"] # Phasenrichtige Kodierung: keine Tricks!
            ffmpeg_cmd += ["-f", "ogg", str(tmp_file)]

        else: # target_codec == 'flac'
            ffmpeg_cmd = ["ffmpeg", "-y"]
            ffmpeg_cmd += ["-i", str(audio_file), "-c:a", "flac"]
            ffmpeg_cmd += ["-compression_level", str(flac_compression_level)]
            ffmpeg_cmd += [str(tmp_file)]

        # start encoding into temp file and wait until finished
        subprocess.run(ffmpeg_cmd, check=True)

        # copy tmp-file data into the array and remove the temp file
        with open(tmp_file, "rb") as f:
            for buffer in iter(lambda: f.read(chunk_size), b''):
                audio_blob_array.append(np.frombuffer(buffer, dtype="u1"))
        tmp_file.unlink()

        # add encoding attributes to this array
        attrs = {
                "codec": target_codec,
                "nb_channels": source_params["nb_channels"],
                "sample_rate": target_sample_rate,
                "sampling_rescale_factor": sampling_rescale_factor,
                "container_type": "flac-nativ",
                "first_sample_time_stamp": first_sample_time_stamp
                }
        if target_codec == "opus":
            attrs["container_type"] = "ogg"
            attrs["opus_bitrate"] = opus_bitrate,
        else: # target_codec == "flac"
            attrs["compression_level"] = flac_compression_level
        audio_blob_array.attrs.update(attrs)
        
        # 7) Create and save index inside the group (as array, not attribute since the size of structured data)
        if target_codec == 'opus':
            indexer = OggOpusIndexer(new_original_audio_grp, audio_blob_array)
        else:
            indexer = FLACIndexer(new_original_audio_grp, audio_blob_array)
        # build index    
        indexer.build_index()

    except Exception:
        # Full-Rollback: In case of any exception: we remove the group with all content.
        if created:
            remove_zarr_group_recursive(zarr_original_audio_group.store, new_original_audio_grp.path)
        raise  # raise original exception




def _get_source_params(input_file: pathlib.Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate:stream=codec_name:stream=sample_fmt:stream=bit_rate:stream=channels",
        "-of", "json", str(input_file)
    ]
    out = subprocess.check_output(args=cmd)
    info = json.loads(out)
    source_params = {
                "sampling_rate": int(info['streams'][0]['sample_rate']) if 'sample_rate' in info['streams'][0] else None,
                "is_opus": info['streams'][0]['codec_name'] == "opus",
                "sample_format": info['streams'][0]['sample_fmt'],
                "bit_rate": int(info['streams'][0]['bit_rate']) if 'bit_rate' in info['streams'][0] else None,
                "nb_channels": int(info['streams'][0]['channels'])
            }
    return source_params



def _get_ffmpeg_sample_fmt(source_sample_fmt: str, target_codec: str) -> str:
    """
    Gibt das beste sample_fmt-Argument für ffmpeg zurück, basierend auf Quellformat und Zielcodec.
    
    Args:
        source_sample_fmt (str): Sample-Format der Quelle laut ffprobe, z.B. "fltp", "s16", "s32", "flt"
        target_codec (str): "flac" oder "opus"
    
    Returns:
        str: Der passende Wert für "-sample_fmt" bei ffmpeg (z.B. "s32", "flt", "s16")
    """

    # Mappings nach Fähigkeiten der Codecs
    flac_supported = ["s8", "s12", "s16", "s24", "s32"]
    opus_supported = ["s16", "s24", "flt"]

    # Auflösen von planar zu packed
    normalized_fmt = source_sample_fmt.rstrip("p") if source_sample_fmt.endswith("p") else source_sample_fmt

    if target_codec == "flac":
        if normalized_fmt in flac_supported:
            return normalized_fmt
        # fallback
        return "s32"

    elif target_codec == "opus":
        # Opus arbeitet intern mit 16-bit, akzeptiert aber auch float input
        if normalized_fmt in opus_supported:
            return normalized_fmt
        # fallback auf float oder s16
        return "flt" if normalized_fmt.startswith("f") else "s16"

    else:
        raise NotImplementedError(f"Unsupported codec: {target_codec}")
