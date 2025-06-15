import numpy as np
import subprocess
import pathlib
import hashlib
import json
import datetime
import time
import yaml
import tempfile
from mutagen import File as MutagenFile
import zarr
import zarrcompatibility as zc
from .utils import file_size

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Module loading...")

# Enable universal serialization
zc.enable_universal_serialization()

# Import FLAC functions from flac_access module
from .flac_access import (  # noqa: E402
    import_flac_to_zarr,
    extract_audio_segment_flac, 
    parallel_extract_audio_segments_flac
)

from .aac_access import (
    import_aac_to_zarr,
    extract_audio_segment_aac, 
    parallel_extract_audio_segments_aac
)

from .utils import check_ffmpeg_tools, next_numeric_group_name, remove_zarr_group_recursive, safe_int_conversion  # noqa: E402
from .config import Config  # noqa: E402
from .exceptions import Doublet, ZarrComponentIncomplete, ZarrComponentVersionError, ZarrGroupMismatch, OggImportError  # noqa: E402
#from .packagetypes import AudioFileBaseFeatures, AudioSampleFormatMap  # noqa: E402
from .audio_coding import AudioCompressionBaseType, audio_codec_compression

STD_TEMP_DIR = "/tmp"
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array" 


def init_original_audio_group(store_path: str|pathlib.Path, group_path: str|pathlib.Path|None = None) -> zarr.Group:
    """
    Erstellt eine neue Original-Audio-Gruppe im Zarr-Store oder prüft eine existierende.
    
    Args:
        store_path: Pfad zum Zarr-Store
        group_path: Pfad zur Gruppe innerhalb des Stores (optional)
    """
    logger.trace(f"create_original_audio_group() requested. Parameters: {store_path=}; {group_path=}")
    store_path = pathlib.Path(store_path).resolve()  # Konvertiere zu absolutem Pfad-String
    
    # Zarr-Pfade innerhalb des Stores müssen als Strings vorliegen
    zarr_group_path = None
    if group_path is not None:
        zarr_group_path = str(group_path)  # Zarr erwartet String-Pfade
    
    logger.trace(f"Try to open store {store_path=} and audio group path {group_path=}...")
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store, mode='a')
    
    def _initialize_new_original_audio_group(grp: zarr.Group):
        logger.trace("Initialize Zarr group containing all original audio imports...")
        grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        grp.attrs["version"] = Config.original_audio_group_version
        logger.trace(f"Initializing done. {grp.attrs['magic_id']=} and {grp.attrs['version']=}")
    
    group = None
    if zarr_group_path is not None:
        if zarr_group_path in root:
            # group exist
            logger.trace(f"Zarr group {zarr_group_path} exist. Check if it is a valid 'original audio' group...")
            grp = root[zarr_group_path]
            assert isinstance(grp, zarr.Group), f"Expected zarr.Group, got {type(grp)=}"
            check_if_original_audio_group(group=grp)
            logger.debug(f"Zarr group {zarr_group_path} is a valid 'original audio' group.")
            return grp
        else:
            # create this group
            logger.trace(f"Zarr group {zarr_group_path} doesn't exist. Try to create...")
            created = False
            try:
                group = root.create_group(zarr_group_path)
                created = True
                _initialize_new_original_audio_group(group)
            except Exception:
                if created:
                    remove_zarr_group_recursive(root.store, group.path)
                raise  # raise original exception
            logger.success(f"Zarr group {zarr_group_path} as 'original audio' group created.")
    else:
        # root is this group
        logger.trace(f"Zarr root {store_path} is given as 'original audio' group. Check if it is a valid 'original audio' group...")
        check_if_original_audio_group(group=root)
        logger.success(f"Checked: Zarr group {zarr_group_path} is a valid 'original audio' group.")
    
    if group is None:
        group = root

    return group


def check_if_original_audio_group(group:zarr.Group):
    """
    Überprüft, ob eine Gruppe eine gültige Original-Audio-Gruppe ist.
    
    Args:
        group: Zu prüfende Zarr-Gruppe
        
    Raises:
        ZarrComponentVersionError: Wenn die Gruppenversion nicht mit der erwarteten Version übereinstimmt
        ZarrGroupMismatch: Wenn die Gruppe keine Original-Audio-Gruppe ist
    """
    logger.trace(f"'ceck if original audio group' with group '{group}' requested.")
    # no logging here: will be done at calling positions
    grp_ok =     ("magic_id" in group.attrs) \
             and (group.attrs["magic_id"] == Config.original_audio_group_magic_id) \
             and ("version" in group.attrs)
    if grp_ok and not (group.attrs["version"] == Config.original_audio_group_version):
        raise ZarrComponentVersionError(f"Original audio group has version {group.attrs['version']} but current zarrwlr needs version {Config.original_audio_group_version} for this group. Please, upgrade group to get access.")
    elif grp_ok:
        return
    raise ZarrGroupMismatch(f"The group '{group}' is not an original audio group.")


def safe_get_sample_format_dtype(sample_fmt, fallback_fmt="s16"):
    """Safe conversion of sample formats to dtype"""
    try:
        if sample_fmt is None or sample_fmt == "":
            sample_fmt = fallback_fmt
        return AudioSampleFormatMap.get(sample_fmt, AudioSampleFormatMap[fallback_fmt])
    except (KeyError, TypeError):
        return AudioSampleFormatMap[fallback_fmt]


# def base_features_from_audio_file(file: str|pathlib.Path) -> AudioFileBaseFeatures:
#     """
#     Ermittelt grundlegende Informationen über eine Audiodatei.
    
#     Args:
#         file: Pfad zur Audiodatei
        
#     Returns:
#         AudioFileBaseFeatures: Objekt mit Basisinformationen über die Audiodatei
        
#     Raises:
#         FileNotFoundError: Wenn die Datei nicht gefunden wird
#     """
#     logger.trace(f"'Base features from audio files' with file '{file}' requested.")
#     file_path = pathlib.Path(file)

#     if not file_path.is_file():
#         logger.error(f"File not found: {file_path}")
#         raise FileNotFoundError(f"File not found: {file_path}")
    
#     # initialize return value
#     logger.trace("Load audio files base features...")
#     base_features = AudioFileBaseFeatures()

#     # name and size
#     logger.trace(f"... and set 'base_features.FILENAME' with '{file_path.name}' and 'base_features.SIZE_BYTES' with '{file_path.stat().st_size}'.")
#     base_features[base_features.FILENAME]   = file_path.name
#     base_features[base_features.SIZE_BYTES] = file_path.stat().st_size

#     # Container und Codec mit ffprobe (aus ffmpeg)
#     try:
#         cmd = [
#             "ffprobe",
#             "-v", "error",
#             "-select_streams", "a",  # Wählt alle Audiostreams aus
#             "-show_entries", "format=format_name",  # Format-Information
#             "-show_entries", "stream=index,codec_name,sample_rate,sample_fmt,channels",  # Stream-Information mit index
#             "-of", "json",
#             str(file_path)
#         ]
#         logger.trace(f"Read meta data from file '{str(file_path)}' by using: '{cmd}'.")
#         result = subprocess.run(cmd,
#                                 stdout=subprocess.PIPE,
#                                 stderr=subprocess.PIPE,
#                                 check=True,
#                                 text=True)
#         metadata = json.loads(result.stdout)
#         logger.trace(f"Loaded metadata from json imported: '{metadata=}'")
#         base_features[base_features.CONTAINER_FORMAT] = \
#                                         metadata.get("format", {}).get("format_name", "unknown")
        
#         streams                                       = metadata.get("streams", [])

#         base_features[base_features.NB_STREAMS]       = len(streams)
#         logger.trace(f"Meta data: Number of audio streams: {base_features[base_features.NB_STREAMS]}.")
#         base_features[base_features.HAS_AUDIO_STREAM] = len(streams) > 0

#         if not base_features[base_features.HAS_AUDIO_STREAM]:
#             logger.trace("Meta data: No audio streams in data. This can be a problem later.")

#         try:
#             # Codec-Namen (bereits sicher)
#             logger.trace("Try to set base_features[base_features.CODEC_PER_STREAM]...")
#             base_features[base_features.CODEC_PER_STREAM] = list({
#                 stream.get("codec_name", "unknown") for stream in streams
#             })
#             logger.trace("Done.")
            
#             # Sample-Raten mit sicherer int-Konvertierung
#             logger.trace("Try to set base_features[base_features.SAMPLING_RATE_PER_STREAM]...")
#             base_features[base_features.SAMPLING_RATE_PER_STREAM] = list({
#                 safe_int_conversion(stream.get("sample_rate"), 0) for stream in streams
#             })
#             logger.trace("Done.")
            
#             # Sample-Formate mit sicherer dtype-Konvertierung
#             logger.trace("Try to set base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE]...")
#             base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE] = list({
#                 safe_get_sample_format_dtype(stream.get("sample_fmt", "s16")) for stream in streams
#             })
#             logger.trace("Done.")
            
#             # Planar-Check mit sicherer String-Behandlung
#             logger.trace("Try to set base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR]...")
#             base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR] = list({
#                 str(stream.get("sample_fmt", "s16")).endswith("p") for stream in streams
#             })
#             logger.trace("Done.")
            
#             # Kanal-Anzahl mit sicherer int-Konvertierung
#             logger.trace("Try to set base_features[base_features.CHANNELS_PER_STREAM]...")
#             base_features[base_features.CHANNELS_PER_STREAM] = list({
#                 safe_int_conversion(stream.get("channels"), 0) for stream in streams
#             })
#             logger.trace("Done.")
            
#         except Exception as e:
#             logger.warning(f"Error processing stream metadata: {e}. Using default values.")
#             # Fallback-Werte setzen
#             base_features[base_features.CODEC_PER_STREAM] = ["unknown"]
#             base_features[base_features.SAMPLING_RATE_PER_STREAM] = [0]
#             base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE] = [AudioSampleFormatMap["s16"]]
#             base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR] = [False]
#             base_features[base_features.CHANNELS_PER_STREAM] = [0]

#         logger.trace("Try to sort the codecs of streams into UNCOMPRESSED, LOSSLESS_COMPRESSED or LOSSY_COMPRESSED...")
#         base_features[base_features.CODEC_COMPRESSION_KIND_PER_STREAM] = []
#         for codec in base_features[base_features.CODEC_PER_STREAM]:
#             base_features[base_features.CODEC_COMPRESSION_KIND_PER_STREAM].append(
#                                         audio_codec_compression(codec)
#                                         )
#         logger.trace("Done.")

#         # calc hash (SHA256)
#         logger.trace("Try to calculate the hash of the file data...")
#         hasher = hashlib.sha256()
#         with file_path.open("rb") as f:
#             for chunk in iter(lambda: f.read(8192), b""):
#                 hasher.update(chunk)
#         logger.trace("Done.")
#         base_features[base_features.SH256] = hasher.hexdigest()
#         logger.trace(f"All meta data added to 'base_features': {base_features=}")

#     except Exception as e:
#         logger.error(f"Error during extraction of meta data information of file {file_path.name}: {e} – Reset metadata to the initial values of class 'AudioFileBaseFeatures'.")
#         # so reset to default
#         base_features = AudioFileBaseFeatures()
#         logger.trace(f"After error, 'base_features' are initialized with {base_features}.")

#     logger.debug(f"Base features return value of the audio file {file_path.name} read out are: {base_features}")
#     return base_features

def is_audio_in_original_audio_group(zarr_original_audio_group: zarr.Group,
                                     base_features,#: AudioFileBaseFeatures,
                                     sh246_check_only: bool = False) -> bool:
    """
    Überprüft, ob eine Audiodatei mit den angegebenen Eigenschaften bereits in der Gruppe vorhanden ist.
    
    Args:
        zarr_original_audio_group: Zarr-Gruppe mit Original-Audiodateien
        base_features: Eigenschaften der zu prüfenden Audiodatei
        sh246_check_only: Wenn True, wird nur der SHA256-Hash verglichen
        
    Returns:
        bool: True, wenn die Audiodatei bereits in der Gruppe existiert
    """
    logger.trace(f"Check if audio file {base_features.FILENAME} is already in the Zarr database of original audio files...")
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
                    logger.trace(f"Check group '{group_name}' with original file name '{bf.FILENAME}'...")
                    if (base_features.SH256) == bf["SH256"]:
                        logger.warning(f"Hash of audio file {base_features.FILENAME} found in database.")
                        if sh246_check_only:
                            logger.warning(f"Since the 'sh246_check_only' flag of 'is_audio_in_original_audio_group()' is set, we recognize the file {base_features.FILENAME} as 'is already in the database'. Probably, importing of this file will be skipped during the next steps.")
                            return True
                        elif    (base_features.FILENAME   == bf["FILENAME"]) \
                            and (base_features.SIZE_BYTES == bf["SIZE_BYTES"]):
                            logger.warning(f"Not only the hash of {base_features.FILENAME} is the same. Also filename and the size are the same values as an already imported file in the database. Probably, importing of this file will be skipped during the next steps.")
                            return True
                        else:
                            logger.warning(f"Audio file {base_features.FILENAME} has differnces in the file name or file size, but the hash is the same. So, it is not recognized as 'already found in database'. But, be carefully with this import! Maybe, the file was only renamed in the meantime between the firt import and now. It is really very, very (!) extremly seldom that the hash has the same value for different files!")
            logger.trace(f"Check of group {group_name} with original filename '{bf.FILENAME}' done. Is not identical to given audio file.")
    logger.debug(f"Audio file {base_features.FILENAME} finally not recognized as 'already imported'.")
    return False

# class FileMeta:
#     """
#     Vollständige audiorelevante Metainformationen einer Datei über ffprobe und mutagen.
    
#     Args:
#         file: Pfad zur Datei mit Audio Streams
#     """
#     def __init__(self, file: str|pathlib.Path):
#         file = pathlib.Path(file)
#         logger.trace(f"Instance of class FileMeta created. Reading meta information from media file {file.name} requested.")

#         # from ffprobe
#         # ------------
#         logger.trace("Read meta information by using 'ffprobe'...")
#         self.meta = {"ffprobe": self._ffprobe_info(file)}
#         # How many audio streams are in it?
#         nb_streams = len(self.meta["ffprobe"].get("streams", []))
#         if nb_streams == 0:
#             logger.error(f"No audio streams found in file {file.name} by using ffprobe.")
#         elif nb_streams == 1:
#             logger.debug(f"Found exactly 1 audio stream in file {file.name} by using ffprobe.")
#         else:
#             logger.warning(f"Found {nb_streams} audio streams in file {file.name} by using ffprobe. Typically, the first of them will be imported only.")

#         # via mutagen
#         # -----------
#         logger.trace("Read meta information by using 'mutagen'...")
#         self.meta["mutagen"] = self._mutagen_info(file)
#         logger.trace("Done. (Reading meta information by using 'mutagen')")

#     @classmethod
#     def _ffprobe_info(cls, file: pathlib.Path) -> dict:
#         logger.trace("Reading common infos (format, streams, chapters) of file '{file.name}' by using ffprobe requested.")
#         cmd = [
#             "ffprobe",
#             "-v", "error",
#             "-print_format", "json",
#             "-show_format",
#             "-show_streams",
#             "-show_chapters",
#             "-select_streams", "a",  # audio streams only
#             str(file)
#         ]
#         logger.trace(f"Command to read is: {cmd}")
#         result = subprocess.run(cmd,
#                                 stdout=subprocess.PIPE,
#                                 stderr=subprocess.PIPE,
#                                 check=True,
#                                 text=True)
#         info = cls._typing_numbers(json.loads(result.stdout))
#         # remove non-audio streams (but doesn't changes the index number of the audio
#         # streams if other streams are sorted out before)
#         info["streams"] = [
#                                 stream for stream in info.get("streams", [])
#                                 if stream.get("codec_type") == "audio"
#                             ]
#         logger.trace(f"Reading common info from file '{str(file)}' finished.")
#         return info

#     @staticmethod
#     def _mutagen_info(file: pathlib.Path) -> dict:
#         logger.trace(f"Reading info from file '{str(file)}' by using 'mutagen' requested...")
#         try:
#             audio = MutagenFile(str(file), easy=False)
#         except Exception as err:
#             logger.error(f"File {file.name} could not be processed by Mutagen successfully! No meta data from this method. MutagenFile() function from the mutagen modul rasies following exception: '{err}'.")
#             return {}
#         logger.trace("Reading done.")
#         if not audio:
#             logger.error(f"File '{str(file)}' could not read by 'mutagen'. No meta data from this method.")
#             return {}
#         logger.trace("Sort out technical info and tags...")
#         info = {}
#         if audio.info:
#             info['technical'] = {
#                 'length': getattr(audio.info, 'length', None),
#                 'bitrate': getattr(audio.info, 'bitrate', None),
#                 'sample_rate': getattr(audio.info, 'sample_rate', None),
#                 'channels': getattr(audio.info, 'channels', None),
#                 'codec': audio.__class__.__name__,
#             }
#         info['tags'] = {}
#         if audio.tags:
#             for key, value in audio.tags.items():
#                 try:
#                     # Bei manchen Formaten sind die Werte Listen, bei anderen nicht
#                     info['tags'][key] = value.text if hasattr(value, 'text') else str(value)
#                 except Exception:
#                     info['tags'][key] = str(value)
#         logger.trace("Done. (Sort out technical info and tags)")
#         return info
        
#     def __str__(self):
#         return self.as_yaml()
    
#     def as_yaml(self):
#         return yaml.dump(self.meta, sort_keys=False, allow_unicode=True)
    
#     def as_dict(self):
#         return self.meta
    
#     @classmethod
#     def _typing_numbers(cls, obj):
#         """Recursive typing if objects are int or float"""
#         if isinstance(obj, dict):
#             return {k: cls._typing_numbers(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [cls._typing_numbers(i) for i in obj]
#         elif isinstance(obj, str):
#             # Konvertiere Strings, die wie Zahlen aussehen
#             if obj.isdigit():
#                 return int(obj)
#             try:
#                 return float(obj)
#             except ValueError:
#                 return obj
#         else:
#             return obj


def _get_source_params(input_file: pathlib.Path) -> dict:
    """Ermittelt grundlegende Parameter einer Audiodatei mit ffprobe.
    
    Args:
        input_file: Pfad zur Audiodatei
        
    Returns:
        dict: Dictionary mit den grundlegenden Eigenschaften der Audiodatei
    """
    logger.trace(f"'_get_source_params' requested for file '{input_file.name}'...")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate:stream=codec_name:stream=sample_fmt:stream=bit_rate:stream=channels",
        "-of", "json", str(input_file)
    ]
    logger.trace(f"Command to read params with ffprobe: '{cmd}'.")
    out = subprocess.check_output(args=cmd)
    info = json.loads(out)
    if ('streams' not in info) or (len(info["streams"]) == 0):
        logger.error(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
        raise ValueError(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
    logger.trace("Reading done. Sort into the dictionary 'source_params'...")
    source_params = {
                "sampling_rate": int(info['streams'][0]['sample_rate']) if 'sample_rate' in info['streams'][0] else None,
                "sample_format": info['streams'][0]['sample_fmt'] if 'sample_fmt' in info['streams'][0] else "s16",
                "bit_rate": int(info['streams'][0]['bit_rate']) if 'bit_rate' in info['streams'][0] else None,
                "nb_channels": int(info['streams'][0]['channels']) if 'channels' in info['streams'][0] else 1
            }
    logger.debug(f"Read parameters in 'source_params' are: {source_params}")
    return source_params


def import_original_audio_file(
                    audio_file: str|pathlib.Path, 
                    zarr_original_audio_group: zarr.Group,
                    first_sample_time_stamp: datetime.datetime|None,
                    last_sample_time_stamp: datetime.datetime|None = None,
                    target_codec:str = 'flac', # 'flac' or 'aac' 
                    flac_compression_level: int = 4,
                    aac_bitrate:int = 160_000,  # <-- NEU
                    temp_dir="/tmp",
                    chunk_size:int = Config.original_audio_chunk_size
                    ):
    """
    Importiert eine Audiodatei in die Original-Audio-Gruppe und erstellt Index.
    
    SIMPLIFIED VERSION (Step 1.3): 
    Codec-specific logic has been moved to dedicated modules:
    - FLAC: flac_access.import_flac_to_zarr()
    
    This function now serves as a clean orchestrator.
    
    Args:
        audio_file: Pfad zur Audiodatei
        zarr_original_audio_group: Zarr-Gruppe für Original-Audiodateien
        first_sample_time_stamp: Zeitstempel des ersten Samples
        last_sample_time_stamp: Zeitstempel des letzten Samples (optional)
        target_codec: Zielcodec ('flac' oder 'aac')
        flac_compression_level: Kompressionsgrad für FLAC (0-12)
        temp_dir: Verzeichnis für temporäre Dateien
        chunk_size: Chunk-Größe für das Zarr-Array
    
    Raises:
        Doublet: Wenn die Datei bereits in der Datenbank vorhanden ist
        ValueError: Wenn die Datei keinen Audio-Stream enthält
        NotImplementedError: Wenn die Datei mehr als einen Audio-Stream enthält
    """
    logger.trace(f"Import of audio file '{str(audio_file)}' requested with target_codec='{target_codec}'")
    assert target_codec in ('flac', 'aac'), f"Supported codecs: 'flac', 'aac', not '{target_codec}'"
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12), f"The 'flac' compression level must be in the range of 0...12, not {flac_compression_level}."
    assert zarr_original_audio_group is not None, f"Parameter 'zarr_original_audio_group' may not be None."

    audio_file = pathlib.Path(audio_file)

    # 0) Check if group is original audio file group and has the right version
    logger.trace(f"Check if '{zarr_original_audio_group=}' is a valid Zarr group to save audio data...")
    check_if_original_audio_group(zarr_original_audio_group)
    logger.trace("Check done.")

    # 1) Analyze File-/Audio-/Codec-Type
    logger.trace("Check for audio source base features and additional parameters.")
    base_features = base_features_from_audio_file(audio_file)
    source_params = _get_source_params(audio_file)
    logger.trace(f"Done. Found {base_features=} and {source_params=} in file {audio_file.name}.")

    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{audio_file}' has no audio stream")
    if base_features[base_features.NB_STREAMS] > 1:
        raise NotImplementedError("Audio file has more than one audio stream. Import of more than one audio streams (with any number of audio channels) is not yet supported.")

    # 2) Is file yet present in database?
    logger.trace(f"Check if {audio_file.name} was already imported. Use for this '{base_features=}' ...")
    if is_audio_in_original_audio_group(zarr_original_audio_group, base_features):
        raise Doublet("Same file seems yet in database. Same hash, same file name same size.")
    logger.trace("Check done. Not already imported.")

    # 3) Get the complete meta data inside the file.
    logger.trace(f"Read out all meta data from {audio_file.name}...")
    all_file_meta = str(FileMeta(audio_file))
    logger.trace(f"Done. {all_file_meta=}")

    # 4) Calculate the next free 'group-number'
    logger.trace("Calculate next free group number in Zarr for the new audio data...")
    new_audio_group_name = next_numeric_group_name(zarr_group=zarr_original_audio_group)
    logger.trace(f"Done. {new_audio_group_name=}")

    # 5) Create numeric group for the new audio
    logger.trace("Start import process now.")
    created = False
    try:
        logger.trace(f"Try to create Zarr group '{new_audio_group_name}'...")
        new_original_audio_grp = zarr_original_audio_group.require_group(new_audio_group_name)
        created = True
        logger.trace("Created.")

        # add a version attribute to this group
        logger.trace(f"Try to set group attribute original_audio_data_array_version:={Config.original_audio_data_array_version}")
        new_original_audio_grp.attrs["original_audio_data_array_version"] = Config.original_audio_data_array_version
        logger.trace("Done.")

        # Save Original-Audio-Meta data to the group
        logger.trace(f"Try to save some additional attributes (type='original_audio_file', encoding={target_codec}, base_features and meta_data_structure.")
        logger.trace("Set attribute new_original_audio_grp.attrs['type'] := 'original_audio_file' (thats the standard type in order to identify this group as memory for an imported audio file.")
        new_original_audio_grp.attrs["type"]                    = "original_audio_file"
        logger.trace(f"Done. Set now 'new_original_audio_grp.attrs['encoding']' := {target_codec}.")
        new_original_audio_grp.attrs["encoding"]                = target_codec
        logger.trace(f"Done. Set now 'new_original_audio_grp.attrs['base_features']' := {base_features}.")
        new_original_audio_grp.attrs["base_features"]           = base_features
        logger.trace(f"Done. Set now 'new_original_audio_grp.attrs['meta_data_structure']' := {all_file_meta}.")
        new_original_audio_grp.attrs["meta_data_structure"]     = all_file_meta
        logger.trace("Done.")

        # 6) SIMPLIFIED ORCHESTRATOR (Step 1.3): Use dedicated codec modules
        logger.trace(f"Starting import using {target_codec} module...")
        
        if target_codec == 'flac':
            # Use FLAC access module
            logger.trace("Target codec is 'flac'. Using flac_access module for import...")
            audio_blob_array = import_flac_to_zarr(
                zarr_group=new_original_audio_grp,
                audio_file=audio_file,
                source_params=source_params,
                first_sample_time_stamp=first_sample_time_stamp,
                flac_compression_level=flac_compression_level,
                temp_dir=temp_dir
            )
            logger.trace("FLAC import completed via flac_access module.")
            
        elif target_codec == 'aac':
            # Use AAC access module (NEW)
            logger.trace("Target codec is 'aac'. Using aac_access module for import...")
            audio_blob_array = import_aac_to_zarr(
                zarr_group=new_original_audio_grp,
                audio_file=audio_file,
                source_params=source_params,
                first_sample_time_stamp=first_sample_time_stamp,
                aac_bitrate=aac_bitrate,
                temp_dir=temp_dir
            )
            logger.trace("AAC import completed via aac_access module.")
      
        else:
            raise ValueError(f"Unsupported target codec: {target_codec}")



    except Exception as err:
        logger.error(f"An error happens during importing audio file. Error: {err}")
        # Full-Rollback: In case of any exception: we remove the group with all content.
        if created:
            logger.trace("Since the Zarr group was already created until this exception, try to remove the group now...")
            remove_zarr_group_recursive(zarr_original_audio_group.store, new_original_audio_grp.path)
            logger.trace("Removing done.")
        raise  # raise original exception
    
    logger.success(f"Audio data from file '{audio_file.name}' completely imported into Zarr group '{new_audio_group_name}' in '{str(zarr_original_audio_group.store_path)}'. Index for 'random access read' created.")

# Erweitere import_original_audio_file() um Config-Integration:
def import_original_audio_file_with_config(
                    audio_file: str|pathlib.Path, 
                    zarr_original_audio_group: zarr.Group,
                    first_sample_time_stamp: datetime.datetime|None,
                    target_codec:str = None,  # None = use Config default
                    **kwargs):
    """
    Enhanced import with automatic config integration
    """
    # Use config defaults if not specified
    if target_codec is None:
        target_codec = 'aac'  # New default
    
    # Get AAC config if using AAC
    if target_codec == 'aac':
        aac_config = _get_aac_config_for_import()
        if 'aac_bitrate' not in kwargs:
            kwargs['aac_bitrate'] = aac_config['bitrate']
    
    # Call original function with enhanced parameters
    return import_original_audio_file(
        audio_file=audio_file,
        zarr_original_audio_group=zarr_original_audio_group,
        first_sample_time_stamp=first_sample_time_stamp,
        target_codec=target_codec,
        **kwargs
    )

def validate_aac_import_parameters(aac_bitrate: int, source_params: dict) -> dict:
    """
    Validate and optimize AAC import parameters
    
    Args:
        aac_bitrate: Requested bitrate
        source_params: Source audio parameters
        
    Returns:
        Validated and optimized parameters
    """
    from .config import Config
    
    # Validate bitrate range
    min_bitrate = 32000
    max_bitrate = 320000
    
    if aac_bitrate < min_bitrate:
        logger.warning(f"AAC bitrate {aac_bitrate} too low, using minimum {min_bitrate}")
        aac_bitrate = min_bitrate
    elif aac_bitrate > max_bitrate:
        logger.warning(f"AAC bitrate {aac_bitrate} too high, using maximum {max_bitrate}")
        aac_bitrate = max_bitrate
    
    # Optimize bitrate based on channels
    channels = source_params.get("nb_channels", 2)
    if channels == 1:  # Mono
        recommended_max = 128000
        if aac_bitrate > recommended_max:
            logger.info(f"Reducing AAC bitrate from {aac_bitrate} to {recommended_max} for mono audio")
            aac_bitrate = recommended_max
    
    # Check sample rate compatibility
    sample_rate = source_params.get("sampling_rate", 48000)
    supported_rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000]
    
    if sample_rate not in supported_rates:
        closest_rate = min(supported_rates, key=lambda x: abs(x - sample_rate))
        logger.warning(f"Sample rate {sample_rate}Hz not optimal for AAC, closest supported: {closest_rate}Hz")
    
    return {
        'validated_bitrate': aac_bitrate,
        'channels': channels,
        'sample_rate': sample_rate,
        'use_pyav': Config.aac_enable_pyav_native,
        'fallback_ffmpeg': Config.aac_fallback_to_ffmpeg
    }



def extract_audio_segment(zarr_group, start_sample, end_sample, dtype=np.int16):
    """
    Extrahiert ein Audiosegment aus einer Zarr-Gruppe, unabhängig vom verwendeten Codec.
    
    ENHANCED VERSION (Step 2.0): Added AAC-LC support with format auto-detection.
    
    Args:
        zarr_group: Zarr-Gruppe mit den Audiodaten und dem Index
        start_sample: Erstes Sample, das extrahiert werden soll
        end_sample: Letztes Sample, das extrahiert werden soll
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        
    Returns:
        np.ndarray: Extrahiertes Audiosegment
    """  
    if AUDIO_DATA_BLOB_ARRAY_NAME not in zarr_group:
        raise ValueError("No audio data found in zarr_group")
    
    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')

    if codec == 'flac':
        return extract_audio_segment_flac(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    elif codec == 'aac':
        return extract_audio_segment_aac(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def parallel_extract_audio_segments(zarr_group, segments, dtype=np.int16, max_workers=4):
    """
    Extrahiert mehrere Audiosegmente parallel aus einer Zarr-Gruppe.
    
    ENHANCED VERSION (Step 2.0): Added AAC-LC support with format auto-detection.
    """  
    if AUDIO_DATA_BLOB_ARRAY_NAME not in zarr_group:
        raise ValueError("No audio data found in zarr_group")

    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')
    
    if codec == 'flac':
        if audio_blob_array is None:
            raise ValueError("FLAC requires legacy audio_data_blob_array format")
        return parallel_extract_audio_segments_flac(zarr_group, audio_blob_array, segments, dtype, max_workers)
        
    elif codec == 'aac':
        if audio_blob_array is None:
            raise ValueError("AAC requires legacy audio_data_blob_array format")
        return parallel_extract_audio_segments_aac(zarr_group, audio_blob_array, segments, dtype, max_workers)
    else:
        raise ValueError(f"Nicht unterstützter Codec: {codec}")

   
def _get_aac_config_for_import():
    """Get AAC configuration parameters for import operations"""
    from .config import Config
    return {
        'bitrate': Config.aac_default_bitrate,
        'use_pyav': Config.aac_enable_pyav_native,
        'fallback_ffmpeg': Config.aac_fallback_to_ffmpeg,
        'quality_preset': Config.aac_quality_preset,
        'memory_limit': Config.aac_memory_limit_mb
    }


def _log_import_performance(start_time: float, audio_file: pathlib.Path, 
                           target_codec: str, **stats):
    """Log performance metrics for import operations"""
    import_time = time.time() - start_time
    file_size_mb = audio_file.stat().st_size / 1024 / 1024
    
    logger.success(
        f"{target_codec.upper()} import completed: "
        f"{audio_file.name} ({file_size_mb:.1f}MB) in {import_time:.2f}s "
        f"({file_size_mb/import_time:.1f} MB/s)"
    )
    
    if 'compression_ratio' in stats:
        logger.info(f"Compression achieved: {stats['compression_ratio']:.1f}x reduction")

def test_aac_integration():
    """
    Quick integration test for AAC functionality
    This can be called during development to verify AAC integration
    """
    try:
        from .config import Config
        from .aac_access import import_aac_to_zarr
        from .aac_index_backend import build_aac_index
        
        logger.info("AAC integration test started...")
        
        # Test configuration
        original_bitrate = Config.aac_default_bitrate
        Config.set(aac_default_bitrate=128000)
        assert Config.aac_default_bitrate == 128000
        Config.set(aac_default_bitrate=original_bitrate)
        
        logger.success("AAC configuration test passed")
        
        # Test imports
        assert import_aac_to_zarr is not None
        assert build_aac_index is not None
        
        logger.success("AAC module imports test passed")
        logger.success("AAC integration test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"AAC integration test failed: {e}")
        return False

# We do this check during import to brake the program if ffmpeg is missing as soon as possible.
check_ffmpeg_tools()

logger.debug("Module loaded.")
