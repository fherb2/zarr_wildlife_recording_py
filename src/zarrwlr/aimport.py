

import numpy as np
import subprocess
import pathlib
import hashlib
import json
import datetime
import yaml
import tempfile
from mutagen import File as MutagenFile
import zarr

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.debug("Module loading...")


# Import the functions from flacbyteblob and opusbyteblob
from .flacbyteblob import (  # noqa: E402
    build_flac_index, 
    extract_audio_segment_flac, 
    parallel_extract_audio_segments_flac
)
from .opusbyteblob import (  # noqa: E402
    build_opus_index, 
    extract_audio_segment_opus, 
    parallel_extract_audio_segments_opus
)
from .utils import next_numeric_group_name, remove_zarr_group_recursive  # noqa: E402
from .config import Config  # noqa: E402
from .exceptions import Doublet, ZarrComponentIncomplete, ZarrComponentVersionError, ZarrGroupMismatch, OggImportError  # noqa: E402
from .packagetypes import AudioFileBaseFeatures, AudioCompression, AudioSampleFormatMap  # noqa: E402

STD_TEMP_DIR = "/tmp"
AUDIO_DATA_BLOB_ARRAY_NAME = "audio_data_blob_array" 

def _check_ffmpeg_tools():
    """Check if ffmpeg and ffprobe are installed and callable."""
    tools = ["ffmpeg", "ffprobe"]
    logger.debug("Check avalability of ffmpeg and ffprobe tools during import of module...")

    for tool in tools:
        try:
            subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"Missing Command line tool {tool}. Please install ist.")
            exit(1)
    logger.info("ffmpeg and ffprobe tools: Installed and successfully checked: Ok.")




def create_original_audio_group(store_path: str|pathlib.Path, group_path: str|pathlib.Path|None = None):
    """
    Erstellt eine neue Original-Audio-Gruppe im Zarr-Store oder prüft eine existierende.
    
    Args:
        store_path: Pfad zum Zarr-Store
        group_path: Pfad zur Gruppe innerhalb des Stores (optional)
    """
    store_path = pathlib.Path(store_path).resolve()  # Konvertiere zu absolutem Pfad-String
    
    # Zarr-Pfade innerhalb des Stores müssen als Strings vorliegen
    zarr_group_path = None
    if group_path is not None:
        zarr_group_path = str(group_path)  # Zarr erwartet String-Pfade
    
    logger.debug(f"'create_original_audio_group' called. Try to open store {store_path=} and audio group path {group_path=}...")
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store, mode='a')
    
    def initialize_group(grp: zarr.Group):
        logger.debug("Initialize Zarr group containing all original audio imports...")
        grp.attrs["magic_id"] = Config.original_audio_group_magic_id
        grp.attrs["version"] = Config.original_audio_group_version
        logger.debug(f"Initializing done. {grp.attrs['magic_id']=} and {grp.attrs['version']}")
    
    if zarr_group_path is not None:
        if zarr_group_path in root:
            # group exist
            logger.debug(f"Zarr group {zarr_group_path} exist. Check if it is a valid 'original audio' group...")
            grp = root[zarr_group_path]
            assert isinstance(grp, zarr.Group), f"Expected zarr.Group, got {type(grp)=}"
            check_if_original_audio_group(group=grp)
            logger.debug(f"Zarr group {zarr_group_path} is a valid 'original audio' group.")
            return
        else:
            # create this group
            logger.debug(f"Zarr group {zarr_group_path} doesn't exist. Try to create...")
            created = False
            try:
                group = root.create_group(zarr_group_path)
                created = True
                initialize_group(group)
            except Exception:
                if created:
                    remove_zarr_group_recursive(root.store, group.path)
                raise  # raise original exception
            logger.info(f"Zarr group {zarr_group_path} as 'original audio' group created.")
    else:
        # root is this group
        logger.debug(f"Zarr root {store_path} is given as 'original audio' group. Check if it is a valid 'original audio' group...")
        check_if_original_audio_group(group=root)
        logger.debug(f"Zarr group {zarr_group_path} is a valid 'original audio' group.")


def check_if_original_audio_group(group:zarr.Group):
    """
    Überprüft, ob eine Gruppe eine gültige Original-Audio-Gruppe ist.
    
    Args:
        group: Zu prüfende Zarr-Gruppe
        
    Raises:
        ZarrComponentVersionError: Wenn die Gruppenversion nicht mit der erwarteten Version übereinstimmt
        ZarrGroupMismatch: Wenn die Gruppe keine Original-Audio-Gruppe ist
    """
    # no logging here: will be done at calling positions
    grp_ok =     ("magic_id" in group.attrs) \
             and (group.attrs["magic_id"] == Config.original_audio_group_magic_id) \
             and ("version" in group.attrs)
    if grp_ok and not (group.attrs["version"] == Config.original_audio_group_version):
        raise ZarrComponentVersionError(f"Original audio group has version {group.attrs['version']} but current zarrwlr needs version {Config.original_audio_group_version} for this group. Please, upgrade group to get access.")
    elif grp_ok:
        return
    raise ZarrGroupMismatch(f"The group '{group}' is not an original audio group.")

def audio_codec_compression(codec_name: str) -> AudioCompression:
    """
    Erkennt den Kompressionstyp anhand des Codec-Namens.
    
    Args:
        codec_name: Name des Audio-Codecs
        
    Returns:
        AudioCompression: Kompressionstyp (UNCOMPRESSED, LOSSLESS_COMPRESSED, LOSSY_COMPRESSED, UNKNOWN)
    """
    if codec_name.startswith("pcm_"):
        logger.debug(f"Audio compression of {codec_name} checked. Is: uncompressed.")
        return AudioCompression.UNCOMPRESSED
    
    lossless_codecs = {"flac", "alac", "wavpack", "ape", "tak"}
    if codec_name in lossless_codecs:
        logger.debug(f"Audio compression of {codec_name} checked. Is: lossless compressed.")
        return AudioCompression.LOSSLESS_COMPRESSED
    
    lossy_codecs = {"mp3", "aac", "opus", "ogg", "ac3", "eac3", "wma"}
    if codec_name in lossy_codecs:
        logger.debug(f"Audio compression of {codec_name} checked. Is: lossy compressed.")
        return AudioCompression.LOSSY_COMPRESSED
    
    logger.error(f"Audio compression of {codec_name} checked. Is not an implemented code to recognize the type. If necessary to recognize this code, it has to be added to the module function 'audio_codec_compression()'.")
    return AudioCompression.UNKNOWN

def base_features_from_audio_file(file: str|pathlib.Path) -> AudioFileBaseFeatures:
    """
    Ermittelt grundlegende Informationen über eine Audiodatei.
    
    Args:
        file: Pfad zur Audiodatei
        
    Returns:
        AudioFileBaseFeatures: Objekt mit Basisinformationen über die Audiodatei
        
    Raises:
        FileNotFoundError: Wenn die Datei nicht gefunden wird
    """
    file_path = pathlib.Path(file)

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
            "-select_streams", "a",  # Wählt alle Audiostreams aus
            "-show_entries", "format=format_name",  # Format-Information
            "-show_entries", "stream=index,codec_name,sample_rate,sample_fmt,channels",  # Stream-Information mit index
            "-of", "json",
            str(file_path)
        ]
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True,
                                text=True)
        metadata = json.loads(result.stdout)
        base_features[base_features.CONTAINER_FORMAT] = \
                                        metadata.get("format", {}).get("format_name", "unknown")
        streams                                       = metadata.get("streams", [])
        base_features[base_features.NB_STREAMS]       = len(streams)
        base_features[base_features.HAS_AUDIO_STREAM] = len(streams) > 0
        base_features[base_features.CODEC_PER_STREAM] = list({stream.get("codec_name", "unknown") for stream in streams})
        base_features[base_features.SAMPLING_RATE_PER_STREAM] = list({int(stream.get("sample_rate", 0)) for stream in streams})
        base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE] = list({AudioSampleFormatMap.get(stream.get("sample_fmt", "s16")) for stream in streams})
        base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR] = list({stream.get("sample_fmt", "s16").endswith("p") for stream in streams})
        base_features[base_features.CHANNELS_PER_STREAM] = list({int(stream.get("channels", 0)) for stream in streams})

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
        logger.error(f"Fehler beim Extrahieren der Audiodatei-Eigenschaften: {e}")
        # so reset to default
        base_features = AudioFileBaseFeatures()

# verbesserte Version könnte so aussehen:
# try:
#     result = subprocess.run(cmd,
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE,
#                             check=True,
#                             text=True,
#                             timeout=30)  # Timeout hinzufügen
    
#     metadata = json.loads(result.stdout)
    
#     # Format-Informationen
#     base_features[base_features.CONTAINER_FORMAT] = metadata.get("format", {}).get("format_name", "unknown")
    
#     # Stream-Informationen
#     streams = metadata.get("streams", [])
#     base_features[base_features.NB_STREAMS] = len(streams)
#     base_features[base_features.HAS_AUDIO_STREAM] = len(streams) > 0
    
#     # Sicheres Extrahieren von Stream-Eigenschaften
#     codec_names = []
#     sample_rates = []
#     sample_formats_as_dtype = []
#     sample_formats_is_planar = []
#     channels_per_stream = []
    
#     for stream in streams:
#         # Codec-Name
#         codec_names.append(stream.get("codec_name", "unknown"))
        
#         # Sample Rate
#         try:
#             sample_rates.append(int(stream.get("sample_rate", 0)))
#         except (ValueError, TypeError):
#             sample_rates.append(0)
        
#         # Sample Format
#         sample_fmt = stream.get("sample_fmt", "s16")
#         try:
#             sample_formats_as_dtype.append(AudioSampleFormatMap.get(sample_fmt, AudioSampleFormatMap["s16"]))
#         except KeyError:
#             sample_formats_as_dtype.append(AudioSampleFormatMap["s16"])  # Fallback
        
#         sample_formats_is_planar.append(sample_fmt.endswith("p"))
        
#         # Channels
#         try:
#             channels_per_stream.append(int(stream.get("channels", 0)))
#         except (ValueError, TypeError):
#             channels_per_stream.append(0)
    
#     # Einzigartige Werte speichern, falls gewünscht
#     base_features[base_features.CODEC_PER_STREAM] = list(set(codec_names))
#     base_features[base_features.SAMPLING_RATE_PER_STREAM] = list(set(sample_rates))
#     base_features[base_features.SAMPLE_FORMAT_PER_STREAM_AS_DTYPE] = list(set(sample_formats_as_dtype))
#     base_features[base_features.SAMPLE_FORMAT_PER_STREAM_IS_PLANAR] = list(set(sample_formats_is_planar))
#     base_features[base_features.CHANNELS_PER_STREAM] = list(set(channels_per_stream))
    
#     # Optional: Stream-Zuordnung beibehalten
#     base_features[base_features.STREAMS] = [{
#         "index": stream.get("index"),
#         "codec_name": stream.get("codec_name", "unknown"),
#         "sample_rate": sample_rates[i],
#         "sample_fmt": stream.get("sample_fmt", "s16"),
#         "channels": channels_per_stream[i]
#     } for i, stream in enumerate(streams)]
    
# except subprocess.TimeoutExpired:
#     logger.error(f"Timeout beim Ausführen von ffprobe für {file_path}")
#     # Fallback-Werte setzen
    
# except subprocess.SubprocessError as e:
#     logger.error(f"Fehler beim Ausführen von ffprobe: {e}")
#     # Fallback-Werte setzen
    
# except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
#     logger.error(f"Fehler beim Verarbeiten der ffprobe-Ausgabe: {e}")
#     # Fallback-Werte setzen

    logger.debug(f"Base features of the audio file {file_path.name} read out: {base_features}")
    return base_features

def is_audio_in_original_audio_group(zarr_original_audio_group: zarr.Group,
                                     base_features: AudioFileBaseFeatures,
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
    logger.debug(f"Check if audio file {base_features.FILENAME} is already in the Zarr database of original audio files...")
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
    logger.debug(f"Audio file {base_features.FILENAME} finally not recognized as 'already imported'.")
    return False

class FileMeta:
    """
    Vollständige audiorelevante Metainformationen einer Datei über ffprobe und mutagen.
    
    Args:
        file: Pfad zur Datei mit Audio Streams
    """
    def __init__(self, file: str|pathlib.Path):
        file = pathlib.Path(file)
        logger.debug(f"Reading meta information from media file {file.name} requested.")
        # from ffprobe
        # ------------
        self.meta = {"ffprobe": self._ffprobe_info(file)}
        if len(self.meta["ffprobe"].get("streams", [])) == 0:
            logger.warning(f"No audio streams found by ffprobe in file {file.name}.")
        else:
            logger.debug(f"Found {len(self.meta["ffprobe"].get("streams", []))} audio streams found by ffprobe in file {file.name}.")
        # via mutagen
        # -----------
        self.meta["mutagen"] = self._mutagen_info(file)

    @classmethod
    def _ffprobe_info(cls, file: pathlib.Path) -> dict:
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
    def _mutagen_info(file: pathlib.Path) -> dict:
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



def _get_source_params(input_file: pathlib.Path) -> dict:
    """
    Ermittelt grundlegende Parameter einer Audiodatei mit ffprobe.
    
    Args:
        input_file: Pfad zur Audiodatei
        
    Returns:
        dict: Dictionary mit den grundlegenden Eigenschaften der Audiodatei
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate:stream=codec_name:stream=sample_fmt:stream=bit_rate:stream=channels",
        "-of", "json", str(input_file)
    ]
    out = subprocess.check_output(args=cmd)
    info = json.loads(out)
    if ('streams' not in info) or (len(info["streams"]) == 0):
        logger.error(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
        raise ValueError(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
    source_params = {
                "sampling_rate": int(info['streams'][0]['sample_rate']) if 'sample_rate' in info['streams'][0] else None,
                "is_opus": info['streams'][0]['codec_name'] == "opus",
                "sample_format": info['streams'][0]['sample_fmt'] if 'sample_fmt' in info['streams'][0] else "s16",
                "bit_rate": int(info['streams'][0]['bit_rate']) if 'bit_rate' in info['streams'][0] else None,
                "nb_channels": int(info['streams'][0]['channels']) if 'channels' in info['streams'][0] else 1
            }
    return source_params


def _get_ffmpeg_sample_fmt(source_sample_fmt: str, target_codec: str) -> str:
    """
    Gibt das beste sample_fmt-Argument für ffmpeg zurück, basierend auf Quellformat und Zielcodec.
    
    Args:
        source_sample_fmt: Sample-Format der Quelle laut ffprobe, z.B. "fltp", "s16", "s32", "flt"
        target_codec: "flac" oder "opus"
    
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
            retval = normalized_fmt
        else:
            # fallback
            retval = "s32"
    elif target_codec == "opus":
        # Opus arbeitet intern mit 16-bit, akzeptiert aber auch float input
        if normalized_fmt in opus_supported:
            retval = normalized_fmt
        # fallback auf float oder s16
        else:
            retval =  "flt" if normalized_fmt.startswith("f") else "s16"
    else:
        logger.error(f"Could not find any convertion value for '{source_sample_fmt}' as source sample format (function _get_ffmpeg_sample_fmt() ).")
        raise NotImplementedError(f"Unsupported codec: {target_codec}")
    logger.debug(f"Could find the sample format '{retval}' as convertion from '{source_sample_fmt}' as source sample format (function _get_ffmpeg_sample_fmt() ).")
    return retval


def import_original_audio_file(
                    audio_file: str|pathlib.Path, 
                    zarr_original_audio_group: zarr.Group,
                    first_sample_time_stamp: datetime.datetime|None,
                    last_sample_time_stamp: datetime.datetime|None = None,
                    target_codec:str = 'flac', # 'flac' or 'opus'
                    flac_compression_level: int = 4, # 0...12; 4 ist ein guter Kompromiss
                    opus_bitrate:int = 160_000, # bit per second
                    temp_dir="/tmp",
                    chunk_size:int = Config.original_audio_chunk_size
                    ):
    """
    Importiert eine Audiodatei in die Original-Audio-Gruppe und erstellt Index.
    
    Arbeitsablauf:
    1. Analysiere Datei-/Audio-/Codec-Typ
    2. Überprüfe, ob die Datei bereits in der Datenbank vorhanden ist
    3. Erfasse alle Metadaten der Datei
    4. Berechne die nächste freie Gruppen-Nummer
    5. Erstelle die Gruppe, dekodiere die Datei und importiere sie je nach Quell- und Zielcodec
    6. Erstelle den Index und füge ihn als Array hinzu
    7. Füge alle Metadaten als Attribute hinzu
    
    Args:
        audio_file: Pfad zur Audiodatei
        zarr_original_audio_group: Zarr-Gruppe für Original-Audiodateien
        first_sample_time_stamp: Zeitstempel des ersten Samples
        last_sample_time_stamp: Zeitstempel des letzten Samples (optional)
        target_codec: Zielcodec ('flac' oder 'opus')
        flac_compression_level: Kompressionsgrad für FLAC (0-12)
        opus_bitrate: Bitrate für Opus (bit/s)
        temp_dir: Verzeichnis für temporäre Dateien
        chunk_size: Chunk-Größe für das Zarr-Array
    
    Raises:
        Doublet: Wenn die Datei bereits in der Datenbank vorhanden ist
        ValueError: Wenn die Datei keinen Audio-Stream enthält
        NotImplementedError: Wenn die Datei mehr als einen Audio-Stream enthält
    """
    assert target_codec in ('flac', 'opus') 
    assert (flac_compression_level >= 0) and (flac_compression_level <= 12)

    audio_file = pathlib.Path(audio_file)

    # 0) Check if group is original audio file group and has the right version
    check_if_original_audio_group(zarr_original_audio_group)

    # 1) Analyze File-/Audio-/Codec-Type
    base_features = base_features_from_audio_file(audio_file)
    source_params = _get_source_params(audio_file)

    if not base_features[base_features.HAS_AUDIO_STREAM]:
        raise ValueError(f"File '{audio_file}' has no audio stream")
    if base_features[base_features.NB_STREAMS] > 1:
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
        target_sample_rate = source_params["sampling_rate"]
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
                sampling_rescale_factor = float(source_params["sampling_rate"]) / 48000.0
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
            attrs["opus_bitrate"] = opus_bitrate
        else: # target_codec == "flac"
            attrs["compression_level"] = flac_compression_level
        audio_blob_array.attrs.update(attrs)
        
        # 7) Create and save index inside the group (as array, not attribute since the size of structured data)
        # Build index for the appropriate codec - this is the key change from the original code
        if target_codec == 'opus':
            # Using imported build_opus_index function instead of OggOpusIndexer
            build_opus_index(new_original_audio_grp, audio_blob_array)
        else:  # target_codec == 'flac'
            # Using imported build_flac_index function instead of FLACIndexer
            build_flac_index(new_original_audio_grp, audio_blob_array)

    except Exception as e:
        # Full-Rollback: In case of any exception: we remove the group with all content.
        if created:
            remove_zarr_group_recursive(zarr_original_audio_group.store, new_original_audio_grp.path)
        logger.error(f"Error during audio import: {e}")
        raise  # raise original exception


def extract_audio_segment(zarr_group, start_sample, end_sample, dtype=np.int16):
    """
    Extrahiert ein Audiosegment aus einer Zarr-Gruppe, unabhängig vom verwendeten Codec.
    
    Args:
        zarr_group: Zarr-Gruppe mit den Audiodaten und dem Index
        start_sample: Erstes Sample, das extrahiert werden soll
        end_sample: Letztes Sample, das extrahiert werden soll
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        
    Returns:
        np.ndarray: Extrahiertes Audiosegment
        
    Raises:
        ValueError: Wenn der Codec nicht unterstützt wird
    """
    # Codec aus den Attributen holen
    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')
    
    if codec == 'flac':
        return extract_audio_segment_flac(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    elif codec == 'opus':
        return extract_audio_segment_opus(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    else:
        raise ValueError(f"Nicht unterstützter Codec: {codec}")


def parallel_extract_audio_segments(zarr_group, segments, dtype=np.int16, max_workers=4):
    """
    Extrahiert mehrere Audiosegmente parallel aus einer Zarr-Gruppe.
    
    Args:
        zarr_group: Zarr-Gruppe mit den Audiodaten und dem Index
        segments: Liste von (start_sample, end_sample)-Tupeln
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        max_workers: Maximale Anzahl paralleler Worker
        
    Returns:
        Liste von np.ndarray: Extrahierte Audiosegmente
    """
    # Codec aus den Attributen holen
    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')
    
    if codec == 'flac':
        return parallel_extract_audio_segments_flac(zarr_group, audio_blob_array, segments, dtype, max_workers)
    elif codec == 'opus':
        return parallel_extract_audio_segments_opus(zarr_group, audio_blob_array, segments, dtype, max_workers)
    else:
        raise ValueError(f"Nicht unterstützter Codec: {codec}")
    

logger.debug("Module loaded.")
# We do this check during import:
_check_ffmpeg_tools()
