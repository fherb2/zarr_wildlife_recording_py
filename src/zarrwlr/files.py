"""File path types and helpers for Zarr Wildlife Recording library"""

import logging
from typing import Union
from pathlib import Path
import hashlib
import os
import datetime
import subprocess
import json
from zarrwlr.zarropus import is_opus_file

try:
    import mutagen
    MUTAGEN_LOADED = True
except ImportError:
    MUTAGEN_LOADED = False

# get the module logger   
logger = logging.getLogger(__name__)


StrOrPath = Union[str, Path]
ListOrTuple_of_StrOrPath = Union[list[StrOrPath], tuple[StrOrPath, ...]]
SingleOrList_of_FilePaths = Union[StrOrPath, ListOrTuple_of_StrOrPath]

class InvalidFilePathError(ValueError):
    """Value can not be used as valid file path."""
    def __init__(self, message: str = "File path not valid as single path (str|Path) or list of paths ( list|tuple of str|Path )."):
        super().__init__(message)

def is_allowed_sound_file_type(file: Path) -> bool:
    if is_opus_file(file):
        return True
    # add additional allowed file types
    return False

def to_file_path_list(files: SingleOrList_of_FilePaths) -> list[Path]:

    if isinstance(files, (str, Path)):
        files = [Path(files)]
    elif isinstance(files, (list, tuple)):
        files = [Path(item) for item in files]
    else:
        raise InvalidFilePathError
    return files


def file_content_hash(file: StrOrPath) -> tuple[str, str]:

    if isinstance(file, str):
        file = Path(file)

    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while chunk := f.read(4096):  # read in blocks to stay safe.
            sha256.update(chunk)

    return sha256.hexdigest(), 


def filemeta(file:str|Path) -> dict:
    f = Path(file) # to be a pathlib.Path object for following access

    # initialize detailed info dicts
    timestamps = {}
    gen_fileinfo = {}
    ffprobe = {}

    # ###########################
    # os.statinfo
    si = os.stat(file)

    # most_recent_content_modification
    if si.st_mtime_ns:
        timestamps['most_recent_content_modification'] = datetime.datetime.fromtimestamp(si.st_mtime_ns / 1e9)
    elif si.st_mtime:
        timestamps['most_recent_content_modification'] = datetime.datetime.fromtimestamp(si.st_mtime)
    else:
        timestamps['most_recent_content_modification'] = None

    # most_recent_access
    if si.st_atime_ns:
        timestamps['most_recent_access'] = datetime.datetime.fromtimestamp(si.st_atime_ns / 1e9)
    elif si.st_atime:
        timestamps['most_recent_access'] = datetime.datetime.fromtimestamp(si.st_atime)
    else:
        timestamps['most_recent_access'] = None    

    # most_recent_metadata_change
    if si.st_atime_ns:
        timestamps['most_recent_metadata_change'] = datetime.datetime.fromtimestamp(si.st_ctime_ns / 1e9)
    elif si.st_atime:
        timestamps['most_recent_metadata_change'] = datetime.datetime.fromtimestamp(si.st_ctime)
    else:
        timestamps['most_recent_metadata_change'] = None    

    # file_create
    timestamps['file_create'] = None
    try:
        if si.st_birthtime:
            timestamps['file_create'] = datetime.datetime.fromtimestamp(si.st_birthtime)
    except:
        pass
    try:
        if si.st_birthtime_ns:
            timestamps['file_create'] = datetime.datetime.fromtimestamp(si.st_birthtime_ns)
    except:
        pass

    # file_type_and_mode_bits
    if si.st_mode:
        gen_fileinfo['file_type_and_mode_bits'] = si.st_mode
    else:
        gen_fileinfo['file_type_and_mode_bits'] = None

    # size_byte
    if si.st_size:
        gen_fileinfo['size_byte'] = si.st_size
    else:
        gen_fileinfo['size_byte'] = None
    # ending: os.statinfo
    # ###########################

    # ###########################
    # pathlib
    #
    # path_abs, file_name, file_stem, file_suffix
    gen_fileinfo['path_abs'] = f.absolute()
    gen_fileinfo['file_name'] = f.name
    gen_fileinfo['file_stem'] = f.stem
    gen_fileinfo['file_suffix'] = f.suffix
    # ###########################

    # ###########################
    # mutagen
    if MUTAGEN_LOADED:
        mutgn_info = {}
        mutgn_tags = {}
        mutg = mutagen.File(f, easy=True)

        # attrs from .info
        if mutg.info:
            info_attrs = sorted(attr for attr in dir(mutg.info) if not attr.startswith('_'))
            for attr in info_attrs:
                mutgn_info[f"{attr}"] = getattr(mutg.info, attr, None)

        # attrs from .tags
        if mutg.tags:
            tags_attrs = sorted(attr for attr in dir(mutg.tags) if not attr.startswith('_'))
            for attr in tags_attrs:
                value = getattr(mutg.tags, attr, None)
                if callable(value):
                    continue  # Methoden auslassen
                mutgn_tags[f"{attr}"] = value
    # ###########################

    # ###########################
    # ffprope
    try:
        ffprope_results = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format:stream_tags",
            "-show_format", "-show_streams",
            "-print_format", "json",
            f
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffprobe = json.loads(ffprope_results.stdout)
    except:
        ffprobe = None
        logger.warning("'ffprobe' could not be found: Some of sound file meta information can not be recognized.")
    # ###########################

    # ###########################
    # collect detailed information as dict
    meta = {}
    meta['general_fileinfo'] = gen_fileinfo
    meta['file_timestamps'] = timestamps
    meta['mutgn_info'] = mutgn_info
    meta['mutgn_tags'] = mutgn_tags
    meta['ffprobe'] = ffprobe

    # return all collected meta information
    return meta





