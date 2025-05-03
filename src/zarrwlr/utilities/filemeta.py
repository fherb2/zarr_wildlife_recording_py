"""Get meta information from (sound)files
"""
from pathlib import Path 
import os
import datetime
import subprocess
import json

# we support user depending configured logging
import logging
logger = logging.getLogger(__name__)

# application depending installed modules
try:
    import mutagen
    MUTAGEN_READY=True
except:
    MUTAGEN_READY=False
    logger.warning("Python module 'mutagen' missing: Some of sound file meta information can not be recognized.")

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
    if MUTAGEN_READY:
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








# Tests


from rich.pretty import pprint
f="tests/testdata/test1.WAV"
print("--> test1.WAV")
print_meta_structured(f)
print("--> karlinsound_Y2025_dayOfYear123_m05_d03_H15_M00_S00.opus")
f="tests/testdata/karlinsound_Y2025_dayOfYear123_m05_d03_H15_M00_S00.opus"
print_meta_structured(f)