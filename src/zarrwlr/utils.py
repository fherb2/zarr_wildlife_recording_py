"""Mixed general helpers of Zarr Wildlife Recording library"""

import zarr
from pathlib import Path
from io import BufferedReader
import os
from types import MappingProxyType
from collections.abc import Mapping

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

