"""Mixed general helpers of Zarr Wildlife Recording Package"""

import subprocess
import re
import zarr
import numpy as np
from pathlib import Path
from io import BufferedReader
import os
from types import MappingProxyType
from typing import Any, Optional
from collections.abc import Mapping

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.debug("Module loading...")

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

def remove_zarr_group_recursive(store, group_path: str):
    '''Removes a zarr group recursively from a store.'''
    logger.trace(f"Remove zarr group '{group_path}' from store '{store}' requested.")
    
    try:
        # Zarr v3 LocalStore (differs to Version 2)
        import pathlib
        store_path = pathlib.Path(store.root) if hasattr(store, 'root') else pathlib.Path(str(store))
        group_full_path = store_path / group_path
        
        if group_full_path.exists():
            import shutil
            shutil.rmtree(group_full_path)
            logger.trace(f"Zarr group '{group_path}' removed via filesystem")
            return
        else:
            logger.trace(f"Zarr group '{group_path}' does not exist")
            return
    except Exception as e:
        logger.warning(f"Error removing zarr group: {e}")

def zarr_to_dict_snapshot(root_group: zarr.Group, k=32, n=32, d=None) -> dict:
    """
    Erzeugt eine verschachtelte Dictionary-Darstellung einer Zarr V3-Gruppe,
    gekürzt für Snapshot-basierte Tests.

    Args:
        group (zarr.Group): Wurzel der Zarr-Datenstruktur.
        k (int): Max. Länge für Strings und Bytes.
        n (int): Max. Anzahl an Elementen bei Listen, Tuples, Arrays.
        d (int | None): Max. Anzahl Key-Value-Paare in Dictionaries, None = alle.

    Returns:
        dict: Kürzbare, testbare Datenstruktur.
    """
    def truncate_value(value):
        if isinstance(value, dict):
            items = list(value.items())
            if d is not None:
                items = items[:d]
            return {truncate_value(k): truncate_value(v) for k, v in items}
        elif isinstance(value, (list, tuple)):
            truncated = value[:n]
            return [truncate_value(v) for v in truncated]
        elif isinstance(value, (bytes, bytearray)):
            return repr(value[:k])
        elif isinstance(value, str):
            return value[:k]
        elif isinstance(value, np.ndarray):
            flat = np.ravel(value)
            return [truncate_value(v) for v in flat[:n]]
        elif isinstance(value, (int, float, bool, type(None))):
            return value
        else:
            return repr(value)

    def process_array(arr: zarr.Array):
        info = {
            "_type": "array",
            "_shape": list(arr.shape),
            "_dtype": str(arr.dtype),
        }

        try:
            attrs = dict(arr.attrs)
            if attrs:
                info["_attrs"] = truncate_value(attrs)
        except Exception as e:
            info["_attrs"] = f"<Error reading attributes: {e}>"

        try:
            data = arr[:]
            info["_data"] = truncate_value(data)
        except Exception as e:
            info["_data"] = f"<Error reading data: {e}>"

        return info

    def process_group(group: zarr.Group):
        result = {
            "_type": "group"
        }

        try:
            attrs = dict(group.attrs)
            if attrs:
                result["_attrs"] = truncate_value(attrs)
        except Exception as e:
            result["_attrs"] = f"<Error reading attributes: {e}>"

        for name in group:
            try:
                obj = group[name]
                if isinstance(obj, zarr.Group):
                    result[name] = process_group(obj)
                elif isinstance(obj, zarr.Array):
                    result[name] = process_array(obj)
                else:
                    result[name] = f"<Unknown Zarr object type: {type(obj)}>"
            except Exception as e:
                result[name] = f"<Error accessing object: {e}>"

        return result

    return process_group(root_group)

def print_snapshot_dict(d, indent=0):
    pad = '  ' * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                print(f"{pad}{k}:")
                print_snapshot_dict(v, indent + 1)
            else:
                print(f"{pad}{k}: {repr(v)}")
    elif isinstance(d, (list, tuple)):
        for i, item in enumerate(d):
            if isinstance(item, (dict, list, tuple)):
                print(f"{pad}- [{i}]:")
                print_snapshot_dict(item, indent + 1)
            else:
                print(f"{pad}- [{i}]: {repr(item)}")
    else:
        print(f"{pad}{repr(d)}")

def assert_zarr_snapshot_equals(actual: dict, expected: dict, path=""):
    """
    Vergleicht zwei rekursiv erzeugte Snapshot-Dictionaries.
    Gibt bei Abweichungen einen präzisen Pfad aus.

    Args:
        actual (dict): Erzeugter Snapshot.
        expected (dict): Erwarteter Snapshot.
        path (str): interner Pfad für Fehlerausgabe.

    Raises:
        AssertionError: Wenn ein Unterschied gefunden wird.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            raise AssertionError(
                f"Key mismatch at {path or '<root>'}:\n"
                f"  Missing keys: {sorted(missing)}\n"
                f"  Extra keys:   {sorted(extra)}"
            )
        for key in sorted(actual_keys):
            assert_zarr_snapshot_equals(actual[key], expected[key], f"{path}.{key}" if path else key)

    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"Length mismatch at {path}: {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_zarr_snapshot_equals(a, e, f"{path}[{i}]")

    else:
        if actual != expected:
            raise AssertionError(f"Value mismatch at {path or '<root>'}: {actual!r} != {expected!r}")

def safe_int_conversion(value: Any) -> Optional[int]:
    """
    Sichere Konvertierung zu int mit None-Fallback
    
    Args:
        value: Zu konvertierender Wert
        
    Returns:
        int oder None
    """
    if value is None or value == "":
        return None
    
    try:
        if isinstance(value, str):
            # Entferne mögliche Einheiten oder Zusätze
            cleaned = value.split()[0] if ' ' in value else value
            return int(float(cleaned))  # float() für Dezimalzahlen als Strings
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_float_conversion(value: Any) -> Optional[float]:
    """
    Sichere Konvertierung zu float mit None-Fallback
    
    Args:
        value: Zu konvertierender Wert
        
    Returns:
        float oder None
    """
    if value is None or value == "":
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def check_ffmpeg_tools():
    """Check if ffmpeg and ffprobe are installed and callable."""
    logger.trace("'Check for ffmpeg-Tools' requested. Typical position for this during import.")
    tools = ["ffmpeg", "ffprobe"]
    logger.trace("Check avalability of ffmpeg and ffprobe tools during import of module...")

    for tool in tools:
        try:
            subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"Missing Command line tool {tool}. Please install ist.")
            exit(1)
    logger.success("ffmpeg and ffprobe tools: Installed and successfully checked: Ok.")



def can_ffmpeg_decode_codec(codec_name:str) -> bool:
    """Check if a codec can be decoded by installes ffmpeg version."""
    try:
        # Alle verfügbaren Decoder auflisten
        result = subprocess.run(['ffmpeg', '-decoders'], 
                              capture_output=True, text=True, check=True)
        
        # Nach dem Codec suchen (case-insensitive)
        pattern = rf'\b{re.escape(codec_name)}\b'
        return bool(re.search(pattern, result.stdout, re.IGNORECASE))
        
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("ffmpeg nicht gefunden")
        return False



logger.debug("Module loaded.")
