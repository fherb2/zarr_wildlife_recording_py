"""This module includes some general purpose methods for
file access, especially for HDF5 files in the context of
'birdnettools'.
========================================================
"""

import numpy as np
import pickle
import io
import os
import hashlib
import subprocess
import pathlib
import psutil
import h5py
from typing import Any, TypeAlias
from dataclasses import dataclass
from enum import Enum, auto

# class File_path:
#     def __init__(self, 
#                  h5_file_path:File_path|pathlib.Path|str|None = None):
#         if isinstance(h5_file_path, File_path) or isinstance(h5_file_path, pathlib.Path):
#             self.h5_file_path = h5_file_path
#         elif h5_file_path is None:
#             self.h5_file_path = None
#         else:
#             self.h5_file_path = pathlib.Path(h5_file_path)

@dataclass
class H5_OPEN_MODE():
    READ_ONLY_DONT_CREATE = 'r'
    READ_WRITE_DONT_CREATE = 'r+'
    CREATE_OR_TRUNCATE = 'w'
    CREATE_OR_EXCEPTION_IF_EXIST = 'x' # same like 'w-'
    READ_WRITE_CREATE_IF_NOT_EXIST = 'a'

# Define the HDF5 Structur Version
HDF5_STRUCTURE_VERSION_VALUE: dict[int] = {
    "MAJOR": 0,         # -> possible downward incompatible changes
    "MINOR": 1,         # -> downwards compatible changes, new things
    "MAINTENANCE": 1    # -> only corrections or user-invisible changes 
    }
""" Version structure of the actual release
-------------------------------------------

Returns:
    dict[int]:  "MAJOR"        possible downward incompatible changes \n
                "MINOR"        downwards compatible changes, new things \n
                "MAINTENANCE"  only corrections or user-invisible changes \n
"""

# uses always the pickle protocol version 4 in HDF5 attributes
HDF5_ATTRIBUTES_PICKLE_PROTOCOL = 4
""" Used pickle protocol
------------------------

Some attributes or values in the database can be a pickled Python
object. The value of this constant shows the protocol version
of the used Python pickle module.

Returns:
    int: used Python pickle protocol module version
"""

@dataclass
class Dset_compression():
    UNCOMPRESSED = None
    GZIP = "gzip"
    SZIP = "szip"
    LZF  = "lzf"

@dataclass 
class Hd5_folder_structure_Root:
    """ Attributes and values of attributes of the root of birdnettoolbox HDF5 compatible database files
    ---------------------------------------------------------------------------------------------------- 
    """
    COMMENT: str = "comment"
    COMMENT_VALUE: str = "HDF5 file is compatible to 'birdnettoolbox' module" # <- Never change this! It's the recognition
                                                                              #    signal for the right, compatible file.
    HDF5_STRUCTURE_VERSION: str = "version_PICKLEDUINT8"
    HDF5_STRUCTURE_VERSION_VALUE = HDF5_STRUCTURE_VERSION_VALUE
    PICKLED_COMMENT:str = "comment_for_python_pickled_data"
    PICKLED_COMMENT_VALUE:str = "How to decode datasets and attributes ending with '_PICKLEDUINT8':\nUse the 'birdnettoolbox'-'hdf5.py' module method 'unpickle_from_uint8()' for decoding the np.uint8 byte array into a Python object. You will find deeper information about the object in the birdnettoolbox files."
HD5FS_ROOT = Hd5_folder_structure_Root

class Compatibility_state(Enum):
    """Compatibility_state: Possible states of compatibility of a HDF5 file against the actual used
    birdnettoolbox version.

    Can be:

    Args:
        Enum "NO" (int): No major compatibility. API can be confused with this version in read and write access.
        Enum "READ_YES_WRITE_UNCLEAR" (int): Major compatible. Read access without problems, but writes are dangerous.
        Enum "READ_AND_WRITE": Full compatible in read and write access.
    """
    NO = auto()
    READ_YES_WRITE_UNCLEAR = auto()
    READ_AND_WRITE = auto()


def create_h5(hd5_file_path:str|pathlib.Path,
              exist_error:bool = False):
    """create_h5: Create a new HDF5 database if not yet available
    -------------------------------------------------------------
    
    Args:
        file_path (str | pathlib.Path): File path of the new HDF5 file. If file exists,
                                        so it will be not replaced.
        exist_error (bool, optional): This argument takes effect if the file is available:
                                      If 'False', it doesn't replace this file silently. 
                                      If 'True', so it raise a FileExistsError in case the file
                                      is present and can not be created new. Defaults to False.

    Raises:
        FileExistsError: It raise this error if the file exist and the argument 'exist_error' is 'True'.
    """
    hd5_file_path = pathlib.Path(hd5_file_path)
    if hd5_file_path.exists():
        if exist_error:
            raise FileExistsError("Given 'hd5_file_path' exist. Can not create as new database file.")
        else:
            return
    with h5py.File(hd5_file_path, H5_OPEN_MODE.CREATE_OR_EXCEPTION_IF_EXIST) as h5f:
        h5f.attrs[HD5FS_ROOT.COMMENT]                = HD5FS_ROOT.COMMENT_VALUE
        h5f.attrs[HD5FS_ROOT.HDF5_STRUCTURE_VERSION] = pickle_to_uint8(HD5FS_ROOT.HDF5_STRUCTURE_VERSION_VALUE)
        h5f.attrs[HD5FS_ROOT.PICKLED_COMMENT]        = HD5FS_ROOT.PICKLED_COMMENT_VALUE

def delete_all(hd5_file_path:str|pathlib.Path):
    """delete_all: Removes everything except group 'original_sound_files'

    Any data like preprocessing, analyzing, evaluation and so on except the
    group 'original_sound_files' will be removed. The HDF5 file will be
    repacked to reduce file size to the necessary extent.

    Args:
        hd5_file_path (str | pathlib.Path): _description_
    """
    hd5_file_path = pathlib.Path(hd5_file_path)
    # get all groups and datasets in the root
    with h5py.File(hd5_file_path, H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
        for key in h5f.get('/'):
            if key != 'original_sound_files':
                del(h5f[key])
        
def is_toolbox_h5(file:str|pathlib.Path|h5py.File):
    close = False
    if isinstance(file, h5py.File):
        h5f = file
    else:
        h5f = h5py.File(file, mode=H5_OPEN_MODE.READ_WRITE_DONT_CREATE)
        close = True
    try:
        comment = h5f.attrs[HD5FS_ROOT.COMMENT]
        if comment == HD5FS_ROOT.COMMENT_VALUE:
            if close: h5f.close()
            return True
        else:
            if close: h5f.close()
            return False
    except Any as err:
        if close: h5f.close()
        return False
        
def h5f_database_version(h5f:h5py.File) -> dict:
    """hf5_database_version Read ou the data base version of a given HDF5 file
    ---------------------------------------------------------------------------

    Args:
        h5f (h5py._hl.files.File): handle of a opened HDF5 file

    Returns:
        dict: HD5FS_ROOT.HDF5_STRUCTURE_VERSION
    """
    if h5f.attrs[HD5FS_ROOT.COMMENT] != HD5FS_ROOT.COMMENT_VALUE:
        raise TypeError(f"The given HDF5 file {h5f.file} is not compatible to the module 'birdnettoolbox'.")
    return h5f.attrs[HD5FS_ROOT.HDF5_STRUCTURE_VERSION]
    
def h5f_database_compatibility(h5f:h5py.File) -> Compatibility_state:
    hdf5_structure_version_value = unpickle_from_uint8(h5f[HD5FS_ROOT.HDF5_STRUCTURE_VERSION])
    if hdf5_structure_version_value["MAJOR"] != HD5FS_ROOT.HDF5_STRUCTURE_VERSION_VALUE["MAJOR"]:
        return Compatibility_state.NO
    if hdf5_structure_version_value["MINOR"] <= HD5FS_ROOT.HDF5_STRUCTURE_VERSION_VALUE["MINOR"]:
        return Compatibility_state.READ_YES_WRITE_UNCLEAR
    return Compatibility_state.READ_AND_WRITE 
    
def pickle_to_uint8(object: Any):
    bin_stream = io.BytesIO()
    # convert the object into a pickle binary stream
    pickle.dump(object,
                bin_stream,
                protocol=HDF5_ATTRIBUTES_PICKLE_PROTOCOL
                )
    # convert binary stream into a uint8 numpy array
    bin_stream.seek(0)
    # save it into a numpy array
    uint8_stream = np.frombuffer(bin_stream.read(), dtype=np.uint8)
    # we can close the file like object
    bin_stream.close()
    return uint8_stream

def unpickle_from_uint8(np_array:np.ndarray) -> Any:
    # convert a uint8 numpy array into a binary stream
    byte_stream = io.BytesIO(np_array.tobytes())
    # now unpickle
    return pickle.load(file=byte_stream)
    
def in_memory_accepted_file_size_byte(max_free_memory_use_percent:float = 10.0,
                                      at_least_free_ram_size_mbyte:int  = 800
                                      ) -> int:
    svmem = psutil.virtual_memory()
    max_free_memory_use_bytes = int(svmem.available * max_free_memory_use_percent / 100.0)
    usable_ram_size_bytes = svmem.available - at_least_free_ram_size_mbyte*1_000_000
    if usable_ram_size_bytes < 0:
        usable_ram_size_bytes = 0
    return int(min(max_free_memory_use_bytes, usable_ram_size_bytes))

def name_from_link(linkname:str) -> str:
    """name_from_link Gets the last part of a full HDF5 object (group, dataset) link

    Nothing other else: linkname.split('/')[-1]

    Args:
        linkname (str): full link of an HDF5 group or dataset

    Returns:
        str: group or dataset name as last part of a full link
    """
    return linkname.split('/')[-1]

def expand_name_to_link(name:str, parent_link:str) -> str:
    """expand_name_to_link Expands a given name by the parent group to the link in case given name is not yet the full link

    If the given name contains the full link, so this will be returned.
    If the given name contains not the parent_link, so the joined full link will be returned.

    Args:
        name (str): a dataset or group name or a full link to this dataset or group
        parent_link (str): the parent group of 'name'

    Returns:
        str: full link with parent_group and the given dataset or group name
    """
    separator = '/' 
    # in case given name is the link what we would like get:
    if separator.join(name.split('/')[:-1]) == parent_link:
        return name
    else:
        return separator.join([parent_link, name])



class File_like(io.IOBase):
    def __init__(self,
                 hdf5_file_path:str|pathlib.Path|None = None,
                 dataset_link:str|None = None,
                 mode:str = H5_OPEN_MODE.READ_ONLY_DONT_CREATE):
        """Use a dataset of type np.uint8 as file-like object

        Args:
            hdf5_file_path (str | pathlib.Path | None, optional): _description_. Defaults to None.
            dataset_link (str | None, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'r'.
        """
        self._init(hdf5_file_path,
                   dataset_link,
                   mode,
                   readonly = False)

    def _init(self,
              hdf5_file_path:str|pathlib.Path|None = None,
              dataset_link:str|None = None,
              mode:str = H5_OPEN_MODE.READ_ONLY_DONT_CREATE,
              readonly:bool = False
              ):

        self._h5f:h5py.File|None = None # 'None' means also: closed / nothing opened
        self._dset:h5py.Dataset|None = None
        self._len:int           = 0
        self._pointer:int       = 0
        self._readonly          = readonly
        if self._readonly:
            self._mode = H5_OPEN_MODE.READ_ONLY_DONT_CREATE
        else:
            self._mode = mode

        if hdf5_file_path is not None:
            self.open(hdf5_file_path, dataset_link, mode)

    def open(self,
             hdf5_file_path:str|pathlib.Path,
             dataset_link:str,
             mode:str = None
             ):
        if self._h5f is not None:
            raise ValueError("File-like-handle is already open. Close it before open it again. Or use an other/new 'File_like'-object.")

        if mode is not None:
            self._mode = mode
        if self._readonly:
            self._mode = H5_OPEN_MODE.READ_ONLY_DONT_CREATE

        if dataset_link is None:
            raise ValueError("Need HDF5 file and dataset_link to open a dataset.")

        self._h5f = h5py.File(hdf5_file_path, self._mode)
        self._dset = self._h5f[dataset_link]
        if not isinstance(self._dset, h5py.Dataset):
            self.close()
            raise ValueError("Given dataset_link doesn't point on a HDF5-dataset.")
        self._len     = len(self._dset)
        self._pointer = 0
        return self._dset, self._h5f
        
    def close(self):
        if self._h5f is not None:
            self._h5f.close()
            self._h5f  = None
        self._dset = None
    
    @property
    def closed(self) -> bool:
        return self._h5f is None
    
    @property
    def mode(self) -> str:
        if self._h5f is not None:
            return self._h5f.mode
        else:
            return self._mode

    @property
    def readonly(self) -> bool:
        self._readonly
    
    def fileno(self):
        raise OSError
    
    def flush(self):
        if self._readonly:
            return NotImplementedError("File-like object created for read-only access.")
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        self._h5f.flush() 
    
    def isatty(self) -> bool:
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return False
        
    def read(self, size=-1, *_) -> bytes:
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        if size < 0:
            b = self._dset[self._pointer:].tobytes()
            self._pointer += len(b)
        elif size > 0:
            last = self._pointer + size
            if self._pointer > self._len:
                last = self._len
            b = self._dset[self._pointer:last].tobytes()
            self._pointer = last
        else:
            return b''
        return b
        
    def readinto(self, b, *_) -> int:
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        read_size = len(b)
        real_size = self._len - self._pointer
        if real_size < read_size:
            read_size = real_size
        b[:read_size] = self.read(read_size) # see: https://stackoverflow.com/questions/45805111/how-to-implement-readinto-method
        return read_size
    
    def readable(self) -> bool:
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return True
    
    def readline(self, _):
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return NotImplementedError
    
    def readlines(self, _):
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return NotImplementedError
    
    def seek(self, offset:int, whence=os.SEEK_SET):
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        if whence==os.SEEK_SET:
            self._pointer = offset
        elif whence==os.SEEK_CUR:
            self._pointer += offset
        elif whence==os.SEEK_END:
            self._pointer = self._len - offset -1
        else:
            raise ValueError("Argument 'whence' has wrong value.")
        if (self._pointer < 0) or (self._pointer >= self._len):
            raise ValueError("With given offset and whence argument, the file pointer shows out of range.")
        
    def seekable(self):
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return True
    
    def tell(self):
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return self._pointer
    
    def truncate(self, size=None):
        if self._readonly:
            return NotImplementedError("File-like object created for read-only access.")
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        if size is None:
            size = self._pointer
        if size > len(self._dset):
            size = len(self._dset)
        self._dset = self._dset[:size]

    def write(self, b:bytes):
        if self._readonly:
            return NotImplementedError("File-like object created for read-only access.")
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        # It can be, that a part of the new data can be used to overwrite old data
        # and that a part or all new data has to extend the dataset.
        available_part_len = len(self._dset) - self._pointer
        if available_part_len >= len(b):
            self._dset[self._pointer:self._pointer+len(b)] = np.frombuffer(b, dtype=np.uint8)
            self._pointer += len(b)
        else:
            if available_part_len > 0:
                self._dset[self._pointer:] = np.frombuffer(b[:available_part_len], dtype=np.uint8)
                self._dset = np.concatenate(self._dset, np.frombuffer(b[available_part_len:], dtype=np.uint8))
            else:
                self._dset = np.concatenate(self._dset, np.frombuffer(b, dtype=np.uint8))
            self._pointer = len(self._dset)
    
    def writable(self) -> bool:
        if self._readonly:
            return NotImplementedError("File-like object created for read-only access.")
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return self._h5f.writeable()
        
    def writelines(self, *_):
        if self._readonly:
            return NotImplementedError("File-like object created for read-only access.")
        if self._h5f is None:
            raise ValueError("Dataset not opened for file-like read/write.")
        return NotImplementedError

class File_like_ro(File_like):
    def __init__(self,
                 hdf5_file_path:str|pathlib.Path|None = None,
                 dataset_link:str|None = None):
        """Use a dataset of type np.uint8 as file-like object

        Args:
            hdf5_file_path (str | pathlib.Path | None, optional): _description_. Defaults to None.
            dataset_link (str | None, optional): _description_. Defaults to None.
        """
        self._init(hdf5_file_path,
                   dataset_link,
                   mode=H5_OPEN_MODE.READ_ONLY_DONT_CREATE,
                   readonly = True)
        
def calc_hash_from_file(file_path:str|pathlib.Path) -> int:
    """Used method to calculate the hash of a file over all file bytes.

    Args:
        file_path (str | pathlib.Path): File object for calculation of its
                                        hash value

    Returns:
        int: hash as integer value
    """
    h = hashlib.sha1()
    with open(file_path, 'br') as f:
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = f.read(1024)
            h.update(chunk)
    return h.hexdigest() 


class work_arounds():
    @staticmethod
    def require_group(hdf5_file_handle:h5py.File, 
                      name:str, 
                      track_order:bool|None = None):
        """require_group(): Replaces the original to make argument 'track_order' working
        --------------------------------------------------------------------------------

        Documentation see: https://docs.h5py.org/en/stable/high/group.html

        'Parameters as in Group.create_group().'

        """
        if hdf5_file_handle.get(name) is None:
            return hdf5_file_handle.create_group(name, track_order=track_order)
        return hdf5_file_handle.get(name)

    @staticmethod
    def repack(hdf5_file_path:str|pathlib.Path):
        # check if h5repack is installed
        try:
            subprocess.run(["h5repack", "-V"], 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise FileNotFoundError("Missing HDF5 tool 'h5repack'. Please install it.")

        hdf5_file_path = pathlib.Path(hdf5_file_path)
        # temp file in the same directory
        postfix = ".~h5repack"
        tmp_file_path = pathlib.Path(str(hdf5_file_path) + postfix)
        # 1) Copy the current file content by using 'h5repack'.
        #    
        cp = subprocess.run(["h5repack", str(hdf5_file_path), str(tmp_file_path)])
        if cp.returncode != 0:
            # in case the file exist anyway
            subprocess.run(["rm", "-f", str(tmp_file_path)])
        if not tmp_file_path.exists():
            raise IOError("Any problem with h5repack. Repacking stopped.")
        # 2) rename the current file
        cp = subprocess.run(["mv", str(hdf5_file_path), str(hdf5_file_path)+".backup"])
        if cp.returncode != 0:
            # roll back
            subprocess.run(["rm", "-f", str(tmp_file_path)])
            # in case the exist anyway:
            subprocess.run(["mv", str(hdf5_file_path)+".backup", str(hdf5_file_path)])
            raise IOError("Problem with h5repack: Could not move current file as backup.")
        # 3) rename the new file
        cp = subprocess.run(["mv", str(tmp_file_path), str(hdf5_file_path)])
        if cp.returncode != 0:
            # roll back
            subprocess.run(["rm", "-f", str(tmp_file_path)])
            subprocess.run(["mv", str(hdf5_file_path)+".backup", str(hdf5_file_path)])
        subprocess.run(["rm", "-f", str(hdf5_file_path)+".backup"])