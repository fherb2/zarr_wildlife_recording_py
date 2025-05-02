# This module was completed recreated based on the module
# 'soundfile'. So, a lot of file formats and sub-formats
# can be handled and numpy array access is good supported.

import pathlib
import numpy as np
from dataclasses import dataclass
from sunpy.time import TimeRange as sunpy_TimeRange
from astropy.time import Time as astropy_Time
from astropy.units import Unit
import h5py
import os
from typing import Any
import tempfile
import soundfile as sf
from io import BufferedReader

import birdnettoolbox.cores.hdf5 as h5core


# Structure elements, datasets and attributes of
# the HDF5 Database File
# ====================================================
#
# In this Module:
# ---------------
#
# /original_sound_files
#   Folder for "Byte-Blobs" of the original and unchanged sound files 
#   used in data processing results in this HDF5 with some standardized attributes
#   and non-standardized meta data attributes
#   |--1 integer numbered dataset of one file with all attributes
#   |--2                      -"-
#   :
#   |--n dataset, type: bytes (original byte stream of the complete sound file
#          |            with all headers, meta date, ... inside the file; more
#          |            exactly: a binary blob of the file content)
#          |
#          +-- attribute: original_sound_file_name: str # without any path information
#          +-- attribute: pickle of dict {
#          |                              "format": SoundFile.format, 
#          |                              "sub-type": SoundFile.subtype, 
#          |                              "endian": SoundFile.endian,
#          |                              "sections": SoundFile.sections,
#          |                              "seekable": SoundFile.seekable
#          |                              }
#          +-- attribute: file_size_bytes: int # size of the file by using 'os.stat(original sound file).st_size'
#          +-- attribute: content_hash: int # Python hash value over all bytes of the file
#          +-- attribute: sampling_rate: int # sampling rate in samples per second
#          +-- attribute: start_time_astro: pickle of astropy.time.Time # time stamp of the first sample
#          +-- attribute: start_time_readable: str # the string representation of astropy.time.Time
#          +-- attribute: last_frame_time_astro: pickle of astropy.time.Time # time stamp of the last sample
#          +-- attribute: no_channels: int # number of audio channels
#          +-- attribute: no_samples_per_channel: int # number of samples per channel in file
#          +-- attribute: meta: pickle of (dict|None) # some user or case depending meta information about file content
#          +-- attribute: meta_keys: str # String with a list of all most upper keys used in meta dictionary 
#            – no optional attributes –
#   |
#   |
#   |--attribute: time_table Special dataset which summaries sound datasets, recording_arrangements and time 
#         beginnings and endings of these datasets.
#         Its a helping dataset for performant access into the original sound datasets. The content is build on during
#         imports of original sound files.
#   |
#   |--attribute: last_assigned_dset_nb (0 means: no dataset yet)
#   |
#   |--group: recording_arrangements
#         |
#         +--name_a dataset of one recording arrangement
#         +--name_b                      -"-
#         :
#         +--name_n dataset: List of links to the original_sound_files what was created with this 
#              recording_arrangement. Must have a unique name used as link.
#              Note: A recording arrangement is defined as a record device what can record only exactly one sound
#                    file at the same time (but with one or any more synchrony sampled channels) and its special
#                    arrangement like place coordinates or some other place or arrangement context information.
#                    So, all original sound files regarding to a recording_arrangement can not have time-
#                    overlapping records. 
#                    BUT NOTE: It could be possible that the sound frames of such a unique arrangement overlap
#                    anyway: If the ADC clock produces samples asynchronous to the computer clock and so the result
#                    sampling rate is also asynchronous to the file time stamp system of the recorder, it can happen,
#                    that after a file changing to the next record file the file time stamp of the first frame points
#                    a little bit before the last frame of the former file.
#                    By using the recording_arrangement we can recognize this between original sound files: If the
#                    time stamp of a sound file of the first frame is a bit before the last calculated time stamp
#                    of an former sound file AND is the recording arrangement the same, so we know that this is a correct
#                    frame after frame step from the first file to the next, since the recording device clock is
#                    a bit asynchron to the sample clock of the ADC of this recording device. But in case both 
#                    sound files have different recording devices/arrangements, so we know that it is possible, that
#                    the second sound file can start earlier the other sound file ends.
#                    Ergo: The assignment of recoding arrangement to sound files helps to recognize real frame-following
#                    cases between files even if the memorized file time stamps and sampling rates shown a small
#                    time overlap or time gap between the last and first sample of two files.
#                    And at the end: If you use the same device at different recording places over the time, so
#                    it is not necessary to reference the device here as actually being one and the same device
#                    across these use cases. Thats the reason to name this not recording_device but instead 
#                    recording_arrangement. 
#             |
#             +-- attribute: location_longitude_degree: float # -180°...+180°
#             +-- attribute: location_latitude_degree: float # -90°...+90°
#             +-- attribute: meta_data: a pickled dictionary of an meta information for this special arrangement
#             +-- attribute: original_sound_file_dataset_list: A list of all original sound file datasets regarding to 
#                            this recording arrangement. 



@dataclass 
class Hd5_folder_structure_Original_sound_files:
    @dataclass
    class __Grp_original_sound_files:
        NAME        = "original_sound_files"
        FULL_PATH   = "/" + NAME
        @dataclass
        class __Grp_attributes():
            COMMENT_ATTR_NAME       = "comment"
            COMMENT_VALUE           = "Contains the original sound files as byte streams in data sets. Each dataset in this group contains exactly one original sound file and the attributes of each dataset contains defined meta data regarding this file."
            TIME_TABLE_ATTR_NAME    = "time_table_PICKLEDUINT8"
            LAST_ASSIGNED_DSET_NB   = "last_assigned_dset_nb"
        GRP_ATTRIBUTES = __Grp_attributes
        @dataclass
        class __Dataset_original_sound_file():
            # NAME =  -> will be generated as unique number
            FULL_PATH = "/original_sound_files/" #  + the dynamic build NAME
            class __Dset_attributes:
                CONTENT_HASH_ATTR_NAME    = "content_hash"
                ORIGINAL_SOUND_FILE_NAME_ATTR_NAME = "original_sound_file_name"
                RECORDING_DEVICE_LINK_ATTR_NAME    = "recording_device_link"
                FILE_SIZE_ATTR_NAME       = "file_size_bytes"
                SOUNDFILE_TYPE_ATTR_NAME  = "soundfile_type_PICKLEDUINT8"
                SAMPLING_RATE_ATTR_NAME   = "sampling_rate"
                START_TIME_ATTR_NAME      = "start_time_astro_PICKLEDUINT8"
                START_TIME_READABLE_ATTR_NAME = "start_time_readable"
                NUMBER_CHANNELS_ATTR_NAME = "no_channels"
                NUMBER_SAMPLES_PER_CHANNEL_ATTR_NAME = "no_samples_per_channel"
                META_DATA_ATTR_NAME       = "meta_PICKLEDUINT8"
                META_DATA_KEYS_ATTR_NAME  = "meta_keys"
            DSET_ATTRIBUTES = __Dset_attributes
        DSET_OSF = __Dataset_original_sound_file
        @dataclass
        class __Grp_recording_arrangements():
            NAME        = "recording_arrangements"
            FULL_PATH   = "/original_sound_files/" + NAME
            @dataclass
            class __Grp_attributes:
                COMMENT_ATTR_NAME   = "comment"
                COMMENT_VALUE       = "Each original sound file is assigned to a recording arrangement. This recording arrangement ensures that the individual original sound files belong together in terms of location and time."
            GRP_ATTRIBUTES = __Grp_attributes
            @dataclass
            class __Dataset_attributes:
                LOCATION_LONGITUDE_DEGREE_ATTR_NAME = "location_longitude_degree"
                LOCATION_LATITUDE_DEGREE_ATTR_NAME  = "location_latitude_degree"
                META_DATA_ATTR_NAME                 = "meta_data"
            DSET_ATTRIBUTES = __Dataset_attributes
        GRP_RECORDING_ARRANGEMENTS = __Grp_recording_arrangements
    GRP_ORIGINAL_SOUND_FILES = __Grp_original_sound_files
HD5FS = Hd5_folder_structure_Original_sound_files


class Recording_arrangement():
    # we can initialize it with data for a new object, or: see "from_dataset"
    def __init__(self,
                 unique_name:str,
                 location_longitude_degree:float,
                 location_latitude_degree:float,
                 meta_data:dict[str, Any]|None = None,
                 original_sound_file_dataset_list:list[int]|int|None = None
                 ):
        """Recording_arrangement() – A recording device at a special location what can record
        exactly on audio stream with an arbitrary number of synchronous sampled audio channels.
        --------------------------------------------------------------------------------------

        Note::
        ------
            A recording arrangement is defined as a record device what can record only exactly one sound
            file at the same time (but with one or any more synchrony sampled channels) and its special
            arrangement like place coordinates or some other place or arrangement context information.
            So, all original sound files regarding to a recording_arrangement can not have time-
            overlapping records. 

        BUT NOTE also::
        ---------------
            It could be possible that the sound frames of such a unique arrangement overlap
            anyway: If the ADC clock produces samples asynchronous to the computer clock and so the result
            sampling rate is also asynchronous to the file time stamp system of the recorder, it can happen,
            that after a file changing to the next record file the file time stamp of the first frame points
            a little bit before the last frame of the former file.
            By using the recording_arrangement we can recognize this between original sound files: If the
            time stamp of a sound file of the first frame is a bit before the last calculated time stamp
            of an former sound file AND is the recording arrangement the same, so we know that this is a correct
            frame after frame step from the first file to the next, since the recording device clock is
            a bit asynchron to the sample clock of the ADC of this recording device. But in case both 
            sound files have different recording devices/arrangements, so we know that it is possible, that
            the second sound file can start earlier the other sound file ends.

        Ergo::
        ------
            The assignment of recoding arrangement to sound files helps to recognize real frame-following
            cases between files even if the memorized file time stamps and sampling rates shown a small
            time overlap or time gap between the last and first sample of two files.
            And at the end: If you use the same device at different recording places over the time, so
            it is not necessary to reference the device here as actually being one and the same device
            across these use cases. Thats the reason to name this not recording_device but instead 
            recording_arrangement. 

        Args::
        ------
            unique_name (str): Each recording_arrangement dataset MUST HAVE a unique name, in HDF5 used as link.
            location_longitude_degree (float): float # -180°...+180°
            location_latitude_degree (float): float # -90°...+90°
            meta_data (dict | None, optional): A pickled dictionary of an meta information for this special arrangement.
                                               Defaults to None.
            original_sound_file_dataset_list (list[int] | int | None, optional): A list of all original sound file
                                                                                 datasets regarding to 
                                                                                 this recording arrangement.
                                                                                 Defaults to None.
        """
        self._unique_name               = h5core.name_from_link(unique_name)
        self.location_longitude_degree  = location_longitude_degree
        self.location_latitude_degree   = location_latitude_degree
        self.meta_data                  = meta_data
        self.osf_datasets:list[int]     = []
        self.add_osf_dataset_links(original_sound_file_dataset_list)


    def __str__(self) -> str:
        """__str__ Get back the unique name string of the recording arrangement.

        Returns:
            str: unique name
        """
        return self.unique_name
    

    @property
    def unique_name(self) -> str:
        """Property: The unique name of the recording arrangement, used as dataset name in the HDF5 file."""
        return self._unique_name


    # we can also initialize by a recording_arrangements dataset from HDF5 file
    @classmethod 
    def read_dataset(cls,
                     hdf5_file_path:pathlib.Path|str,
                     recording_arrangements_unique_name:str
                     ) -> 'Recording_arrangement': # thats a so named Factory method
        """read_dataset – Read the recording arrangement dataset from HDF5 file and initialize 
        the Recording_arrangement.
        --------------------------

        Can be used instead the __init__ method.

        Args::
        ------
            hdf5_file_path (pathlib.Path|str): HDF5 file what should be used.
            recording_arrangements_unique_name (str): HDF5 links (string) to the
                                               dataset what should be used for initializing.

        Raises::
        --------
            FileExistsError: In case the HDF5 file is missing.

        Returns::
        ---------
            Recording_arrangement: Returns an initialized Recording_arrangement-object.
        """
        hdf5_file_path = pathlib.Path(hdf5_file_path)
        if not hdf5_file_path.exists():
            raise FileExistsError(f"Given HDF5 File {str(hdf5_file_path)} doesn't exist.")
        with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            recording_arrangements_dataset_link = h5core.expand_name_to_link(recording_arrangements_unique_name, HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.FULL_PATH)
            dset = h5f.get(recording_arrangements_dataset_link)
            if dset is None:
                raise ValueError("Given recording arrangement '{recording_arrangements_unique_name}' is missing in file {hdf5_file_path}")
            unique_name                         = h5core.name_from_link(dset.name)
            original_sound_file_dataset_list    = h5core.unpickle_from_uint8(dset[:])
            loc_long = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.LOCATION_LONGITUDE_DEGREE_ATTR_NAME]
            loc_lat  = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.LOCATION_LATITUDE_DEGREE_ATTR_NAME]
            meta     = h5core.unpickle_from_uint8(dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.META_DATA_ATTR_NAME])
        return cls(unique_name,
                   loc_long,
                   loc_lat,
                   meta,
                   original_sound_file_dataset_list
                   )
    

    def write_dataset(self,
                       hdf5_file_path:h5py._hl.files.File|pathlib.Path|str,
                       ):
        """write_dataset – Writes the recording arrangement dataset to the given HDF5 file
        ----------------------------------------------------------------------------------

        Args:
            hdf5_file_path (h5py._hl.files.File|pathlib.Path|str): HDF5 file path to write the recording arrangement dataset
                                               or a open file handle with write access.
        """
        # open file if not yet open
        if isinstance(hdf5_file_path, h5py._hl.files.File):
            h5f = hdf5_file_path
            do_close = False
        else:
            h5f = h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE)
            do_close = True
           
        grp = h5f.require_group(HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.FULL_PATH)
        dset_data_pickled = h5core.pickle_to_uint8(self.osf_datasets)
        # create dataset if not exist
        if not self._unique_name in grp.keys():
            dset = grp.create_dataset(self._unique_name, shape=dset_data_pickled.shape, dtype=np.uint8, maxshape=(None,))
        else:
            dset = grp[self._unique_name]
        # fill dataset and set attributes
        dset.resize(dset_data_pickled.shape)
        dset[:] = dset_data_pickled
        dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.LOCATION_LATITUDE_DEGREE_ATTR_NAME] = self.location_latitude_degree
        dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.LOCATION_LONGITUDE_DEGREE_ATTR_NAME] = self.location_longitude_degree
        dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_RECORDING_ARRANGEMENTS.DSET_ATTRIBUTES.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(self.meta_data)
        # close file if it was not open
        if do_close:
            h5f.close()


    @classmethod
    def append_osf_dataset_links(cls,
                                 hdf5_file_path:pathlib.Path|str,
                                 recording_arrangement_dset_link:str,
                                 original_sound_file_dataset_list:list[int]|int # link list or a single link
                                 ):
        """append_osf_dataset_links – Assigns an original sound file record to a recording arrangement
        in an HDF5 file.
        ----------------

        Summarizes the entire assignment process without creating an object: Loading the previous
        recording arrangement, adding the original sound file data set(s) and saving the actualized recording
        arrangement.

        Args:
            hdf5_file_path (pathlib.Path|str): HDF5 file what should be used.
            recording_arrangement_dset_link (str): HDF5 links (string) to the
                                dataset what should be used to add the original
                                sound file dataset(s).
            original_sound_file_dataset_list (list[int] | int): The dataset to be added. Or
                                a list of datasets to be added.
        """
        rec_arr = cls().read_dataset(hdf5_file_path, recording_arrangement_dset_link)
        rec_arr.add_osf_dataset_links(original_sound_file_dataset_list)
        rec_arr.write_dataset(hdf5_file_path)


    def add_osf_dataset_links(self,
                          original_sound_file_dataset_list:list[int]|int|None # link list or a single link
                          ):
        """add_osf_dataset_links – Add original sound file datasets to this recording arrangement.
        ----------------------------------------------------------------------------------------

        The difference to the class method 'append_osf_dataset_links' is that::
            - The recording arrangement must be exist as object.
            - After the original sound file dataset is added, the actualized recording
              arrangement is not written into the HDF5 file automatically.

        Args::
            original_sound_file_dataset_list (list[int] | int | None): HDF5 dataset link list or a single
                                                       link to original sound file datasets which are to
                                                       be assigned to this recording arrangement.
        """
        if isinstance(original_sound_file_dataset_list, list):
            self.osf_datasets.extend(original_sound_file_dataset_list)
        elif isinstance(original_sound_file_dataset_list, int):
            self.osf_datasets.append(original_sound_file_dataset_list)


class Time_table:
    def __init__(self,
                 hdf5_file_path:pathlib.Path|str):
        """Time_table – This table summarizes all the essential information of the stored original
        sound file records in order to facilitate the selection of accesses to these records.
        ------------------------------------------------------------------------------------------

        The time table is a single attribute of the group of all original sound file datasets. It
        summarizes following information about the original sound file datasets:

            - Time stamp of the the first and last audio sample in each original sound file dataset.
            - The corresponding data set number.
            - The unique name (HDF5 link) to the corresponding recording arrangement.
            - The number of frames in this audio file data set.
            - The nominal sampling frequency of the audio data.

        The time table is a helper for access and filtering of the original sound file data sets.

        The time table can be used as iterator object. Additionally a filter mechanism is implemented.
        If filtering methods are called, so the iterator get back only such Time_table elements which
        are included ba the filter. You can reset the filter by calling 'clear_filter'.
        In case one or some more entries are added, the filter will be reset also.

        Args::
        ------
            hdf5_file_path (pathlib.Path|str): Time table will be initialized by the data from this file
                                               or, if no time table in file, this table will be initialized.
        """
        self.hdf5_file_path = pathlib.Path(hdf5_file_path)
        
        # init time table
        if self._is_time_table_in_file():
            self._init_from_file()
        else:
            self._init_as_new()
            self.save() # first write of an empty time table

        # init some stuff:
        self._filtered_indices:list[int]    = []
        self._iter_index:int                = 0
        self.clear_filter() # init filter and generator

    def _is_time_table_in_file(self):
        with h5py.File(self.hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            attrs = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH).attrs
            return HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.TIME_TABLE_ATTR_NAME in attrs

    def _init_as_new(self):
        """_init_as_new – Initializes the object with empty information
        ------------------------------------------------------------------------
        """
        # Time table columns are lists of row entries. Only the first
        # frame time and last frame time is organized in a astropy.time.Time
        # object.
        self.osf_dset_number_list:list[int]                     = []
        self.recording_arrangement_unique_name_list:list[str]   = []
        self.time_range_list:list[sunpy_TimeRange]              = []
        self.number_of_frames_list:list[int]                    = []
        self.nominal_sampling_frequency_list:list[int]          = []

    def _init_from_file(self):
        """_init_from_file – Initializes the object by using the given HDF5 file.
        -------------------------------------------------------------------------

        Raises:
            ValueError: In case, the HDF5 file is missing.
        """
        if self.hdf5_file_path is None:
            raise ValueError("Missing hdf5_file_path.")
        with h5py.File(self.hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            grp = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH)
            self._overwrite_time_table_from_dict(h5core.unpickle_from_uint8(grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.TIME_TABLE_ATTR_NAME]))
    
    def get_dict_from_time_table(self) -> dict[str, list[Any]]:
        return {
                "original_dset_number_list" :               self.osf_dset_number_list,
                "recording_arrangement_unique_name_list" :  self.recording_arrangement_unique_name_list,
                "time_range_list" :                         self.time_range_list,
                "number_of_frames_list":                    self.number_of_frames_list,
                "nominal_sampling_frequency_list" :         self.nominal_sampling_frequency_list
               }
    
    def _overwrite_time_table_from_dict(self, dict:dict[str, list[Any]]):
        # check lengths before overwriting the time table data
        if len(dict["original_dset_number_list"]) == \
           len(dict["recording_arrangement_unique_name_list"]) == \
           len(dict["time_range_list"]) == \
           len(dict["number_of_frames_list"]) == \
           len(dict["nominal_sampling_frequency_list"]):
            # overwrite
            self.osf_dset_number_list                   = dict["original_dset_number_list"]
            self.recording_arrangement_unique_name_list = dict["recording_arrangement_unique_name_list"]
            self.time_range_list                        = dict["time_range_list"]
            self.number_of_frames_list                  = dict["number_of_frames_list"]
            self.nominal_sampling_frequency_list        = dict["nominal_sampling_frequency_list"]
        else:
            raise ValueError("Elements of given dict must ne lists of the same length. But lengths are different.")

    def save(self):
        """save – Write the time table into the HDF5 file.
        ---------------------------------------------------

        The data are coded by self.get_dict_from_time_table() and pickle_to_uint8).

        Args::
        ------
            hdf5_file_path (pathlib.Path|str | None, optional): HDF5 file to save. If not given,
                                                        parameter 'self.hdf5_file_path' is used.
                                                        Defaults to None.

        Raises::
        --------
            ValueError: In case there is no HDF5 file known.
        """
        # either the file is given or it should be memorized in self.hdf5_file_path
        if self.hdf5_file_path is None:
            raise ValueError("Missing hdf5_file_path.")
        with h5py.File(self.hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            grp = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH)
            grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.TIME_TABLE_ATTR_NAME] = \
                h5core.pickle_to_uint8(self.get_dict_from_time_table())
            

    def append_soundfile(self,
               osf_dset_number:int,
               recording_arrangement:str,
               time_range:sunpy_TimeRange,
               number_of_frames:int,
               nominal_sampling_frequency:int
               ):
        """append – Adds a single entry in the time table by given data for a single original sound file.
        -------------------------------------------------------------------------------------------------

        If you want to import more than one original sound file data set, so use 'extend'.

        Args::
            first_last_frame_time_stamps (astropy.time.Time): This Time object contains exactly two time stamps
                of a special original sound file dataset::
                    - the time stamp of the first sample frame
                    - the time stamp of the last sample frame
            osf_dset_number (int): The dataset number of the original sound file.
            recording_arrangement (str): The HDF5 link string to the assigned recording arrangement.
            number_of_frames (int): The number of frames (samples per audio channel) for the assigned original
                                    sound file dataset.
            nominal_sampling_frequency (int): Defined sampling frequency of the assigned original sound file dset.
        """
        self.time_range_list.append(time_range)
        self.osf_dset_number_list.append(osf_dset_number)
        self.recording_arrangement_unique_name_list.append(recording_arrangement)
        self.number_of_frames_list.append(number_of_frames)
        self.nominal_sampling_frequency_list.append(nominal_sampling_frequency)
        # and we actualize in HDF5 file (we can do this always without a performance problem,
        # since 'append' is done during an import of a sound file whats seldom and time consuming)
        if self.hdf5_file_path is not None:
            self.save()
        self.clear_filter()

    def extend_soundfiles(self,
               osf_dset_number_list:list[int],
               recording_arrangement_unique_name_list:list[str],
               time_range_list:list[sunpy_TimeRange],
               numbers_of_frames_list:list[int],
               nominal_sampling_frequency_list:list[int]
               ):
        """extend – Adds a list of entries in the time table by given data for a list of original sound files.
        ------------------------------------------------------------------------------------------------------

        If you want to import only one original sound file data set, so use 'append'.

        Args::
            first_last_frame_time_stamps_list (list[astropy.time.Time]): Each Time object in the list contains exactly two time stamps
                of a special original sound file dataset::
                    - the time stamp of the first sample frame
                    - the time stamp of the last sample frame
            osf_dset_number_list (list[int]): The list of dataset numbers of the original sound files.
            recording_arrangement_list (list[str]): A list of the HDF5 link strings to the assigned recording arrangements.
            numbers_of_frame_list (list[int]): List of the number of frames (samples per audio channel) for the assigned original
                                    sound file dataset.
            nominal_sampling_frequency_list (list[int]): List of defined sampling frequency of the assigned original sound file dset.

        Raises:
            ValueError: The index of all parameters points to the same original sound file datasets and assigned the data in all lists here
                        to these datasets. All list must have the same number of elements. ValueError in case the lengths are different.
        """
        # length must be the same over all given lists
        lengths = np.array([
                            len(time_range_list),
                            len(osf_dset_number_list),
                            len(recording_arrangement_unique_name_list),
                            len(numbers_of_frames_list),
                            len(nominal_sampling_frequency_list)
                            ], dtype=np.uint32)
        if lengths.min() != lengths.max():
            raise ValueError(f"Lengths of all given lists must be the same. But given lengths are {lengths}")
        # ok, we can extend
        self.time_range_list.extend(time_range_list)
        self.osf_dset_number_list.extend(osf_dset_number_list)
        self.recording_arrangement_unique_name_list.extend(recording_arrangement_unique_name_list)
        self.number_of_frames_list.extend(numbers_of_frames_list)
        self.nominal_sampling_frequency_list.extend(nominal_sampling_frequency_list)
        # and we actualize in HDF5 file (we can do this always without a performance problem,
        # since 'extend' is done during an import of sound files whats seldom and time consuming)
        if self.hdf5_file_path is not None:
            self.write()
        self.clear_filter()


    def clear_filter(self):
        """clear_filter – After calling this, the iterator gets back all unfiltered entries.
        -----------------------------------------------------------------------------------
        """
        if len(self) == 0:
            self._filtered_indices:list[int] = []
        else:
            self._filtered_indices:list[int] = range(len(self))
        self._length:int = len(self._filtered_indices)

    @property
    def filtered_indices(self) -> list[int]:
        """filtered_indices – Gets back a list of all indices after filtering.

        Returns:
            list[int]: Indices of the time table.
        """
        return self._filtered_indices

    @filtered_indices.setter
    def filtered_indices(self,
                         indices_list:list[int]):
        """filtered_indices – Overwrite the indices list of filtering.
        --------------------------------------------------------------

        Use this carefully by your own filter functions. 

        Args:
            indices_list (list[int]): Given lists of indices which point to
                                      time table entries by calling the iterator.
        """
        self._filtered_indices = indices_list
    
    @property
    def unfiltered_len(self) -> int:
        """unfiltered_len – Number of all entries in the time table.
        ------------------------------------------------------------

        Returns:
            int: Number of all entries in the time table.
        """
        return len(self.osf_dset_number_list)
    
    @property
    def filtered_len(self) -> int:
        """filtered_len – Number of all entries in the time table after filtering.
        --------------------------------------------------------------------------

        Returns:
            int: Number of all entries in the time table which are included by the filters.
        """
        return len(self._filtered_indices)
    
    def __len__(self) -> int:
        """__len__ – Number of all entries in the time table after filtering.
        --------------------------------------------------------------------------

        Used for iterating calls.

        Returns:
            int: Number of all entries in the time table which are included by the filters.
        """
        return self.filtered_len

    def __iter__(self):
        """__iter__ – Initializes Python like iterating
        -----------------------------------------------

        Used for iterating calls.
        """
        self._iter_index = 0
        return self
    
    def __next__(self):
        """__next__ – Get the next output during iterating
        --------------------------------------------------

        Raises:
            StopIteration: In case, there are no more elements.

        Returns:
            tuple(dict, int): Tuple of
                                 - Elements of the entry in the time table. For details see: get_dset_by_index
                                 - Index of this inside time table.
        """
        if self._iter_index < self.filtered_len:
            i = self._iter_index
            self._iter_index += 1
            return self.get_dset_by_index(self._filtered_indices[i]), self._filtered_indices[i] # return selected data row as dictionary and its index number
        else: 
            raise StopIteration

    def get_entry_by_index(self,
                          index:int) -> dict:
        """get_entry_by_index – Gets back a special entry in time table as dictionary.

        Args:
            index (int): The index (row) in the time table.

        Returns:
            dict: Elements of the entry:
                "start_frame_astro_time"
                "last_frame_astro_time"
                "osf_dset_number" (original sound file dataset number)
                "recording_arrangement_unique_name"
                "number_of_frames"
                "nominal_sampling_frequency"
        """
        return {
            "start_frame_astro_time": self.time_range_list[index].start,
            "last_frame_astro_time":  self.time_range_list[index].end,
            "osf_dset_number":        self.osf_dset_number_list[index],
            "recording_arrangement_unique_name":  self.recording_arrangement_unique_name_list[index],
            "number_of_frames":       self.number_of_frames_list[index],
            "nominal_sampling_frequency": self.nominal_sampling_frequency_list[index]
        }
    
    def get_filtered_entries(self) -> tuple[list[dict], list[dict]]|None:
        """get_filtered_entries – Gets back the filtered time table entries and its indices
         
        Note: Returned indices belongs to the unfiltered time table.

        Returns:
            tuple[list[dict], list[dict]]|None: List of entries from timetable, list of indices of these entries.
                                       'None' in case there are no entries after filtering.
        """
        if self.filtered_len == 0:
            return None
        dset_list  = []
        index_list = []
        for dset, index in self:
            dset_list.append(dset)
            index_list.append(index)
        return dset_list, index_list

    def filter_recording_arrangement_unique_name(self,
                                                 recording_arrangement_unique_name:str) -> int:
        """filter_recording_arrangement_unique_name Filters for given recording arrangement.

        Args::
            recording_arrangement_unique_name (str): name of the recording arrangement used for filtering

        Returns:
            int: number of entries after this filter step
        """
        filtered_indices = []
        for dset, index in self:
            if dset["recording_arrangement_unique_name"] == recording_arrangement_unique_name:
                filtered_indices.append(index)
        self.filtered_indices = filtered_indices
        return len(self.filtered_indices)

    def filter_first_frame_time_stamp(self,
                                      time_range:sunpy_TimeRange) -> int:
        """filter_first_frame_time_stamp Filters start_frame_time_astro by given range

        Args:
            astro_time_range (astropy.time.Time): Astropy Time object with two time values gives
                                                  time range for filtering

        Returns:
            int: number of entries after this filter step
        """
        filtered_indices = []
        for dset, index in self:
            if dset["start_frame_astro_time"] >= time_range[0] \
               and dset["start_frame_astro_time"] <= time_range[1]:
                filtered_indices.append(index)
        self.filtered_indices = filtered_indices
        return len(self.filtered_indices)


class Original_sound_file():
    # Static and Class-methods
    # ========================
    @staticmethod
    def check_compatibility(file_path:pathlib.Path|str) -> bool:
        """check_compatibility: Checks if the given sound file
        format can be used directly with the toolbox
        ======================================================

        The toolbox uses module 'soundfile' for file import and random access into
        this file or its copy inside the database. This function returns with 'True'
        if the file format is compatible. Otherwise try to convert your files into
        one of the compatible formats of the module 'soundfile'.
        You can list the compatible formats with function 'compatibility_formats()'.

        Args:
            file_path (str | pathlib.Path): your sound file to check

        Raises:
            ValueError: Description in case if no compatible.

        Returns:
            bool: 'True' in case of compatibility, ValueError otherwise
        """
        try:
            f = sf.SoundFile(pathlib.Path|str(file_path))
        except:
            raise ValueError("File format is not compatible with used Python module 'soundfile'.")
        if f.seekable():
            f.close()
            return True
        raise ValueError("File format is not compatible since it is not seekable for random access.")

    @classmethod
    def upgrade_h5_file(cls,
                        file_path:pathlib.Path|str
                        ):
        # TODO: Fill out this method!
        # Check if upgrade is needed
        # ... and do upgrades step by step
        pass

    @staticmethod
    def available_formats(verbose:bool = False) -> dict:
        """available_formats Lists supported sound file formats
        and sub-types as dictionary

        Returns all supported sound file formats and sub-types
        as dictionary and, if verbose is True, as text to stdout.

        Args:
            verbose (bool, optional): Print the information additional
                                        to stdout. Defaults to False.

        Returns:
            dict: Dictionary. Keys are the format and value is
                    a list of sub-types for this format.
        """
        if verbose:
            print("List of formats and sub-formats:")
            print("--------------------------------")
            for format in sf.available_formats():
                print(f"* {format} with following sub-types:")
                for subtype in sf.available_subtypes(format):
                    print(f"   - {subtype}")
        formats = {}
        for format in sf.available_formats():
            formats[format] = sf.available_subtypes(format).keys()
        return formats
    
    @classmethod
    def initialize_hdf5_file(cls,
                             hdf5_file_path:pathlib.Path|str,
                             dont_create:bool = False
                             ) -> pathlib.Path:
        # initialize HDF5 file stuff
        # --------------------------
        #   create if needed, upgrade if needed; 
        hdf5_file_path = pathlib.Path(hdf5_file_path)
        if hdf5_file_path.exists():
            # check common compatibility
            assert h5core.is_toolbox_h5(hdf5_file_path), \
                f"Expect a birdnet toolbox compatible file. But given file seems to be not compatible."
            cls.upgrade_h5_file(hdf5_file_path) # with actual version, these method doesn't change the file
        elif dont_create:
            raise FileNotFoundError(f"Given {hdf5_file_path} doesn't exist.")
        h5core.create_h5(hdf5_file_path)
        # create groups and group attributes if not yet done
        with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:

            # Group: /original_sound_files
            grp = h5core.work_arounds.require_group(h5f, HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH, track_order=True) # see: https://docs.h5py.org/en/stable/high/attr.html ("Large attributes")
            # Group-Attributes: 
            if not HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.COMMENT_ATTR_NAME in grp.attrs:
                grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.COMMENT_ATTR_NAME] = HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.COMMENT_VALUE
            if not HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.LAST_ASSIGNED_DSET_NB in grp.attrs:
                grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.LAST_ASSIGNED_DSET_NB] = 0

        return hdf5_file_path
            
    # Non-static-/class-methods
    # ========================
    def __init__(self,
                 hdf5_file_path:pathlib.Path|str,
                 dont_create:bool = False
                 ):
        """Original_sound_file is the class to handle the HDF5 entries under '/original_sound_files'
        
        Each imported sound file is saved as dataset with a unique number under '/original_sound_files'.
        So all soundfile bytes are saved as 1D np.array with dtype np.uint8 with names as e.g.
        '/original_sound_files/4' . The used unique numbers can have gaps in case data sets are removed any
        time after its import.
        
        Each such dataset has attributes with all meta data for this dataset.
        
        For more details, see the method 'import_soundfile()' also.
        
        For the evaluation of recognition tasks it is important to hold the results and used processing
        method next by the original data. So some restrictions are explicitly implemented:
        
          - Imported sound files can not be changed.
          - Meta data can not be over written
          - In case there are good reasons to remove sound files, the dataset will not be changed. Only
            the file bytes will be set to an empty data stream. So any references to this dataset will
            not point into to void: All meta data are kept. 
            
        But this is save only, if the group '/original_sound_files' will be changed by this 
        class only.

        Args:
            hdf5_file_path (str | pathlib.Path): HDF5 file to use as database
            create_if_not_available (bool, optional): The HDF5 file will be created if not available (true)
                                                      or initialization raises an FileNotFoundError if this
                                                      argument is False. 
                                                      Defaults to True.

        Raises:
            FileNotFoundError: Raises if given file ist not found and create_if_not_available is False.
        """
        # initialization of file content
        self._hdf5_file_path = self.initialize_hdf5_file(hdf5_file_path = hdf5_file_path,
                                                         dont_create    = dont_create)
        # initialize other stuff
        self._dset_handle:h5py.Dataset|None = None

    @property
    def hdf5_file_path(self) -> pathlib.Path:
        """File and path used as HDF5 database file

        Returns:
            pathlib.Path: filename including the path
        """
        return self._hdf5_file_path
    
    @property
    def nb_dsets(self) -> int:
        """Number of datasets; means number of imported sound files
        
        All sound files can be addressed by the link '/original_sound_files/##'
        with '##' as unique dataset number in this group.

        Returns:
            int: Number of datasets; means number of imported sound files
        """
        print("Request for number of dsets...")
        count:int = 0
        with h5py.File(self._hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            grp = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH)
            for key in grp.keys():
                dset = grp.get(key)
                if isinstance(dset, h5py.Dataset):
                    count += 1
        print(f"{count=}")
        return count 

    @property
    def dsets_list(self) -> list[str]:
        """Get a list of dataset links for each imported sound file.
        
        This function is good to loop through all imported sound files.

        Returns:
            list[str]: List of strings of the form like ['/original_sound_files/1', '/original_sound_files/2']
                       and so on.
        """
        print("Request for dset-list...")
        dset_list = []
        with h5py.File(self.hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            grp = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH)
            for key in grp.keys():
                dset = grp.get(key)
                if isinstance(dset, h5py.Dataset):
                    components = {}
                    components['dataset_link'] = dset.name
                    components['dataset_attributes'] = self.dset_attributes(dset)
                    dset_list.append(components)
        return dset_list  
    
    def dset_attributes(self,
                        dset:str|h5py.Dataset
                        ) -> dict[str, Any]:
        """Get all attributes of a dataset/sound-file inside the HDF5 file under

        Args:
            dset (str | h5py.Dataset): Either a link to a dataset (e.g. '/original_sound_files/5') 
                                       (HDF5 file will be short opened to read the attributes) or
                                       a dataset object of an opened HDF5 file.

        Returns:
            dict[Any]: Dictionary of all attributes of the given dataset
        """
        h5f = None
        if isinstance(dset, str):
            h5f = h5py.File(self._hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE)
            dset = h5f.get(dset)
        attrs = {}
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.FILE_SIZE_ATTR_NAME]      = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.FILE_SIZE_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.CONTENT_HASH_ATTR_NAME]   = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.CONTENT_HASH_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SOUNDFILE_TYPE_ATTR_NAME] = h5core.unpickle_from_uint8(dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SOUNDFILE_TYPE_ATTR_NAME])
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SAMPLING_RATE_ATTR_NAME]  = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SAMPLING_RATE_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_CHANNELS_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_CHANNELS_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_SAMPLES_PER_CHANNEL_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_SAMPLES_PER_CHANNEL_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_READABLE_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_READABLE_ATTR_NAME]
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_ATTR_NAME]      = h5core.unpickle_from_uint8(dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_ATTR_NAME])
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_KEYS_ATTR_NAME] = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_KEYS_ATTR_NAME]                  
        attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_ATTR_NAME]      = h5core.unpickle_from_uint8(dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_ATTR_NAME])
        if h5f is not None:
            h5f.close()
        return attrs

    def import_soundfile(self,
                         sound_file_path: str|pathlib.Path,
                         recording_arrangement_name: str,
                         start_time: astropy_Time,
                         meta: dict[str, Any]|None = None,
                         dset_compression: h5core.Dset_compression = None,
                         compression_opts = None
                         ):
        """Import a new sound file
        
        The import function makes a byte-by-byte-copy of the original sound file. So it can be restored as 
        1:1 copy of the original file. Each imported sound file results in a dataset with some attributes
        as meta information. For a restore of the file only the dataset bytes and one attribute, the original
        file name, is needed. All other meta information are additional extracted from the file, like the
        channel number or frame rate. But such information are redundant.
        
        During import it will be checked if the sound file was imported any time before. The import stops
        in this case. For this check the hash is used and in case the hash is identically, the file
        content will be compared byte by byte. Not important is the file name, path or any other meta
        information.
        
        As start time stamp an object of astro.time.Time must be given. See:
        https://docs.astropy.org/en/stable/time/index.html
        
        The recording location of the file is saved at the given recording arrangement. So we don't save it 
        in the original sound file dataset.
        
        Its possible to save any kind and any amount of additional meta data to the file. All meta information
        must be packed into a Dictionary. But the content of this Dictionary can be all what you think to save
        together with the original sound file. In case of difficulty data, make a byte stream or encode it
        in any other manner.
        
        In case the sound file is not a lossy or lossless format, so switch on the dataset compression. See
        the HDF5 documentation what is possible. Standard HDF5 compression is fully transparent. All methods
        work like the saved data are uncompressed. But the HDF5 file will be smaller.

        Args:
            soundfile_path (str | pathlib.Path): File path of the sound file to import.
            start_time (astro.time.Time): _description_
            meta (dict | None, optional): _description_. Defaults to None.
            dset_compression (Dset_compression, optional): _description_. Defaults to None.
            compression_opts (_type_, optional): _description_. Defaults to None.

        """
        # We don't want do put to much actions in one function call. So creating of the sound recording arrangement
        # is not part of this method. This must be done before. We check this and raise an Error if this
        # recording arrangement ist missing.
        rec_arr = Recording_arrangement.read_dataset(self._hdf5_file_path, recording_arrangement_name)

        # We process the soundfile now:
        given_sound_file_path = sound_file_path
        file_byte_hash = h5core.calc_hash_from_file(sound_file_path)
        
        # check file and content before, maybe there is a version difference
        # -> we don't need this: It was done during initializing the class object
        with h5py.File(self.hdf5_file_path, h5core.H5_OPEN_MODE.READ_WRITE_CREATE_IF_NOT_EXIST) as h5f:
            # Check if this file was already imported before
            # ----------------------------------------------
            for soundfile_dset_name in h5f[HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH].keys():
                soundfile_dset = h5f[str(HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH) + "/" + str(soundfile_dset_name)] # convert to full path ob the object
                if isinstance(soundfile_dset, h5py.Dataset):
                    # The check is positive, if the file name is the same and
                    # also the content is the same.
                    #  1) Check file name
                    same_file_name = \
                        soundfile_dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME] \
                        == given_sound_file_path.name
                    #  2) Check the hash
                    if same_file_name \
                    and (file_byte_hash == soundfile_dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.CONTENT_HASH_ATTR_NAME]):
                        # Oh! Same hash: We have to check byte by byte:
                        if self._is_file_content_identically(given_sound_file_path, soundfile_dset):
                            raise ValueError(f"Given sound file {given_sound_file_path} is already in HDF5 database file. File name, length and content are identically to a former imported file.")
            #
            # If we are here, so the file ist not yet part of the HDF5 File: We can import now.
            # ---------------------------------------------------------------------------------
            #
            # get a free number as dataset name for this file: we rely on the attribute 
            grp = h5f[HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH]
            dataset_name = str(grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.LAST_ASSIGNED_DSET_NB] + 1)
            # We save the full file, not only the sound data.
            # Explore the byte size of the content of the file and create the new dataset.
            file_content_size = os.stat(given_sound_file_path).st_size
            if     (dset_compression == h5core.Dset_compression.LZF) \
                or (dset_compression == h5core.Dset_compression.GZIP):
                shuffle = True
            else:
                shuffle = False
            dset = h5f[HD5FS.GRP_ORIGINAL_SOUND_FILES.FULL_PATH].create_dataset(dataset_name,  
                                                                           (file_content_size,), 
                                                                           dtype=np.uint8,
                                                                           compression=dset_compression,
                                                                           compression_opts = compression_opts,
                                                                           shuffle = shuffle)
            # if creation dset successful, we actualize HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.LAST_ASSIGNED_DSET_NB
            if isinstance(dset, h5py._hl.dataset.Dataset):
                # actualize LAST_ASSIGNED_DSET_NB with new value
                grp.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.GRP_ATTRIBUTES.LAST_ASSIGNED_DSET_NB] = int(dataset_name)
                # fill the file content as byte stream into the dataset
                chunk_size = 100_000 # read in ~100 kB chunks to let file size independent of available RAM
                with open(given_sound_file_path, 'rb') as f:
                    chunk = np.empty(shape=(1,), dtype=np.uint8)
                    index = 0
                    # transfer the file bytes into the hdf5-File
                    while chunk.shape[0] > 0:
                        chunk = np.fromfile(file = given_sound_file_path,
                                            dtype = np.uint8,
                                            count = chunk_size,
                                            offset = index)
                        dset[index:index+chunk.shape[0]] = chunk
                        index += chunk.shape[0]
                # Add the defined attributes of this dataset.
                # For this, we have to read file type specific content.
                f = sf.SoundFile(str(given_sound_file_path), 'r')
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME] = given_sound_file_path.name
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SOUNDFILE_TYPE_ATTR_NAME]  = h5core.pickle_to_uint8({ "format": f.format, 
                                                                                            "sub-type": f.subtype,
                                                                                            "endian": f.endian,
                                                                                            "sections": f.sections,
                                                                                            "seekable": f.seekable()
                                                                                            })
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.FILE_SIZE_ATTR_NAME]       = file_content_size
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.CONTENT_HASH_ATTR_NAME]    = file_byte_hash
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SAMPLING_RATE_ATTR_NAME]   = f.samplerate
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_ATTR_NAME]      = h5core.pickle_to_uint8(start_time)
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_READABLE_ATTR_NAME] = str(start_time)
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_CHANNELS_ATTR_NAME] = f.channels
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_ATTR_NAME]       = h5core.pickle_to_uint8(meta)
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.NUMBER_SAMPLES_PER_CHANNEL_ATTR_NAME] = f.frames
                meta_data_key_list:str = ""
                if meta is not None:
                    for key in meta.keys():
                        meta_data_key_list += str(key) + ", " 
                dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.META_DATA_KEYS_ATTR_NAME]  = meta_data_key_list
                # Add dataset into the record arrangement list
                rec_arr.add_osf_dataset_links(dset.name)
                rec_arr.write_dataset(self._hdf5_file_path)
            # create entry in time table
            time_table = Time_table(self.hdf5_file_path)
            time_table.append_soundfile(osf_dset_number               = int(dataset_name),
                            recording_arrangement         = rec_arr.unique_name,
                            time_range                    = sunpy_TimeRange(start_time, (f.frames-1)/f.samplerate * Unit('s')),
                            number_of_frames              = f.frames,
                            nominal_sampling_frequency    = f.samplerate 
                            )
            time_table.save()
                
    def remove_sound_data(self,
                          dset_link:str,
                          original_file_name:str
                          ):
        """remove_sound_data: Removes the saved original sound file byte stream.
        ------------------------------------------------------------------------
        
        Note: The sound file data can be removed in case to save memory, but the database
              keeps the dataset with all the given attributes.

        Args:
            dset_link (str): link of the dataset what keep this sound file
            original_file_name (str): for safety: the original file name like contained in the
                                      attribute of the dataset.

        Raises:
            ValueError: In case, the given sound file name is not the same as in the dataset.
        """
        with h5py.File(self._hdf5_file_path, h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            grp = h5f.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.PATH)
            dset = grp.get(dset_link)
            if dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME] == original_file_name:
                dset = np.array([], dtype=np.uint8)
            else:
                raise ValueError("Given original file name is not the same like in given data set.")
 
    def open_as_file_like(self,
                          dset:str|h5py.Dataset
                          ) -> h5core.File_like_ro:
        """Creates a Python file-like object with the original sound file data of given dataset
        ---------------------------------------------------------------------------------------

        => Thats the preferred method to work with the sound of a dataset as byte stream to save
        memory (RAM or file system) during processing: Keeps the data inside the HDF file without
        a temporary file in the file system or an data stream in the RAM.
        
        You can work with this 'file' like any other opened file. But access is read-only (mode='r').
        
        This is the preferred method to work with the sound data without to copy the full file
        bytes into the memory (like as_ndarray does). Only in case you uses functions which work
        only with a real file in the file system, use the 'open_as_tmp_file()' method or export
        the sound data with 'extract_as_original_file' before the access to the data.
        
        Note: 'open_as_file_like()' opens the sound data with the same coding like the original
              file. Use the 'soundfile' module to decode the data. For example, a typical access
              would be (for access of the sound file in dataset number 4):
              
              import soundfile as sf
              
              osf = Original_sound_file(hdf5_file_path=' ... ')
              pcm_numpy_array = sf.read(osf.open_as_file_like(HD5FS.GRP_ORIGINAL_SOUND_FILES.PATH + '/4'))
              
              With 'soundfile.read()' you have arguments to read parts of the PSC data without
              to load the full memory into the memory. It is very performant to extract chunks of 
              sound (tested with opus files).

        Args:
            dset (str | h5py.Dataset): _description_

        Returns:
            _File_like: _description_
        """
        return h5core.File_like_ro(self._hdf5_file_path, dset)            
    
    def as_ndarray(self,
                   dset:str|h5py.Dataset,
                   frames=-1, # This and following parameters are the same as 'soundfile.read()'
                   start=0,   # See: https://python-soundfile.readthedocs.io/en/0.11.0/#module-soundfile
                   stop=None, 
                   dtype='float64', 
                   always_2d=False, 
                   fill_value=None, 
                   out=None, 
                   samplerate=None, 
                   channels=None, 
                   format=None, 
                   subtype=None, 
                   endian=None, 
                   closefd=True
                   ):
        """as_ndarray(): gets back a soundfile or a part of them from a data set as numpy array (decoded)
        -------------------------------------------------------------------------------------------------

        Shape of returned array: [number samples, number channels]

        => Prefer this method if you work with a lot of random access into the sound data and you process
        with numpy based methods. It's also a good start point if you move the data into the GPU memory 
        for GPU based processing. 
        Based on 'open_as_file_like': If you use 'start' end 'stop' arguments (or 'frames'), only the
        requested data will be read, decoded and moved into RAM.
        
        All parameters beginning with 'frames' are identically to the parameters in 'soundfile.read()'.

        Args:
            dset (str | h5py.Dataset): _description_
            frames (int, optional): _description_. Defaults to -1.
            dtype (str, optional): _description_. Defaults to 'float64'.
            always_2d (bool, optional): _description_. Defaults to False.
            fill_value (_type_, optional): _description_. Defaults to None.
            out (_type_, optional): _description_. Defaults to None.
            samplerate (_type_, optional): _description_. Defaults to None.
            channels (_type_, optional): _description_. Defaults to None.
            format (_type_, optional): _description_. Defaults to None.
            subtype (_type_, optional): _description_. Defaults to None.
            endian (_type_, optional): _description_. Defaults to None.
            closefd (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray, int: requested and decoded sound data as numpy array, sample-rate
        """
        with self.open_as_file_like(dset) as fl:
            samples_ndarray, sample_rate = sf.read( fl, 
                                                    frames=frames,
                                                    start=start, 
                                                    stop=stop, 
                                                    dtype=dtype, 
                                                    always_2d=always_2d, 
                                                    fill_value=fill_value, 
                                                    out=out, 
                                                    samplerate=samplerate, 
                                                    channels=channels, 
                                                    format=format, 
                                                    subtype=subtype, 
                                                    endian=endian, 
                                                    closefd=closefd
                                                    )
        
        return samples_ndarray, sample_rate
         
    def extract_as_original_file(self,
                                 dset:str|h5py.Dataset,
                                 file_path:pathlib.Path|str
                                 ) -> pathlib.Path:
        """Extract the original sound file bytes from the database into a new file.

        It includes all bytes of the original sound file as byte-to-byte copy (not only
        the raw sound data).

        Args:
            dset (str | h5py.Dataset): dataset with the original sound file data
            file_path (File_path): Path with or without a file name. If no
                                       file name is given, so the original file name is used.

        Returns:
            pathlib.Path: _description_
        """
        file_path = pathlib.Path(file_path)
        h5f = None
        if isinstance(dset, str):
            h5f = h5py.File(self._hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE)
            dset = h5f.get(dset)
        if not file_path.is_file():
            # only path, no file name: we take the original name
            file_path /= pathlib.Path(dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME])
        with open(file_path, mode="bw") as sound_f:
            # fill the file with the original file bytes from dataset
            sound_f.write(np.array(dset, dtype=np.uint8).tobytes())
        if h5f is not None:
            h5f.close()
        return file_path
    
    def open_as_tmp_file(self,
                         dset:str|h5py.Dataset,
                         ) -> BufferedReader:
        """Opens the original sound file data as temporary file in binary read mode.

        It includes all bytes of the original sound file as byte-to-byte copy (not only
        the raw sound data).
        
        Use this only in case the calling function can not work with Python file-like
        objects. If this is possible, so use 'open_as_file_like()'. This doesn't need
        a copy of the full file bytes into a physically file directory or RAM.
        """
        if isinstance(dset, h5py.Dataset):
            dset_link = dset.name
        else:
            dset_link = dset
        # prepare as temporary file and get back the file path/name
        tmp_file = _TempSoundFile(self.hdf5_file_path,
                                 dset_link).as_pathlib_Path
        f = open(self.extract_as_original_file(dset_link,
                                               tmp_file), mode='rb')
        return f
              
    def _is_file_content_identically(self, 
                                     file: pathlib.Path, 
                                     dset: str|h5py.Dataset
                                     ) -> bool:
        is_identical = True
        h5f = None
        if isinstance(dset, str):
            h5f = h5py.File(self._hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE)
            dset = h5f.get(dset)
        if os.stat(file).st_size != dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.FILE_SIZE_ATTR_NAME]:
            is_identical = False
        else:
            with open(file, 'rb') as f:
                chunk_size = 100_000 # read in ~100 kB chunks to let file size independent of available RAM
                chunk = np.empty(shape=(1,), dtype=np.uint8)
                index = 0
                # transfer the file bytes into the hdf5-File
                while chunk.shape[0] > 0:
                    chunk = np.fromfile(file = file,
                                        dtype = np.uint8,
                                        count = chunk_size,
                                        offset = index)
                    if not np.all(np.equal(dset[index:index+chunk.shape[0]], chunk)):
                        is_identical = False
                        break
                    index += chunk.shape[0]
        if h5f is not None:
            h5f.close()
        return is_identical



# Some sound libraries like libopus can not handle file-like objects. Only real files with a string as
# file path are possible to open such files by the special libraries.
# This is the reason to replace here the typical NamedTemporaryFile object from Python tempfile module or,
# alternative, an in memory file-like object with a real file in a given temporary file folder. 
# If the object is deleted, also the file will be removed if not done before.
class _TempSoundFile():
    def __init__(self,
                 hdf5_file_path:str|pathlib.Path, 
                 hd5_original_sound_file_dataset_link:str,
                 tmp_folder:str|pathlib.Path = tempfile.gettempdir()
                 ):
        self.tmp_file = None
        self.soundfile_type = None
        self.original_file_name = None
        with h5py.File(hdf5_file_path, h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as hd5f:
            dset:h5py.Dataset = hd5f[hd5_original_sound_file_dataset_link]
            # get the soundfile type from attributes
            self.soundfile_type = h5core.unpickle_from_uint8(dset.attrs.get(HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SOUNDFILE_TYPE_ATTR_NAME))
            # get the original name (we use it as temporary name for better debugging)
            self.original_file_name:str = dset.attrs[HD5FS.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME]
            # Create the temporary file in given folder and make it save that it is a unique file name in order
            # to avoid to overwrite any other file.
            postfix:int|None = None
            tmp_file_path:pathlib.Path = pathlib.Path(tmp_folder) / pathlib.Path(self.original_file_name)
            base_file_path:pathlib.Path = tmp_file_path
            while True:
                try:
                    tmp_sound_f = tmp_file_path.open(mode='bx')
                    break
                except FileExistsError:
                    if postfix is None:
                        postfix:int = 1
                    else:
                        postfix += 1
                    tmp_file_path = base_file_path.parent / pathlib.Path(str(base_file_path.stem) + '-' + str(postfix) + base_file_path.suffix)
            del(base_file_path)
            self.tmp_file = tmp_file_path
            # fill the file with the original file bytes from dataset
            tmp_sound_f.write(np.array(dset, dtype=np.uint8).tobytes())
            tmp_sound_f.close()
    
    def original_file_name(self) -> str:
        return self.original_file_name
    
    @property
    def as_str(self) -> str:
        return str(self.tmp_file)
    
    @property
    def as_pathlib_Path(self):
        return self.tmp_file 


    