# Save unprocessed recording detections in a HDF results directory (group)

import pathlib
from dataclasses import dataclass
from typing import Any
import h5py
import numpy as np

from birdnettoolbox.cores.original_sound_files import HD5FS as HD5FS_OSF
import birdnettoolbox.cores.hdf5 as h5core

@dataclass 
class PredictionDictKeys:
    COMMON_NAME = 'common_name'
    SCIENTIFIC_NAME = 'scientific_name'
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    CONVFIDENCE = 'confidence'
    LABEL = 'label'

# How to save detections as dataset?
# ----------------------------------
#
#
# column 1: osf dset number
# column 2: channel (may be the stereo channel of the original recording or any channel of a preprocessed sound location detection)
# column 2: absolute start time of analyzed sound chunk
# column 3: absolute end time of analyzed sound chunk
# column 4: taxonomy_label index number
# column 5: confidence of the recognition

class RecordingDetections:
    # Contains, manages, writes recording detections of exactly one original sound file
    def __init__(self,
                 hdf5_file_path:pathlib.Path|str,
                 osf_dset_name_number:int|str,
                 full_dset_link:str
                 ):
        self.hdf5_file_path = hdf5_file_path
        self.recording_detections_dset_link = dset_link
        # self.osf_dset_number (may be given as full link, the name or the name as integer
        assert isinstance(osf_dset_name_number, int|str), \
            f"osf_dset_name_number must be the full dset link, the dset name as character string or the dset name as integer"
        if isinstance(osf_dset_name_number, int):
            self.osf_dset_number = osf_dset_name_number
        elif '/' in osf_dset_name_number:
            self.osf_dset_number = int(osf_dset_name_number.rsplit('/', 1)[-1])
        else:
            self.osf_dset_number = int(osf_dset_name_number)
        # this dataset as full link:
        self.osf_dset_link = HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH + '/' + str(self.osf_dset_number)
        # start time of the recording
        with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            dset = h5f.get(self.osf_dset_link)
            # get as astropy Time object:
            self.osf_start_time = dset.attrs[HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.START_TIME_ATTR_NAME]
        # create the result dset if needed
        with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            h5f.require_dataset(full_dset_link, shape=(0,5), dtype=np.float32, exact=True, maxshape=(None, 5))

    def add_detections(birdnet_result_list:list[dict[str, Any]]|dict[str, Any]):
        # convert a detection or a list of detections into a dset line(s) and save it
        pass


def save_recording_detections(hdf5_file_path:pathlib.Path|str,
                              hdf5_result_group:str,
                              osf_dset_number:int,
                              birdnetlib_recording_detections: list[dict[str, Any]],
                              ):
    # do it in class RecordingDetections?
    pass
    
