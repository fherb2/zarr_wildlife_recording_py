import pathlib
import re
import os
from astropy.time import Time
from sunpy.time import TimeRange as sunpy_TimeRange
import soundfile as sf
import h5py
from time import sleep
import soundfile as sf
from datetime import datetime

from birdnetlib import RecordingFileObject, Recording, RecordingBuffer

from birdnetlib.analyzer import Analyzer

import birdnettoolbox.cores.hdf5 as h5core

from birdnettoolbox.cores.original_sound_files import   Original_sound_file, \
                                                        Recording_arrangement, \
                                                        Time_table
                                                            
from birdnettoolbox.cores.original_sound_files import HD5FS as HD5FS_OSF

from birdnettoolbox.cores.project_analyser import GenericAnalysis, UsedModules

def import_karlincamsound_folder(soundfile_folder: str|pathlib.Path,
                                 hdf5_file_path: str|pathlib.Path,
                                 recording_arrangement: Recording_arrangement|str,
                                 dset_compression: h5core.Dset_compression|None = None,
                                 dset_compression_opts = None
                                 ):
    # type adaption
    # -------------
    soundfile_folder = pathlib.Path(soundfile_folder)
    assert soundfile_folder.is_dir(), f"Given argument 'soundfile_folder' is not a directory."
    if isinstance(recording_arrangement, str):
        recording_arrangement = Recording_arrangement.read_dataset(hdf5_file_path, recording_arrangement)
    #
    # read soundfile names from folder and prepare import
    # ---------------------------------------------------
    # We use only files of given folder and don't take files from deeper
    # subdirectories.
    # We use Opus in Karlincam sound files. The name convention
    # with time-data-content is defined, so we can rely on it.

    # We have 2 channels in this data.
    # 
    filelist = []
    for entry in soundfile_folder.iterdir():
        if entry.is_file():
            if entry.suffix == '.opus':
                filelist.append(entry.name)
    if len(filelist) == 0:
        raise ValueError("No opus files found in given 'soundfile_folder'.")   
    filelist.sort()
    #
    # Go throughout the list and import
    # ---------------------------------
    for file in filelist:
        filepath = soundfile_folder / file
        # create the start time stamp from file name
        # -> Time is coded in filename as UTC.
        p = re.compile("^karlinsound_Y(\d{4})_dayOfYear\d+_m(\d{2})_d(\d{2})_H(\d{2})_M(\d{2})_S(\d{2}).opus$")
        match = p.fullmatch(file)
        if match is None:
            raise ValueError("Problem with file {file}: The file name doesn't match the given time stamp structure!")
        print(f"match[0]: {match[0]}")
        year  = match[1]
        month = match[2]
        day   = match[3]
        hour  = match[4]
        min   = match[5]
        sec   = match[6]
        # convert to astropy.time.Time
        time_string = f"{year}-{month}-{day}T{hour}:{min}:{sec}" 
        astro_time = Time([time_string], format='isot', scale='utc')[0]
        # ready to import 
        osfs = Original_sound_file(hdf5_file_path)
        osfs.import_soundfile(filepath,
                              recording_arrangement.unique_name,
                              astro_time,
                              meta = None,
                              dset_compression = dset_compression,
                              compression_opts = dset_compression_opts
                              )
        
def main():
    soundfolder="/workspace/test_zone/audio_source"
    hd5_file_path="/workspace/src/birdnettoolbox/karlincam/test.hdf5"
    # test:
    try:
        os.remove(hd5_file_path)
    except:
        pass
    
    print(f"\n=================================================")
    print(f"We use soundfile module. The available formats as")
    print(f"printed information (verbose=True) and as list:")
    l = Original_sound_file.available_formats(verbose = True)
    print(l)
    print(f"=================================================\n")

    print(f"We check our sound files if they are compatible:")
    sfolder = pathlib.Path(soundfolder)
    for element in sfolder.iterdir():
        if element.is_file():
            try:
                print(f"File {element.name}: {Original_sound_file.check_compatibility(element)}")
            except ValueError as err:
                print(f"File {element.name}: {err}")
    print(f"=================================================\n")
    


    print(f"In order to import sound files, we have to create a")
    print(f"Original_sound_file object. This object needs a new")
    print(f"or existing HDF5 file.")
    osf = Original_sound_file(hd5_file_path)
    print(f"osf: {osf}")
    print(f"=================================================\n")

    
    print(f"Are there some datasets in {osf.hdf5_file_path.name}?")
    print(f"{osf.dsets_list}")
    print(f"=================================================\n")
    
    print(f"Create recording arrangement...")
    rec_arr = Recording_arrangement(unique_name="karlin_8_garden_stereo_setting",
                                    location_longitude_degree=50.995,
                                    location_latitude_degree=14.311,
                                    meta_data={'comment': 'Position under the umbrella on the ground',
                                               'important_noise_sources': ['wind in the trees', 
                                                                           'drops of water falling onto the umbrella surface'
                                                                          ]
                                              }
                                   )
    rec_arr.write_dataset(hd5_file_path)
    print(f"=================================================\n") 



    print(f"We import the sound files...")
    import_karlincamsound_folder(soundfolder, 
                                 hd5_file_path, 
                                 recording_arrangement=str(rec_arr),
                                 dset_compression=h5core.Dset_compression.GZIP
                                 )   

    print(f"We reopen the HDF5 file...")
    osf = Original_sound_file(hd5_file_path)
    for dset in osf.dsets_list:
        print(f"{dset}\n")
    print(f"Number of datasets: {osf.nb_dsets}")
    print(f"=================================================\n")
    



    print(f"Test with the 4th Dataset...")
    print(f"It works. It was tested.")
    #  osf.extract_as_original_file(HD5S.GRP_ORIGINAL_SOUND_FILES.PATH + '/4',
    #                             '/workspace/test_zone')
    print(f"=================================================\n")

    print(f"Open and work as file-like object.")
    osf = Original_sound_file(hd5_file_path)
    # we use 4th of the imported data sets:
    dset_link = HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH + "/4"
    f = osf.open_as_file_like(dset_link)
    bytestream_a = f.read()
    tf = osf.open_as_file_like(dset_link)
    bytestream_b = tf.read()
    tf.close()
    print(f"{len(bytestream_a)=} -- {len(bytestream_b)=}")  
    print(f"{bytestream_a==bytestream_b=}")
    print(f"{bytestream_a[:20]=}")
    print(f"{bytestream_b[:20]=}")
    f.seek(0)
    bytestream_c = f.read(10)
    print(f"{bytestream_c=}")
    f.close()
    print(f"=================================================\n")
    
    print(f"Open and work with soundfile by using datasets as")
    print(f"file-like object.")
    osf = Original_sound_file(hd5_file_path)
    # we use 4th of the imported data sets:
    dset_link = HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH + "/4"
    f = osf.open_as_file_like(dset_link)
    data, samplerate = sf.read(f, frames=10, start=42119600)
    f.close()
    print(f"{samplerate=}")
    print(f"{len(data)=}")
    
    h5f = h5py.File(hd5_file_path)
    grp = h5f.get(HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH)
    dset = grp.get("4")
    tmpf = osf.open_as_tmp_file(dset)
    tmpf.close()
    h5f.close()
    print(f"=================================================\n")

    with h5py.File(hd5_file_path, mode='r+') as h5f:
        h5f.create_group('test')
        h5f.create_dataset('dset_test', dtype='int16')
        h5f.attrs['test'] = 'testattribut' 

    h5core.delete_all(hd5_file_path)

    # Following is only a test for 'h5core.work_arounds.repack)'.
    # The removing of HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES is not
    # intended to protect the original data from destruction.
    print(f"Test if HDF file will be smaller if data are removed.")
    h5f = h5py.File(hd5_file_path, mode='r+')
    del(h5f[HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH])
    h5f.close()
    h5core.work_arounds.repack(hd5_file_path)

    # Now we fill the file again.
    rec_arr.write_dataset(hd5_file_path)
    import_karlincamsound_folder(soundfolder, 
                                 hd5_file_path, 
                                 recording_arrangement=str(rec_arr),
                                 dset_compression=h5core.Dset_compression.GZIP
                                 )   


    class Analysis(GenericAnalysis):
        def __init__(self, 
                     hdf5_file_path:pathlib.Path|str
                     ):
            super().__init__(hdf5_file_path)

    an = Analysis(hd5_file_path)
    an.write_analysis_class_code_lines(comment="My class comment.")
    print(f"{an.analysis_group_comment=}")
    print(f"{an.analysis_group_meta_data=}")
    print(f"{an.processing_documentation_comment=}")
    print(f"{an.processing_documentation_meta_data=}")
    print(f"{an.read_analysis_class_code_lines()=}")

    um = UsedModules()
    an.write_used_modules(um, comment="My comment to the used modules.")
    um1 = an.read_used_modules()
    print(f"Read used modules. Module list: {um1.modules_name_list}")
  #  um1.export('/workspace')
    
    print("\n\n")

    # Test some analysis now
    # ======================

    class AnalysisTest(GenericAnalysis):
        def __init__(self, 
                     hdf5_file_path:pathlib.Path|str
                     ):
            super().__init__(hdf5_file_path)

        def run(self):
            print("===============================\nStart to run analysis.\n")
            # read the time table to prepare the loop
            time_table = Time_table(self.hdf5_file_path)
            tt_dict = time_table.get_dict_from_time_table()
            print(f"{tt_dict=}")
            # walk trough the original_dsets
            with_file_ok_counter = 0
            with_file_false_counter = 0
            with_h5_ok_counter = 0
            with_h5_false_counter = 0
            exeption_list_file = []
            exeption_list_h5 = []
            for i in range(len(tt_dict['original_dset_number_list'])):
                original_dset_number = tt_dict['original_dset_number_list'][i]
                recording_arrangement_unique_name = tt_dict['recording_arrangement_unique_name_list'][i]
                rec_arrgmt = Recording_arrangement.read_dataset(self.hdf5_file_path,
                                                                recording_arrangement_unique_name)
                lat = rec_arrgmt.location_latitude_degree
                lon = rec_arrgmt.location_longitude_degree
                print(f"{original_dset_number=}\n   {lat=}\n   {lon=}")          
                osf = Original_sound_file(hdf5_file_path=self.hdf5_file_path, dont_create=True)
                dset_link = HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.FULL_PATH + '/' + str(original_dset_number)
                print(f"{dset_link=}")
                with h5py.File(self.hdf5_file_path) as h5f:
                    dset =  h5f.get(dset_link)
                    filename = dset.attrs[HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.ORIGINAL_SOUND_FILE_NAME_ATTR_NAME]
                    print(f"Name of file was: {filename=}\n") 
                    samplerate = dset.attrs[HD5FS_OSF.GRP_ORIGINAL_SOUND_FILES.DSET_OSF.DSET_ATTRIBUTES.SAMPLING_RATE_ATTR_NAME]
                analyzer = Analyzer()

                print(f"We use osf.as_ndarray()...")
                sleep(3)
                print(f"We take: {dset_link=}")
                
                
                # following, we separate the stereo channels
                try:
                    sound_data, samplerate = osf.as_ndarray(dset_link)
                    print(f"{samplerate=}")
                    print(f"{sound_data.shape=}")
                    left = sound_data[:,0]
                    print(f"Left samples: {left.shape=}")
                    recording = RecordingBuffer(analyzer=analyzer,
                                                buffer = left,
                                                rate = samplerate,
                                                week_48=-1
                                                )
                    recording.analyze()
                    print(recording.detections)
                    print("look if ok")
                    sleep(3)
                    right = sound_data[:,1]
                    print(f"Right samples: {right.shape=}")
                    recording = RecordingBuffer(analyzer=analyzer,
                                                buffer = right,
                                                rate = samplerate,
                                                week_48=-1
                                                )
                    recording.analyze()
                    print(recording.detections)
                    print("look if ok")
                    sleep(3)
                    with_h5_ok_counter += 1
                except Exception as err:
                    print(f"Exception happened: {err}")
                    sleep(10.0)
                    with_h5_false_counter += 1
                    exeption_list_h5.append(i)

            print(f"\nAuswertung:")
            print(f"   {with_file_ok_counter=}")
            print(f"   {with_file_false_counter=}")
            print(f"   {with_h5_ok_counter=}")
            print(f"   {with_h5_false_counter=}")
            print(f"   {exeption_list_file=}")
            print(f"   {exeption_list_h5=}")

    analysis = AnalysisTest(hd5_file_path)
    analysis.write_used_modules(UsedModules(), comment="My comment to the used modules.")
    analysis.run()
            
if __name__ == "__main__":
    main()     