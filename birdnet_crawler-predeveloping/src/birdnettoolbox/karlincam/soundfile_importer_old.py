import pathlib
import re
import os
from astropy.time import Time
import soundfile as sf
import h5py

from birdnettoolbox.cores.hdf5 import Dset_compression, \
                                      h5py_work_arounds, \
                                      delete_all

from birdnettoolbox.cores.original_sound_files import  Original_sound_file, \
                                                            Hd5_structure_Original_sound_files
                                                            

from birdnettoolbox.cores.hdf5_preprocessed_sound_files import Hd5_structure_Preprocessed_sound_files, \
                                                          Preprocessed_sound_file

HD5S_ORG = Hd5_structure_Original_sound_files
HD5S_PRESOUND = Hd5_structure_Preprocessed_sound_files

def import_karlincamsound_folder(soundfile_folder: str|pathlib.Path,
                                 hdf5_file_path: str|pathlib.Path,
                                 dset_compression: Dset_compression|None = None,
                                 dset_compression_opts = None
                                 ):
    # type adaption
    # -------------
    if type(soundfile_folder) != pathlib.Path:
        soundfile_folder = pathlib.Path(soundfile_folder)
    assert soundfile_folder.is_dir(), f"Given argument 'soundfile_folder' is not a directory."
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
    
    print(f"We import the sound files...")
    import_karlincamsound_folder(soundfolder, hd5_file_path, Dset_compression.GZIP, dset_compression_opts=9)

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
    dset_link = osf.HD5S.GRP_ORIGINAL_SOUND_FILES.PATH + "/4"
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
    dset_link = osf.HD5S.GRP_ORIGINAL_SOUND_FILES.PATH + "/4"
    f = osf.open_as_file_like(dset_link)
    data, samplerate = sf.read(f, frames=10, start=42119600)
    f.close()
    print(f"{samplerate=}")
    print(f"{len(data)=}")
    
    
    h5f = h5py.File(hd5_file_path)
    grp = h5f.get(HD5S_ORG.GRP_ORIGINAL_SOUND_FILES.PATH)
    dset = grp.get("4")
    tmpf = osf.open_as_tmp_file(dset)
    tmpf.close()
    h5f.close()
    print(f"=================================================\n")

    with h5py.File(hd5_file_path, mode='r+') as h5f:
        h5f.create_group('test')
        h5f.create_dataset('dset_test', dtype='int16')
        h5f.attrs['test'] = 'testattribut'

    delete_all(hd5_file_path)


    print(f"Test if HDF file will be smaller if data are removed.")
#    h5f = h5py.File(hd5_file_path, mode='r+')
#    del(h5f[Hd5_structure_Original_sound_files.GRP_ORIGINAL_SOUND_FILES.PATH])
#    h5f.close()
#    h5py_work_arounds.repack(hd5_file_path)
    

    # ppsf = Preprocessed_sound_file(hd5_file_path,
    #                             'test',
    #                             'postfixcomment ist das',
    #                             'proc-meth Beschreibung')

        
if __name__ == "__main__":
    main()     