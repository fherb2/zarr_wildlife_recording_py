
from zarrwlr.oggfileblob import import_audio_to_blob, _get_source_params, \
    create_index, extract_audio_segment_from_blob
import os
import pathlib
import zarr
import numpy as np
from zarrwlr.utils import zarr_to_dict_snapshot, print_snapshot_dict

# ##########################################################
#
# Helpers
# =======
#
# ##########################################################

TEST_RESULTS_DIR = pathlib.Path(__file__).parent.resolve() / "testresults"
ZARR3_STORE_DIR = TEST_RESULTS_DIR / "zarr3-store"

def prepare_testresult_dir():
    # remove content in result directory or create directory
    os.system(f"rm -rf {str(TEST_RESULTS_DIR)}/*")
    os.system(f"mkdir -p {str(TEST_RESULTS_DIR)}")

def prepare_zarr_database() -> zarr.Group:
    # We don't need to prepare the root directory: LocalStore does it
    # by using root as directory name.
    store = zarr.storage.LocalStore(root=ZARR3_STORE_DIR)
    root = zarr.create_group(store=store)
    audio_import_grp = root.create_group('audio_imports')
    return audio_import_grp

def get_test_files() -> list[pathlib.Path]:
    test_files = [
                    "testdata/audiomoth_long_snippet.wav",
                    "testdata/audiomoth_long_snippet_converted.opus",
                    "testdata/audiomoth_long_snippet_converted.flac",
                    "testdata/audiomoth_short_snippet.wav",
                    "testdata/bird1_snippet.mp3",
                    "testdata/camtrap_snippet.mov" # mp4 coded video with audio stream
                ]
    return [pathlib.Path(__file__).parent.resolve() / file for file in test_files]

# ###########################################################
#
# Tests
# =====
#
# ###########################################################

def test__get_source_params():
    # preparations
    prepare_testresult_dir()
    test_files = get_test_files()

    # tests
    print("\n\n========== test__get_source_params() =============")
    for file in test_files:
        params = _get_source_params(file)
        print(f"File: {file}; {params=}")

test__get_source_params()

def test_oggfileblob():
    # preparations
    prepare_testresult_dir()
    test_files = get_test_files()
    audio_import_grp = prepare_zarr_database()
    
    # tests
    print("\n\n========== test_oggfileblob() =============")
    i = -1
    for file in test_files:
        print(f"\n\nFile: {file}\n+++++++++++++++++++++++++++++")
        params = _get_source_params(file)
        print(f"Source parameters:\{params}\n---------\n")
        i += 1
        print(f"FLAC-Import  {i=}\n------------------\n")
        new_audio_file_grp = audio_import_grp.create_group(str(i))
        audio_blob_array = new_audio_file_grp.create_array(name="audio_data_blob_array", 
                                                    shape=(0,),
                                                    dtype=np.uint8,
                                                    chunks=(int(1e6),),
                                                    shards=(int(20*1e6),))
        import_audio_to_blob(file, 
                            audio_blob_array,
                            'flac')
        i += 1
        print(f"\n\nOPUS-Import {i=}\n--------------------\n")
        
        new_audio_file_grp = audio_import_grp.create_group(str(i))
        audio_blob_array = new_audio_file_grp.create_array(name="audio_data_blob_array", 
                                                    shape=(0,),
                                                    dtype=np.uint8,
                                                    chunks=(int(1e6),),
                                                    shards=(int(20*1e6),))
        import_audio_to_blob(file, 
                            audio_blob_array,
                            'opus')
        
    print("\n\n====================================================")
    print("\n --- ZARR-Struktur ---\n")
    print_snapshot_dict(zarr_to_dict_snapshot(root_group = audio_import_grp, k=32, n=32, d=None), 3)


   
test_oggfileblob()

