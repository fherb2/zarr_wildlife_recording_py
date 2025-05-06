"""High level objects and functions for Zarr Wildlife Recording library"""

import logging
import zarr
import numpy as np
from zarrwlr.files import SingleOrList_of_FilePaths, to_file_path_list, file_content_hash, is_allowed_sound_file_type
from zarrwlr.zarropus import is_opus_file
from zarrwlr.utils import next_numeric_group_name, file_size
from zarrwlr.module_config import ModuleConfig, ModuleStaticConfig
from zarrwlr.file_blob import is_file_blob_present

# get the module logger
logger = logging.getLogger(__name__)


def import_audio_file(files: SingleOrList_of_FilePaths, target_group: zarr.Group) -> None:

    files = to_file_path_list(files) # to pathlib.Path list if there are str-paths given

    # run throughout the file list...
    for file in files:

        # Skip if file is not allowed
        if not is_allowed_sound_file_type(file):
            logger.warning(f"File {file.name} is not allowed as valid sound file for import.")
            continue

        # Skip, if file content already in database
        content_hash = file_content_hash(file) # we need this later for a second time; Its a bytewise content hash of the full file.
        if is_file_blob_present(target_group, 
                                file_name=file.name, 
                                file_size=file_size(file), 
                                file_hash=content_hash):
            logger.warning(f"File {file.name} with same size and hash found in database. Skipping.")
            continue
        
        # new group for this file blob with ascending numeric value
        group_name = next_numeric_group_name(target_group)

        # go into this new group
        current_grp = target_group.require_group(group_name)
        current_grp.attrs["file_blob_group_version"] = ModuleStaticConfig.versions["file_blob_group_version"]

        # Open file for reading into zarr
        with open(file, "rb") as f:

            # create the file blob array
            blob_size = file_size(f)
            chunk_size = ModuleConfig.new_file_blob_chunk_size
            if chunk_size > blob_size:
                chunk_size = blob_size

            shard_size = ModuleConfig.new_file_blob_shard_size

            file_blob = current_grp.create_array(
                name="file_blob",
                shape=(blob_size,),
                chunks=(chunk_size,),
                shards=(shard_size,),
                dtype="u1"
            )

            # fill file blob array step by step
            # to avoid memory problems if file is very big
            offset = 0
            while True:
                buffer = f.read(size = 1*1024*1024)
                if not buffer:
                    break
                arr = np.frombuffer(buffer, dtype="u1")
                file_blob[offset : offset + len(arr)] = arr
                offset += len(arr)
                
            # Add general attributes to file_blob array
            file_blob.attrs['file_name'] = file.name
            file_blob.attrs['file_size'] = blob_size
            file_blob.attrs['file_hash'] = content_hash
            file_blob.attrs['file_import_path'] = file.resolve().parent
            # Add the version of importing this array to the array containing group
            current_grp.attrs['file_blob_group_version'] = ModuleStaticConfig.versions["file_blob_group_version"]


            # following finally tasks are file type depending
            if is_opus_file(f):


# Die File-abhängigen Codezeilen sollten in jeweils zum File passendes Modul, wie z.B. zarropus,
# ausgelagert werden. Dann herrscht hier mehr Ordnung.

                # create the important package index from blob
                index = build_packet_index(blob)


                


    # Neue Erkenntnis: Wir legen die Chunk-Size dynamisch fest. Definition:
    #
    # 1) Der Nutzer gibt die Chunk-size vor: Wir verwenden diese. (allgemeine Konfiguration oder Import-Parameter?)
    # 2) Der Import hat eine Länge von > 2-facher Default-Chunk-Size: Wir legen die Chunk-Size auf den Default-Wert fest.
    # 3) anonsten: Chunksize := tatsächlicher Datenlänge. Also exakt 1 Chunk pro Import.

                # Zarr Array mit einer festen Chunk-Größe von 8 MB erstellen
                zarr.array(
                    blob_arr,
                    store=target_group.store,
                    path=f"{group_path}/file_blob",
                    chunks=(DEFAULT_CHUNK_SIZE,),  # Chunk-Größe als Tuple (8 MB)
                    overwrite=False  # Standardmäßig nicht überschreiben, um Fehler zu vermeiden
                )

                # Index für Pakete speichern
                for serial, packets in index.items():
                    packet_dtype = np.dtype([
                        ("start", np.int64),
                        ("end", np.int64),
                        ("granule", np.int64),
                        ("page_sequence", np.int32)
                    ])

                    # Strukturierte Paket-Daten erstellen
                    packet_data = np.array(
                        [(p.start, p.end, p.granule, p.page_sequence) for p in packets],
                        dtype=packet_dtype
                    )

                    # Index-Daten als Dataset in Zarr speichern
                    subgrp = target_group.create_group(f"{group_path}/index/{serial}")
                    subgrp.create_dataset(f"packet_data", data=packet_data, chunks=(DEFAULT_CHUNK_SIZE,), overwrite=False)

        # elif ...other format ...
        else:
            logger.error("File 'file_path' in given file list is not supported to be imported from module 'zarrwlr'")

