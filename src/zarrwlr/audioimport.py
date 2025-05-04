
from typing import Union
from pathlib import Path
import numpy as np
import zarr

from zarrwlr.zarropus import is_opus_file, build_packet_index, import_opus_file

# we support user depending configured logging
import logging
logger = logging.getLogger(__name__)

# Standard-Chunkgröße: 8 MB
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB in Bytes

StrOrPath = Union[str, Path]
ListOrTupleOfStrOrPath = Union[list[StrOrPath], tuple[StrOrPath, ...]]
AllowedInputType_import_file_files = Union[StrOrPath, ListOrTupleOfStrOrPath]

def next_numeric_group_name(zarr_group: zarr.Group) -> str:

    # Files are imported into numbered Zarr groups. In order
    # to add a new file, we have to find the next free number.
    existing = [int(k) for k in zarr_group.group_keys() if k.isdigit()]
    next_index = max(existing, default=-1) + 1
    return str(next_index)

def import_file(files: AllowedInputType_import_file_files, target_group: zarr.Group) -> None:

    # make files to type List[Path]
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    elif isinstance(files, (list, tuple)):
        files = [Path(item) for item in files]

    # run throughout the file list...
    for file_path in files:
        if is_opus_file(file_path):
            # its an OPUS-File

            # Read file as blob
            with open(file_path, "rb") as f:
                blob = f.read()

            # create the important package index from blob
            index = build_packet_index(blob)

            # Zielgruppe mit aufsteigendem numerischem Namen finden/anlegen
            group_name = next_numeric_group_name(target_group)
            group_path = f"{target_group.path}/{group_name}" if target_group.path else group_name

            # nur als Idee
            # if zarr_group.store.exists(f"{group_path}/file_blob"):
            #     raise RuntimeError(f"Ziel '{group_path}/file_blob' existiert bereits – Import abgebrochen.")

            # Bytes als uint8 in file_blob speichern
            blob_arr = np.frombuffer(blob, dtype=np.uint8)


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



