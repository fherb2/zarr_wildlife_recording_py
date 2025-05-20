# veraltet
# --------

import logging
from pathlib import Path
from io import BytesIO, BufferedReader
import wave
from typing import TypedDict

# get the module logger   
logger = logging.getLogger(__name__)

def is_wav_file(file: str | Path | BufferedReader) -> bool:
    
    if isinstance(file, str):
        file = Path(file)
    
    is_wav = False
    if isinstance(file, BufferedReader):
        # read from start position of opened file and reset the
        # pointer to the value before
        current_position = file.tell()
        with wave.open(file, 'rb') as wf:
            try:
                wf.getparams()
                is_wav = True
            except (wave.Error, EOFError):
                pass
        file.seek(current_position)
    elif isinstance(file, Path):
        with wave.open(file, 'rb') as wf:
            try:
                wf.getparams()
                is_wav = True
            except (wave.Error, EOFError):
                pass
    else:
        raise ValueError("file must be of type: str, Path (both as a valid file path) or BufferedReader (as already opened file).")
    
    return is_wav


class WavMetadata(TypedDict, total=False):

def extract_wav_metadata(blob: bytes) -> WavMetadata: