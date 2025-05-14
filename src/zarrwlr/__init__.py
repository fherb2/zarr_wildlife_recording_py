

from .logsetup import setup_module_logging

# Diese Funktionen werden zur Verfügung gestellt, wenn man 'import zarrwlr' angibt.
from .aimport import create_original_audio_group, \
                     import_original_audio_file
from .config import Config
from .exceptions import *
from .types import LogLevel, \
                   AudioFileBaseFeatures, \
                   AudioCompression

# Beispiel: Was bei "from zarrwlr import *" geladen wird:
# __all__ = [
#     "CoreClass",
#     "useful_function",
#     "AudioHandler",
#     "extract_features",
# ]

# Relative Pfade verwenden ist sicherer!

# initialize logging
setup_module_logging()