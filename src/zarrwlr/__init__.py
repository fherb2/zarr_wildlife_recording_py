

from .logging_setup import setup_module_logging

# Beispiel: Diese Funktionen werden zur Verfügung gestellt, wenn man 'import zarrwlr' angibt.
# from .core import CoreClass, useful_function
# from .audio import AudioHandler
# from .features import extract_features

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