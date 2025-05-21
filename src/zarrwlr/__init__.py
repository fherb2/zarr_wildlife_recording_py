

"""zarrwlr package initialization."""

# Logging wird automatisch bei der ersten Verwendung initialisiert
# Kein expliziter setup_module_logging() Aufruf nötig!

# Optional: Paket-Logger für Initialisierungs-Messages
from .logsetup import package_logger
package_logger.info("zarrwlr package loaded")


# Diese Funktionen werden zur Verfügung gestellt, wenn man 'import zarrwlr' angibt.
# from .aimport import create_original_audio_group, \
#                      import_original_audio_file
# from .config import Config
# from .exceptions import *
# from .types import LogLevel, \
#                    AudioFileBaseFeatures, \
#                    AudioCompression

# Beispiel: Was bei "from zarrwlr import *" geladen wird:
# __all__ = [
#     "CoreClass",
#     "useful_function",
#     "AudioHandler",
#     "extract_features",
# ]

# Relative Pfade verwenden ist sicherer!

