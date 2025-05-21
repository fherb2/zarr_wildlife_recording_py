"""Package logging module"""

import logging
import sys
import inspect
from .config import Config

class ModuleLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        level_name = getattr(Config, "log_level", None)
        if level_name is None:
            return True
        # Nutze den Enum-Wert direkt mit logging._nameToLevel
        min_level = logging._nameToLevel.get(str(level_name), logging.WARNING)
        return record.levelno >= min_level

class DebugPrintHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if getattr(Config, "debug", False):
            try:
                msg = self.format(record)
                print(f"[DEBUG-PRINT] {msg}")
            except Exception:
                self.handleError(record)

def setup_module_logging():
    # Root-Logger für das Modul
    logger = logging.getLogger("zarrwlr")
    logger.setLevel(logging.NOTSET)  # Alle Nachrichten erfassen
    
    # Verhindere, dass Logs zum Root-Logger propagiert werden
    logger.propagate = False
    
    # Filter hinzufügen
    logger.addFilter(ModuleLogFilter())
    
    # Standard-Handler hinzufügen (immer aktiv, unabhängig von debug)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Debug-Handler nur für zusätzliche Debug-Ausgaben
    debug_handler = DebugPrintHandler()
    debug_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(debug_handler)
    
    # Logging-Initialisierung bestätigen
    logging.getLogger("zarrwlr").debug("zarrwlr logging initialized")

def get_module_logger():
    """Automatisch einen Logger für das aufrufende Modul zurückgeben."""
    frame = inspect.currentframe().f_back
    module = inspect.getmodule(frame)
    return logging.getLogger(module.__name__)

# Hilfsfunktion, die direkt einen konfigurierten Logger zurückgibt
def get_logger():
    """Shortcut zum Erstellen eines Loggers für das aufrufende Modul."""
    logger = get_module_logger()
    return logger

# Initialize during importing:
setup_module_logging()