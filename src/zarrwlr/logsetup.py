"""Package logging module"""

"""Einfaches, robustes Logging-System für zarrwlr"""
import logging
import sys
from .config import Config

def get_logger(name: str = None):
    """
    Gibt einen konfigurierten Logger zurück.
    
    Args:
        name: Logger-Name (optional, wird automatisch bestimmt wenn None)
    """
    if name is None:
        # Automatisch den Modulnamen bestimmen
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else "zarrwlr"
    
    logger = logging.getLogger(name)
    
    # Nur konfigurieren, wenn noch nicht geschehen
    if not logger.handlers:
        _configure_logger(logger)
    
    return logger

def _configure_logger(logger):
    """Konfiguriert einen einzelnen Logger."""
    
    # Level setzen
    logger.setLevel(logging.DEBUG)  # Immer alle Messages erfassen
    
    # Handler erstellen
    handler = logging.StreamHandler(sys.stdout)
    
    # Formatter erstellen
    formatter = logging.Formatter(
        fmt='%(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Filter hinzufügen (prüft Config-Werte zur Laufzeit)
    handler.addFilter(_create_config_filter())
    
    # Handler zum Logger hinzufügen
    logger.addHandler(handler)
    
    # Debug-Handler hinzufügen (wenn Config.debug aktiv)
    debug_handler = _DebugPrintHandler()
    debug_handler.addFilter(_create_config_filter())
    logger.addHandler(debug_handler)
    
    # Propagation verhindern, um Duplikate zu vermeiden
    logger.propagate = False

def _create_config_filter():
    """Erstellt einen Filter, der Config-Werte zur Laufzeit prüft."""
    def config_filter(record):
        # Log-Level aus Config holen
        config_level = getattr(Config, 'log_level', None)
        if config_level is None:
            return True
        
        # String-Wert des Enums holen und in numerischen Level umwandeln
        try:
            min_level = getattr(logging, str(config_level))
        except AttributeError:
            min_level = logging.WARNING
        
        return record.levelno >= min_level
    
    return config_filter

class _DebugPrintHandler(logging.Handler):
    """Handler für zusätzliche Debug-Ausgaben."""
    
    def emit(self, record):
        # Nur ausgeben, wenn Config.debug True ist
        if getattr(Config, 'debug', False):
            try:
                msg = self.format(record)
                print(f"[DEBUG] {msg}")
            except Exception:
                self.handleError(record)

# Paket-weiter Logger für allgemeine Verwendung
package_logger = get_logger("zarrwlr")