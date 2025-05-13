"""Package logging module"""

import logging
from zarrwlr.config import ModuleConfig

class ModuleLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        level_name = getattr(ModuleConfig, "log_level", None)
        if level_name is None:
            return True
        min_level = logging._nameToLevel.get(level_name.upper(), logging.WARNING)
        return record.levelno >= min_level

class DebugPrintHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if getattr(ModuleConfig, "debug", False):
            try:
                msg = self.format(record)
                print(f"[DEBUG-PRINT] {msg}")
            except Exception:
                self.handleError(record)

def setup_module_logging():
    logger = logging.getLogger("zarrwlr")  # Root-Logger for module
    logger.setLevel(logging.NOTSET)
    logger.addFilter(ModuleLogFilter())
    logger.addHandler(DebugPrintHandler())
