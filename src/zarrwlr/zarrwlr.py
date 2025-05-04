
import logging

class ModuleConfig:
    log_level: str | None = None  # None: no internal lof filtering
    debug: bool = False # special debugging mode for additional deep inspecting via print/log messages
    std_zarr_chunk_size: int = 8*1024*1024 # 8 MByte is a good value for access for both: local resources and network

class ModuleLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        level_name = getattr(ModuleConfig, "log_level", None)
        if level_name is None:
            return True  # Keine zusätzliche Filterung: alles durchlassen
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  
logger.addFilter(ModuleLogFilter())
logger.addHandler(DebugPrintHandler())

