

class ModuleConfig:
    # Default-Werte (Klassenattribute)
    debug: bool = False
    log_level: str = "INFO"
    sample_rate: int = 48000
    codec: str = "opus"

    # Optional: Validierung oder Hilfsmethoden
    @classmethod
    def set(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Invalid config key: {key}")
