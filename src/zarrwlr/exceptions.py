
from zarr.errors import BaseZarrError

class ZarrComponentVersionError(BaseZarrError):
    """Error of a special component version inside a Zarr database"""
    def __init__(self, message: str = "Version error of a requested database element."):
        super().__init__(message)

class Doublet(Exception):
    """Error of some dublicat."""
    def __init__(self, message: str = "Doublet error."):
        super().__init__(message)
