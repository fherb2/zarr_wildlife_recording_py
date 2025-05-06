
from zarr.errors import ZarrError

class ZarrComponentVersionError(ZarrError):
    """Error of a special component version inside a Zarr database"""
    def __init__(self, message: str = "Version error of a requested database element."):
        super().__init__(message)