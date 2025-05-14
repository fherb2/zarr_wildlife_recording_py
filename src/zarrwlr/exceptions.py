
from zarr.errors import BaseZarrError


class Doublet(Exception):
    """Error of some dublicat."""
    def __init__(self, message: str = "Doublet error."):
        super().__init__(message)


class ZarrComponentVersionError(BaseZarrError):
    """Error of a special component version inside a Zarr database"""
    def __init__(self, message: str = "Version error of a requested database element."):
        super().__init__(message)

class ZarrComponentIncomplete(BaseZarrError):
    """Error that a database element is not completely created."""
    def __init__(self, message: str = "Requested database element is incomplete created."):
        super().__init__(message)

class ZarrGroupMismatch(BaseZarrError):
    """Error that the requested group is not the expected group."""
    def __init__(self, message: str = "Requested group is not the expected group (clearly recognised by scheme, key, id...)."):
        super().__init__(message)