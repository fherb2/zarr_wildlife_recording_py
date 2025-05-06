
import logging
import zarr
from zarr.errors import PathNotFoundError
from zarrwlr.module_config import ModuleStaticConfig
from zarrwlr.exceptions import ZarrComponentVersionError

# get the module logger   
logger = logging.getLogger(__name__)

def is_file_blob_present(target_group: zarr.Group, 
                         file_name: str, 
                         file_size: int, 
                         file_hash: str
                         ) -> bool:

    for group_name in target_group:
        # we need check this group only the name is only a number
        if group_name.isdigit():
            # this should be a file_blob group...
            sub_group = target_group[group_name]
            # ... and should have a version attribute:
            if "file_blob_group_version" in sub_group.attrs:
                # is compatible?
                major_setpoint, _ = ModuleStaticConfig.versions["file_blob_group_version"]
                major, _ = sub_group.attrs["file_blob_group_version"]
                if major_setpoint == major:
                    # its the right version
                    # an array with name 'file_blob' should be existing
                    if 'file_blob' in sub_group:
                        # ok, found a right group
                        file_blob_array = sub_group['file_blob']
                        # Do we find all attributes to check for identically files already imported?
                        if 'file_name' in file_blob_array.attrs and 'file_size' in file_blob_array.attrs and 'file_hash' in file_blob_array.attrs:
                            file_name = file_blob_array.attrs['file_name']
                            file_size = file_blob_array.attrs['file_size']
                            file_hash = file_blob_array.attrs['file_hash']
                            # Compare values
                            if (file_name == file_name and 
                                file_size == file_size and 
                                file_hash == file_hash):
                                return True
                        else:
                            raise PathNotFoundError(f"Missing 'file_name', 'file_size' or/and 'file_hash' as attribute(s) of a file_blob array in group {group_name}.")
                    else:
                        raise PathNotFoundError(f"Missing 'file_blob' array in group {group_name}.")
                else:
                    raise ZarrComponentVersionError(f"Found wrong version of group {group_name} attribute 'file_blob_group_version'. Major value is {major} but should be {major_setpoint}")
            else:
                raise PathNotFoundError(f"Could not find group attribute 'file_blob_group_version' in group {group_name}.")
    return False
