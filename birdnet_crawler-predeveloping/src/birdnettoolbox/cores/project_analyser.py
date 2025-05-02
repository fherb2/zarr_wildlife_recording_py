
import pathlib
from dataclasses import dataclass
import h5py
import zipfile
import numpy as np
import pathlib
import inspect
import zipfile
import importlib
import io
from birdnetlib.species import SpeciesList


import birdnettoolbox.cores.hdf5 as h5core

@dataclass 
class Hd5_folder_structure_Analyses:
    @dataclass
    class __Grp_analyses:
        NAME        = "analyses"
        FULL_PATH   = "/" + NAME
        @dataclass
        class __Grp_attributes():
            COMMENT_ATTR_NAME       = "comment"
            COMMENT_VALUE           = "Contains results and metadata of automated analyzing processes. Each analyzing process creates an own group inside this group. Groups are numbered, but a readable name is assigned to each group."
            LAST_ASSIGNED_GRP_NB_ATTR_NAME = "last_assigned_group_nb" # (analyses groups will be sequentially numbered)
        GRP_ATTRIBUTES = __Grp_attributes
        @dataclass
        class __Grp_analysis:
            # NAME -> will be generated as sequentially numbered unique number
            @dataclass
            class __Grp_attributes():
                COMMENT_ATTR_NAME       = "comment"
                META_DATA_ATTR_NAME     = "meta_data"
            GRP_ATTRIBUTES = __Grp_attributes
            @dataclass
            class __Grp_process_doc:
                NAME = "processing_documentation"
                # FULL_PATH = FULL_PATH of __Grp_analysis + NAME
                COMMENT_ATTR_NAME       = "comment"
                COMMENT_VALUE           = "Contains an automatic source text copy of the analyzing process."
                META_DATA_ATTR_NAME     = "meta_data"
                @dataclass
                class __Dset_analyser:
                    NAME = 'analyzer_class'
                    COMMENT_ATTR_NAME       = "comment"
                    META_DATA_ATTR_NAME     = "meta_data"
                    # content is pickled Analyzer(GenericAnalysis)
                DSET_ANALYZER = __Dset_analyser
                @dataclass
                class __Dset_modules:
                    NAME = 'used_modules'
                    COMMENT_ATTR_NAME       = "comment"
                    MODULES_LIST_ATTR_NAME  = "pickled_modules_list"
                    META_DATA_ATTR_NAME     = "meta_data"
                    # Content is a zip archive containing module files
                DSET_MODULES = __Dset_modules
            GRP_PROCESS_DOC = __Grp_process_doc
            @dataclass
            class __Grp_results:
                NAME = "results"
                # FULL_PATH = FULL_PATH of __Grp_analysis + NAME
                COMMENT_ATTR_NAME       = "comment"
                COMMENT_VALUE           = "Contains all produced results depending on the analyzing process."
                META_DATA_ATTR_NAME     = "meta_data" # contains any data as pickled dictionary
                TAXONOMY_LABEL_TO_INDEX_DICT_ATTR_NAME = "taxonomy_label_to_index_dict"
                INDEX_TO_TAXONOMY_LABEL_DICT_ATTR_NAME = "index_to_taxonomy_label_dict"
            GRP_RESULTS = __Grp_results
        GRP_ANALYSIS = __Grp_analysis
    GRP_ANALYZES = __Grp_analyses
HD5FS = Hd5_folder_structure_Analyses


def get_taxonomy_dicts(hdf5_file_path:pathlib.Path|str,
                        grp_results_link:str) -> tuple[dict[str, int], dict[int, str]]:
    taxonomy_label_to_index_dict:dict[str, int] = {}
    index_to_taxonomy_label_dict:dict[int, str] = {}
    # create taxonomy attributes if needed or import
    with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
        grp_results = h5f.get(grp_results_link)
        if not HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.TAXONOMY_LABEL_TO_INDEX_DICT_ATTR_NAME in grp_results.attrs:
            grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.TAXONOMY_LABEL_TO_INDEX_DICT_ATTR_NAME] = h5core.pickle_to_uint8(taxonomy_label_to_index_dict) # write empty dictionary
        else:
            taxonomy_label_to_index_dict = h5core.unpickle_from_uint8(grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.TAXONOMY_LABEL_TO_INDEX_DICT_ATTR_NAME])
        if not HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.INDEX_TO_TAXONOMY_LABEL_DICT_ATTR_NAME in grp_results.attrs:
            grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.INDEX_TO_TAXONOMY_LABEL_DICT_ATTR_NAME] = h5core.pickle_to_uint8(index_to_taxonomy_label_dict) # write empty dictionary
        else:
            index_to_taxonomy_label_dict = h5core.unpickle_from_uint8(grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.INDEX_TO_TAXONOMY_LABEL_DICT_ATTR_NAME])
    return taxonomy_label_to_index_dict, index_to_taxonomy_label_dict


def actualize_taxonomy_labels(hdf5_file_path:pathlib.Path|str,
                              grp_results_link:str) -> tuple[dict[str, int], dict[int, str]]:
    taxonomy_label_to_index_dict, index_to_taxonomy_label_dict = get_taxonomy_dicts(hdf5_file_path, grp_results_link)
    # consistence check
    assert len(taxonomy_label_to_index_dict) == len(index_to_taxonomy_label_dict), \
        "Consistence check failed: 'taxonomy_label_to_index_dict' and 'index_to_taxonomy_label_dict' have different length."
    if len(taxonomy_label_to_index_dict) > 0:
        # We have entries in the dictionary. Check the maximum index number.
        last_index = max(index_to_taxonomy_label_dict.keys()) # thats is the last index in the first dictionary
        #   now look for the last index in the other dictionary:
        max_index = 0
        for _, i in taxonomy_label_to_index_dict.items():
            if i > max_index:
                max_index = i
        assert last_index == max_index, \
            "Consistence check failed: 'taxonomy_label_to_index_dict' and 'index_to_taxonomy_label_dict' have different indices."
    # ok, consistence check done; our last (highest) index is last_index
    # Now we can add new items from the birdnetlib:
    something_added = False
    for species in SpeciesList().return_list_for_analyzer(threshold=0.0):
        if species not in taxonomy_label_to_index_dict.keys():
            something_added = True
            last_index += 1
            taxonomy_label_to_index_dict[species]    = last_index
            index_to_taxonomy_label_dict[last_index] = species
    if something_added:
        # so we should it write into the file
        with h5py.File(hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            grp_results = h5f.get(grp_results_link)
            grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.TAXONOMY_LABEL_TO_INDEX_DICT_ATTR_NAME] = h5core.pickle_to_uint8(taxonomy_label_to_index_dict)
            grp_results.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.INDEX_TO_TAXONOMY_LABEL_DICT_ATTR_NAME] = h5core.pickle_to_uint8(index_to_taxonomy_label_dict)
    return taxonomy_label_to_index_dict, index_to_taxonomy_label_dict


class UsedModules():
    def __init__(self,
                 modules_name_list:list[pathlib.Path]|None = None):
        birdnet_toolbox_list = ['birdnettoolbox.cores.hdf5',
                                'birdnettoolbox.cores.original_sound_files',
                                'birdnettoolbox.cores.project_analyser']
        self._modules_name_list = []
        self.modules_name_list = birdnet_toolbox_list
        self._hdf5_file_path = None
        self._full_used_modules_link = None
        if modules_name_list is not None:        
            self.modules_name_list = modules_name_list            

    @property
    def modules_name_list(self) -> list[str]|None:
        return self._modules_name_list
    @modules_name_list.setter
    def modules_name_list(self, module_name_list:list[pathlib.Path]|list[str]|pathlib.Path|str):
        # make list in case we get a singleton
        if isinstance(module_name_list, str|pathlib.Path):
            module_name_list = [module_name_list]
        # convert elements to str if needed
        path_list = []
        for module in module_name_list:
            path_list.append(str(module))
        # add if not already included
        for path in path_list:
            if path not in self._modules_name_list:
                self._modules_name_list.append(path)
    
    def zipped_modules(self) -> tuple[io.BytesIO, list[str]]:
        if (self._modules_name_list) is None or (len(self._modules_name_list) == 0):
            return None, None
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zip:
            for module_name in self.modules_name_list:
                path = inspect.getsourcefile(importlib.import_module(str(module_name)))
                zip.write(path)
        raw.seek(0)
        return raw, self.modules_name_list
        
    def _init_from_HDF5(self,
                       hdf5_file_path:pathlib.Path|str,
                       full_used_modules_link:str,
                       modules_name_list):
        self._hdf5_file_path = hdf5_file_path
        self.modules_name_list = modules_name_list
        self._full_used_modules_link = full_used_modules_link

    def export(self,
               target_folder_path:pathlib.Path|str):
        target_folder_path = pathlib.Path(target_folder_path)
        print(f"{self._hdf5_file_path=}")
        print(f"{self._full_used_modules_link=}")
        assert  (self._hdf5_file_path is not None) and \
                (self._full_used_modules_link is not None), \
            "UsedModules object has to be created by the (Generic)Analysis object in order to export the listed modules."
        assert target_folder_path.exists(), \
            f"Target folder '{target_folder_path}' doesn't exist."
        with open(target_folder_path/'modules.zip', mode='xb') as modules_zip_file:
            with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
                dset = h5f.get(self._full_used_modules_link)
                modules_zip_file.write(np.uint8(dset).tobytes())
 


        

class GenericAnalysis:
    def __init__(self, 
                 hdf5_file_path:pathlib.Path|str,
                 full_analysis_group_name:str|None = None
                 ):
        self._hdf5_file_path            = pathlib.Path(hdf5_file_path)
        self._full_analysis_group_name  = full_analysis_group_name

        self._full_processing_documentation_group_name   = None
        self._full_analysis_modules_zip_dataset_name     = None
        self._full_results_group_name            = None

        self._analysis_group_comment             = None
        self._analysis_group_meta_data           = None
        self._processing_documentation_comment   = None
        self._processing_documentation_meta_data = None
        self._analysis_class_code_lines          = None
        self._results_group_comment              = None
        self._results_meta_data                  = None

        if full_analysis_group_name is not None:
            # we read the essential data from this group
            self._read_group_essentials(full_analysis_group_name)
        else:
            # we create and read back the new data
            full_analysis_group_name = self.create_analysis_group()
            self._read_group_essentials(full_analysis_group_name)
            
    def create_analysis_group(self) -> str:
        assert self._full_analysis_group_name is None, \
            "The analysis group already exists. Create a new Analyzer object for a new analysis group."
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            # Create high leven group '/analyzes' if not yet done
            # ---------------------------------------------------
            analyzes_grp = h5core.work_arounds.require_group(h5f, HD5FS.GRP_ANALYZES.FULL_PATH, track_order=True)
            if not HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.COMMENT_ATTR_NAME in analyzes_grp.attrs.keys():
                 analyzes_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.COMMENT_ATTR_NAME] = HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.COMMENT_VALUE
            if not HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.LAST_ASSIGNED_GRP_NB_ATTR_NAME in analyzes_grp.attrs.keys():
                 analyzes_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.LAST_ASSIGNED_GRP_NB_ATTR_NAME] = str(0)
            # Adding new analysis group with attributes and sub groups
            # --------------------------------------------------------
            # get the new analysis group name
            new_analysis_grp_name = str(int(analyzes_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.LAST_ASSIGNED_GRP_NB_ATTR_NAME]) + 1)
            analyzes_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ATTRIBUTES.LAST_ASSIGNED_GRP_NB_ATTR_NAME] = new_analysis_grp_name
            # new analysis group
            analysis_grp = analyzes_grp.create_group(new_analysis_grp_name)
            self._full_analysis_group_name = analysis_grp.name # save the full link of this group
            # new process documentation group
            processing_documentation_grp = analysis_grp.create_group(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.NAME)
            self._full_processing_documentation_group_name = processing_documentation_grp.name
            # add attributes
            processing_documentation_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.COMMENT_ATTR_NAME] = HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.COMMENT_VALUE
               # attribute meta_data will be filled during adding the documentation datasets
            # new results group
            processing_results_grp = analysis_grp.create_group(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.NAME)
            self.processing_results_grp = processing_results_grp.name
            # add attributes
            processing_results_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.COMMENT_ATTR_NAME] = HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.COMMENT_VALUE
               # attribute meta_data will be filled during adding the results 
        return(self._full_analysis_group_name)

    def _read_group_essentials(self,
                               full_analysis_group_name:str):
        # collect HDF5 links and essential attributes of objects of chosen analysis group
        self._full_analysis_group_name = full_analysis_group_name
        self._full_processing_documentation_group_name = full_analysis_group_name + '/' + HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.NAME
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_ONLY_DONT_CREATE) as h5f:
            # analysis group
            analysis_grp = h5f.get(full_analysis_group_name)
            try:
                self._analysis_group_comment = analysis_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_ATTRIBUTES.COMMENT_ATTR_NAME]
            except KeyError:
                self._analysis_group_comment = None
            try:
                self._analysis_group_meta_data = h5core.unpickle_from_uint8(analysis_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_ATTRIBUTES.META_DATA_ATTR_NAME])
            except KeyError:
                self._analysis_group_meta_data = None
            # processing documentation group
            processing_documentation_grp = h5f.get(self._full_processing_documentation_group_name)
            try:
                self._processing_documentation_comment = processing_documentation_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.COMMENT_ATTR_NAME]
            except KeyError:
                self._processing_documentation_comment = None
            try:
                self._processing_documentation_meta_data = h5core.unpickle_from_uint8(processing_documentation_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.META_DATA_ATTR_NAME])
            except KeyError:
                self._processing_documentation_meta_data
            # Code lines of the Analysis class
            try:
                dset = processing_documentation_grp.get(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.NAME)
                if dset is not None:
                    self._analysis_class_code_lines = h5core.unpickle_from_uint8(dset)
            except KeyError:
                self._analysis_class_code_lines = None
            # full link to the modules zip dataset
            self._full_analysis_modules_zip_dataset_name = self._full_processing_documentation_group_name + '/' + HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.NAME
            # full link to the result group
            self._full_results_group_name = full_analysis_group_name + '/' + HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.NAME
            # common meta information of result group
            results_grp = h5f.get(self._full_results_group_name)
            try:
                self._results_group_comment = results_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.COMMENT_ATTR_NAME]
            except KeyError:
                self._results_group_comment = None
            try:
                self._results_meta_data = h5core.unpickle_from_uint8(results_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.META_DATA_ATTR_NAME])
            except KeyError:
                self._results_meta_data = None

    def write_used_modules(self,
                           used_modules:UsedModules,
                           comment:str          = '',
                           meta_data:dict|None  = None):
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            doc_grp = h5f.get(self._full_processing_documentation_group_name)
            used_modules_dset = doc_grp.get(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.NAME)
            del used_modules_dset
            zip, _ = used_modules.zipped_modules()
            dset = doc_grp.create_dataset(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.NAME,
                                        dtype = np.uint8,
                                        data  = np.frombuffer(zip.getbuffer(), dtype=np.uint8))
            dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.COMMENT_ATTR_NAME] = comment
            if meta_data is not None:
                dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(meta_data)
            if used_modules.modules_name_list is not None:
                dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.MODULES_LIST_ATTR_NAME] = h5core.pickle_to_uint8(used_modules.modules_name_list)

    def read_used_modules(self) -> UsedModules:
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            doc_grp = h5f.get(self._full_processing_documentation_group_name)
            used_modules_dset = doc_grp.get(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.NAME)
            if used_modules_dset is None:
                raise ValueError("Missing 'used_modules' dataset.")
            modules_list = h5core.unpickle_from_uint8(used_modules_dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_MODULES.MODULES_LIST_ATTR_NAME])
            used_modules = UsedModules()
            used_modules._init_from_HDF5(self._hdf5_file_path,
                                         used_modules_dset.name,
                                         modules_list)
        return used_modules


    def write_analysis_class_code_lines(self,
                                        comment:str|None    = None,
                                        meta_data:dict|None = None):
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            doc_grp = h5f.get(self._full_processing_documentation_group_name)
            analyser = doc_grp.get(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.NAME)   
            if analyser is not None:
                del analyser
            dset = doc_grp.create_dataset(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.NAME,
                                          dtype=np.uint8,
                                          data=h5core.pickle_to_uint8(inspect.getsource(self.__class__)))
            dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.COMMENT_ATTR_NAME] = comment
            dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(meta_data)

    def read_analysis_class_code_lines(self):
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            doc_grp = h5f.get(self._full_processing_documentation_group_name)
            comment = None
            meta_data = None
            analyser_code = None
            try:
                analyser_dset = doc_grp.get(HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.NAME)
                analyser_code = h5core.unpickle_from_uint8(analyser_dset[:])
            except KeyError:
                return analyser_code, comment, meta_data
            # ok, we have this dataset
            try:
                comment = analyser_dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.COMMENT_ATTR_NAME]
            except KeyError:
                pass
            try:
                meta_data = h5core.unpickle_from_uint8(analyser_dset.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.DSET_ANALYZER.META_DATA_ATTR_NAME])
            except KeyError:
                pass
        return analyser_code, comment, meta_data

    @property
    def hdf5_file_path(self) -> pathlib.Path:
        return self._hdf5_file_path

    @property
    def analysis_group_comment(self) -> str:
        return self._analysis_group_comment
    @analysis_group_comment.setter
    def analysis_group_comment(self, comment:str):
        self._analysis_group_comment = comment
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            analysis_group = h5f.get(self._full_analysis_group_name)
            analysis_group.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_ATTRIBUTES.COMMENT_ATTR_NAME] = self._analysis_group_comment

    @property
    def analysis_group_meta_data(self):
        return self._analysis_group_meta_data
    @analysis_group_meta_data.setter
    def analysis_group_meta_data(self, meta_data):
        self._analysis_group_meta_data = meta_data
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            analysis_group = h5f.get(self._full_analysis_group_name)
            analysis_group.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_ATTRIBUTES.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(self._analysis_group_meta_data)

    @property
    def processing_documentation_comment(self) -> str:
        return self._processing_documentation_comment
    @processing_documentation_comment.setter
    def processing_documentation_comment(self, comment:str):
        self._processing_documentation_comment = comment
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            processing_documentation_grp = h5f.get(self._full_processing_documentation_group_name)
            processing_documentation_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.COMMENT_ATTR_NAME] = self._processing_documentation_comment

    @property
    def processing_documentation_meta_data(self):
        return self._processing_documentation_meta_data
    @processing_documentation_meta_data.setter
    def processing_documentation_meta_data(self, meta_data):
        self._processing_documentation_meta_data = meta_data
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            processing_documentation_grp = h5f.get(self._full_processing_documentation_group_name)
            processing_documentation_grp.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_PROCESS_DOC.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(self._processing_documentation_meta_data)

    @property
    def results_group_comment(self) -> str:
        return self._results_group_comment
    @results_group_comment.setter
    def results_group_comment(self, comment:str):
        self._results_group_comment = comment
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            results_group = h5f.get(self._full_results_group_name)
            results_group.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.COMMENT_ATTR_NAME] = self._results_group_comment

    @property
    def results_meta_data(self):
        return self._results_meta_data
    @results_meta_data.setter
    def results_meta_data(self, meta_data):
        self._results_meta_data = meta_data
        with h5py.File(self._hdf5_file_path, mode=h5core.H5_OPEN_MODE.READ_WRITE_DONT_CREATE) as h5f:
            results_group = h5f.get(self._full_results_group_name)
            results_group.attrs[HD5FS.GRP_ANALYZES.GRP_ANALYSIS.GRP_RESULTS.META_DATA_ATTR_NAME] = h5core.pickle_to_uint8(self._results_meta_data)




