# Common planning of the software parts

## Including data structure into Zarr

* Start point is a group somewhere in the data base: representing variable: *wlrec_grp*, type: Zarr-group-object, default value: under root of Zarr: "/wlrecsrc" (wildlife-recording-source)
* Under this start point: we implement the designed structure from hdf5. Datasets are named "array" in Zarr.
* First point of this next level: representing as variable: original_sound_file_grp, type: Zarr-group-object, default value: '/original_sound_files'
