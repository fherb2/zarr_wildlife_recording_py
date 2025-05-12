
from pathlib import Path
from zarrwlr.aimport import base_features_from_audio_file, import_audio_file
import zarr

path = Path(__file__).parent / "testdata"
path = Path(path)

store = zarr.storage.LocalStore('./tests/testdata/zarr3/original_audio', read_only=False)
original_audio_grp = zarr.open_group(store=store, mode='w')

for file in path.iterdir():
    if file.is_file() and file.suffix==".WAV":
        print("--Start----------------------")
        print(f"File: {file}")
        print(base_features_from_audio_file(file))
        import_audio_file(file, original_audio_grp, 'opus')
        print("--End------------------------")


# path=Path(__file__).parent.parent.parent / "tests/testdata"
# path = Path(path)
# for file in path.iterdir():
#     if file.is_file():
#         import_audio_file(file)