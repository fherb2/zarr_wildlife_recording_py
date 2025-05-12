
from pathlib import Path
from zarrwlr.aimport import base_features_from_audio_file, import_audio_file


path = Path(__file__).parent / "testdata"
print(f"{path=}")
path = Path(path)
print(f"{path=}")
for file in path.iterdir():
    if file.is_file():
        print("--Start----------------------")
        print(f"File: {file}")
        print(base_features_from_audio_file(file))
        import_audio_file(file)
        print("--End------------------------")


# path=Path(__file__).parent.parent.parent / "tests/testdata"
# path = Path(path)
# for file in path.iterdir():
#     if file.is_file():
#         import_audio_file(file)