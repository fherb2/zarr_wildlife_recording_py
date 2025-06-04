# Import Audio

From – To:

Any –> FLAC (lossless compressed; Tested Compession Rate: 4 is fast and higher compress rates doesn't reduce size not significantly more)

Opus –> Opus (byte-copy)

maybe later:

MP3 –> MP3 (byte-copy)

## Orininal audio file database

This is the part inside of a Zarr database, where the imported audio data are saved. Inclusive meta data and time random access index.

```
│original_audio [group]
    ├── '1' [group]
    ├── '2'
    :
    └── n
        ├── 'audio' [array]
        ├── 'index' [array]
        ├── meta data...
    
```

### Audio-File-Import

*   Analyze File-/Audio-/Codec-Type 'analyze_audio_file'
    *   'original file name'
    *   'original file size'
    *   'original file type
    *   'original codec name'
    *   'original file byte-Hash'
*   Is 'original file name' and 'original size' known in database?
    *   If yes: 
        *   Is 'original hash' the same as in database?
            *   If yes: get warning and raise Exception
            *   If no: get warning only and continue
*   Look for all other meta data inside the file.
*   Calculate the next free 'array-number'.
*   Create array, decode file and import depending on source and target codec
*   Create Index for random time access and put it into array
*   Put all meta data as attributes.
*   Set attribute 'import\_finalized' with true. Thats the marker for a completed import.

## Clean up original-file database

It can happen that an audio file import runs not until the ende. So we should have a function to clean up incomplete imports:

*   Go through all imported audio data and check:
*   Is there an attribute 'import\_finalized' (true)?
    *   If no, so remove this data set.
    *   Otherwise continue.