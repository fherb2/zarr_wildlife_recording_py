# Python Zarr Wildlife Recording Package Project – Concept

## Project Aim

The project aims to provide a Python package that allows audio recordings of animals living in the wild to be handled in a uniform manner. This includes a database that not only serves as a sound archive, but also allows fast and parallel read access to arbitrarily small or large sections for processing this data. The associated storage of meta information is a matter of course. For further processing of the audio data, a Numpy-compatible interface to the sample data is provided so that all common scientific Python libraries can be used immediately. Data import is also carried out within this project, so that scientists do not need to familiarise themselves with Python sound libraries.

Extending handling to image and video data is conceivable, but is not currently planned or in development. However, video data can already be used as a source if it contains audio data.

## Overview of the package

### Audio Import

- Most audio data can be imported directly (anything that FFMPEG can decode).
- Metadata from the audio files is automatically transferred as dictionary.
- Additional user metadata can be added as a dictionary.
- Audio data is stored in relation to the source file. This preserves the link to external information that is not transferred as metadata during import.
- Audio data is stored in a flat structure starting from an ‘attache point’ in the database (see next point). If the structure information of the source files is required for referencing external information, this structure information should either be included as user metadata during import or the following procedure should be used:
- With regard to the Zarr database system, the user can create a nested, structured database of audio data of course! The API of this package allows you to ‘attach’ yourself to a specific location in this structure. However, starting from this hook-in point, the API only allows a flat hierarchy. When using the API, the user can, however, repeatedly ‘attache’ the entire database at different points. For more information, please refer to the Zarr V3 documentation and read the following section on audio storage.

### Audio Storage

#### Zarr Databases

The Zarr system, currently in version 3, is used to store the audio data and meta information. The Zarr design was based on the well-known HDF5 database standard, but unlike HDF5, it is designed to support massive parallel access during writing and reading, which is why, unlike HDF5, it is also suitable for processing in high-performance clusters (HPC). Advantages:

As with HDF5:
- Zarr is suitable for the main data in array structures ranging from very small to very large, and thus for any type of sampled physical signals, including audio. An array is treated as a Numpy array at the API.
- Zarr allows data to be structured into arbitrarily nested groups and metadata to be stored that can be linked to these groups or arrays.
Difference to HDF5:
- Unlike HDF5, the structure is not organised within a file, but in storages, which in the case of file systems also use the file system for structuring. This prevents global write locks, which would make massive parallel processing virtually impossible.
Additionally:
- Zarr databases can be created not only in the file system. Supported are: ‘local file systems, Zip files, remote stores via fsspec (S3, HTTP, etc.), and in-memory storage’.

#### Zarr Inside This Project

- Access to the audio and metadata is via the package's API. Users do not need to be familiar with the structure used. If the audio data can be structured completely flat (via index), the user does not need any knowledge of the database used.

With Zarr databases, access can also take place within the database structure at a group level. Therefore:
- The Zarr database used here can be part of a larger storage system. When using the API, the user is free to connect to different locations in the Zarr storage system, which means that deeper data structuring by the user is possible at any time. The API of the package then works flat in this mount point, but also builds a structure for its data, since metadata and an index for granular access are created in addition to the pure audio data as an array. In such a use case, the user needs basic knowledge of the Zarr database system. This is made easier for the user by the fact that Zarr itself has a concise and clear API that does not require in-depth IT knowledge.

#### Audio Inside Data Base

##### Audio Encoding – From Low Frequency To Ultrasonic: Compressed With Minimal Or No Loss

Audio files are stored in one of the encoding formats supported by the package when they are imported. The user can select the format during import. The formats were chosen (and limited) specifically to meet the user's audio requirements, to keep the storage footprint as small as possible through compression, especially for large sound databases, while still allowing individual sample-accurate snippets to be extracted efficiently and enabling massive parallel processing during access: This package therefore implements a separate high-performance index system for each of the supported encodings.

**FLAC (Free Lossless Audio Codec), Container Less**

- native library used; so all features of flac usable
- wide range of common and uncommon sample rates from 1Hz until about 1MHz (native ultrasonic support)
- wide range of bit-width of the samples
- 1 until 8 channels nativ
- lossless compression

Note: Can be used for small array microphones until 8 channels. An update to use more channels by saving of more than one Flac stream in parallel, is in scope.

**Numpy-Array**

In pipeline!

- fall back if any other method not possible
- each column is a channel: unlimited array microphones can be used
- using segmented entropy encoding to save memory but allow fast access to small sample ranges

**AAC-LC, ADTS Container**

- lossy high quality compression
- good usable for 1-4 channels: all channels are on equal terms during compression until 4 channels
- fixed 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000 Samples/s
- all amplitude bit-widths usable: works internal with floating point
- In Pipeline: ultrasonic signals by sample rate reinterpreting behind the scences



#### Metadaten
