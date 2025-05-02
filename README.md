# wildlifeRecords2Hdf (Python module)

**Python module for summarizing environmental sounds (or/and images) and metadata for scientific analysis by using HDF5 databases.**

> **Framework with methods for organising mass source data, in particular environmental sound recordings or wildlife camera images, including other recorded data that is directly related to this mass source data, in such a way that the data can be easily accessed at any later point in time. The magic words are "self-describing database".**

**If you are thinking about how you or your workgroup should store a large amount of sound or/and image data for subsequent analysis,** **then this module could be the answer.** **The module also provides methods to work with these data directly with the [opensoundscape](https://opensoundscape.org) module for analysis. And it includes importing methodes.**

## Typical use cases for which this Python module will be made

### Environmental sound recording

Environmental sound recording is a frequently used application in field research to detect the presence of species in a certain area or to monitor their presence in the long term if they make themselves heard. This is common for the group of birds (also night bird migration), for bats and sometimes for acoustically active insects such as cicadas. In areas where other animal groups, such as monkeys, also frequently make their presence and activity known through vocalizations, these methods is in use, of course.

A 24/7 recording will yield a lot of sample data. For bird calls and songs as example, up to 12 kHz and 4-fold oversampling is useful, so 48 kS/s and a resolution of 16 bits would be a good choice: The data volume per month is 1/4 TByte. It is commen that recorder systems split the data into smaller chunks to protect it against a variety of error cases, from write errors to power outages and so on. Splitting into 15-minute chunks would give you 86.4 MB per file and 2880 files per month. With two-channel audio (or some mor channels) for better extraction of simultaneously calling or singing individuals (as a later pre-processing step), it results in half a terrabyte per month and field recorder or more. This data volume is of course somewhat lower if specialised devices such as AudioMoth are used, which can only record time segments across the day.

At the latest when you use several such field recorders and perhaps also want to combine the data with metadata, such as the current weather, the question arises as to how this data can be meaningfully merged and stored. This is especially true if you want to preserve the data for future tasks. At the latest when you use several such field recorders and perhaps also want to combine the data with metadata, such as the current weather, the question arises as to how you can reasonably bring together and store this data

### Additional data acquisition

The objectives of field research can vary considerably, which is why it can be necessary and is common practice to collect additional information alongside the sound data. In the following, this data is called ‘metadata’, although it can be more valuable in terms of content than simply providing context for the sound data. However, as the core of the module is essentially used to summarize sound data, we will continue to use the term ‘metadata’ for this data. This generally and commonly includes weather data, current light brightness, e.g. the actual twilight brightness at the location of the sound recording. Particularly near breeding sites, however, movement data of individuals is also of interest and must be recorded synchronously with the sound and possibly also in parallel with recordings from image recording devices. This means that these metadata are not always just simple attributes of the sound data, but can result in completely independent data series. Although in many cases the metadata actually represent little more than attributes of the sound data, we would at least like to leave open the possibility here that metadata can also be time-stamped information that occurs in parallel to the sound data.

### Pragmatic solutions, their problems and the aim of the project

Not every scientific working group has the opportunity to appoint an experienced data specialist with software expertise as a member of their group. Depending on the experience of the members of the working group, pragmatic solutions are chosen to summarize the data from the data recorders. The primary, minimum goal is to summarize the data in such a way that it can be used for at least a single analysis with subsequent publication of discoveries. The consequence of this is that the data is lost for subsequent use at a later date or by other working groups if not at least one member of the original working group is not available to help interpret the data files. This is a common and recurring problem in science.

While openly accessible and well-documented software tools are usually used for the actual evaluation of the data, it is rather difficult to access the source data again later and extract it from the collected data series due to the pragmatic isolated solutions described above.

There are now good examples where such mass data series can be stored in a format that provides the desired conditions for reusability of the data. In the field of artificial intelligence, the need for well-structured mass data has become established and has led to solutions such as Pandas, which actually enable the data to be reused beyond the working group that generated it.

However, data sampled in great detail over time, such as environmental audio data, has special requirements, as this data is usually not fed to AI applications as a pure data series, but is selected and pre-processed in pre-processing phases and sometimes converted into completely different mathematical spaces. All the more so if metadata containing more information than a simple singular attribute for a sound data series over a specific time range is also recorded in parallel.

This project is intended to address precisely this specificity in order to ensure the implementation of ‘good scientific practice’ (publication of reusable source data in parallel with the publication of results obtained from it) for scientific data series with the content character described above, but also to provide working groups with limited resources with tools for data structuring in addition to the analysis tools that are usually available.

### What is the right data format for audio and meta data?

First of all: This project specialises in sound records. Recordings of all other data are useful and often important. But we interpret sound data as primary data and all other recorded data as secondary "metadata". There is of course no reason to reduce the content of this metadata to simple attributes if this data is not specifically recorded to be truly only metadata in the sense of attributes. However, this means that we concentrate on preparing our interfaces in such a way that they are particularly well suited for use with opensoundscape:

> **Opensoundscape is the leading Python project for processing environmental audio data.**

### Images or videos as meta or main data

With regard to the question of metadata, image data plays a special role, however, since it can also be available as mass data and, in the context of audio data, it must also undergo similarly complex recognition algorithms. For this reason, it is planned at a later point in time to also organise image data within this framework in a comparable way and to organise it together with audio data in a common database.

## Databases for use with fine-sampled recording data

> **The core question over all is, how we should save our data to be optimal usable over a long time?**

Many people start by saving the recorded data in folder structures and good named files on their hard drive in the use cases considered here. The data is therefore stored in a tree-like structure. Alternatively, you could also store all the data in a single level and enter all the context information about how this data is related in tables. You can certainly do that. And supporters of SQL databases will do just that and possibly even enter the source data itself in special tables whose columns are defined as a so-called blob data type, where the raw data is copied into. These so-called relational databases are not a bad choice for pure machine data processing in many cases, since they usually have numerous search and filter methods implemented. However, the ‘relations’ of the data become difficult to understand even with simple relationships without the aid of detailed documentation. You can also create groupings and hierarchies here, but these are not visible in the tables at first glance.

The folder structure mentioned above is quite different. The way the data is organised is immediately apparent. The nature of the relationship between the items can often be inferred from the names of the folders. Metadata can also be easily classified as separate files in such a structure. And if the data is not too complexly interwoven and the file and folder names are meaningful, such a structure can be self-explanatory. If you also insert a few README files in the appropriate places to describe the final context and details of the data, you actually have a self-explanatory database with such a folder structure.

In principle, this is sufficient. And that is why it is often practised this way. If the data is also to be archived in a coherent manner, then it is compressed into a zip file (or another common format) and the content can be passed on in one piece. Doing it this way is perfectly legitimate and usually much more user-friendly than a solution with table-based databases.

> **So we won't deviate from a tree-like data structure here.**

However, the folder solution has two major disadvantages:

1. The file system management of the operating system used is misused as a database. This is not what it was actually designed for.
2. Such use, without standards or an API, represents a proprietary solution in which no access methods are defined to match the content of the stored data.

The two issues are unrelated. In principle, it would be sufficient to standardise point 2 with a suitable framework. However, point 1 has a significant disadvantage: when dealing with large amounts of data, distributed across many individual files, the operating system's file system works relatively inefficiently. The data set transferred via compression as a whole unit must first be unzipped to enable direct access to the individual files. And the mechanisms in the operating system, in particular caching, are not necessarily ideal for use as a database system.

This problem was identified many years ago, particularly in scientific applications, and the U.S. [National Center for Supercomputing Applications](https://en.wikipedia.org/wiki/National_Center_for_Supercomputing_Applications) (NSCA) made a name for itself many years ago by developing **a so-called hierarchical data format (HDF**) and a **high-performance API** to go with it. The establishment of the HDF Group as a non-profit company in 2006 professionalised this basis and led to the version HDF5 that is in use today. Professionalised means that there is

* hardly a system with a better performance (based on the type of data that is typically stored in this system)
* the library is available for an exceptionally large number of programming languages

What type of data is the system optimised for?

* records and metadata can be organised hierarchically, enabling a first level of self-explanation.
* mass data in the form of tabular data, one-dimensional data blobs or arbitrary dimensional arrays, stored in the structure as data records
* assignment of any amount of metadata to the tree-like structures and to the mass data
* relationships between records can ultimately be defined based on structure and metadata
* descriptions can also be attached to any object in the form of metadata

Another optimisation is that access to the data can be indexed in data sets, so that it is not necessary to load a data set completely into the main memory in order to work with it. Furthermore, file access is optimised to such an extent that it is clearly superior in terms of performance to the above-described variant of storing the data in individual files in a file folder structure.

However, HDF has a significant disadvantage compared to relational databases: it is not designed to process write operations from parallel processes! It is hardly useful as a database for reading AND writing data with high performance. The design of HDF essentially meets the requirements of our application for storing extensive source data and metadata. Using the same HDF5 database file to store the results of the recognition process is possible, but not performance-optimised. Unlike SQL databases, HDF5 does not have an engine that receives and cleverly sorts data queries and write requests. **HDF is designed to combine (scientific) mass and source data with all its meta information and to standardise access to it.**

So in the end, it is designed to store our wildlife data together with all meta information in a structured and ‘easy to read’ way. If the analysis of the data from the HDF5 file produces relatively little data, or data that is well compressed in terms of content, it is of course no problem to store these results in the same HDF5 file. However, we are only considering the case of summarising the source data well structured and with metadata in an HDF5 file. With the following objectives:

* data that belongs together is also stored together in a file
* the fact that the data belongs together is self-evident from the structure and any ‘descriptive’ metadata that may have been added.
* chunkwise access, even across data that was originally recorded in separate files
* metadata describes the data (e.g. which physical unit is to be used or which conversion into a physical unit is necessary; e.g. with which device combination and at which location the data was generated; how the individual data series can be historically aligned, etc.)
* the associated framework (developed and published here) provides standardisation of the data

The goals show that reusability with the help of the HDF5 format and the framework published here is only possible if the framework itself is accepted and used as a de facto standard by many working groups. Without this effect, the good self-description of the data in the HDF5 files remains. However, each new data set also requires the development of new access methods.

> **The main purpose of this project is therefore to establish the access methods as an API as a quasi-standard.**

> **Accordingly, the project is open to contributors for continuous development and improvement.**

### Explore any HDF5 files

There are a number of tools for opening and exploring any HDF5 file. Generally speaking, there is no problem in recognising the structure and details of data sets and metadata in order to create suitable evaluation scripts. That said, exploring a previously unknown HDF5 file and writing suitable evaluation scripts from it is a straightforward process with a low barrier to entry.

# Module Documentation

[Development process and documentation are in progress. – **Not yet ready tu use!**]

## HDF5 file structure and documentation of groups, datasets and attributes

### Structure (over all)

```
Designation regarding optional and mandatory attributes:
  [O] - optional
  [M] - mandatory

Structure of the HDF5 file (part for original sound file data and metadata inclusing a concatenation
of splittet sound files). 
Following, we decribe only our minimum "standard". Everybode can add any other components to the HDF5
file. This "standard" has to be implemented to can be found by the methodes of the module. However,
additional elements within the file (groups, data sets or attributes) are possible and do 
not pose any risk whatever in relation to the functioning of the module. The approach is 
thus open to any extensions.
------------------------------------------------------------------------------------------

/ # root of HDF5 file
|
+––/original_sound_files # group # Contains all original sound files as file-byte copied data
|  |          # sets and their metadata.
|  |          #
|  |          # Group for "Byte-Blobs" of the original and unchanged sound files, some
|  |          # standardized and non-standardized meta data attributes, and all data needed
|  |          # in order to generate the original sound files from its byte blobs again.
|  |          # The samples are completely unchanged and not resampled and not decompressed
|  |          # if the files were originally compressed. So: A truly pure byte-for-byte copy
|  |          # of the sound file.
|  |
|  +–– 1 # (first imported sound file) dataset  # numbered dataset of one file with all 
|  |                                            # attributes (for details, see dataset 'n')
|  |
|  +–– 2 # (second imported sound file) dataset #                    –"–
|  |
|  +–– 3 # (third imported sound file) dataset  #                    –"–
|  :
|  +–– n # (last imported sound file) dataset   #                    –"–
|  |   |        # Documentation of all datasets 1 ... n:
|  |   |        # type: byte array 'uint8' (original byte stream of the complete sound file
|  |   |        #       with all headers, meta date, ... inside the file; more exactly: a
|  |   |        #       binary blob of the file byte stream)
|  |   |
|  |   +–– original_sound_file_name # [M] attribute:str # without any path information
|  |   |
|  |   +–– file_formate_description # [M] attribute:pickle-of-dict 
|  |   |              #                              { "format": SoundFile.format, 
|  |   |              #                                "sub-type": SoundFile.subtype, 
|  |   |              #                                "endian": SoundFile.endian,
|  |   |              #                                "sections": SoundFile.sections,
|  |   |              #                                "seekable": SoundFile.seekable }
|  |   |              # Parameters used by Python to understand the sound file byte stream
|  |   |              # and to recreate the original sound files from this byte stream.
|  |   +–– file_size_bytes # [M] attribute:int # size of the file by using 'os.stat(original
|  |   |                                       # sound file).st_size'
|  |   +–– content_hash # [M] attribute:int # Python hash value over all bytes of the file;Coded as unicode. 
|  |   |                                    # usable to make a check if file content of a
|  |   |                                    # new import yet exists
|  |   +–– device_id # [O] attribute:str    # An identifier for the device where sound was recorded.
|  |   |                                    # Kind of device-coding is not standardized.
|  |   +–– sampling_rate # [M] attribute:int # sampling rate in samples per second
|  |   |
|  |   +–– no_channels # [M] attribute:int # number of audio channels; common are 1 or 2; but
|  |   |                                   # can be more by using of microphone arrays
|  |   +–– start_time_dt # [M] attribut:bytestream # Time stamp of the first sample as pickle.dump
|  |   |                                           # of a localizied Python datetime.datetime object.
|  |   |                                           # Note: The resolution is micro second. This is
|  |   |                                           #       sufficient for acoustic phases evaluation,
|  |   |                                           #       since the equivalent is a path length of 
|  |   |                                           #       only 0,334mm in air.
|  |   +–– start_time_iso861_localizied # [M] attribute:str # Same as start_time_dt, but human readable.
|  |   |                                       # created with: 
|  |   |                                       # start_time_dt.strftime('%Y-%m-%d %H:%M:%S') + \
|  |   |                                       #                        f" {start_time_dt.tzinfo.key}"
|  |   +–– no_samples_per_channel # [M] attribute:int # number of samples per channel in file
|  |   |                          # The (single or multi channel) samples have to be equidistant, so that
|  |   |                          # the the time stamp of the last sample is:
|  |   |                          # start_time_dt + (no_samples_per_channel – 1)/sampling_rate
|  |   +–– maximum_timeshift_percent # [M/O] attribute:float
|  |   |                             # – Not required if we have 'last_sample_time_dt'.
|  |   |                             # – Mandatory if we don't have 'last_sample_time_dt' and we want
|  |   |                             #   to concatenate data sets to stream in group 'concatenations'.
|  |   |                             # This value is important for the continuity of individual files
|  |   |                             # to form a continuous sound and is evaluated to summarise the
|  |   |                             # files that immediately follow each other in time in the 
|  |   |                             # ‘/concatenation’ group.
|  |   |                             # Background and configuration of this value: Please, see the
|  |   |                             # documentation, chapter "maximum_timeshift_percent".
|  |   +–– last_sample_time_dt # [O] attribute:bytestream 
|  |   |                             # Time stamp of the last sample as pickle.dump
|  |   |                             # of a localizied Python datetime.datetime object.
|  |   |                             # Note: The resolution is micro second.
|  |   |                             # If the recording device has an exactly clock and can get the time
|  |   |                             # stamp of the last sample in this data set, so this would be 
|  |   |                             # greate for concatening. In this case we have exact parameters for
|  |   |                             # and we can forego the 'maximum_timeshift_percent' value
|  |   |
|  |   |   # Following all file or user depending meta information in two versions:
|  |   |   #    1) as Python dictionary (pickle serialized)
|  |   |   #    2) as dataset attribute: Each individual piece of meta-information is also listed below
|  |   |   #       as an attribute with str data type, so that it is also human-readable when looking
|  |   |   #       into the database file. Each meta information is named as "meta_" plus the name of the
|  |   |   #       given meta values by the user.
|  |   +–– meta_dict #  [M] attribute:pickle-of-dict # Some user or case depending meta information about
|  |   |                                        # file content. This includes audio file information like
|  |   |                                        # artist, song name... 
|  |   |                                        # This attribute is added by the import methodes. In case
|  |   |                                        # there are no such meta information, this contains a
|  |   |                                        # pickled empty dictionary.
|  |   |
|  |   +–– meta_...  # [O] attribute:str # Added bei the import methode if there is at least one meta
|  |   |                                 # information value.
|  |   +–– meta_...  # [O] attribute:str # Added bei the import methode if there are at least two meta
|  |   |                                 # information values.
|  |   :...          # and so on...
|  |
|  |+––/concatenations # group # [O] Contains one or more data sets that describe which sound file data
|      |                       # sets can be chained together immediately in time.
|      |                       # These data sets are generated by a separate method independing on the import
|      |                       # of sound file data.
|      |                       # The presence of the group /concatenations is optional. However, if it is
|      |                       # present, all files from /original_sound_files must be integrated into the data 
|      |                       # sets inside this group accordingly.
|      |                       # In practical terms, this means that for evaluation and recognition processes:
|      |                       # If the group /concatenations is missing, the individual sound files in
|      |                       # /original_sound_files must be processed. However, if the group /concatenations
|      |                       # is present, the sounds can be evaluated in a concatenated form by using
|      |                       # the datasets in this group. 
|      |                       #
|      |                       # How to use
|      |                       # ----------
|      |                       # 1) Add sound files to concatenations.
|      |                       #    This is possible only with imported sound files in '/original_sound_files'
|      |                       #    since some attributes are important for this process.
|      |                       #    With the method 'concatenate_original_sound_files()' new datasets will be
|      |                       #    created here. This method attempts to concatenate a given list of sound files
|      |                       #    saved in '/original_sound_files'. 'Try' means, that a concatenation works
|      |                       #    only under certain conditions: The concatenation walks throughout the given
|      |                       #    sound data sets (or all data sets) and looks for the right time order. The
|      |                       #    primary attribute is the start time and the time of the last sample of each
|      |                       #    sound file. However, we can expect that due to the lack of synchronisation
|      |                       #    between audio sampling and NTP time or system time, the samples of consecutive
|      |                       #    sound files will not match exactly in terms of the time stamps in the individual
|      |                       #    sound files.
|      |                       #    That is the reason for introducing the 'maximum_timeshift_percent' attribute
|      |                       #    for each original sound file. This value determines the maximum time range in
|      |                       #    which the sound may be expand or compress in order to be considered an
|      |                       #    uninterrupted data stream with the samples of the following sound file. If the
|      |                       #    concatenation is allowed under these conditions, an entry is made in the data
|      |                       #    set with the necessary correction parameters. If concatenation is not possible,
|      |                       #    the previous data set under ‘/concatenations’ is closed and a new data set is
|      |                       #    opened for the following audio file. 
|      |                       #
|      |                       # 2) Evaluation (sound detection) with these concatenations.
|      |                       #    Just as a method makes accessing the sound datasets in ‘/original_sound_files’
|      |                       #    transparent and continuously returns the individual samples, there is another
|      |                       #    method that handles access to the related data in a dataset under ‘/concatenations’
|      |                       #    in the same way. Here, too, the individual samples of a data set can be accessed
|      |                       #    freely, although the individual samples come from different sound files in 
|      |                       #    ‘/original_sound_files’, but they are in the correct order and, if necessary, 
|      |                       #    time-corrected.
|      |   
|      +––1 # stream list dataset # first concatened sound data stream (for details, see dataset 'n')
|      |
|      +––2 # stream list dataset # second concatened sound data stream (for details, see dataset 'n')
|      :
|      +––n # stream list dataset # last concatened sound data stream
|         |         # Documentation of all groups 1 ... n:
|         |         # type: int (Sound file dataset numbers from 'original_sound_files' describing a
|         |         #            continuouse sound data stream by concatening the sound data of the file-
|         |         #            blobs referenced by this numbers.)
|         |         #            "Stream" means, that all the containing samples are recorded without breaks.
|         |
|         +–– time_scaling_percent # [O] attribute:float # In case of long time sound streams, we can take
|                                  # into account the difference between the sampling oscillator missalignment
|                                  # during recording and the really elapsed time during recording. Thats
|                                  # a mean value over the full streaming time of the dataset. It can not be
|                                  # correct any kind of jitter of the sampling osszillator, of course.
|
| # You are free to use all HDF5 elements to save additional meta information or additional primary information!
```

## Precise interpretations of samples, time and periods of time

### In less words

##### Metainformation of sound recording data (original sound file data copy)

1) **Machine coded time stamps and time differences as a unique reference**

* For the exactly encoded **time stamps** of the recorded files, we use **POSIX time with signed integer 64 Bit with the time base of number of nano seconds since 1.1.1970 00:00:00 UTC**. This has the best support over a width range of programming languages and system time call functions. And the resolution for selected recording samples is exactly enough. Note: POSIX times doesn't include leap seconds.
* For **time differences** like the length of a recording, we use **nanoseconds** as an integer.

2) **Human readable time stamps**

* Additionally, we write time stamps also in a human readable format, so we can get the information about the recorded data also with any HDF5 visualisation tool.

3) **Python pickled datetime.datetime**

* Additional, we write the **timestamps as bytestream of a pickled Python datetime.datetime objects** and **time differences as byte stream of a pickled Python datetime.timedelta objects**. So the import can hapen by unpickling these byte streams only.

4) **Daylight and time zones**

   An universal time stamp code can be not enought for schientific results. At least, we need the time zone where the data was recorded. Better, we have also the coordinates of the recording:

* The Python pickled **datetime.datetime** contains **time zone information**.
* Additionally, we save the **geografic cordinates as meta data**.

##### TimeTable dataset for analysis of the recorded data

There is a special time table dataset what is created and expanded during the import of the original sound data samples. **This is created as intermediate layer for analysis procedures only.**

During the import, a special analysing procedure tries to recognise the immediate sequence of sound files. The documentation for this is at an other place. But what we should know here: These time table concatenates sequential "original" file samples to a continuous audio stream. The source file data will be never changed. But with this table we can create an intermediate layer to use the record data independing of the size and slicing of the files what contains these data.

Since this table ist prepared for automatic concatenating of samples coming from different original sound files by the Python program, we use datetime.datetime and datetime.timedelta as pickled Python objects only. A humand readable format is not needed in this case.

### Time zones and daylight saving time

To take this into account, when importing files, the interpretation of the timestamp is queried in addition to the timestamp itself. This includes which time zone is to be used and whether the timestamp provided is to be interpreted as daylight saving time if it falls within the daylight saving time period for the time zone in question.

### Leap seconds and time drift

Everything in this section is relevant when we process recordings that took place continuously but are split into individual files and the time of the first sample in each file is known. (This is not always an exact precondition: if the file timestamp is used, it is not the exact time when the first sample in the file took place, but when the file was created. Both values can differ by a few seconds.

The assumptions and associated solutions discussed below are therefore also only approximations. If these approximations are considered to be too ‘rough’ or even inadmissible in a specific application, alternative solutions are required, which often arise from the special case under consideration.

If the actual alignment of the Earth with the Sun deviates by more than half a second, leap seconds are inserted into or removed from a minute. The time of such a correction occurs at midnight at the turn of the year or at midnight at the midpoint of the year.

For a recording system that is running, the following cases arise:

1) The recording system uses an NTP service. Usually, the insertion or removal of a second by the system's NTP service is done ‘skulking’ by slightly increasing or decreasing the system clock by a small amount and for a certain time. All minutes, even in such a case where the leap second is taking place, will still have exactly 60 seconds.
2) The recording system uses an NTP service. In special cases, the NTP service can be configured to actually insert an additional 61th second in the relevant minute or it switches to the next minut after 59 second. A lot of software can't handle a minute lasting 61 seconds, which is why this is more of an exception.
3) The recording system has a clock that is not connected to an NTP time server (e.g. via network or GPS). During the entire recording period, the timestamp depends on the time when the recording system's clock was synchronised.

In many cases, the fact that a leap second takes place is not important. But what if a user has a use case where this could cause database consistency issues? This module should therefore at least take this special case into account. The best strategy for solving the problem also depends on the length of a continuous recording.

1. We assume that the sample clock is quite accurate, but that the recording is only a few seconds or a very less minutes long. To combine these samples of the relevant file with the immediately following samples of the next file, a resampling with a time stretch or compression would result in an impermissible frequency shift or (with frequency-preserving stretching and compression) an impermissible change in the timing of sounds. A sensible approach could be not to change the timing, but to bridge the sound with silence between the file in question and the immediately following file.
2. Now, let's assume that we have longer recordings per file. If there is an extra second or a second is missing, we have a similar case to when the ADC's sampling clock is not absolutely exact. We can't really distinguish here whether a leap second has occurred or the sample clock is slightly off. Unless, of course, we have the fact that a leap second has occurred as meta information available for the file in question. In this case, we cannot tell whether we might get a frequency shift or a gap in time if we apply the exact sampling clock.
3. What we do know is that a leap second occurs relatively rarely. As a rule, we are dealing with deviations of the actual sampling frequency during recording from the nominal value. Or with a difference in time between the first sample and the timestamp of the file creation. However, the latter is prevented in those recording systems that store the timestamp for the first sample of a file separately and do not abuse the file system driver function, which records when the associated file was created. So, with regard to the mass of data that we use to recognise biologically interesting details, we make the smallest error if we assume that the sampling frequency of the recording does not exactly match the nominal value. The fact that a leap second is occasionally inserted every few years can be neglected for plausible reasons.
   The appropriate strategy would therefore be to assume that the presumably inexact sampling frequency is to be corrected and that we therefore resample the original data in such a way that the correct time interval of the recording is reproduced in a file. The corresponding frequency shift corresponds to the value that was incorrectly encoded by the not quite exact beat of the ADC. We therefore shift the samples to the exact time base without obtaining an error in the frequency range. The strategy would only lead to a frequency shift in the case of a rare leap second.
   **We will therefore ignore the leap second.** If someone wants to take the leap second into account in their own preprocessing of the data, please refer to the *astropy.time* module. This module can be used to calculate timestamp differences taking a leap second into account.

### Time drift

As introduced in the previous section, there are basically two causes of a time offset between the samples of different files:

1. A sound data file's creation time is used as the timestamp. This time does not have to correspond exactly to the time of the first sample.
2. The actual frequency of the ADC, which generates the samples from the analogue sound data, does not have to match the assumed nominal frequency stored with the file. Deviations in the lower percentage range are possible. This fact is often overlooked if files are sampled and saved, but it is taken into account in media streaming tools such as gstreamer or ffmpeg.

When the recording in a file is finished and a new file is automatically created to store the next samples, it may happen that the effort of this file change is so great that samples have to be discarded and there is actually a gap in the data. In many cases this does not happen and the data is consecutive. Even if the timestamps of the files suggest otherwise. However, there may also be constellations where this condition is not met. We will support both options by allowing you to select whether the samples can be considered consecutive or whether a pause is to be expected when reading the samples. For the second case, methods will be implemented to concatenate files by detecting zero crossings and adding a few samples of complete silence if necessary to avoid acoustic artefacts (cracking) when transitioning from one file to the next. This is especially important if slicing is used to analyse the sounds and the synchronisation is not exactly at the start and end of the file.
The aim is to allow the evaluation to be carried out in time slices that were independent of the time slices present during sound data recording, especially in the event that the database only contains sound files that represent an uninterrupted data recording across multiple files. Accordingly, overlaps are also possible across files during the evaluation.

### maximum_timeshift_percent

---

TODO: Add the alternativ last_sample_time_dt parameter into the documentation.

---

Audio data are sampled by an analogue-digital converter (ADC) in the sound device. The frequency of the sampling is specified by driver configurations. However, the sample clock is generated by an oscillator in the audio device that is not synchronised to the system clock and therefore not synchronised to an NTP server. The actual clock generated by the oscillator depends on the individual sound device (electronic component tolerances) and the temperature of the electronic components involved in the clock generation. Apart from the fact that there are also time shifts between the start time measured in the system time and the actual time of the first – and, if ultimately not even the exact system time of the start of the first sample is stored, but the timestamp of the creation of the sound file is used as a reference by storing the first sample, then it is clear that time differences arise here that are considerably greater than the time between two samples.

If we later want to combine the samples of the consecutive files into one uninterrupted sound, for example, to let the chunks run through the analysis independently of the actual individual file beginnings and file ends across all samples, then we will naturally run into problems with the time alignment of the samples over longer uninterrupted recording periods.
To restore the missing synchronicity of the ADC over long periods of time when recording with the actual system time (at the end the atomic time, via which the system synchronises itself using NTP), the samples must be corrected over long recording periods with regard to time. This results in new samples that remains synchronised with the running (NTP-synchronised) time over a longer period of time. If the resampling is not carried out, the frequencies in the sound will shift slightly because the playback clock does not correspond exactly to the recording clock. Since we know that the system clock has very good long-term stability due to the NTP-based mechanism, but that the ADC's sample clock does not, this means that the sound from the recorded samples has to be converted into a sound that corresponds to sample times that are very accurate and permanently corrected with atomic time using NTP.
Attention: None of this applies if the entire recording system itself does not have NTP synchronisation. Here, too, the sample clock is slightly asynchronous with the actual time. However, the system time itself is also not particularly stable, which means that the time stamps are also created asynchronously with atomic time. Such conditions are present, for example, in microcontrollers. For example, the Audiomoth device. – Since there is basically no synchronicity in this case, it does no harm if data from such systems are also treated equally, like systems with system clock synchronisation to an atomic clock.
If we want to put together uninterrupted samples in consecutive files, we can basically ignore this. However, there are scenarios where gaps in the samples actually occur. Power failures, reboots... The reasons for this do not matter. But the fact is that the files may actually contain gaps in the recording where concatenating the samples makes no sense and concatenating them would create an artefact in the form of an acoustic signal. Also, time allocations no longer match.
If we want samples that immediately follow one another and are stored in different audio files to be reconnected to a continuous audio data stream, we have to recognise whether or not there is actually a continuous audio stream. And that is under the premise that the sampling frequency and real time do not run absolutely parallel.
The maximum_timeshift_percent parameter is used to decide whether two consecutive sound files can be concatenated or not. It is used to determine whether the files are (most likely) an uninterrupted sequence of sampling data or whether there is obviously a gap in the sampling data and therefore the files should not be concatenated.
Inside the application it must take into account what needs to be used as a timestamp. If the actual time of the first sample of the files was stored very accurately, the theoretically possible deviation of the ADC sample clock can be entered into this parameter. However, if the timestamp itself can also have a significant deviation from the first sample, this must also be included in the tolerable range, i.e. the parameter maximum_timeshift_percent.
If you want to use the concatenation functionality in the HDF5 file under /original_sound_files/concatenation, then select a reasonable value for this percentage.

# Project State

Documentation in progress. Since a pilot implementation was developed, these documentation will be used to convert these pilot implementation into a final implementation as defined here and now.

Development for a first version with a reasonable range of functions in progress (based on a pilot implementation). Image data are not yet considered.

# Development

This project is managed by

* Poetry and
* pyenv

Use the *setup-dev-tooling.sh* to set up the full environment. The Python virtual environment will be placed inside the project path and also the right standard Python version will be downloaded by this setup-script. Poetry installs all requirements at the end of this script

Since including of some development tools for good code quality, the 'sandbox' directory can be used to add quick and dirty code during development steps. This directory is not part of the official source code and can be removed everey time in the main branch while merging.

# Licence

The MIT licence was chosen to be fully compatible with opensoundscape.
