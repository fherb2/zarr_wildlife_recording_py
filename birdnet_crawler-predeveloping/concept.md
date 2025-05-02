# Initial situation

*Note: This is the beginning to write a concept.*

## Given sounds from outside. The input interface.

* A (nested) folder of audio files with special name convention and any audio format is given by the user.
* We need at first an import into the database by annotation of time stamp information and,
  maybe, other context information. And, of course the real sound data.
* For a common API this is a good point to use a callback function given by the user
  for its special situation of recorded sound data.
  Sound files will be read from a folder (or an expression) and for each file a callback
  function will be called with the sound file path. 

Ok. What would be a good database for the audio data?

Point 1 is a really base point: We have scientific date and it makes sense to put this in an *completely open format*.

Next, the possibilities depending on the content of the data: We have audio data. How to save this?

These data are analog vibration (sound) streams of one or more channels. Typically like other physically measurement data streams. But not really like any physically standard measurement data: Sound data will be interpreted by ears. And the content of sound will be recognize behind the biology processing of ear mechanics, sensor cells, nerves in the brain. In the raw principle, these recognition system seems the same for human and animals like birds. 

### Differences of sound recognition over all species

The hearing abilities of humans and birds cannot be different enough. We are descended from mammals in the long line. Birds share a lineage with dinosaurs. But independent of the evolution, sound and sound conversion is a physically process. If we assume, that it is difficulty to recognize background sounds in case the loudness and frequency is small different to foreground sound data independent of different species, so it could be a good assumption, that compressing methods for human purposes can also be used for other animals.

But independent of this, we should think about that smaller animals in particular have a much more detailed resolution of sound nuances in the higher frequency range than humans due to shorter signal propagation times when processing data to and in the brain.

As a result of this, it is not yet clear, how good audio compression methods works to separate essential and non-essential acoustic signals.

If we compress such acoustic data, we should use

* high enough sampling rates and
* lossless processing.

With an AudioMoth system at 96kS/s and 16Bit, 1 channel, we get 0.691 GByte per hour. Recording 10 hours per day, we get around 7 GByte/day. After 3 months we get about 640 GByte. And if we assume, we have 30 of these AudioMoths in our research region, we get 19 TBytes in 3 months. A bit compression of the date would be a good idea.

So we have think about compression methods and compression ratio.

### Sound differences between individuals

Such differences could be a reference point for the recognition sensitivity of birds. But this field is not yet good researched. We don't know the recognition resolution of individual sound articulation. Wir wissen, dass viele Vögel als Pärchen ein Nest anlegen und die Jungvögel groß ziehen. Aber wissen wir, wie gut sich Männchen und Weibchen an Äußerlichkeiten und Gesang erkennen? Es ist bekannt, dass sich Weibchen auch mit im Revier benachbarten Männchen paaren. Evolutionsbiologisch ist das sogar eher vorteilhaft. Aber erkennt ein Weibchen "sein" Männchen eindeutig? Und wenn ja, an was? Es kann sein, dass die Partnertreue viel weniger auf optische und akustische Merkmale beruht, als auf den Zusammenhang im System: Das Männchen, dass um "mein"/"unser" Nest herum singt ist mein Männchen, weil es das tut. Und nicht weil "sie" es am Gesang und am äußeren Erscheinungsbild erkennt.

Insofern ist also völlig unklar, die weit die Differenzierung am Gesang von der eigenen Spezies tatsächlich unterscheidbar ist.


## Old:

After this process we have concatenate sound data with contexts about the recording time.
Some other context information about recording world coordinates or some more has to be
added.

Now we can run in the problem, that the database files will be very big. Its a good idea
to have compressing ideas. What kind of compressing are possible?

* We could record a lot of sound data with no relevant information. This could be
  * Record times in the night.
  * Recording in winter, when birds do not sing and only occasionally passing groups
    of birds exchange vocal calls with their conspecifics.
  * Record times during rain (maybe with some noise from the rain drops)
  * Record times during very windy situations (a lot of background noise)
  We can remove these sound data completely since there are no relevant information inside.
* We can compress the data with some less loss. (mp3 and other methods)
* We could think about remove noise between calls or songs with filtering or amplification
  control. -> But this is not really simples: The AI model learned with a noise part in the
  data. By removing noise data we would produce additional "image information". Why "image"?
  Currently models are working very often with image information. BirdNET too. The image is
  a special created sonogram image. If we remove noise from the sound, we would add
  dark surfaces in these sonograms. In parts of the diagram, where such noise canceling is
  not full active (since some bird calls are present), we would draw a halo around the bird
  call. The AI model would "see" this and could be impermissible influenced!

So we must be carefully to remove any data between bird calls in order to avoid artifacts which
can be misinterpreted by the AI model! 

If we have prepared the audio data, we can let the AI run over the data.


## ToDo

1) Crawl through the given (nested) folder by using the given callback function to collect context
   information and put source and context into the HDF5 database file.
2) Remove "empty information" from the sound to save memory capacity.
2) Go through the HDF5 database and call Stefan Krahls analyzer function to get analyzing results.


## Result after processing

* One (or some) HDF5 database file(s) containing all data:
  * source audio data in a pre-defined compression
  * results of recognizing (inclusive meta data) with links to start and stop point of the audio
    source belonging to each recognition result
  * optimal would be to save also the second and third choice in the probability of species recognition
  * (put in the HDF5 file: self-documenting about structure and content of the HDF5 file)

### Why the sound is included into the database?

In order to reuse audio data sources with contextual information about the record situation and what is to hear in the sound (analyzing data), a file system as database could be used. If you understand 'database' as a set of tables and functional rules (relational databases systems), so the file system as database would be a better idea. But, the problem are the links between these data. This file system orientated method is possible, no question. But, do you know HDF (Hierarchical Data Format)?

HDF5 (version 5) is not a database engine. It is a data format. A file format. Importantly, it gives us a folder-like structure of data AND allows us to assign contextual information within that structure. This format is very often specified as a mandatory data storage format for big data projects like in astronomy. It combines structure with context and massive columns of data.

To compensate, that the sound data are included in such a file, a simple export function to convert this data back into a sound file is a must have, of course. But all other contextual information can be read out with any tool for the HDF format. No a-priori information is required to read the structure and all context information.

Such an HDF file is therefore a container that contains all related information and remains consistent even when copied, moved or archived. This is the reason to use HDF5 here instead a lot of folders of sound files of any format combined with special structured contextual information as separated files.

## Tools

If we have all results in the HDF data base, we should have some helpers. Let us thing about this as second step.

### BirdNET-Analyzing

#### There is a library: bitdnetlib (https://github.com/joeweiss/birdnetlib)

Instead a repository what contains an output of some doing of the creators of the BirdNET model inclusive an analyzing tool with an GUI, this project contains the core of all as Python library for development projects. But, if we have progress, we should ask Joe Weiss and some development colleagues if the future of this project. Since, we would be depending on, if we use the library.


