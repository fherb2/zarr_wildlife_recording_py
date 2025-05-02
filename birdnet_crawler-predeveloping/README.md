# birdnettoolbox

## State: Preparation of a first usable version of the toolbox. -> A pure state of development -> Follwing comments are prepared for the future after finishing these initial steps.

## What is it?

Of course, it based on bird sound analyzing by using the Python 'birdnetlib'.

The birdnettoolbox is a kind of framework for scientific bird song and sound analyzing of recorded audio as mass data. The toolbox is no replacement of any published birdnet tools and libraries. The aims of the toolbox are:

* A box with tested sound processing methods to prepare mono and stereo sound recordings before analyzing with the birdnetlib AI model.
* A fundamentally connected recording of all initial data, methods and results in well self-documenting HDF5 database files. The basis of good scientific practice.
* A toolbox to bring the bridnetlib analyzing results into a good format for evaluation.

## Status, tidiness vs. chaos in this toolbox

Dear user,

The fundamental basis for this project is to hold together:

* A growing collection of algorithmic tools to preprocess, interprete, recognize and evaluate sound data by using the birdnet AI model
* and link this together with a well documenated manner to ensure reproducibility at all times. 

Of course, methodes have a lot of experimental flexibility. So, in detail, each processing step can be extended by any other useful algorithmic step or method. And so it is possible, that such a toolbox could shoot up into the weeds after short time.

How can we counteract this?

* Either we limit ourselves to a few algorithms that can be pulled from the toolbox in a well-documented manner in order to analyse sound data.

* Or we find a good method for reliably documenting algorithmic-experimental quick shots with the source data and the results.

I think, the first point would be a good shot in the neck for this project.

==> Let us find a way to bundle all possible Python processing scipts together with the source and resulting data: How can we collect not only sources and results inside a HDF5 data base? How can we also implement a kind of automatic documentation of the processing inside the same HDF5 data base? For a maximum of reproducibility.

  **The state of this project is, that we have to find the practical details of this way. – In the meantime, some more or less chaotic collected tools could help for first results and to find the right way.**

Frank