Helper scripts for Taskmaster MT system
=======================================

`taskmaster.py` - to extract data from the Taskmaster release

`test-coref.py` - annotate Taskmaster training data with coreference using AllenNLP model

To use, clone the Taskmaster release from `https://github.com/google-research-datasets/Taskmaster.git`
into a subdirectory of this folder.

`test-coref.py` also requires AllenNLP to be installed. `pip install allennlp` didn't work for me
as the version it installs doesn't seem to be compatible with their latest coreference model,
so I manually cloned the most recent version from their Github repository and followed their
instructions to install from source.