aopy

Package of python code for reading and processing data collected by the aoLab @UW.


Michael Nolan
Created: 20190930
Updated:

Installation:

After forking, cloning and pulling the repo, move to the root dir and run:

pip install .

From that point, your local conda environment will have access to the files.


Package Contents:

- datareaders: contains code for reading structured binary data from recNNN.___.___.dat files
- datafilters: contains code for creating junk data masks
- brpylib: contains code for reading Blackrock Microsystems ___.nev and ___.nsX data
- brMiscFxns: contains auxillary functionality to brpylib

Blackrock Microsystems code source available at https://www.blackrockmicro.com/wp-content/software/brPY.zip
(current version: 1.0.0.0)

