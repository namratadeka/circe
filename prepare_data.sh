#!/bin/sh

cd data

if [ "$1" = "dsprites" ]; then
    git clone https://github.com/deepmind/dsprites-dataset.git
    cd dsprites-dataset
    rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
    cd ..
    python split_dsprites.py

elif [ "$1" = "yale-b" ]; then
    wget http://vision.ucsd.edu/extyaleb/ExtendedYaleB.tar.bz2
    tar -xvf ExtendedYaleB.tar.bz2
    python split_yale_b.py

cd ..
