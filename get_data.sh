#!/bin/bash
# @Ruosi Wang, Nov 9, 2018

if [ ! -d data ]; then
   mkdir data
fi

cd data

if [ ! -f groups.json.gz ]; then
   kaggle datasets download -d ruosiwang/meetup
   unzip meetup.zip
   rm meetup.zip
fi

if [ ! -d glove.6B ]; then
   mkdir glove.6B
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   mv glove.6B.*.txt glove.6B
   rm glove.6B.zip
fi
cd ..

if [ ! -d models ]; then
    mkdir models
    cd models
    mkdir sklearn keras
fi

cd ..
