#!/bin/bash
# @Ruosi Wang, Nov 9, 2018

if [ ! -r data ]; then
mkdir data
fi

cd data

kaggle datasets download -d ruosiwang/meetup
unzip meetup.zip
rm meetup.zip
