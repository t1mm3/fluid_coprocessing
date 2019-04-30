#!/bin/bash
# Proper header for a Bash script.

if [ ! -d "amsfilter" ]
then
	git clone git@github.com:harald-lang/amsfilter.git
	cd amsfilter
	git submodule update --remote --recursive --init
fi
cd amsfilter
mkdir -p build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 6
cd amsfilter
# Make sure that the machine is idle during calibration. 
./amsfilter_calibration
