#!/bin/bash
# Copyright (c) 2019 by Tim Gubner
#
# Generates data sets in parallel
#
# It takes two environment variables as arguments:
# - PROBE: probe size
# - BUILD: build size
#
#
SPROBE="${PROBE:-0}"
SBUILD="${BUILD:-0}"
CMD="build/release/main_cu"

$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=1 &

for ((i=0;i<=100;i+=5)); 
do
	$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=${i} &
done

wait 