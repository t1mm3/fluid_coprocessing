#!/bin/bash
# Copyright (c) 2019 by Tim Gubner
#
# Script for smaller machines which only generate three files in parallel
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
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=5 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=10 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=15 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=20 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=25 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=30 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=35 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=40 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=45 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=50 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=55 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=60 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=65 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=70 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=75 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=80 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=85 &
wait
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=90 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=95 &
$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=100 &
wait


for ((i=0;i<=100;i+=5)); 
do
	$CMD --only_generate=1 --probe_size=${SPROBE} --build_size=${SBUILD} --selectivity=${i} &
done

wait 