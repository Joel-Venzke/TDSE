#!/bin/bash

TDSE_ROOT_DIR=/Users/jvenzke/CU/RA/Code/TDSE

ANALYSIS_DIR=${TDSE_ROOT_DIR}/analysis

# make a directory for the images
mkdir figs

# animate wavefunction
echo "Creating Wavefunction animation"
python ${ANALYSIS_DIR}/wave_animation.py

# plot pulses
echo "Creating pulse plots"
python ${ANALYSIS_DIR}/pulse_plot.py
