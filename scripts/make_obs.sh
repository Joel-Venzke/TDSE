#!/bin/bash

# remove Observables.h5 if it exists

h5copy -i Pulse.h5 -o Observables.h5 -s /Pulse -d /Pulse
h5copy -i TDSE.h5 -o Observables.h5 -s /Observables -d /Observables
h5copy -i TDSE.h5 -o Observables.h5 -s /Parameters -d /Parameters
h5copy -i TDSE.h5 -o Observables.h5 -p  -s /Wavefunction/norm -d /Wavefunction/norm
h5copy -i TDSE.h5 -o Observables.h5 -s /Wavefunction/num_x -d /Wavefunction/num_x
h5copy -i TDSE.h5 -o Observables.h5 -s /Wavefunction/projections -d /Wavefunction/projections
h5copy -i TDSE.h5 -o Observables.h5 -s /Wavefunction/time -d /Wavefunction/time
h5copy -i TDSE.h5 -o Observables.h5 -s /Wavefunction/x_value_0 -d /Wavefunction/x_value_0

