#!/usr/bin/bash
#PBS -l nodes=1:ppn=32
#PBS -N Venzke
#PBS -j oe


TDSE_ROOT=/home/becker/jove7731/Repos/TDSE

RUN_FILE=${TDSE_ROOT}/src/TDSE

module unload intel
module load gcc
module load hdf5
module list

cd $PBS_O_WORKDIR


export OMP_NUM_THREADS=1
time ${RUN_FILE}>run_01.log

export OMP_NUM_THREADS=32
time ${RUN_FILE}>run_32.log
