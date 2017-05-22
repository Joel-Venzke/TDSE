#!/usr/bin/bash
#PBS -l nodes=32:ppn=1
#PBS -N Venzke
#PBS -j oe
#PBS -q nistQ

module purge 
module load intel 
module load openmpi
module load hdf5 
module load boost
module load cmake
module list

TDSE_ROOT=/home/becker/jove7731/Repos/TDSE

RUN_FILE=${TDSE_ROOT}/src/TDSE

module unload intel
module load gcc
module load hdf5
module list

cd $PBS_O_WORKDIR

mpiexec -n 32 ${RUN_FILE}  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist
