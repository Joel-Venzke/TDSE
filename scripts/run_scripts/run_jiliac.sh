#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -N Venzke
#PBS -j oe
#PBS -q nistQ

TDSE_ROOT=$HOME/Repos/TDSE

RUN_FILE=${TDSE_ROOT}/bin/TDSE
#RUN_FILE=./TDSE

module purge 
module load intel 
module load openmpi
module load hdf5 
module load boost
module load cmake
module load blas
module list

cd $PBS_O_WORKDIR

pwd 
export TMPDIR=/scratch/becker/jove7731
echo $TMPDIR
echo ${RUN_FILE}
mpiexec -n 16 ${RUN_FILE}  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist > run.log

