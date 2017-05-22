#!/usr/bin/bash
#PBS -l nodes=1:ppn=04
#PBS -N Venzke
#PBS -j oe
#PBS -q xeon 

echo $HOSTNAME

TDSE_ROOT=/home/becker/jove7731/Repos/TDSE

RUN_FILE=${TDSE_ROOT}/bin/TDSE

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
mpiexec -n 4 ${RUN_FILE} -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist > run.log

