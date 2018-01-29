#!/bin/bash
export SLEPC_DIR=/projects/cogo4490/repos/slepc

module load gcc
module load impi 
module load szip 
module load zlib
module load boost
module load gsl
module load mkl
module load valgrind
module load fftw
module load cmake 
module list

./configure \
  --prefix=/projects/cogo4490/local/gcc/slepc


