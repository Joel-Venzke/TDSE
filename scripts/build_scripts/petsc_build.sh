#!/bin/bash
export PETSC_DIR=`pwd`

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
  CXXOPTFLAGS=“-mtune=haswell -O2” \
  COPTFLAGS=“-mtune=haswell -O2” \
  FOPTFLAGS=“-mtune=haswell -O2” \
  --prefix=/projects/cogo4490/local/gcc/petsc \
  --with-shared-libraries=1 \
  --with-pthread=0 \
  --with-openmp=0 \
  --with-debugging=0 \
  --with-ssl=0 \
  --with-x=0 \
  --with-valgrind=1 \
  --with-fortran-kernels=1 \
  --with-cxx-dialect=c++11 \
  --with-scalar-type=complex \
  --with-hdf5-dir=/projects/cogo4490/local/gcc/hdf5 \
  --with-blas-lapack-dir=${CURC_MKL_LIB} \
  --download-parmetis=yes \
  --download-fftw \
  --download-metis 

make PETSC_DIR=/projects/cogo4490/repos/petsc PETSC_ARCH=arch-linux2-c-opt all
make PETSC_DIR=/projects/cogo4490/repos/petsc PETSC_ARCH=arch-linux2-c-opt install

