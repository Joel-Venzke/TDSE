#!/bin/bash

module purge
module load gcc
module load impi
module load szip 
module load zlib 
module list 


/projects/cogo4490/repos/hdf5-1.10.0-patch1/configure \
  --prefix=/projects/cogo4490/local/gcc/hdf5 \
  --enable-build-mode=production \
  --disable-dependency-tracking \
  --with-zlib=${CURC_ZLIB_ROOT} \
  --with-szlib=${CURC_SZIP_ROOT} \
  --enable-static=yes \
  --enable-shared=yes \
  --enable-unsupported \
  --enable-cxx \
  --enable-fortran \
  --enable-parallel
make 
make check 
make install
