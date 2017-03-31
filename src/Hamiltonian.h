#pragma once
#include <math.h>
#include <petsc.h>
#include <stdlib.h>
#include <boost/mpi.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <iostream>
#include <vector>
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Wavefunction.h"

class Hamiltonian
{
 private:
  mpi::communicator world;
  int num_dims;
  int *num_x;
  int num_psi;
  int num_psi_12;
  double z;      // atomic number
  double alpha;  // soft core atomic
  double *a_field;
  double *delta_x;
  double **x_value;
  Mat time_independent;
  Mat time_dependent;
  Mat total_hamlitonian;

  void CreateTimeIndependent();
  void CreateTimeDependent();
  void CreateTotalHamlitonian();

 public:
  // Constructor
  Hamiltonian(Wavefunction &w, Pulse &pulse, HDF5Wrapper &data_file,
              Parameters &p);

  Mat *GetTotalHamiltonian(int time_idx);
  Mat *GetTimeIndependent();

  ~Hamiltonian();
};