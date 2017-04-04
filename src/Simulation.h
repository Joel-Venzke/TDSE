#pragma once
#include <time.h>
#include <boost/mpi.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <iostream>
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Wavefunction.h"

class Simulation
{
 private:
  mpi::communicator world;
  Hamiltonian *hamiltonian;
  Wavefunction *wavefunction;
  Pulse *pulse;
  Parameters *parameters;
  HDF5Wrapper *h5_file;
  ViewWrapper *viewer_file;
  Vec *psi;
  double *time;
  int time_length;

  /* destroys psi_old*/
  bool CheckConvergance(Vec &psi_1, Vec &psi_2, double tol);
  void ModifiedGramSchmidt(std::vector<Vec> &states);

 public:
  // Constructor
  Simulation(Hamiltonian &h, Wavefunction &w, Pulse &pulse_in,
             HDF5Wrapper &h_file, ViewWrapper &v_file, Parameters &p);

  void PowerMethod(int num_states);
  void Propagate();

  void CheckpointState(HDF5Wrapper &h_file, ViewWrapper &v_file, int write_idx);
};