/**
 * @file Simulation.h
 * @brief Propagation and Eigen State calculations
 * @author Joel Venzke
 * @date 06/13/2017
 */
#pragma once
#include <time.h>
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "PETSCWrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Utils.h"
#include "Wavefunction.h"

/**
 * @brief Ground state calculations and time propagation
 * @details Allows for ground state calculations and time propagation. It has
 * support for split operator propagation, full Hamiltonian propagation, power
 * method eigen state calculations, and slepc support for eigen state
 * calculations
 *
 */
class Simulation : protected Utils
{
 private:
  PETSCWrapper p_wrap;
  Hamiltonian *hamiltonian;
  Wavefunction *wavefunction;
  Pulse *pulse;
  Parameters *parameters;
  HDF5Wrapper *h5_file;
  ViewWrapper *viewer_file;
  Vec *psi;
  Vec psi_right;
  double *time;
  PetscInt time_length;
  Mat *h;
  Mat left;                   ///< matrix on left side of Ax=b
  Mat right;                  ///< matrix on left side of Ax=Cb
  KSP ksp;                    ///< solver for Ax=b
  KSPConvergedReason reason;  ///< reason for convergence check

  /* destroys psi_old*/
  bool CheckConvergance(Vec &psi_1, Vec &psi_2, double tol);
  void ModifiedGramSchmidt(std::vector< Vec > &states);

 public:
  // Constructor
  Simulation(Hamiltonian &h, Wavefunction &w, Pulse &pulse_in,
             HDF5Wrapper &h_file, ViewWrapper &v_file, Parameters &p);
  ~Simulation();

  void EigenSolve(PetscInt num_states, PetscInt return_state_idx = 0);
  void PowerMethod(PetscInt num_states, PetscInt return_state_idx = 0);
  void Propagate();
  void SplitOpperator();
  void CrankNicolson(double dt, PetscInt time_idx, PetscInt dim_idx = -1);

  void CheckpointState(HDF5Wrapper &h_file, ViewWrapper &v_file,
                       PetscInt write_idx);
};