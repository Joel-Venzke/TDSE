#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Utils.h"
#include "Wavefunction.h"

class Hamiltonian : protected Utils
{
 private:
  int num_dims;
  int num_electrons;
  int num_nuclei; /* number of nuclei in potential */
  int *num_x;
  int num_psi;
  int num_psi_build;
  double *z;         /* atomic number of each nuclei */
  double **location; /* location of each nuclei */
  double alpha;      /* soft core atomic */
  double alpha_2;    /* square of soft core atomic */
  double *a_field;
  double *delta_x;
  double *delta_x_2;
  double **x_value;
  Mat time_independent;
  Mat time_dependent;
  Mat total_hamlitonian;

  void CreateTimeIndependent();
  void CreateTimeDependent();
  void CreateTotalHamlitonian();

 public:
  /* Constructor */
  Hamiltonian(Wavefunction &w, Pulse &pulse, HDF5Wrapper &data_file,
              Parameters &p);

  Mat *GetTotalHamiltonian(int time_idx);
  Mat *GetTimeIndependent();

  dcomp GetVal(int idx_i, int idx_j, bool time_dep);
  dcomp GetOffDiagonal(std::vector<int> &idx_array,
                       std::vector<int> &diff_array, bool time_dep);
  dcomp GetDiagonal(std::vector<int> &idx_array, bool time_dep);
  dcomp GetKineticTerm();
  dcomp GetNucleiTerm(std::vector<int> &idx_array);
  dcomp GetElectronElectronTerm(std::vector<int> &idx_array);
  int GetOffset(int elec_idx, int dim_idx);
  double SoftCoreDistance(double *location, std::vector<int> &idx_array,
                          int elec_idx);
  double SoftCoreDistance(std::vector<int> &idx_array, int elec_idx_1,
                          int elec_idx_2);
  std::vector<int> GetIndexArray(int idx_i, int idx_j);
  std::vector<int> GetDiffArray(std::vector<int> &idx_array);

  ~Hamiltonian();
};