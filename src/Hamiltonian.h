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

  /* SAE stuff */
  double **a;    /* SAE a for each nuclei (coefficient of exponential) */
  double **b;    /* SAE b for each nuclei (in exponential) */
  double *r0;    /* SAE r_0 for each nuclei */
  double *c0;    /* SAE C_0 for each nuclei */
  double *z_c;   /* SAE z_0 for each nuclei */
  int *sae_size; /* number of elements in a and b */

  double alpha;   /* soft core atomic */
  double alpha_2; /* square of soft core atomic */
  double **field;
  double *delta_x;
  double *delta_x_2;
  double **x_value;
  Mat hamiltonian;
  int **gobbler_idx; /* distance that starts gobbler */
  double eta;        /* value in exponent for ECS */

  void CreateHamlitonian();
  void CalculateHamlitonian(int time_idx);

 public:
  /* Constructor */
  Hamiltonian(Wavefunction &w, Pulse &pulse, HDF5Wrapper &data_file,
              Parameters &p);

  Mat *GetTotalHamiltonian(int time_idx);
  Mat *GetTimeIndependent();

  dcomp GetVal(int idx_i, int idx_j, bool time_dep, int time_idx,
               bool &insert_val);
  dcomp GetOffDiagonal(std::vector<int> &idx_array,
                       std::vector<int> &diff_array, bool time_dep,
                       int time_idx);
  dcomp GetDiagonal(std::vector<int> &idx_array, bool time_dep, int time_idx);
  dcomp GetKineticTerm(std::vector<int> &idx_array);
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