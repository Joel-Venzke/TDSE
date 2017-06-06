#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Utils.h"
#include "Wavefunction.h"

class Hamiltonian : protected Utils
{
 private:
  PetscInt num_dims;
  PetscInt num_electrons;
  PetscInt num_nuclei; /* number of nuclei in potential */
  PetscInt coordinate_system_idx;
  PetscInt *num_x;
  PetscInt num_psi;
  PetscInt num_psi_build;
  double *z;         /* atomic number of each nuclei */
  double **location; /* location of each nuclei */

  /* SAE stuff */
  double **a;         /* SAE a for each nuclei (coefficient of exponential) */
  double **b;         /* SAE b for each nuclei (in exponential) */
  double *r0;         /* SAE r_0 for each nuclei */
  double *c0;         /* SAE C_0 for each nuclei */
  double *z_c;        /* SAE z_0 for each nuclei */
  PetscInt *sae_size; /* number of elements in a and b */

  double alpha;   /* soft core atomic */
  double alpha_2; /* square of soft core atomic */
  double **field;
  double *delta_x;
  double *delta_x_2;
  double **x_value;
  Mat hamiltonian;
  PetscInt **gobbler_idx; /* distance that starts gobbler */
  double eta;             /* value in exponent for ECS */

  PetscInt order;
  PetscInt order_middle_idx;
  /* *_ecs_coef[dim_idx][discontinuity_idx][derivative][index] */
  std::vector< std::vector< std::vector< std::vector< dcomp > > > >
      left_ecs_coef;
  std::vector< std::vector< std::vector< std::vector< dcomp > > > >
      right_ecs_coef;
  /* radial_bc_coef[discontinuity_idx][derivative][index] */
  std::vector< std::vector< std::vector< dcomp > > > radial_bc_coef;
  /* real_coef[dim_idx][derivative][index] */
  std::vector< std::vector< std::vector< dcomp > > > real_coef;

  void CreateHamlitonian();
  void CalculateHamlitonian(PetscInt time_idx);
  void SetUpCoefficients();

 public:
  /* Constructor */
  Hamiltonian(Wavefunction &w, Pulse &pulse, HDF5Wrapper &data_file,
              Parameters &p);

  Mat *GetTotalHamiltonian(PetscInt time_idx);
  Mat *GetTimeIndependent();

  dcomp GetVal(PetscInt idx_i, PetscInt idx_j, bool time_dep, PetscInt time_idx,
               bool &insert_val);
  void FDWeights(std::vector< dcomp > &x_vals, PetscInt max_derivative,
                 std::vector< std::vector< dcomp > > &coef,
                 PetscInt z_idx = -1);
  dcomp GetOffDiagonal(std::vector< PetscInt > &idx_array,
                       std::vector< PetscInt > &diff_array, bool time_dep,
                       PetscInt time_idx);
  dcomp GetDiagonal(std::vector< PetscInt > &idx_array, bool time_dep,
                    PetscInt time_idx);
  dcomp GetKineticTerm(std::vector< PetscInt > &idx_array);
  dcomp GetNucleiTerm(std::vector< PetscInt > &idx_array);
  dcomp GetElectronElectronTerm(std::vector< PetscInt > &idx_array);
  PetscInt GetOffset(PetscInt elec_idx, PetscInt dim_idx);
  double SoftCoreDistance(double *location, std::vector< PetscInt > &idx_array,
                          PetscInt elec_idx);
  double SoftCoreDistance(std::vector< PetscInt > &idx_array,
                          PetscInt elec_idx_1, PetscInt elec_idx_2);
  std::vector< PetscInt > GetIndexArray(PetscInt idx_i, PetscInt idx_j);
  std::vector< PetscInt > GetDiffArray(std::vector< PetscInt > &idx_array);

  ~Hamiltonian();
};