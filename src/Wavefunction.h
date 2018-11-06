#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"
#include "ViewWrapper.h"

class Wavefunction : protected Utils
{
 private:
  PetscInt ierr;
  PetscInt num_dims;         /* number of dimensions */
  PetscInt num_electrons;    /* number of electrons in the system */
  double *dim_size;          /* sizes of each dimension in a.u. */
  double *delta_x_min;       /* step sizes of each dimension in a.u. */
  double *delta_x_min_end;   /* step sizes of each dimension in a.u. */
  double *delta_x_max;       /* step sizes of each dimension in a.u. */
  double *delta_x_max_start; /* step sizes of each dimension in a.u. */
  double delta_t;            /* time step in a.u. */
  PetscInt coordinate_system_idx;
  std::string target_file_name;
  PetscInt num_states;
  PetscInt *num_x;        /* number of grid points in each dimension */
  PetscInt num_psi_build; /* number of points in psi_1 and psi_2 */
  PetscInt num_psi;       /* number of points in psi */
  double **x_value;       /* location of grid point in each dimension */
  dcomp ***psi_build;     /* used for allocating new wave functions */
  Vec psi;                /* wavefunction for 2 electron system */
  Vec psi_small;          /* wavefunction for spherical eigen states */
  Vec psi_tmp;            /* wavefunction for 2 electron system */
  Vec psi_proj;           /* wavefunction for 2 electron system */
  Vec psi_tmp_cyl;        /* wavefunction for 2 electron system */
  Vec jacobian;
  Vec ECS;
  Vec *position_expectation;
  Vec *dipole_acceleration;
  bool psi_alloc_build;
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma;           /* std of gaussian guess */
  PetscInt **gobbler_idx; /* distance that starts gobbler */
  PetscInt order;

  PetscInt write_counter_checkpoint;
  PetscInt write_counter_observables;
  PetscInt write_counter_projections;

  PetscLogEvent time_norm;
  PetscLogEvent time_energy;
  PetscLogEvent time_position;
  PetscLogEvent time_dipole_acceration;
  PetscLogEvent time_gobbler;
  PetscLogEvent time_projections;

  /* hidden from user for safety */
  void CreateGrid();
  void CreatePsi();
  void CreateObservables();
  void CleanUp();

  dcomp GetPsiVal(dcomp ***data, PetscInt idx);
  dcomp GetPositionVal(PetscInt idx, PetscInt elec_idx, PetscInt dim_idx,
                       bool integrate);
  dcomp GetDipoleAccerationVal(PetscInt idx, PetscInt elec_idx,
                               PetscInt dim_idx);
  dcomp GetGobblerVal(PetscInt idx);
  dcomp GetVolumeElement(PetscInt idx);

  double GetDistance(std::vector< PetscInt > idx_array, PetscInt elec_idx);

  std::vector< PetscInt > GetIntArray(PetscInt idx);

  void LoadRestart(HDF5Wrapper &h5_file, ViewWrapper &viewer_file,
                   PetscInt write_frequency_checkpoint,
                   PetscInt write_frequency_observables);

  void InsertRadialPsi(Vec &psi_radial, Vec &psi_total, PetscInt l_val);

 public:
  /* Constructor */
  Wavefunction(HDF5Wrapper &h5_file, ViewWrapper &view_file, Parameters &p);

  /* destructor */
  ~Wavefunction();

  /* IO */
  void Checkpoint(HDF5Wrapper &data_file, ViewWrapper &view_file, double time,
                  PetscInt checkpoint_psi = 0);
  void CheckpointPsi(ViewWrapper &view_file, PetscInt write_idx);
  void CheckpointPsiSmall(ViewWrapper &view_file, PetscInt write_idx,
                          PetscInt l_val);

  /* tools */
  void Normalize();
  void Normalize(Vec &data, double dx);
  double Norm();
  double Norm(Vec &data, double dx);
  double GetEnergy(Mat *h);
  double GetEnergy(Mat *h, Vec &p);
  double GetPosition(PetscInt elec_idx, PetscInt dim_idx);
  double GetDipoleAcceration(PetscInt elec_idx, PetscInt dim_idx);
  double GetGobbler();
  std::vector< dcomp > Projections(std::string file_name);
  void ProjectOut(std::string file_name, HDF5Wrapper &h5_file,
                  ViewWrapper &viewer_file, double time);
  void LoadPsi(std::string file_name, PetscInt num_states,
               PetscInt num_start_state, PetscInt *start_state_idx,
               double *start_state_amplitude, double *start_state_phase);
  void LoadPsi(std::string file_name, PetscInt num_states,
               PetscInt num_start_state, PetscInt *start_state_idx,
               PetscInt *start_state_l_idx, double *start_state_amplitude,
               double *start_state_phase);
  void ResetPsi();

  PetscInt *GetNumX();
  PetscInt GetNumPsi();
  PetscInt GetNumPsiBuild();
  Vec *GetPsi();
  Vec *GetPsiSmall();
  double **GetXValue();
  PetscInt **GetGobblerIdx();
  PetscInt GetWriteCounterCheckpoint();
};
