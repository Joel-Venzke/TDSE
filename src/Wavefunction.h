#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"
#include "ViewWrapper.h"

class Wavefunction : protected Utils
{
 private:
  PetscInt ierr;
  PetscInt num_dims;      /* number of dimensions */
  PetscInt num_electrons; /* number of electrons in the system */
  double *dim_size;       /* sizes of each dimension in a.u. */
  double *delta_x;        /* step sizes of each dimension in a.u. */
  PetscInt coordinate_system_idx;
  std::string target_file_name;
  PetscInt num_states;
  PetscInt *num_x;        /* number of grid points in each dimension */
  PetscInt num_psi_build; /* number of points in psi_1 and psi_2 */
  PetscInt num_psi;       /* number of points in psi */
  double **x_value;       /* location of grid point in each dimension */
  dcomp ***psi_build;     /* used for allocating new wave functions */
  Vec psi;                /* wavefunction for 2 electron system */
  Vec psi_tmp;            /* wavefunction for 2 electron system */
  Vec psi_tmp_cyl;        /* wavefunction for 2 electron system */
  bool psi_alloc_build;
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma;           /* std of gaussian guess */
  PetscInt **gobbler_idx; /* distance that starts gobbler */
  PetscInt order;

  PetscInt write_counter_checkpoint;
  PetscInt write_counter_observables;

  /* hidden from user for safety */
  void CreateGrid();
  void CreatePsi();
  void CreateObservable(PetscInt observable_idx, PetscInt elec_idx,
                        PetscInt dim_idx);
  void CleanUp();

  dcomp GetPsiVal(dcomp ***data, PetscInt idx);
  dcomp GetPositionVal(PetscInt idx, PetscInt elec_idx, PetscInt dim_idx,
                       bool integrate);
  dcomp GetDipoleAccerationVal(PetscInt idx, PetscInt elec_idx,
                               PetscInt dim_idx);
  dcomp GetGobblerVal(PetscInt idx);

  double GetDistance(std::vector< PetscInt > idx_array, PetscInt elec_idx);

  std::vector< PetscInt > GetIntArray(PetscInt idx);

  void LoadRestart(HDF5Wrapper &h5_file, ViewWrapper &viewer_file,
                   PetscInt write_frequency_checkpoint,
                   PetscInt write_frequency_observables);

 public:
  /* Constructor */
  Wavefunction(HDF5Wrapper &h5_file, ViewWrapper &view_file, Parameters &p);

  /* destructor */
  ~Wavefunction();

  /* IO */
  void Checkpoint(HDF5Wrapper &data_file, ViewWrapper &view_file, double time,
                  bool checkpoint_psi = true);
  void CheckpointPsi(ViewWrapper &view_file, PetscInt write_idx);

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
                  ViewWrapper &viewer_file);
  void ResetPsi();

  PetscInt *GetNumX();
  PetscInt GetNumPsi();
  PetscInt GetNumPsiBuild();
  Vec *GetPsi();
  double **GetXValue();
  PetscInt **GetGobblerIdx();
  PetscInt GetWrieCounterCheckpoint();
};
