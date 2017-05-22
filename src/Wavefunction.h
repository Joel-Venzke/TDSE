#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"
#include "ViewWrapper.h"

class Wavefunction : protected Utils
{
 private:
  PetscInt ierr;
  int num_dims;       /* number of dimensions */
  int num_electrons;  /* number of electrons in the system */
  double *dim_size;   /* sizes of each dimension in a.u. */
  double *delta_x;    /* step sizes of each dimension in a.u. */
  int *num_x;         /* number of grid points in each dimension */
  int num_psi_build;  /* number of points in psi_1 and psi_2 */
  int num_psi;        /* number of points in psi */
  double **x_value;   /* location of grid point in each dimension */
  dcomp ***psi_build; /* used for allocating new wave functions */
  Vec psi;            /* wavefunction for 2 electron system */
  Vec psi_tmp;        /* wavefunction for 2 electron system */
  bool psi_alloc_build;
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma; /* std of gaussian guess */

  int write_counter_checkpoint;
  int write_counter_observables;

  /* hidden from user for safety */
  void CreateGrid();
  void CreatePsi();
  void CleanUp();

  dcomp GetVal(dcomp ***data, int idx);

 public:
  /* Constructor */
  Wavefunction(HDF5Wrapper &h5_file, ViewWrapper &view_file, Parameters &p);

  /* destructor */
  ~Wavefunction();

  /* IO */
  void Checkpoint(HDF5Wrapper &data_file, ViewWrapper &view_file, double time,
                  bool checkpoint_psi = true);
  void CheckpointPsi(ViewWrapper &view_file, int write_idx);

  /* tools */
  void Normalize();
  void Normalize(Vec &data, double dx);
  double Norm();
  double Norm(Vec &data, double dx);
  double GetEnergy(Mat *h);
  double GetEnergy(Mat *h, Vec &p);
  void ResetPsi();

  int *GetNumX();
  int GetNumPsi();
  int GetNumPsiBuild();
  Vec *GetPsi();
  double **GetXValue();
};
