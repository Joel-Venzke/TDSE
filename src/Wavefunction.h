#pragma once
#include <math.h>
#include <petsc.h>
#include <boost/mpi.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <complex>
#include <iostream>
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "ViewWrapper.h"

namespace mpi = boost::mpi;

typedef std::complex<double> dcomp;

class Wavefunction
{
 private:
  mpi::communicator world;
  const double pi = 3.1415926535897;
  PetscInt ierr;
  int num_dims;               /* number of dimensions */
  int num_electrons;          /* number of electrons in the system */
  double *dim_size;           /* sizes of each dimension in a.u. */
  double *delta_x;            /* step sizes of each dimension in a.u. */
  int *num_x;                 /* number of grid points in each dimension */
  int num_psi_build;          /* number of points in psi_1 and psi_2 */
  int num_psi;                /* number of points in psi */
  double **x_value;           /* location of grid point in each dimension */
  dcomp ***psi_build;         /* used for allocating new wave functions */
  dcomp ***psi_gobbler_build; /* used for allocating new wave functions */
  Vec psi;                    /* wavefunction for 2 electron system */
  Vec psi_tmp;                /* wavefunction for 2 electron system */
  Vec psi_gobbler;            /* boundary for 2 electron system */
  bool psi_alloc_build;
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma;   /* std of gaussian guess */
  double *offset; /* distance that starts gobbler */
  double *width;  /* width of gobbler */

  int write_counter;

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
  void Checkpoint(HDF5Wrapper &data_file, ViewWrapper &view_file, double time);
  void CheckpointPsi(ViewWrapper &view_file, int write_idx);

  /* tools */
  void Normalize();
  void Normalize(Vec &data, double dx);
  double Norm();
  double Norm(Vec &data, double dx);
  double GetEnergy(Mat *h);
  double GetEnergy(Mat *h, Vec &p);
  void ResetPsi();
  void GobblePsi();

  int *GetNumX();
  int GetNumPsi();
  int GetNumPsiBuild();
  Vec *GetPsi();
  double *GetDeltaX();
  double **GetXValue();

  /* error handling */
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};
