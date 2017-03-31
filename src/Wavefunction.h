#pragma once
#include <petsc.h>
#include <petscviewerhdf5.h>
#include <boost/mpi.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <complex>
#include <iostream>
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "ViewWrapper.h"

namespace mpi = boost::mpi;

#define dcomp std::complex<double>

class Wavefunction
{
 private:
  mpi::communicator world;
  const double pi = 3.1415926535897;
  int num_dims;         /* number of dimensions */
  double *dim_size;     /* sizes of each dimension in a.u. */
  double *delta_x;      /* step sizes of each dimension in a.u. */
  int *num_x;           /* number of grid points in each dimension */
  int num_psi_12;       /* number of points in psi_1 and psi_2 */
  int num_psi;          /* number of points in psi */
  double **x_value;     /* location of grid point in each dimension */
  dcomp *psi_1;         /* wavefunction for electron 1 */
  dcomp *psi_1_gobbler; /* boundary for electron 1 */
  dcomp *psi_2;         /* wavefunction for electron 2 */
  dcomp *psi_2_gobbler; /* boundary for electron 2 */
  Vec psi;              /* wavefunction for 2 electron system */
  dcomp *psi_gobbler;   /* boundary for 2 electron system */
  bool psi_12_alloc;    /* true if psi_1 and psi_2 are allocated */
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma;  /* std of gaussian guess */
  double offset; /* distance that starts gobbler */
  double width;  /* width of gobbler */

  int write_counter;

  /* hidden from user for safety */
  void CreateGrid();
  void CreatePsi();
  void Cleanup();

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
  // double get_energy(Eigen::SparseMatrix<dcomp> *h);
  void reset_psi();
  void GobblePsi();

  int *GetNumX();
  int GetNumPsi();
  int GetNumPsi12();
  Vec *GetPsi();
  double *GetDeltaX();
  double **GetXValue();

  /* error handling */
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};
