#include "Wavefunction.h"
#include <math.h> /* ceil() */
#include <complex>
#include <iostream>

#define dcomp std::complex<double>

/* prints error message, kills code and returns -1 */
void Wavefunction::EndRun(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  exit(-1);
}

/* prints error message, kills code and returns exit_val */
void Wavefunction::EndRun(std::string str, int exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  exit(exit_val);
}

Wavefunction::Wavefunction(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                           Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Wavefunction\n";

  /* initialize values */
  psi_alloc_build = false;
  psi_alloc       = false;
  first_pass      = true;
  num_dims        = p.GetNumDims();
  num_electrons   = p.GetNumElectrons();
  dim_size        = p.dim_size.get();
  delta_x         = p.delta_x.get();
  sigma           = p.GetSigma();
  num_psi_build   = 1.0;
  write_counter   = 0;

  offset = new double[num_dims];
  width  = new double[num_dims];
  for (int i = 0; i < num_dims; ++i)
  {
    offset[i] = dim_size[i] / 2.0 * p.GetGobbler();
    width[i]  = pi / (2.0 * (dim_size[i] / 2.0 - offset[i]));
  }

  /* allocate grid */
  CreateGrid();
  if (world.rank() == 0) std::cout << "grid built\n";

  /* allocate psi_1, psi_2, and psi */
  CreatePsi();

  /* write out data */
  Checkpoint(h5_file, viewer_file, -1.0);

  /* delete psi_1 and psi_2 */
  CleanUp();

  if (world.rank() == 0) std::cout << "Wavefunction created\n";
}

void Wavefunction::Checkpoint(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                              double time)
{
  if (world.rank() == 0)
    std::cout << "Checkpointing Wavefunction: " << write_counter << "\n"
              << std::flush;
  std::string str;
  const char* tmp;
  std::string name;
  std::string group_name = "/Wavefunction/";

  /* only write out at start */
  if (first_pass)
  {
    viewer_file.Open("a");
    /* move into group */
    viewer_file.PushGroup(group_name);
    viewer_file.WriteObject((PetscObject)psi_gobbler);
    /* set time step */
    viewer_file.SetTime(write_counter);
    /* write vector */
    viewer_file.WriteObject((PetscObject)psi);
    /* get object name */
    PetscObjectGetName((PetscObject)psi, &tmp);
    name = tmp;
    viewer_file.WriteAttribute(name, "Attribute",
                               "Wavefunction for the two electron system");
    PetscObjectGetName((PetscObject)psi_gobbler, &tmp);
    name = tmp;
    viewer_file.WriteAttribute(name, "Attribute", "boundary potential for psi");
    /* close file */
    viewer_file.Close();

    /* size of each dim */
    h5_file.WriteObject(num_x, num_dims, "/Wavefunction/num_x",
                        "The number of physical dimension in the simulation");

    /* write each dims x values */
    for (int i = 0; i < num_dims; i++)
    {
      str = "x_value_" + std::to_string(i);
      h5_file.WriteObject(
          x_value[i], num_x[i], "/Wavefunction/" + str,
          "The coordinates of the " + std::to_string(i) + " dimension");
    }

    /* write psi_1 and psi_2 if still allocated */
    h5_file.WriteObject(psi_build[0][0], num_x[0], "/Wavefunction/psi_1",
                        "Wavefunction of first electron");
    h5_file.WriteObject(psi_build[1][0], num_x[0], "/Wavefunction/psi_2",
                        "Wavefunction of second electron");

    // h5_file.WriteObject(psi_1_gobbler, num_psi_build,
    //                     "/Wavefunction/psi_1_gobbler",
    //                     "Wavefunction of second electron");
    // h5_file.WriteObject(psi_2_gobbler, num_psi_build,
    //                     "/Wavefunction/psi_2_gobbler",
    //                     "Boundary potential of second electron");

    // h5_file.WriteObject(psi_gobbler->data(), num_psi,
    //                     "/Wavefunction/psi_gobbler",
    //                     "boundary potential for the two electron system");

    // /* write time and attribute */
    h5_file.WriteObject(time, "/Wavefunction/psi_time",
                        "Time step that psi was written to disk",
                        write_counter);

    /* write time and attribute */
    h5_file.WriteObject(Norm(), "/Wavefunction/norm", "Norm of wavefunction",
                        write_counter);

    /* allow for future passes to write psi only */
    first_pass = false;
  }
  else
  {
    viewer_file.Open("a");
    /* set time step */
    viewer_file.SetTime(write_counter);
    /* move into group */
    viewer_file.PushGroup(group_name);
    /* write vector */
    viewer_file.WriteObject((PetscObject)psi);
    /* close file */
    viewer_file.Close();

    /* write time */
    h5_file.WriteObject(time, "/Wavefunction/psi_time", write_counter);

    h5_file.WriteObject(Norm(), "/Wavefunction/norm", write_counter);
  }

  /* keep track of what index we are on */
  write_counter++;
}

void Wavefunction::CheckpointPsi(ViewWrapper& viewer_file, int write_idx)
{
  // if (world.rank() == 0)
  if (world.rank() == 0)
    std::cout << "Checkpointing Wavefunction in " << viewer_file.file_name
              << ": "
              << " " << write_idx << "\n";
  viewer_file.Open("a");
  /* set time step */
  viewer_file.SetTime(write_idx);
  /* write vector */
  viewer_file.WriteObject((PetscObject)psi);
  /* close file */
  viewer_file.Close();
}

void Wavefunction::CreateGrid()
{
  int center;       /* idx of the 0.0 in the grid */
  double current_x; /* used for setting grid */

  /* allocation */
  num_x   = new int[num_dims];
  x_value = new double*[num_dims];

  /* initialize for loop */
  num_psi_build = 1.0;

  /* build grid */
  for (int i = 0; i < num_dims; i++)
  {
    num_x[i] = ceil((dim_size[i]) / delta_x[i]) + 1;

    /* odd number so it is even on both sides */
    if (num_x[i] % 2 == 0) num_x[i]++;

    /* size of 1d array for psi */
    num_psi_build *= num_x[i];

    /* find center of grid */
    center = num_x[i] / 2 + 1;

    /* allocate grid */
    x_value[i] = new double[num_x[i]];

    /* store center */
    x_value[i][center] = 0.0;

    /* loop over all others */
    for (int j = center - 1; j > 0; j--)
    {
      /* get x value */
      current_x = (j - center) * delta_x[i];

      /* double checking index */
      if (j - 1 < 0 || num_x[i] - j >= num_x[i])
      {
        EndRun("Allocation error in grid");
      }

      /* set negative side */
      x_value[i][j - 1] = current_x;
      /* set positive side */
      x_value[i][num_x[i] - j] = -1 * current_x;
    }
  }
}

/* builds psi from 2 Gaussian psi (one for each electron) */
void Wavefunction::CreatePsi()
{
  double sigma2; /* variance squared for Gaussian in psi */
  double x;      /* x value squared */
  double x2;     /* x value squared */
  int rank;
  PetscComplex val;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  sigma2 = sigma * sigma;

  if (!psi_alloc_build)
  {
    psi_build         = new dcomp**[num_electrons];
    psi_gobbler_build = new dcomp**[num_electrons];

    for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    {
      psi_build[elec_idx]         = new dcomp*[num_dims];
      psi_gobbler_build[elec_idx] = new dcomp*[num_dims];

      for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        psi_build[elec_idx][dim_idx]         = new dcomp[num_x[dim_idx]];
        psi_gobbler_build[elec_idx][dim_idx] = new dcomp[num_x[dim_idx]];
      }
    }
    psi_alloc_build = true;
  }

  for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      for (int i = 0; i < num_x[dim_idx]; i++)
      {
        /* get x value squared */
        x  = x_value[dim_idx][i];
        x2 = x * x;

        /* Gaussian centered around 0.0 with variation sigma */
        psi_build[elec_idx][dim_idx][i] =
            dcomp(exp(-1 * x2 / (2 * sigma2)), 0.0);
        if (std::abs(x) - offset[dim_idx] > 0)
        {
          psi_gobbler_build[elec_idx][dim_idx][i] = std::pow(
              cos((std::abs(x) - offset[dim_idx]) * width[dim_idx]), 1.0 / 8.0);
        }
        else
        {
          psi_gobbler_build[elec_idx][dim_idx][i] = dcomp(1.0, 0.0);
        }
      }
    }
  }

  /* get size of psi */
  num_psi = 1.0;
  for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    num_psi *= num_psi_build;
  }

  /* allocate psi */
  if (!psi_alloc)
  {
    VecCreate(PETSC_COMM_WORLD, &psi);
    VecSetSizes(psi, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi);
    ierr = PetscObjectSetName((PetscObject)psi, "psi");

    VecCreate(PETSC_COMM_WORLD, &psi_tmp);
    VecSetSizes(psi_tmp, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_tmp);

    VecCreate(PETSC_COMM_WORLD, &psi_gobbler);
    VecSetSizes(psi_gobbler, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_gobbler);
    ierr = PetscObjectSetName((PetscObject)psi_gobbler, "psi_gobbler");

    psi_alloc = true;
  }

  /* tensor product of psi_1 and psi_2 */
  for (int idx = 0; idx < num_psi; idx++)
  {
    if (rank == 0)
    {
      /* set psi */
      val = GetVal(psi_build, idx);
      VecSetValues(psi, 1, &idx, &val, INSERT_VALUES);

      /* set psi */
      val = GetVal(psi_gobbler_build, idx);
      VecSetValues(psi_gobbler, 1, &idx, &val, INSERT_VALUES);
    }
  }
  VecAssemblyBegin(psi);
  VecAssemblyBegin(psi_gobbler);
  VecAssemblyEnd(psi);

  /* normalize all psi */
  Normalize();

  /* psi_gobbler not used in Normalize */
  VecAssemblyEnd(psi_gobbler);
}

void Wavefunction::CleanUp()
{
  if (psi_alloc_build)
  {
    for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    {
      for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        delete psi_build[elec_idx][dim_idx];
        delete psi_gobbler_build[elec_idx][dim_idx];
      }
      delete[] psi_build[elec_idx];
      delete[] psi_gobbler_build[elec_idx];
    }
    delete[] psi_build;
    delete[] psi_gobbler_build;
    psi_alloc_build = false;
  }
}

dcomp Wavefunction::GetVal(dcomp*** data, int idx)
{
  /* Value to be returned */
  dcomp ret_val(1.0, 0.0);
  /* Total number of dims for total system*/
  int total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector<int> num(total_dims);
  /* idx for return */
  std::vector<int> idx_array(total_dims);
  for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      num[elec_idx * num_dims + dim_idx] = num_x[dim_idx];
    }
  }
  for (int i = total_dims - 1; i >= 0; --i)
  {
    idx_array[i] = idx % num[i];
    idx /= num[i];
  }
  for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      ret_val *=
          data[elec_idx][dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
    }
  }
  return ret_val;
}

/* normalize psi_1, psi_2, and psi */
void Wavefunction::Normalize() { Normalize(psi, delta_x[0]); }

void Wavefunction::GobblePsi() { VecPointwiseMult(psi, psi_gobbler, psi); }

/* normalizes the array provided */
void Wavefunction::Normalize(Vec& data, double dx)
{
  PetscReal total = Norm(data, dx);
  VecScale(data, 1.0 / total);
}

/* returns norm of psi */
double Wavefunction::Norm() { return Norm(psi, delta_x[0]); }

/* returns norm of array using trapezoidal rule */
double Wavefunction::Norm(Vec& data, double dx)
{
  double total = 0;
  VecNorm(data, NORM_2, &total);
  return total * dx;
}

double Wavefunction::GetEnergy(Mat* h) { return GetEnergy(h, psi); }

double Wavefunction::GetEnergy(Mat* h, Vec& p)
{
  PetscComplex energy;
  VecDot(p, p, &energy);
  MatMult(*h, p, psi_tmp);
  VecDot(p, psi_tmp, &energy);
  return energy.real() * (delta_x[0] * delta_x[0]);
}

void Wavefunction::ResetPsi()
{
  CreatePsi();
  CleanUp();
}

int* Wavefunction::GetNumX() { return num_x; }

int Wavefunction::GetNumPsi() { return num_psi; }

int Wavefunction::GetNumPsiBuild() { return num_psi_build; }

Vec* Wavefunction::GetPsi() { return &psi; }

double* Wavefunction::GetDeltaX() { return delta_x; }

double** Wavefunction::GetXValue() { return x_value; }

/* destructor */
Wavefunction::~Wavefunction()
{
  if (world.rank() == 0) std::cout << "Deleting Wavefunction\n";
  /* do not delete dim_size or delta_x since they belong to the Parameter
   * class and will be freed there*/
  delete num_x;
  for (int i = 0; i < num_dims; i++)
  {
    delete x_value[i];
  }
  delete[] x_value;
  CleanUp();
  VecDestroy(&psi);
  VecDestroy(&psi_tmp);
  VecDestroy(&psi_gobbler);
}
