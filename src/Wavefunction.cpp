#include "Wavefunction.h"

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

  /* allocate grid */
  CreateGrid();

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
    /* set time step */
    viewer_file.SetTime(write_counter);
    /* write vector */
    viewer_file.WriteObject((PetscObject)psi);
    /* get object name */
    PetscObjectGetName((PetscObject)psi, &tmp);
    name = tmp;
    viewer_file.WriteAttribute(name, "Attribute",
                               "Wavefunction for the two electron system");
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
    // if (psi_alloc_build)
    // {
    //   for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    //   {
    //     for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
    //     {
    //       h5_file.WriteObject(
    //           psi_build[elec_idx][dim_idx], num_x[dim_idx],
    //           "/Wavefunction/psi_build_dim_" + std::to_string(dim_idx),
    //           "Guess for then nth dimension of the wavefunction. "
    //           "Order is electron 0 the electron 1",
    //           elec_idx);
    //     }
    //   }
    // }

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
  PetscInt low, high;
  PetscComplex val;

  sigma2 = sigma * sigma;

  if (!psi_alloc_build)
  {
    psi_build = new dcomp**[num_electrons];

    for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    {
      psi_build[elec_idx] = new dcomp*[num_dims];

      for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        psi_build[elec_idx][dim_idx] = new dcomp[num_x[dim_idx]];
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

    psi_alloc = true;
  }

  VecGetOwnershipRange(psi, &low, &high);
  for (int idx = low; idx < high; idx++)
  {
    /* set psi */
    val = GetVal(psi_build, idx);
    VecSetValues(psi, 1, &idx, &val, INSERT_VALUES);
  }
  VecAssemblyBegin(psi);
  VecAssemblyEnd(psi);
  /* normalize all psi */
  Normalize();
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
      }
      delete[] psi_build[elec_idx];
    }
    delete[] psi_build;
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

/* normalizes the array provided */
void Wavefunction::Normalize(Vec& data, double dv)
{
  PetscReal total = Norm(data, dv);
  VecScale(data, 1.0 / total);
}

/* returns norm of psi */
double Wavefunction::Norm() { return Norm(psi, delta_x[0]); }

/* returns norm of array using trapezoidal rule */
double Wavefunction::Norm(Vec& data, double dv)
{
  double total = 0;
  VecNorm(data, NORM_2, &total);
  /* TODO(jove7731): normalize with correct dx term */
  // return total * dv;
  return total;
}

double Wavefunction::GetEnergy(Mat* h) { return GetEnergy(h, psi); }

double Wavefunction::GetEnergy(Mat* h, Vec& p)
{
  PetscComplex energy;
  MatMult(*h, p, psi_tmp);
  VecDot(p, psi_tmp, &energy);
  return energy.real();
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
}
