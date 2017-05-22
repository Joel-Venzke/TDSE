#include "Wavefunction.h"

Wavefunction::Wavefunction(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                           Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Wavefunction\n";

  /* initialize values */
  psi_alloc_build           = false;
  psi_alloc                 = false;
  first_pass                = true;
  num_dims                  = p.GetNumDims();
  num_electrons             = p.GetNumElectrons();
  dim_size                  = p.dim_size.get();
  delta_x                   = p.delta_x.get();
  sigma                     = p.GetSigma();
  num_psi_build             = 1.0;
  write_counter_checkpoint  = 0;
  write_counter_observables = 0;

  /* allocate grid */
  CreateGrid();

  /* allocate psi_1, psi_2, and psi */
  CreatePsi();

  /* allocate psi_1, psi_2, and psi */
  CreateObservables();

  /* write out data */
  Checkpoint(h5_file, viewer_file, -1.0);

  /* delete psi_1 and psi_2 */
  CleanUp();

  if (world.rank() == 0) std::cout << "Wavefunction created\n";
}

void Wavefunction::Checkpoint(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                              double time, bool checkpoint_psi)
{
  if (world.rank() == 0)
  {
    if (checkpoint_psi)
    {
      std::cout << "Checkpointing Psi: " << write_counter_checkpoint << "\n"
                << std::flush;
    }
  }
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
    viewer_file.SetTime(write_counter_checkpoint);
    /* write vector */
    viewer_file.WriteObject((PetscObject)psi);
    /* get object name */
    PetscObjectGetName((PetscObject)psi, &tmp);
    name = tmp;
    viewer_file.WriteAttribute(name, "Attribute",
                               "Wavefunction for the two electron system");
    for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        viewer_file.WriteObject(
            (PetscObject)position_expectation[elec_idx * num_dims + dim_idx]);
        PetscObjectGetName(
            (PetscObject)position_expectation[elec_idx * num_dims + dim_idx],
            &tmp);
        name = tmp;
        viewer_file.WriteAttribute(
            name, "Attribute",
            "Vector for measuring position expectation value");
      }
    }
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
    h5_file.WriteObject(time, "/Wavefunction/time",
                        "Time step that psi was written to disk",
                        write_counter_checkpoint);

    /* write observables */
    h5_file.WriteObject(time, "/Observables/time",
                        "Time step that the observables were written to disk",
                        write_counter_observables);
    h5_file.WriteObject(Norm(), "/Observables/norm", "Norm of wavefunction",
                        write_counter_observables);
    for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        h5_file.WriteObject(GetPosition(elec_idx, dim_idx),
                            "/Observables/position_expectation_" +
                                std::to_string(elec_idx) + "_" +
                                std::to_string(dim_idx),
                            "Expectation value of position for the " +
                                std::to_string(elec_idx) + " electron and " +
                                std::to_string(dim_idx) + " dimension",
                            write_counter_observables);
        h5_file.WriteObject(
            GetDipoleAcceration(elec_idx, dim_idx),
            "/Observables/dipole_acceleration_" + std::to_string(elec_idx) +
                "_" + std::to_string(dim_idx),
            "Expectation value of the dipole acceleration for the " +
                std::to_string(elec_idx) + " electron and " +
                std::to_string(dim_idx) + " dimension",
            write_counter_observables);
      }
    }

    /* allow for future passes to write psi or observables only */
    first_pass = false;
    write_counter_checkpoint++;
    write_counter_observables++;
  }
  else
  {
    if (checkpoint_psi)
    {
      viewer_file.Open("a");
      /* set time step */
      viewer_file.SetTime(write_counter_checkpoint);
      /* move into group */
      viewer_file.PushGroup(group_name);
      /* write vector */
      viewer_file.WriteObject((PetscObject)psi);
      /* close file */
      viewer_file.Close();

      /* write time */
      h5_file.WriteObject(time, "/Wavefunction/time", write_counter_checkpoint);
      write_counter_checkpoint++;
    }
    else
    {
      h5_file.WriteObject(time, "/Observables/time", write_counter_observables);
      h5_file.WriteObject(Norm(), "/Observables/norm",
                          write_counter_observables);
      for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
        {
          h5_file.WriteObject(GetPosition(elec_idx, dim_idx),
                              "/Observables/position_expectation_" +
                                  std::to_string(elec_idx) + "_" +
                                  std::to_string(dim_idx),
                              write_counter_observables);
          h5_file.WriteObject(GetDipoleAcceration(elec_idx, dim_idx),
                              "/Observables/dipole_acceleration_" +
                                  std::to_string(elec_idx) + "_" +
                                  std::to_string(dim_idx),
                              write_counter_observables);
        }
      }
      write_counter_observables++;
    }
  }
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
    if (num_x[i] % 2 != 0) num_x[i]++;

    /* size of 1d array for psi */
    num_psi_build *= num_x[i];

    /* find center of grid */
    center = num_x[i] / 2;

    /* allocate grid */
    x_value[i] = new double[num_x[i]];

    /* store center */
    x_value[i][center] = 0.0;

    /* loop over all others */
    for (int j = center; j >= 0; j--)
    {
      /* get x value */
      current_x = (j - center) * delta_x[i] + delta_x[i] / 2.0;

      /* double checking index */
      if (j < 0 || num_x[i] - j - 1 >= num_x[i])
      {
        EndRun("Allocation error in grid");
      }

      /* set negative side */
      x_value[i][j] = current_x;
      /* set positive side */
      x_value[i][num_x[i] - j - 1] = -1 * current_x;
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
    val = GetPsiVal(psi_build, idx);
    VecSetValues(psi, 1, &idx, &val, INSERT_VALUES);
  }
  VecAssemblyBegin(psi);
  VecAssemblyEnd(psi);
  /* normalize all psi */
  Normalize();
}

void Wavefunction::CreateObservables()
{
  PetscInt low, high;
  PetscComplex val;
  /* allocate arrays */
  position_expectation = new Vec[num_dims * num_electrons];
  dipole_acceleration  = new Vec[num_dims * num_electrons];
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      /* create position vector */
      VecCreate(PETSC_COMM_WORLD,
                &position_expectation[elec_idx * num_dims + dim_idx]);
      VecSetSizes(position_expectation[elec_idx * num_dims + dim_idx],
                  PETSC_DECIDE, num_psi);
      VecSetFromOptions(position_expectation[elec_idx * num_dims + dim_idx]);
      ierr = PetscObjectSetName(
          (PetscObject)position_expectation[elec_idx * num_dims + dim_idx],
          ("position_expectation_" + std::to_string(elec_idx) + "_" +
           std::to_string(dim_idx))
              .c_str());

      /* Fill position vector*/
      VecGetOwnershipRange(position_expectation[elec_idx * num_dims + dim_idx],
                           &low, &high);
      for (int idx = low; idx < high; idx++)
      {
        val = GetPositionVal(idx, elec_idx, dim_idx);
        VecSetValues(position_expectation[elec_idx * num_dims + dim_idx], 1,
                     &idx, &val, INSERT_VALUES);
      }
      /* Assemble position vector */
      VecAssemblyBegin(position_expectation[elec_idx * num_dims + dim_idx]);
      VecAssemblyEnd(position_expectation[elec_idx * num_dims + dim_idx]);

      /* create dipole acceleration vector */
      VecCreate(PETSC_COMM_WORLD,
                &dipole_acceleration[elec_idx * num_dims + dim_idx]);
      VecSetSizes(dipole_acceleration[elec_idx * num_dims + dim_idx],
                  PETSC_DECIDE, num_psi);
      VecSetFromOptions(dipole_acceleration[elec_idx * num_dims + dim_idx]);
      ierr = PetscObjectSetName(
          (PetscObject)dipole_acceleration[elec_idx * num_dims + dim_idx],
          ("dipole_acceleration_" + std::to_string(elec_idx) + "_" +
           std::to_string(dim_idx))
              .c_str());

      /* fill dipole acceleration vector */
      VecGetOwnershipRange(dipole_acceleration[elec_idx * num_dims + dim_idx],
                           &low, &high);
      for (int idx = low; idx < high; idx++)
      {
        val = GetDipoleAccerationVal(idx, elec_idx, dim_idx);
        VecSetValues(dipole_acceleration[elec_idx * num_dims + dim_idx], 1,
                     &idx, &val, INSERT_VALUES);
      }
      /* assemble dipole acceleration vector */
      VecAssemblyBegin(dipole_acceleration[elec_idx * num_dims + dim_idx]);
      VecAssemblyEnd(dipole_acceleration[elec_idx * num_dims + dim_idx]);
    }
  }
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

/* returns values for global psi */
dcomp Wavefunction::GetPsiVal(dcomp*** data, int idx)
{
  /* Value to be returned */
  dcomp ret_val(1.0, 0.0);
  /* idx for return */
  std::vector<int> idx_array = GetIntArray(idx);
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

/* returns values for global position vector */
dcomp Wavefunction::GetPositionVal(int idx, int elec_idx, int dim_idx)
{
  /* Value to be returned */
  dcomp ret_val(0.0, 0.0);
  /* idx for return */
  std::vector<int> idx_array = GetIntArray(idx);
  ret_val += x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
  return ret_val;
}

/* returns values for global dipole acceleration */
dcomp Wavefunction::GetDipoleAccerationVal(int idx, int elec_idx, int dim_idx)
{
  /* Value to be returned */
  dcomp ret_val(0.0, 0.0);
  double r;
  /* idx for return */
  std::vector<int> idx_array = GetIntArray(idx);
  r                          = GetDistance(idx_array, elec_idx);
  ret_val +=
      x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]] / (r * r * r);
  return ret_val;
}

/* Returns r component of that electron */
double Wavefunction::GetDistance(std::vector<int> idx_array, int elec_idx)
{
  double r = 0.0;
  double x;
  for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
  {
    x = x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
    r += x * x;
  }
  return sqrt(r);
}

std::vector<int> Wavefunction::GetIntArray(int idx)
{
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
  return idx_array;
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

/* returns position expectation value <psi|x_{elec_idx,dim_idx}|psi> */
double Wavefunction::GetPosition(int elec_idx, int dim_idx)
{
  PetscComplex expectation;
  VecPointwiseMult(psi_tmp, position_expectation[elec_idx * num_dims + dim_idx],
                   psi);
  VecDot(psi, psi_tmp, &expectation);
  return expectation.real();
}

/* returns dipole acceleration value <psi|x_{elec_idx,dim_idx}/r^3|psi> */
double Wavefunction::GetDipoleAcceration(int elec_idx, int dim_idx)
{
  PetscComplex expectation;
  VecPointwiseMult(psi_tmp, dipole_acceleration[elec_idx * num_dims + dim_idx],
                   psi);
  VecDot(psi, psi_tmp, &expectation);
  return expectation.real();
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
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      /* destroy position vector */
      VecDestroy(&position_expectation[elec_idx * num_dims + dim_idx]);

      /* destroy dipole acceleration vector */
      VecDestroy(&dipole_acceleration[elec_idx * num_dims + dim_idx]);
    }
  }
  delete position_expectation;
  delete dipole_acceleration;
}
