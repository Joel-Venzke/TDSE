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
  coordinate_system_idx     = p.GetCoordinateSystemIdx();
  target_file_name          = p.GetTarget() + ".h5";
  num_states                = p.GetNumStates();
  sigma                     = p.GetSigma();
  num_psi_build             = 1.0;
  write_counter_checkpoint  = 0;
  write_counter_observables = 0;
  order                     = p.GetOrder();

  /* allocate grid */
  CreateGrid();

  /* allocate psi_1, psi_2, and psi */
  CreatePsi();

  gobbler_idx = new PetscInt*[num_dims];
  for (PetscInt i = 0; i < num_dims; ++i)
  {
    gobbler_idx[i] = new PetscInt[2];
    if (coordinate_system_idx == 1 and i == 0)
    {
      gobbler_idx[i][0] = (num_x[i] - PetscInt(num_x[i] * p.GetGobbler())) - 1;
      gobbler_idx[i][1] = num_x[i] - 1 - gobbler_idx[i][0];
    }
    else
    {
      gobbler_idx[i][0] =
          (num_x[i] - PetscInt(num_x[i] * p.GetGobbler())) / 2 - 1;
      gobbler_idx[i][1] = num_x[i] - 1 - gobbler_idx[i][0];
    }
  }

  if (p.GetRestart() == 1)
  {
    LoadRestart(h5_file, viewer_file, p.GetWriteFrequencyCheckpoint(),
                p.GetWriteFrequencyObservables());
  }
  else
  {
    /* write out data */
    Checkpoint(h5_file, viewer_file, -1.0);
  }

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
                               "Wavefunction for the n electron system");
    /* close file */
    viewer_file.Close();

    /* size of each dim */
    h5_file.WriteObject(num_x, num_dims, "/Wavefunction/num_x",
                        "The number of physical dimension in the simulation");

    /* write each dims x values */
    for (PetscInt i = 0; i < num_dims; i++)
    {
      str = "x_value_" + std::to_string(i);
      h5_file.WriteObject(
          x_value[i], num_x[i], "/Wavefunction/" + str,
          "The coordinates of the " + std::to_string(i) + " dimension");
    }

    /* write psi_1 and psi_2 if still allocated */
    // if (psi_alloc_build)
    // {
    //   for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    //   {
    //     for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
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
    h5_file.WriteObject(Norm(), "/Wavefunction/norm", "Norm of wavefunction",
                        write_counter_checkpoint);

    std::vector< dcomp > projections;
    projections.resize(num_states, dcomp(0.0, 0.0));
    h5_file.WriteObject(
        &projections[0], projections.size(), "/Wavefunction/projections",
        "Projection onto the various excited states", write_counter_checkpoint);

    /* write observables */
    h5_file.CreateGroup("/Observables/");
    h5_file.WriteObject(time, "/Observables/time",
                        "Time step that the observables were written to disk",
                        write_counter_observables);
    h5_file.WriteObject(Norm(), "/Observables/norm", "Norm of wavefunction",
                        write_counter_observables);
    h5_file.WriteObject(
        GetGobbler(), "/Observables/gobbler",
        "Amount of wavefunction in absorbing boundary potential",
        write_counter_observables);
    for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
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
      h5_file.WriteObject(Norm(), "/Wavefunction/norm",
                          write_counter_checkpoint);
      std::vector< dcomp > projections = Projections(target_file_name);
      h5_file.WriteObject(&projections[0], projections.size(),
                          "/Wavefunction/projections",
                          write_counter_checkpoint);
      write_counter_checkpoint++;
    }
    else
    {
      h5_file.WriteObject(time, "/Observables/time", write_counter_observables);
      h5_file.WriteObject(Norm(), "/Observables/norm",
                          write_counter_observables);
      h5_file.WriteObject(GetGobbler(), "/Observables/gobbler",
                          write_counter_observables);
      for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
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

void Wavefunction::LoadRestart(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                               PetscInt write_frequency_checkpoint,
                               PetscInt write_frequency_observables)
{
  first_pass             = false;
  std::string group_name = "/Wavefunction/";
  /* Get write index for last checkpoint */
  write_counter_checkpoint = h5_file.GetTime("/Wavefunction/psi");
  /* Open file */
  viewer_file.Open("r");
  /* Set time idx */
  viewer_file.SetTime(write_counter_checkpoint);
  /* Push group */
  viewer_file.PushGroup(group_name);
  /* Read psi*/
  viewer_file.ReadObject(psi);
  /* Close file */
  viewer_file.Close();
  /* Calculate observable counters */
  write_counter_observables =
      ((write_counter_checkpoint - 1) * write_frequency_checkpoint) /
      write_frequency_observables;

  /* Increment both counters */
  write_counter_observables++;
  write_counter_checkpoint++;
}

/**
 * @brief Returns a vector of projections based on the states the given file
 *
 * @param file_name name of the file containing the eigen states
 * @return A vector of projections corresponding to that state
 */
std::vector< dcomp > Wavefunction::Projections(std::string file_name)
{
  /* Get write index for last checkpoint */
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);

  PetscInt file_states = h5_file.GetTime("/psi/") + 1;
  if (file_states < num_states)
  {
    EndRun("Not enough states in the target file");
  }
  std::vector< dcomp > ret_vec;
  dcomp projection_val;

  viewer_file.Open("r");
  for (int state_idx = 0; state_idx < num_states; ++state_idx)
  {
    /* Set time idx */
    viewer_file.SetTime(state_idx);

    if (coordinate_system_idx == 1)
    {
      /* Read psi*/
      viewer_file.ReadObject(psi_tmp_cyl);
      Normalize(psi_tmp_cyl, 0.0);
      CreateObservable(2, 0, 0);
      VecPointwiseMult(psi_tmp, psi_tmp, psi_tmp_cyl);
    }
    else
    {
      viewer_file.ReadObject(psi_tmp);
      Normalize(psi_tmp, 0.0);
    }
    VecDot(psi, psi_tmp, &projection_val);
    ret_vec.push_back(projection_val);
  }

  /* Close file */
  viewer_file.Close();
  return ret_vec;
}

/**
 * @brief Returns a vector of projections based on the states the given file
 *
 * @param file_name name of the file containing the eigen states
 * @return A vector of projections corresponding to that state
 */
void Wavefunction::ProjectOut(std::string file_name, HDF5Wrapper& h5_file_in,
                              ViewWrapper& viewer_file_in)
{
  /* Get write index for last checkpoint */
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);

  PetscInt file_states = h5_file.GetTime("/psi/") + 1;
  if (file_states < num_states)
  {
    EndRun("Not enough states in the target file");
  }
  std::vector< dcomp > ret_vec;
  dcomp projection_val;
  dcomp sum = 0.0;

  ierr = PetscObjectSetName((PetscObject)psi_tmp_cyl, "psi");
  ierr = PetscObjectSetName((PetscObject)psi_tmp, "psi");

  viewer_file.Open("r");
  for (int state_idx = 0; state_idx < num_states; ++state_idx)
  {
    /* Set time idx */
    viewer_file.SetTime(state_idx);
    viewer_file.ReadObject(psi_tmp_cyl);
    Normalize(psi_tmp_cyl, 0.0);
    if (coordinate_system_idx == 1)
    {
      CreateObservable(2, 0, 0);
      VecPointwiseMult(psi_tmp, psi_tmp, psi_tmp_cyl);
    }
    else
    {
      viewer_file.ReadObject(psi_tmp);
      Normalize(psi_tmp, 0.0);
    }
    VecDot(psi, psi_tmp, &projection_val);
    ret_vec.push_back(projection_val);
    if (world.rank() == 0)
      std::cout << std::norm(projection_val) << " " << projection_val << " "
                << -1.0 * ret_vec[state_idx] << "\n";
    sum += projection_val;
    viewer_file.SetTime(state_idx);
    viewer_file.ReadObject(psi_tmp_cyl);
    Normalize(psi_tmp_cyl, 0.0);
    VecAXPY(psi, -1.0 * ret_vec[state_idx], psi_tmp_cyl);
    Checkpoint(h5_file_in, viewer_file_in, -1 * state_idx);
  }
  if (world.rank() == 0) std::cout << sum << " sum\n";
  /* Close file */
  viewer_file.Close();
}

void Wavefunction::CheckpointPsi(ViewWrapper& viewer_file, PetscInt write_idx)
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
  PetscInt center;  /* idx of the 0.0 in the grid */
  double current_x; /* used for setting grid */

  /* allocation */
  num_x   = new PetscInt[num_dims];
  x_value = new double*[num_dims];

  /* initialize for loop */
  num_psi_build = 1.0;

  /* build grid */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
  {
    num_x[dim_idx] = ceil((dim_size[dim_idx]) / delta_x[dim_idx]);

    /* odd number so it is even on both sides */
    if (num_x[dim_idx] % 2 != 0) num_x[dim_idx]++;

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    if (coordinate_system_idx == 1 and dim_idx == 0)
    {
      for (int x_idx = 0; x_idx < num_x[dim_idx]; ++x_idx)
      {
        if (order > 2)
        {
          x_value[dim_idx][x_idx] = (x_idx + 1) * delta_x[dim_idx];
        }
        else
        {
          x_value[dim_idx][x_idx] =
              (x_idx)*delta_x[dim_idx] + delta_x[dim_idx] / 2.0;
        }
      }
    }
    else
    {
      /* find center of grid */
      center = num_x[dim_idx] / 2;

      /* store center */
      x_value[dim_idx][center] = 0.0;

      /* loop over all others */
      for (PetscInt x_idx = center; x_idx >= 0; x_idx--)
      {
        /* get x value */
        current_x =
            (x_idx - center) * delta_x[dim_idx] + delta_x[dim_idx] / 2.0;

        /* double checking index */
        if (x_idx < 0 || num_x[dim_idx] - x_idx - 1 >= num_x[dim_idx])
        {
          EndRun("Allocation error in grid");
        }

        /* set negative side */
        x_value[dim_idx][x_idx] = current_x;
        /* set positive side */
        x_value[dim_idx][num_x[dim_idx] - x_idx - 1] = -1 * current_x;
      }
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

    for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    {
      psi_build[elec_idx] = new dcomp*[num_dims];

      for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        psi_build[elec_idx][dim_idx] = new dcomp[num_x[dim_idx]];
      }
    }
    psi_alloc_build = true;
  }

  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      for (PetscInt i = 0; i < num_x[dim_idx]; i++)
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
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
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
    ierr = PetscObjectSetName((PetscObject)psi_tmp, "psi");

    VecCreate(PETSC_COMM_WORLD, &psi_tmp_cyl);
    VecSetSizes(psi_tmp_cyl, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_tmp_cyl);
    ierr = PetscObjectSetName((PetscObject)psi_tmp_cyl, "psi");

    psi_alloc = true;
  }

  VecGetOwnershipRange(psi, &low, &high);
  for (PetscInt idx = low; idx < high; idx++)
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

void Wavefunction::CreateObservable(PetscInt observable_idx, PetscInt elec_idx,
                                    PetscInt dim_idx)
{
  PetscInt low, high;
  PetscComplex val;

  /* Fill position vector*/
  VecGetOwnershipRange(psi_tmp, &low, &high);
  if (observable_idx == 0) /* Position operator */
  {
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = GetPositionVal(idx, elec_idx, dim_idx, false);
      VecSetValues(psi_tmp, 1, &idx, &val, INSERT_VALUES);
    }
  }
  else if (observable_idx == 1) /* Dipole acceleration */
  {
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = GetDipoleAccerationVal(idx, elec_idx, dim_idx);
      VecSetValues(psi_tmp, 1, &idx, &val, INSERT_VALUES);
    }
  }
  else if (observable_idx == 2) /* r */
  {
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = GetPositionVal(idx, 0, 0, true);
      VecSetValues(psi_tmp, 1, &idx, &val, INSERT_VALUES);
    }
  }
  else if (observable_idx == 3) /* ecs */
  {
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = GetGobblerVal(idx);
      VecSetValues(psi_tmp, 1, &idx, &val, INSERT_VALUES);
    }
  }
  else
  {
    EndRun("Bad observable index in Wavefunction");
  }
  /* Assemble position vector */
  VecAssemblyBegin(psi_tmp);
  VecAssemblyEnd(psi_tmp);
}

void Wavefunction::CleanUp()
{
  if (psi_alloc_build)
  {
    for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
    {
      for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
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
dcomp Wavefunction::GetPsiVal(dcomp*** data, PetscInt idx)
{
  /* Value to be returned */
  dcomp ret_val(1.0, 0.0);
  /* idx for return */
  std::vector< PetscInt > idx_array = GetIntArray(idx);
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      ret_val *=
          data[elec_idx][dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
    }
  }
  return ret_val;
}

/* returns values for global position vector */
dcomp Wavefunction::GetPositionVal(PetscInt idx, PetscInt elec_idx,
                                   PetscInt dim_idx, bool integrate)
{
  /* Value to be returned */
  dcomp ret_val(0.0, 0.0);
  /* idx for return */
  std::vector< PetscInt > idx_array = GetIntArray(idx);
  ret_val = x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
  if (integrate and order > 2 and dim_idx == 0 and coordinate_system_idx == 1)
  {
    // std::cout << "using integrate\n";
    /* see appendix A of https://arxiv.org/pdf/1604.00947.pdf using Lagrange
     * interpolation polynomials and
     * http://slideflix.net/doc/4183369/gregory-s-quadrature-method*/
    // if (idx_array[elec_idx * num_dims + dim_idx] == 0) ret_val *= 13.0 /
    // 12.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 7.0 / 6.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 23.0 / 24.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 299.0 / 240.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 211.0 / 240.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
    //   ret_val *= 739.0 / 720.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 317.0 / 240.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 23.0 / 30.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
    //   ret_val *= 793.0 / 720.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 3)
    //   ret_val *= 147.0 / 160.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 84199.0 / 60480.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 18869.0 / 30240.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
    //   ret_val *= 37621.0 / 30240.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 3)
    //   ret_val *= 55031.0 / 60480.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 4)
    //   ret_val *= 61343.0 / 60480.0;

    if (idx_array[elec_idx * num_dims + dim_idx] == 0)
      ret_val *= 22081.0 / 15120.0;
    else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
      ret_val *= 54851.0 / 120960.0;
    else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
      ret_val *= 103.0 / 70.0;
    else if (idx_array[elec_idx * num_dims + dim_idx] == 3)
      ret_val *= 89437.0 / 120960.0;
    else if (idx_array[elec_idx * num_dims + dim_idx] == 4)
      ret_val *= 16367.0 / 15120.0;
    else if (idx_array[elec_idx * num_dims + dim_idx] == 5)
      ret_val *= 23917.0 / 24192.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 5537111.0 / 3628800.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 103613.0 / 403200.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
    //   ret_val *= 261115.0 / 145152.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 3)
    //   ret_val *= 298951.0 / 725760.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 4)
    //   ret_val *= 515677.0 / 403200.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 5)
    //   ret_val *= 3349879.0 / 3628800.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 6)
    //   ret_val *= 3662753.0 / 3628800.0;

    // if (idx_array[elec_idx * num_dims + dim_idx] == 0)
    //   ret_val *= 1153247.0 / 725760.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 1)
    //   ret_val *= 130583.0 / 3628800.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 2)
    //   ret_val *= 903527.0 / 403200.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 3)
    //   ret_val *= -797.0 / 5670.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 4)
    //   ret_val *= 6244961.0 / 3628800.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 5)
    //   ret_val *= 56621.0 / 80640.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 6)
    //   ret_val *= 3891877.0 / 3628800.0;
    // else if (idx_array[elec_idx * num_dims + dim_idx] == 7)
    //   ret_val *= 1028617.0 / 1036800.0;
  }
  return ret_val;
}

dcomp Wavefunction::GetGobblerVal(PetscInt idx)
{
  /* Value to be returned */
  dcomp ret_val(0.0, 0.0);
  /* idx for return */
  std::vector< PetscInt > idx_array = GetIntArray(idx);
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      PetscInt current_idx = idx_array[elec_idx * num_dims + dim_idx];
      if (current_idx >= gobbler_idx[dim_idx][1] or
          (dim_idx != 0 and coordinate_system_idx != 1 and
           current_idx <= gobbler_idx[dim_idx][0]))
      {
        ret_val = dcomp(1.0, 0.0);
      }
    }
  }
  return ret_val;
}

/* returns values for global dipole acceleration */
dcomp Wavefunction::GetDipoleAccerationVal(PetscInt idx, PetscInt elec_idx,
                                           PetscInt dim_idx)
{
  /* Value to be returned */
  dcomp ret_val(0.0, 0.0);
  double r;
  /* idx for return */
  std::vector< PetscInt > idx_array = GetIntArray(idx);
  r                                 = GetDistance(idx_array, elec_idx);
  ret_val +=
      x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]] / (r * r * r);
  return ret_val;
}

/* Returns r component of that electron */
double Wavefunction::GetDistance(std::vector< PetscInt > idx_array,
                                 PetscInt elec_idx)
{
  double r = 0.0;
  double x;
  for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
  {
    x = x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
    r += x * x;
  }
  return sqrt(r);
}

std::vector< PetscInt > Wavefunction::GetIntArray(PetscInt idx)
{
  /* Total number of dims for total system*/
  PetscInt total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector< PetscInt > num(total_dims);
  /* idx for return */
  std::vector< PetscInt > idx_array(total_dims);
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      num[elec_idx * num_dims + dim_idx] = num_x[dim_idx];
    }
  }
  for (PetscInt i = total_dims - 1; i >= 0; --i)
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
  dcomp dot_product;
  double total = 0;
  if (coordinate_system_idx == 1)
  {
    CreateObservable(2, 0, 0);
    VecPointwiseMult(psi_tmp, psi_tmp, data);
    VecDot(data, psi_tmp, &dot_product);
    total = sqrt(dot_product.real());
  }
  else
  {
    VecNorm(data, NORM_2, &total);
  }
  return total;
}

double Wavefunction::GetEnergy(Mat* h) { return GetEnergy(h, psi); }

double Wavefunction::GetEnergy(Mat* h, Vec& p)
{
  PetscComplex energy;
  if (coordinate_system_idx == 1)
  {
    MatMult(*h, p, psi_tmp_cyl);
    CreateObservable(2, 0, 0); /* pho */
    VecPointwiseMult(psi_tmp_cyl, psi_tmp, psi_tmp_cyl);
    VecDot(p, psi_tmp_cyl, &energy);
  }
  else
  {
    MatMult(*h, p, psi_tmp);
    VecDot(p, psi_tmp, &energy);
  }
  return energy.real();
}

/* returns position expectation value <psi|x_{elec_idx,dim_idx}|psi> */
double Wavefunction::GetPosition(PetscInt elec_idx, PetscInt dim_idx)
{
  PetscComplex expectation;
  if (coordinate_system_idx == 1)
  {
    CreateObservable(2, elec_idx, dim_idx); /* pho */
    VecPointwiseMult(psi_tmp_cyl, psi_tmp, psi);
    CreateObservable(0, elec_idx, dim_idx); /* dim */
    VecPointwiseMult(psi_tmp, psi_tmp, psi_tmp_cyl);
  }
  else
  {
    CreateObservable(0, elec_idx, dim_idx); /* dim */
    VecPointwiseMult(psi_tmp, psi_tmp, psi);
  }
  VecDot(psi, psi_tmp, &expectation);
  return expectation.real();
}

/* returns dipole acceleration value <psi|x_{elec_idx,dim_idx}/r^3|psi> */
double Wavefunction::GetDipoleAcceration(PetscInt elec_idx, PetscInt dim_idx)
{
  PetscComplex expectation;
  if (coordinate_system_idx == 1)
  {
    CreateObservable(2, elec_idx, dim_idx); /* pho */
    VecPointwiseMult(psi_tmp_cyl, psi_tmp, psi);
    CreateObservable(1, elec_idx, dim_idx); /* dim */
    VecPointwiseMult(psi_tmp, psi_tmp, psi_tmp_cyl);
  }
  else
  {
    CreateObservable(1, elec_idx, dim_idx); /* dim */
    VecPointwiseMult(psi_tmp, psi_tmp, psi);
  }
  VecDot(psi, psi_tmp, &expectation);
  return expectation.real();
}

double Wavefunction::GetGobbler()
{
  PetscComplex expectation;
  if (coordinate_system_idx == 1)
  {
    CreateObservable(2, 0, 0); /* pho */
    VecPointwiseMult(psi_tmp_cyl, psi_tmp, psi);
    CreateObservable(3, 0, 0); /* gobbler */
    VecPointwiseMult(psi_tmp, psi_tmp, psi_tmp_cyl);
  }
  else
  {
    CreateObservable(3, 0, 0); /* gobbler */
    VecPointwiseMult(psi_tmp, psi_tmp, psi);
  }
  VecDot(psi, psi_tmp, &expectation);
  return expectation.real();
}

void Wavefunction::ResetPsi()
{
  CreatePsi();
  CleanUp();
}

PetscInt* Wavefunction::GetNumX() { return num_x; }

PetscInt Wavefunction::GetNumPsi() { return num_psi; }

PetscInt Wavefunction::GetNumPsiBuild() { return num_psi_build; }

Vec* Wavefunction::GetPsi() { return &psi; }

double** Wavefunction::GetXValue() { return x_value; }

PetscInt** Wavefunction::GetGobblerIdx() { return gobbler_idx; }

PetscInt Wavefunction::GetWrieCounterCheckpoint()
{
  return write_counter_checkpoint;
}

/* destructor */
Wavefunction::~Wavefunction()
{
  if (world.rank() == 0) std::cout << "Deleting Wavefunction\n";
  /* do not delete dim_size or delta_x since they belong to the Parameter
   * class and will be freed there*/
  delete num_x;
  for (PetscInt i = 0; i < num_dims; i++)
  {
    delete x_value[i];
  }
  delete[] x_value;
  CleanUp();
  VecDestroy(&psi);
  VecDestroy(&psi_tmp);
}
