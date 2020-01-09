#include "Wavefunction.h"

Wavefunction::Wavefunction(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                           Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Wavefunction\n";
  double ecs = ((1 - p.GetGobbler()) / 2.0) + p.GetGobbler();

  /* initialize values */
  position_mat_alloc        = false;
  psi_alloc_build           = false;
  psi_alloc                 = false;
  first_pass                = true;
  num_dims                  = p.GetNumDims();
  num_electrons             = p.GetNumElectrons();
  dim_size                  = p.dim_size.get();
  delta_x_min               = p.delta_x_min.get();
  delta_x_min_end           = p.delta_x_min_end.get();
  delta_x_max               = p.delta_x_max.get();
  delta_x_max_start         = p.delta_x_max_start.get();
  delta_t                   = p.GetDeltaT();
  coordinate_system_idx     = p.GetCoordinateSystemIdx();
  target_file_name          = p.GetTarget() + ".h5";
  num_states                = p.GetNumStates();
  sigma                     = p.GetSigma();
  num_psi_build             = 1.0;
  write_counter_checkpoint  = 0;
  write_counter_observables = 0;
  write_counter_projections = 0;
  order                     = p.GetOrder();
  max_block_size            = 0;

  /* SAE stuff */
  z                      = p.z.get();
  location               = p.GetLocation();
  exponential_r_0        = p.GetExponentialR0();
  exponential_amplitude  = p.GetExponentialAmplitude();
  exponential_decay_rate = p.GetExponentialDecayRate();
  exponential_size       = p.exponential_size.get();
  gaussian_r_0           = p.GetGaussianR0();
  gaussian_amplitude     = p.GetGaussianAmplitude();
  gaussian_decay_rate    = p.GetGaussianDecayRate();
  gaussian_size          = p.gaussian_size.get();
  square_well_r_0        = p.GetSquareWellR0();
  square_well_amplitude  = p.GetSquareWellAmplitude();
  square_well_width      = p.GetSquareWellWidth();
  square_well_size       = p.square_well_size.get();
  yukawa_r_0             = p.GetYukawaR0();
  yukawa_amplitude       = p.GetYukawaAmplitude();
  yukawa_decay_rate      = p.GetYukawaDecayRate();
  yukawa_size            = p.yukawa_size.get();
  alpha                  = p.GetAlpha();
  alpha_2                = alpha * alpha;
  ee_soft_core           = p.GetEESoftCore();
  ee_soft_core_2         = ee_soft_core * ee_soft_core;
  num_nuclei             = p.GetNumNuclei();

  if (coordinate_system_idx == 3)
  {
    l_max = p.GetLMax();
    m_max = p.GetMMax();
    k_max = 0;
  }
  else if (coordinate_system_idx == 4)
  {
    l_max = p.GetLMax();
    m_max = p.GetMMax();
    k_max = p.GetKMax();
  }
  else
  {
    l_max = 0;
    m_max = 0;
    k_max = 0;
  }

  PetscLogEventRegister("WaveNorm", PETSC_VIEWER_CLASSID, &time_norm);
  PetscLogEventRegister("WaveEner", PETSC_VIEWER_CLASSID, &time_energy);
  PetscLogEventRegister("WavePos", PETSC_VIEWER_CLASSID, &time_position);
  PetscLogEventRegister("WaveDA", PETSC_VIEWER_CLASSID,
                        &time_dipole_acceration);
  PetscLogEventRegister("WaveECS", PETSC_VIEWER_CLASSID, &time_gobbler);
  PetscLogEventRegister("WaveProj", PETSC_VIEWER_CLASSID, &time_projections);
  PetscLogEventRegister("WaveRadialPsi", PETSC_VIEWER_CLASSID,
                        &time_insert_radial_psi);

  /* allocate grid */
  CreateGrid();

  gobbler_idx = new PetscInt*[num_dims];
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    gobbler_idx[dim_idx]    = new PetscInt[2];
    gobbler_idx[dim_idx][0] = -1;
    gobbler_idx[dim_idx][1] = num_x[dim_idx];

    if (ecs < 1.0)
    {
      if (coordinate_system_idx == 1 and dim_idx == 0)
      {
        if (dim_size[dim_idx] * ecs < delta_x_max_start[dim_idx])
        {
          EndRun("ECS (gobbler) starts inside delta_x_max_start");
        }
        if (delta_x_min_end[dim_idx] / delta_x_min[dim_idx] < p.GetOrder() + 1)
        {
          EndRun(
              "delta_x_min_end must be more than $order + 1$ grid points from "
              "the origin in the radial component");
        }
      }
      else
      {
        if (dim_size[dim_idx] / 2.0 * ecs < delta_x_max_start[dim_idx])
        {
          EndRun("ECS (gobbler) starts inside delta_x_max_start");
        }
      }
      if (not((coordinate_system_idx == 3 or coordinate_system_idx == 4) and
              dim_idx != 2))
      {
        gobbler_idx[dim_idx][0] = -1;
        for (int i = 0; i < num_x[dim_idx]; ++i)
        {
          if (x_value[dim_idx][i] < dim_size[dim_idx] * ecs)
          {
            gobbler_idx[dim_idx][1] = i;
          }
        }
        if ((coordinate_system_idx == 3 or coordinate_system_idx == 4) and
            num_x[dim_idx] - gobbler_idx[dim_idx][1] < order)
        {
          EndRun(
              "The Gobbler is not big enough to fit the full "
              "stencil_size.\nEither decrease the real part of the grid or set "
              "gobbler=1.0");
        }
      }
    }
    else
    {
      if (coordinate_system_idx == 1 and dim_idx == 0)
      {
        if (delta_x_min_end[dim_idx] / delta_x_min[dim_idx] < p.GetOrder() + 1)
        {
          EndRun(
              "delta_x_min_end must be more than $order + 1$ grid points from "
              "the origin in the radial component");
        }
      }
    }
  }

  /* allocate psi_1, psi_2, and psi */
  CreatePsi();

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
}

void Wavefunction::Checkpoint(HDF5Wrapper& h5_file, ViewWrapper& viewer_file,
                              double time, PetscInt checkpoint_psi)
{
  clock_t checkpoint_time = clock();
  if (world.rank() == 0)
  {
    if (checkpoint_psi == 0)
    {
      std::cout << "Checkpointing Psi: " << write_counter_checkpoint << "\n"
                << std::flush;
    }
    else if (checkpoint_psi == 2)
    {
      std::cout << "Checkpointing Psi Projections: "
                << write_counter_projections << "\n"
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
    group_name = "/Wavefunction/";
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
    viewer_file.PopGroup(); /* close file */

    group_name = "/Projections/";
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
    viewer_file.PopGroup(); /* close file */
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

    if (coordinate_system_idx == 3)
    {
      h5_file.WriteObject(
          l_values, num_x[1], "/Wavefunction/l_values",
          "l values for each point in dimension 1. This combines both the l "
          "and m values to avoid issues with tensor grids");
      h5_file.WriteObject(
          m_values, num_x[1], "/Wavefunction/m_values",
          "m values for each point in dimension 1. This combines both the l "
          "and m values to avoid issues with tensor grids");
    }
    if (coordinate_system_idx == 4)
    {
      int* cur_vals;
      for (int i = 0; i < num_x[1]; ++i)
      {
        cur_vals = eigen_values[i];
        h5_file.WriteObject(
            &cur_vals[0], 6, "/Wavefunction/eigen_values",
            "spherical harmonic eigen values k_val, n, l_1, l_2, L, M", i);
      }
      h5_file.WriteObject(l_block_size, k_max + 1, "/Wavefunction/l_block_size",
                          "number of elements in blocks with L eigen value");
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
    PetscInt projection_size = GetProjectionSize();
    projections.resize(projection_size, dcomp(0.0, 0.0));
    h5_file.WriteObject(
        &projections[0], projections.size(), "/Wavefunction/projections",
        "Projection onto the various excited states", write_counter_checkpoint);

    h5_file.WriteObject(time, "/Projections/time",
                        "Time step that psi was written to disk",
                        write_counter_checkpoint);

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
    if (world.rank() == 0)
      std::cout << "Checkpoint time: "
                << ((float)clock() - checkpoint_time) / (CLOCKS_PER_SEC) << "\n"
                << std::flush;
    write_counter_checkpoint++;
    write_counter_observables++;
    write_counter_projections++;
  }
  else
  {
    if (checkpoint_psi == 0)
    {
      group_name = "/Wavefunction/";
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
      std::vector< dcomp > projections;
      if (time < 1e-14)
      {
        PetscInt projection_size = GetProjectionSize();
        projections.resize(projection_size, dcomp(0.0, 0.0));
      }
      else
      {
        projections = Projections(target_file_name);
      }
      h5_file.WriteObject(&projections[0], projections.size(),
                          "/Wavefunction/projections",
                          write_counter_checkpoint);
      write_counter_checkpoint++;
      if (world.rank() == 0)
        std::cout << "Checkpoint time: "
                  << ((float)clock() - checkpoint_time) / (CLOCKS_PER_SEC)
                  << "\n"
                  << std::flush;
    }
    else if (checkpoint_psi == 1) /* Observables */
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
    else if (checkpoint_psi == 2) /* Projections */
    {
      group_name = "/Projections/";
      viewer_file.Open("a");
      /* set time step */
      viewer_file.SetTime(write_counter_projections);
      /* move into group */
      viewer_file.PushGroup(group_name);
      /* write vector */
      viewer_file.WriteObject((PetscObject)psi);
      /* close file */
      viewer_file.Close();

      /* write time */
      h5_file.WriteObject(time, "/Projections/time", write_counter_projections);
      write_counter_projections++;
    }
    else
    {
      EndRun("Bad index in Wavefunction's Checkpoint function");
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
  write_counter_checkpoint  = h5_file.GetTimeIdx("/Wavefunction/psi");
  write_counter_projections = h5_file.GetTimeIdx("/Projections/psi");
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

  /* Calculate observable counter */
  write_counter_observables =
      std::round(h5_file.GetLast("/Wavefunction/time") / delta_t);
  write_counter_observables /= write_frequency_observables;

  /* Increment both counters */
  write_counter_observables++;
  write_counter_checkpoint++;
  write_counter_projections++;
}

/**
 * @brief Returns a vector of projections based on the states the given file
 *
 * @param file_name name of the file containing the eigen states
 * @return A vector of projections corresponding to that state
 */
std::vector< dcomp > Wavefunction::Projections(std::string file_name)
{
  PetscLogEventBegin(time_projections, 0, 0, 0, 0);
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);
  std::vector< dcomp > ret_vec;

  if (coordinate_system_idx == 3)
  {
    Vec psi_small_local;
    VecCreateSeq(PETSC_COMM_SELF, num_x[2], &psi_small_local);
    VecSetFromOptions(psi_small_local);
    ierr = PetscObjectSetName((PetscObject)psi_small_local, "psi");

    PetscInt file_states = h5_file.GetTimeIdx("/psi_l_0/psi/") + 1;
    if (file_states < num_states)
    {
      EndRun("Not enough states in the target file");
    }
    dcomp projection_val;

    viewer_file.Open("r");
    for (int n_index = 1; n_index <= num_states; ++n_index)
    {
      for (int l_idx = 0; l_idx < fmin(n_index, l_max + 1); ++l_idx)
      {
        /* Set time idx */
        viewer_file.SetTime(n_index - l_idx - 1);
        viewer_file.PushGroup("psi_l_" + std::to_string(l_idx));
        for (int m_idx = -1. * std::min(m_max, l_idx);
             m_idx < std::min(m_max, l_idx) + 1; ++m_idx)
        {
          viewer_file.ReadObject(psi_small_local);
          InsertRadialPsi(psi_small_local, psi_proj, l_idx, m_idx);
          Normalize(psi_proj, 0.0);
          VecPointwiseMult(psi_tmp, jacobian, psi_proj);
          VecDot(psi, psi_tmp, &projection_val);
          ret_vec.push_back(projection_val);
        }
        viewer_file.PopGroup();
      }
    }
    /* Close file */
    viewer_file.Close();

    VecDestroy(&psi_small_local);
  }
  else
  {
    PetscInt file_states = h5_file.GetTimeIdx("/psi/") + 1;
    if (file_states < num_states)
    {
      EndRun("Not enough states in the target file");
    }
    dcomp projection_val;

    viewer_file.Open("r");
    for (int state_idx = 0; state_idx < num_states; ++state_idx)
    {
      /* Set time idx */
      viewer_file.SetTime(state_idx);
      viewer_file.ReadObject(psi_proj);
      Normalize(psi_proj, 0.0);
      VecPointwiseMult(psi_tmp, jacobian, psi_proj);
      VecDot(psi, psi_tmp, &projection_val);
      ret_vec.push_back(projection_val);
    }
    /* Close file */
    viewer_file.Close();
  }

  PetscLogEventEnd(time_projections, 0, 0, 0, 0);
  return ret_vec;
}

/**
 * @brief Returns a vector of projections based on the states the given file
 *
 * @param file_name name of the file containing the eigen states
 * @return A vector of projections corresponding to that state
 */
void Wavefunction::ProjectOut(std::string file_name, HDF5Wrapper& h5_file_in,
                              ViewWrapper& viewer_file_in, double time)
{
  /* Get write index for last checkpoint */
  std::vector< dcomp > ret_vec = Projections(file_name);
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);

  if (coordinate_system_idx == 3)
  {
    Vec psi_small_local;
    PetscInt state_idx = 0;
    VecCreateSeq(PETSC_COMM_SELF, num_x[2], &psi_small_local);
    VecSetFromOptions(psi_small_local);
    ierr = PetscObjectSetName((PetscObject)psi_small_local, "psi");

    /* Save psi from projection death */
    VecCopy(psi, psi_tmp_cyl);
    viewer_file.Open("r");
    for (int n_index = 1; n_index <= num_states; ++n_index)
    {
      for (int l_idx = 0; l_idx < fmin(n_index, l_max + 1); ++l_idx)
      {
        /* Set time idx */
        viewer_file.SetTime(n_index - l_idx - 1);
        viewer_file.PushGroup("psi_l_" + std::to_string(l_idx));
        viewer_file.ReadObject(psi_small_local);
        viewer_file.PopGroup();
        for (int m_idx = -1. * std::min(m_max, l_idx);
             m_idx < std::min(m_max, l_idx) + 1; ++m_idx)
        {
          InsertRadialPsi(psi_small_local, psi_proj, l_idx, m_idx);
          Normalize(psi_proj, 0.0);
          VecAXPY(psi_tmp_cyl, -1.0 * ret_vec[state_idx], psi_proj);
          state_idx++;
        }
      }
    }
    /* Save psi from projection death */
    VecCopy(psi, psi_proj);
    VecCopy(psi_tmp_cyl, psi);
    Checkpoint(h5_file_in, viewer_file_in, time, 2);
    VecCopy(psi_proj, psi);

    /* Close file */
    viewer_file.Close();

    VecDestroy(&psi_small_local);
  }
  else
  {
    /* Save psi from projection death */
    VecCopy(psi, psi_tmp_cyl);
    viewer_file.Open("r");
    for (int state_idx = 0; state_idx < num_states; ++state_idx)
    {
      viewer_file.SetTime(state_idx);
      viewer_file.ReadObject(psi_proj);
      Normalize(psi_proj, 0.0);
      VecAXPY(psi_tmp_cyl, -1.0 * ret_vec[state_idx], psi_proj);
    }

    /* Save psi from projection death */
    VecCopy(psi, psi_proj);
    VecCopy(psi_tmp_cyl, psi);
    Checkpoint(h5_file_in, viewer_file_in, time, 2);
    VecCopy(psi_proj, psi);

    /* Close file */
    viewer_file.Close();
  }
}

/**
 * @brief Uses the "target".h5 file to read in the ground state
 * @details Uses the "target".h5 file to read in the ground state. It also
 * makes sure you have enough states for the projections
 *
 * @param num_states The number of states you wish to use for projections
 * @param return_state_idx the index of the state you want the Wavefunction
 * class to be set to upon return
 */
void Wavefunction::LoadPsi(std::string file_name, PetscInt num_states,
                           PetscInt num_start_state, PetscInt* start_state_idx,
                           double* start_state_amplitude,
                           double* start_state_phase)
{
  if (world.rank() == 0)
    std::cout << "Loading wavefunction from " << file_name << "\n";
  /* Get write index for last checkpoint */
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);

  PetscInt file_states = h5_file.GetTimeIdx("/psi/") + 1;
  if (file_states < num_states)
  {
    EndRun("Not enough states in the target file");
  }

  for (int idx = 0; idx < num_start_state; ++idx)
  {
    if (start_state_idx[idx] >= num_states)
    {
      EndRun("The start state must be less than the total number of states");
    }
  }

  /* Open File */
  viewer_file.Open("r");
  Vec psi_super_pos;
  VecDuplicate(psi, &psi_super_pos);
  ierr = PetscObjectSetName((PetscObject)psi_super_pos, "psi");

  /* Read the first psi */
  viewer_file.SetTime(start_state_idx[0]);
  viewer_file.ReadObject(psi_super_pos);
  Normalize(psi_super_pos, 0.0);
  VecScale(psi_super_pos, dcomp(start_state_amplitude[0], 0.0) *
                              std::exp(dcomp(0.0, start_state_phase[0])));
  VecCopy(psi_super_pos, psi);

  /* add on all other psi in super position*/
  for (int idx = 1; idx < num_start_state; ++idx)
  {
    viewer_file.SetTime(start_state_idx[idx]);
    viewer_file.ReadObject(psi_super_pos);
    Normalize(psi_super_pos, 0.0);
    VecAXPY(psi,
            dcomp(start_state_amplitude[idx], 0.0) *
                std::exp(dcomp(0.0, start_state_phase[idx])),
            psi_super_pos);
  }

  VecDestroy(&psi_super_pos);

  /* Normalize */
  Normalize(psi, 0.0);

  /* Close file */
  viewer_file.Close();
}

void Wavefunction::LoadPsi(std::string file_name, PetscInt num_states,
                           PetscInt num_start_state, PetscInt* start_state_idx,
                           PetscInt* start_state_l_idx,
                           PetscInt* start_state_m_idx,
                           double* start_state_amplitude,
                           double* start_state_phase)
{
  if (world.rank() == 0)
    std::cout << "Loading wavefunction from " << file_name << "\n";
  /* Get write index for last checkpoint */
  HDF5Wrapper h5_file(file_name);
  ViewWrapper viewer_file(file_name);

  PetscInt file_states = h5_file.GetTimeIdx("/psi_l_0/psi/") + 1;
  if (file_states < num_states)
  {
    EndRun("Not enough states in the target file");
  }

  for (int idx = 0; idx < num_start_state; ++idx)
  {
    if (start_state_idx[idx] - 1 >= num_states)
    {
      EndRun("The start state must be less than the total number of states");
    }
  }

  /* Open File */
  viewer_file.Open("r");

  /* Produce wavefunctions */
  Vec psi_super_pos;
  Vec psi_small_local;
  VecDuplicate(psi, &psi_super_pos);
  ierr = PetscObjectSetName((PetscObject)psi_super_pos, "psi");

  VecCreateSeq(PETSC_COMM_SELF, num_x[2], &psi_small_local);
  VecSetFromOptions(psi_small_local);
  ierr = PetscObjectSetName((PetscObject)psi_small_local, "psi");

  /* Read the first psi */
  viewer_file.SetTime(start_state_idx[0] - start_state_l_idx[0] - 1);
  viewer_file.PushGroup("psi_l_" + std::to_string(start_state_l_idx[0]));
  viewer_file.ReadObject(psi_small_local);
  viewer_file.PopGroup();
  InsertRadialPsi(psi_small_local, psi_super_pos, start_state_l_idx[0],
                  start_state_m_idx[0]);
  Normalize(psi_super_pos, 0.0);
  VecScale(psi_super_pos, dcomp(start_state_amplitude[0], 0.0) *
                              std::exp(dcomp(0.0, start_state_phase[0])));
  VecCopy(psi_super_pos, psi);

  /* add on all other psi in super position*/
  for (int idx = 1; idx < num_start_state; ++idx)
  {
    viewer_file.SetTime(start_state_idx[idx] - start_state_l_idx[idx] - 1);
    viewer_file.PushGroup("psi_l_" + std::to_string(start_state_l_idx[idx]));
    viewer_file.ReadObject(psi_small_local);
    viewer_file.PopGroup();
    InsertRadialPsi(psi_small_local, psi_super_pos, start_state_l_idx[idx],
                    start_state_m_idx[idx]);
    Normalize(psi_super_pos, 0.0);
    VecAXPY(psi,
            dcomp(start_state_amplitude[idx], 0.0) *
                std::exp(dcomp(0.0, start_state_phase[idx])),
            psi_super_pos);
  }

  VecDestroy(&psi_super_pos);
  VecDestroy(&psi_small_local);

  /* Normalize */
  Normalize(psi, 0.0);

  /* Close file */
  viewer_file.Close();
}

void Wavefunction::InsertRadialPsi(Vec& psi_radial, Vec& psi_total,
                                   PetscInt l_val, PetscInt m_val)
{
  PetscLogEventBegin(time_insert_radial_psi, 0, 0, 0, 0);
  PetscInt low, high;
  PetscComplex val;

  VecGetOwnershipRange(psi_total, &low, &high);
  for (PetscInt idx = low; idx < high; idx++)
  {
    std::vector< PetscInt > idx_array = GetIntArray(idx);
    if (l_values[idx_array[1]] == l_val and m_values[idx_array[1]] == m_val)
    {
      VecGetValues(psi_radial, 1, &idx_array[2], &val);
    }
    else
    {
      val = dcomp(0.0, 0.0);
    }
    VecSetValues(psi_total, 1, &idx, &val, INSERT_VALUES);
  }
  VecAssemblyBegin(psi_total);
  VecAssemblyEnd(psi_total);
  PetscLogEventEnd(time_insert_radial_psi, 0, 0, 0, 0);
}

void Wavefunction::CheckpointPsi(ViewWrapper& viewer_file, PetscInt write_idx)
{
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

void Wavefunction::CheckpointPsiSmall(ViewWrapper& viewer_file,
                                      PetscInt write_idx, PetscInt l_val)
{
  /* create group name */
  std::string group_name = "/psi_l_" + std::to_string(l_val) + "/";
  if (world.rank() == 0)
    std::cout << "Checkpointing Wavefunction in " << viewer_file.file_name
              << ": "
              << " " << write_idx << "\n";

  /* open file */
  viewer_file.Open("a");
  /* push group */
  viewer_file.PushGroup(group_name);
  /* set time step */
  viewer_file.SetTime(write_idx);
  /* write vector */
  viewer_file.WriteObject((PetscObject)psi_small);
  /* pop group */
  viewer_file.PopGroup();
  /* close file */
  viewer_file.Close();
}

void Wavefunction::CreateGrid()
{
  /* allocation */
  num_x   = new PetscInt[num_dims];
  x_value = new double*[num_dims];

  /* RBF */
  if (coordinate_system_idx == 2)
  {
    /* load node set hdf5 file*/
    HDF5Wrapper node_set("nodes.h5", "r");

    /* Find out how many nodes we need*/
    num_psi = node_set.GetLast("/parameters/num_nodes");

    /* allocate data */
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      x_value[dim_idx] = new double[num_psi];
      num_x[dim_idx]   = num_psi;
    }

    /* load x data */
    double* nodes_x = node_set.GetFirstN("/node_set/x", num_psi);
    for (int idx = 0; idx < num_psi; ++idx)
    {
      x_value[0][idx] = nodes_x[idx];
    }
    delete nodes_x;

    /* load y data */
    double* nodes_y = node_set.GetFirstN("/node_set/y", num_psi);
    for (int idx = 0; idx < num_psi; ++idx)
    {
      x_value[1][idx] = nodes_y[idx];
    }
    delete nodes_y;

    /* load z data */
    double* nodes_z = node_set.GetFirstN("/node_set/z", num_psi);
    for (int idx = 0; idx < num_psi; ++idx)
    {
      x_value[2][idx] = nodes_z[idx];
    }
    delete nodes_z;
  }
  else if (coordinate_system_idx == 3) /* spherical code*/
  {
    double slope;
    double x_total;
    double max_x;
    double amplitude;
    double w;
    double s1;
    PetscInt count;
    PetscInt dim_idx;
    PetscInt index;

    /* initialize for loop */
    num_psi_build = 1.0;

    /* build grid */
    /********************************************************************/
    /********************************************************************/
    /*                                                                  */
    /* This dimension is used to tell the code that we are working in 3 */
    /* spacial dimensions. It is set to size one to avoid unneeded      */
    /* computation                                                      */
    /*                                                                  */
    /********************************************************************/
    /********************************************************************/
    dim_idx = 0;

    /* m goes from -m_max to m_max giving 2*m_max+1 terms */
    num_x[dim_idx] = 1;

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    x_value[dim_idx][0] = 0;

    /**************************************************************************/
    /**************************************************************************/
    /*                                                                        */
    /* l and m values */
    /* Since l and m cannot be written as a tensor product we combine them */
    /* into one dimension. This avoids having |m| > l which would happen with
     */
    /* a tensor like grid */
    /*                                                                        */
    /**************************************************************************/
    /**************************************************************************/
    dim_idx = 1;

    /* combines both m and l in one dimension  */
    num_x[dim_idx] = GetIdxFromLM(l_max, m_max, m_max) + 1;

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];
    /* these vectors are small, so it is easier (and possible faster) to store
     * them rather than convert and index to l and m values */
    l_values = new PetscInt[num_x[dim_idx]];
    m_values = new PetscInt[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    for (int l_val = 0; l_val < l_max + 1; ++l_val)
    {
      for (int m_val = -1. * std::min(m_max, l_val);
           m_val < std::min(m_max, l_val) + 1; ++m_val)
      {
        index                   = GetIdxFromLM(l_val, m_val, m_max);
        l_values[index]         = l_val;
        m_values[index]         = m_val;
        x_value[dim_idx][index] = index;
      }
    }

    /****************************************************************/
    /****************************************************************/
    /*                                                              */
    /* handle radial direction                                      */
    /* This is a tensor grid with respect to the l and m dimension  */
    /*                                                              */
    /****************************************************************/
    /****************************************************************/
    dim_idx = 2;

    /* get change in dx */
    amplitude = delta_x_max[dim_idx] - delta_x_min[dim_idx];
    w = pi / (2.0 * (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]));

    /* start at dx */
    x_total = delta_x_min[dim_idx];
    count   = 0;
    max_x   = dim_size[dim_idx];

    while (x_total < max_x)
    {
      if (x_total < delta_x_min_end[dim_idx])
      {
        x_total += delta_x_min[dim_idx];
      }
      else if (x_total < delta_x_max_start[dim_idx])
      {
        s1 = std::sin(w * (x_total - delta_x_min_end[dim_idx]));
        x_total += amplitude * s1 * s1 + delta_x_min[dim_idx];
      }
      else
      {
        x_total += delta_x_max[dim_idx];
      }
      count++;
    }
    num_x[dim_idx] = count;

    if (num_x[dim_idx] % 2 != 0)
    {
      num_x[dim_idx]++;
    }
    if (num_x[dim_idx] <= order + 1)
    {
      EndRun(
          "Not enough gird points to support this order of Finite "
          "Difference. "
          "Please increase grid size.");
    }

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    slope = (delta_x_max[dim_idx] - delta_x_min[dim_idx]) /
            (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]);

    x_value[dim_idx][0] = delta_x_min[dim_idx];
    for (int x_idx = 1; x_idx < num_x[dim_idx]; ++x_idx)
    {
      if (x_value[dim_idx][x_idx - 1] < delta_x_min_end[dim_idx])
      {
        x_value[dim_idx][x_idx] =
            x_value[dim_idx][x_idx - 1] + delta_x_min[dim_idx];
      }
      else if (x_value[dim_idx][x_idx - 1] < delta_x_max_start[dim_idx])
      {
        s1                      = std::sin(w *
                      (x_value[dim_idx][x_idx - 1] - delta_x_min_end[dim_idx]));
        x_value[dim_idx][x_idx] = x_value[dim_idx][x_idx - 1] +
                                  amplitude * s1 * s1 + delta_x_min[dim_idx];
      }
      else
      {
        x_value[dim_idx][x_idx] =
            x_value[dim_idx][x_idx - 1] + delta_x_max[dim_idx];
      }
    }
  }
  else if (coordinate_system_idx == 4) /* hyperspherical code*/
  {
    double slope;
    double x_total;
    double max_x;
    double amplitude;
    double w;
    double s1;
    PetscInt count;
    PetscInt dim_idx;
    PetscInt index;
    PetscInt block_size;

    /* initialize for loop */
    num_psi_build = 1.0;

    /* build grid */
    /********************************************************************/
    /********************************************************************/
    /*                                                                  */
    /* This dimension is used to tell the code that we are working in 3 */
    /* spacial dimensions. It is set to size one to avoid unneeded      */
    /* computation                                                      */
    /*                                                                  */
    /********************************************************************/
    /********************************************************************/
    dim_idx        = 0;
    num_x[dim_idx] = 1;

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    x_value[dim_idx][0] = 0;

    /**************************************************************************/
    /**************************************************************************/
    /*                                                                        */
    /* k, n, l_1, l_2, L, and m values */
    /* Since l and m cannot be written as a tensor product we combine them */
    /* into one dimension. This avoids having |m| > l which would happen with
     */
    /* a tensor like grid */
    /*                                                                        */
    /**************************************************************************/
    /**************************************************************************/
    dim_idx = 1;

    /* combines both m and l in one dimension  */
    num_x[dim_idx] = GetHypersphereSize(k_max, l_max);

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];
    /* these vectors are small, so it is easier (and possible faster) to store
     * them rather than convert and index to l and m values */
    eigen_values = new PetscInt*[num_x[dim_idx]];
    l_block_size = new PetscInt[k_max + 1];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];
    index = 0;
    for (int L_val = 0; L_val < l_max + 1; ++L_val)
    {
      block_size = 0;
      for (int k_val = 0; k_val < k_max + 1; ++k_val)
      {
        for (int l_1 = 0; l_1 < k_max + 1; l_1 += 2)
        {
          for (int l_2 = 0; l_2 < k_max + 1; ++l_2)
          {
            for (int n = 0; n < k_val / 2 + 1; ++n)
            {
              if (k_val == l_1 + l_2 + 2 * n)
              {
                for (int L = abs(l_1 - l_2); L < l_1 + l_2 + 1; ++L)
                {
                  if (L == L_val and k_val % 2 == L_val % 2)
                  {
                    eigen_values[index]    = new PetscInt[6];
                    eigen_values[index][0] = k_val;
                    eigen_values[index][1] = n;
                    eigen_values[index][2] = l_1;
                    eigen_values[index][3] = l_2;
                    eigen_values[index][4] = L;
                    eigen_values[index][5] =
                        0; /* M value only m=0 for linear */
                    x_value[dim_idx][index] = index;
                    index++;
                    block_size++;
                  }
                }
              }
            }
          }
        }
      }
      l_block_size[L_val] = block_size;
      max_block_size      = max(max_block_size, block_size);
    }
    /****************************************************************/
    /****************************************************************/
    /*                                                              */
    /* handle radial direction                                      */
    /* This is a tensor grid with respect to the l and m dimension  */
    /*                                                              */
    /****************************************************************/
    /****************************************************************/
    dim_idx = 2;

    /* get change in dx */
    amplitude = delta_x_max[dim_idx] - delta_x_min[dim_idx];
    w = pi / (2.0 * (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]));

    /* start at dx */
    x_total = delta_x_min[dim_idx];
    count   = 0;
    max_x   = dim_size[dim_idx];

    while (x_total < max_x)
    {
      if (x_total < delta_x_min_end[dim_idx])
      {
        x_total += delta_x_min[dim_idx];
      }
      else if (x_total < delta_x_max_start[dim_idx])
      {
        s1 = std::sin(w * (x_total - delta_x_min_end[dim_idx]));
        x_total += amplitude * s1 * s1 + delta_x_min[dim_idx];
      }
      else
      {
        x_total += delta_x_max[dim_idx];
      }
      count++;
    }
    num_x[dim_idx] = count;

    if (num_x[dim_idx] % 2 != 0)
    {
      num_x[dim_idx]++;
    }
    if (num_x[dim_idx] <= order + 1)
    {
      EndRun(
          "Not enough gird points to support this order of Finite "
          "Difference. "
          "Please increase grid size.");
    }

    /* allocate grid */
    x_value[dim_idx] = new double[num_x[dim_idx]];

    /* size of 1d array for psi */
    num_psi_build *= num_x[dim_idx];

    slope = (delta_x_max[dim_idx] - delta_x_min[dim_idx]) /
            (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]);

    x_value[dim_idx][0] = delta_x_min[dim_idx];
    for (int x_idx = 1; x_idx < num_x[dim_idx]; ++x_idx)
    {
      if (x_value[dim_idx][x_idx - 1] < delta_x_min_end[dim_idx])
      {
        x_value[dim_idx][x_idx] =
            x_value[dim_idx][x_idx - 1] + delta_x_min[dim_idx];
      }
      else if (x_value[dim_idx][x_idx - 1] < delta_x_max_start[dim_idx])
      {
        s1                      = std::sin(w *
                      (x_value[dim_idx][x_idx - 1] - delta_x_min_end[dim_idx]));
        x_value[dim_idx][x_idx] = x_value[dim_idx][x_idx - 1] +
                                  amplitude * s1 * s1 + delta_x_min[dim_idx];
      }
      else
      {
        x_value[dim_idx][x_idx] =
            x_value[dim_idx][x_idx - 1] + delta_x_max[dim_idx];
      }
    }
  }
  else
  {
    PetscInt center;  /* idx of the 0.0 in the grid */
    double current_x; /* used for setting grid */
    double slope;
    double x_total;
    double max_x;
    double amplitude;
    double w;
    double s1;
    PetscInt count;

    /* initialize for loop */
    num_psi_build = 1.0;

    /* build grid */
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      amplitude = delta_x_max[dim_idx] - delta_x_min[dim_idx];
      w = pi / (2.0 * (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]));
      x_total = delta_x_min[dim_idx] / 2.0;
      count   = 0;
      if (coordinate_system_idx == 1 and dim_idx == 0)
      {
        max_x = dim_size[dim_idx];
      }
      else
      {
        max_x = dim_size[dim_idx] / 2.0;
      }

      while (x_total < max_x)
      {
        if (x_total < delta_x_min_end[dim_idx])
        {
          x_total += delta_x_min[dim_idx];
        }
        else if (x_total < delta_x_max_start[dim_idx])
        {
          s1 = std::sin(w * (x_total - delta_x_min_end[dim_idx]));
          x_total += amplitude * s1 * s1 + delta_x_min[dim_idx];
        }
        else
        {
          x_total += delta_x_max[dim_idx];
        }
        count++;
      }
      if (coordinate_system_idx == 1 and dim_idx == 0)
      {
        num_x[dim_idx] = count;
      }
      else
      {
        num_x[dim_idx] = count * 2.0;
      }

      if (num_x[dim_idx] % 2 != 0)
      {
        num_x[dim_idx]++;
      }

      /* allocate grid */
      x_value[dim_idx] = new double[num_x[dim_idx]];

      /* size of 1d array for psi */
      num_psi_build *= num_x[dim_idx];

      slope = (delta_x_max[dim_idx] - delta_x_min[dim_idx]) /
              (delta_x_max_start[dim_idx] - delta_x_min_end[dim_idx]);

      if (coordinate_system_idx == 1 and dim_idx == 0)
      {
        if (order > 2)
        {
          x_value[dim_idx][0] = delta_x_min[dim_idx];
        }
        else
        {
          x_value[dim_idx][0] = delta_x_min[dim_idx] / 2.0;
        }
        for (int x_idx = 1; x_idx < num_x[dim_idx]; ++x_idx)
        {
          if (x_value[dim_idx][x_idx - 1] < delta_x_min_end[dim_idx])
          {
            x_value[dim_idx][x_idx] =
                x_value[dim_idx][x_idx - 1] + delta_x_min[dim_idx];
          }
          else if (x_value[dim_idx][x_idx - 1] < delta_x_max_start[dim_idx])
          {
            s1 = std::sin(
                w * (x_value[dim_idx][x_idx - 1] - delta_x_min_end[dim_idx]));
            x_value[dim_idx][x_idx] = x_value[dim_idx][x_idx - 1] +
                                      amplitude * s1 * s1 +
                                      delta_x_min[dim_idx];
          }
          else
          {
            x_value[dim_idx][x_idx] =
                x_value[dim_idx][x_idx - 1] + delta_x_max[dim_idx];
          }
        }
      }
      else
      {
        /* find center of grid */
        center = num_x[dim_idx] / 2;

        x_value[dim_idx][center - 1] = -1.0 * delta_x_min[dim_idx] / 2.0;
        x_value[dim_idx][center]     = delta_x_min[dim_idx] / 2.0;

        /* loop over all others */
        for (PetscInt x_idx = center - 1; x_idx >= 0; x_idx--)
        {
          // /* get x value */
          // current_x =
          //     (x_idx - center) * delta_x[dim_idx] + delta_x[dim_idx] / 2.0;

          /* double checking index */
          if (x_idx < 0 || num_x[dim_idx] - x_idx - 1 >= num_x[dim_idx])
          {
            EndRun("Allocation error in grid");
          }

          if (std::abs(x_value[dim_idx][x_idx + 1]) < delta_x_min_end[dim_idx])
          {
            current_x = x_value[dim_idx][x_idx + 1] - delta_x_min[dim_idx];
          }
          else if (std::abs(x_value[dim_idx][x_idx + 1]) <
                   delta_x_max_start[dim_idx])
          {
            s1        = std::sin(w * (std::abs(x_value[dim_idx][x_idx + 1]) -
                               delta_x_min_end[dim_idx]));
            current_x = x_value[dim_idx][x_idx + 1] -
                        (amplitude * s1 * s1 + delta_x_min[dim_idx]);
          }
          else
          {
            current_x = x_value[dim_idx][x_idx + 1] - delta_x_max[dim_idx];
          }

          /* set negative side */
          x_value[dim_idx][x_idx] = current_x;
          /* set positive side */
          x_value[dim_idx][num_x[dim_idx] - x_idx - 1] = -1 * current_x;
        }
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
  if (coordinate_system_idx != 2)
  {
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
  }

  /* allocate psi */
  if (!psi_alloc)
  {
    VecCreate(PETSC_COMM_WORLD, &psi);
    VecSetSizes(psi, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi);
    ierr = PetscObjectSetName((PetscObject)psi, "psi");

    if (coordinate_system_idx == 3)
    {
      VecCreate(PETSC_COMM_WORLD, &psi_small);
      VecSetSizes(psi_small, PETSC_DECIDE, num_x[2]);
      VecSetFromOptions(psi_small);
      ierr = PetscObjectSetName((PetscObject)psi_small, "psi");
    }

    VecCreate(PETSC_COMM_WORLD, &psi_tmp);
    VecSetSizes(psi_tmp, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_tmp);
    ierr = PetscObjectSetName((PetscObject)psi_tmp, "psi");

    VecCreate(PETSC_COMM_WORLD, &psi_tmp_cyl);
    VecSetSizes(psi_tmp_cyl, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_tmp_cyl);
    ierr = PetscObjectSetName((PetscObject)psi_tmp_cyl, "psi");

    VecCreate(PETSC_COMM_WORLD, &psi_proj);
    VecSetSizes(psi_proj, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi_proj);
    ierr = PetscObjectSetName((PetscObject)psi_proj, "psi");
    VecCreate(PETSC_COMM_WORLD, &jacobian);
    VecSetSizes(jacobian, PETSC_DECIDE, num_psi);
    VecSetFromOptions(jacobian);
    ierr = PetscObjectSetName((PetscObject)jacobian, "psi");

    VecCreate(PETSC_COMM_WORLD, &ECS);
    VecSetSizes(ECS, PETSC_DECIDE, num_psi);
    VecSetFromOptions(ECS);
    ierr = PetscObjectSetName((PetscObject)ECS, "psi");

    position_expectation = new Vec[num_dims * num_electrons];
    dipole_acceleration  = new Vec[num_dims * num_electrons];
    for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        VecCreate(PETSC_COMM_WORLD,
                  &position_expectation[elec_idx * num_dims + dim_idx]);
        VecSetSizes(position_expectation[elec_idx * num_dims + dim_idx],
                    PETSC_DECIDE, num_psi);
        VecSetFromOptions(position_expectation[elec_idx * num_dims + dim_idx]);
        ierr = PetscObjectSetName(
            (PetscObject)position_expectation[elec_idx * num_dims + dim_idx],
            "psi");

        VecCreate(PETSC_COMM_WORLD,
                  &dipole_acceleration[elec_idx * num_dims + dim_idx]);
        VecSetSizes(dipole_acceleration[elec_idx * num_dims + dim_idx],
                    PETSC_DECIDE, num_psi);
        VecSetFromOptions(dipole_acceleration[elec_idx * num_dims + dim_idx]);
        ierr = PetscObjectSetName(
            (PetscObject)dipole_acceleration[elec_idx * num_dims + dim_idx],
            "psi");
      }
    }

    psi_alloc = true;
  }

  VecGetOwnershipRange(psi, &low, &high);
  for (PetscInt idx = low; idx < high; idx++)
  {
    /* set psi */
    if (coordinate_system_idx == 2)
    {
      val = dcomp(exp(-1.0 *
                      (x_value[0][idx] * x_value[0][idx] +
                       x_value[1][idx] * x_value[1][idx] +
                       x_value[2][idx] * x_value[2][idx]) /
                      (2.0 * sigma2)),
                  0.0);
    }
    else
    {
      val = GetPsiVal(psi_build, idx);
    }
    VecSetValues(psi, 1, &idx, &val, INSERT_VALUES);
  }
  VecAssemblyBegin(psi);
  VecAssemblyEnd(psi);

  if (coordinate_system_idx == 3)
  {
    VecGetOwnershipRange(psi_small, &low, &high);
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = psi_build[0][2][idx];
      VecSetValues(psi_small, 1, &idx, &val, INSERT_VALUES);
    }
    VecAssemblyBegin(psi_small);
    VecAssemblyEnd(psi_small);
  }

  CreateObservables();

  /* normalize all psi */
  Normalize();
}

void Wavefunction::CreateObservables()
{
  PetscInt low, high;
  PetscComplex val;

  if (coordinate_system_idx == 2) /* RBF */
  {
    /* load node set hdf5 file*/
    HDF5Wrapper node_set("nodes.h5", "r");

    double* weights = node_set.GetFirstN("/node_set/weights", num_psi);
    VecGetOwnershipRange(jacobian, &low, &high);
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = weights[idx];
      VecSetValues(jacobian, 1, &idx, &val, INSERT_VALUES);
    }
    delete weights;
    VecAssemblyBegin(jacobian);
    VecAssemblyEnd(jacobian);

    VecGetOwnershipRange(ECS, &low, &high);
    for (PetscInt idx = low; idx < high; idx++)
    {
      /* TODO add ECS values */
      val = 0.0;
      VecSetValues(ECS, 1, &idx, &val, INSERT_VALUES);
    }
    VecAssemblyBegin(ECS);
    VecAssemblyEnd(ECS);
  }
  else /* not rbf */
  {
    VecGetOwnershipRange(jacobian, &low, &high);
    for (PetscInt idx = low; idx < high; idx++)
    {
      if (coordinate_system_idx == 1)
        val = GetPositionVal(idx, 0, 0, true);
      else
        val = 1.0;
      if (coordinate_system_idx != 3 and coordinate_system_idx != 4)
      {
        val *= GetVolumeElement(idx);
      }
      VecSetValues(jacobian, 1, &idx, &val, INSERT_VALUES);
    }
    VecAssemblyBegin(jacobian);
    VecAssemblyEnd(jacobian);

    VecGetOwnershipRange(ECS, &low, &high);
    for (PetscInt idx = low; idx < high; idx++)
    {
      val = GetGobblerVal(idx);
      VecSetValues(ECS, 1, &idx, &val, INSERT_VALUES);
    }
    VecAssemblyBegin(ECS);
    VecAssemblyEnd(ECS);
  }

  /* same for all coordinate systems */
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      VecGetOwnershipRange(position_expectation[elec_idx * num_dims + dim_idx],
                           &low, &high);
      for (PetscInt idx = low; idx < high; idx++)
      {
        val = GetPositionVal(idx, elec_idx, dim_idx, false);
        VecSetValues(position_expectation[elec_idx * num_dims + dim_idx], 1,
                     &idx, &val, INSERT_VALUES);
      }
      VecAssemblyBegin(position_expectation[elec_idx * num_dims + dim_idx]);
      VecAssemblyEnd(position_expectation[elec_idx * num_dims + dim_idx]);

      VecGetOwnershipRange(dipole_acceleration[elec_idx * num_dims + dim_idx],
                           &low, &high);
      for (PetscInt idx = low; idx < high; idx++)
      {
        val = GetDipoleAccerationVal(idx, elec_idx, dim_idx);
        VecSetValues(dipole_acceleration[elec_idx * num_dims + dim_idx], 1,
                     &idx, &val, INSERT_VALUES);
      }
      VecAssemblyBegin(dipole_acceleration[elec_idx * num_dims + dim_idx]);
      VecAssemblyEnd(dipole_acceleration[elec_idx * num_dims + dim_idx]);
    }
  }
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

dcomp Wavefunction::GetVolumeElement(PetscInt idx)
{
  /* Value to be returned */
  dcomp ret_val(1.0, 0.0);
  if (GetGobblerVal(idx).real())
  {
    return dcomp(0.0, 0.0);
  }
  /* idx for return */
  std::vector< PetscInt > idx_array = GetIntArray(idx);
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      if (idx_array[elec_idx * num_dims + dim_idx] > 0 and
          idx_array[elec_idx * num_dims + dim_idx] < num_x[dim_idx] - 1)
      {
        ret_val *=
            (x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]] -
             x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx] - 1]);
      }
      else if (idx_array[elec_idx * num_dims + dim_idx] == 0 and
               dim_idx == 0 and coordinate_system_idx == 1)
      {
        ret_val *= delta_x_min[dim_idx];
      }
      else
      {
        ret_val *= delta_x_max[dim_idx];
      }
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
  if (coordinate_system_idx == 2) /* RBF */
  {
    ret_val = x_value[dim_idx][idx];
  }
  else
  {
    /* idx for return */
    std::vector< PetscInt > idx_array = GetIntArray(idx);
    ret_val = x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
    if (integrate and order > 2 and dim_idx == 0 and coordinate_system_idx == 1)
    {
      // std::cout << "using integrate\n";
      /* see appendix A of https://arxiv.org/pdf/1604.00947.pdf using Lagrange
       * interpolation polynomials and
       * http://slideflix.net/doc/4183369/gregory-s-quadrature-method */
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
  double r_soft;
  double x;
  double tmp, tmp_soft;
  std::vector< PetscInt > idx_array = GetIndexArray(idx);
  if (coordinate_system_idx == 2) /* RBF */
  {
    r = sqrt(x_value[0][idx] * x_value[0][idx] +
             x_value[1][idx] * x_value[1][idx] +
             x_value[2][idx] * x_value[2][idx]);
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      ret_val -= z[nuclei_idx] * x_value[dim_idx][idx] / (r * r * r);
    }
    if (world.rank() == 0)
    {
      std::cout << "WARNING: RBF grids dipole acceleration only calculates the "
                   "Coulomb potential.\n";
    }
  }
  else
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (coordinate_system_idx == 3)
      {
        /* It is non trivial to account for nuclei not at the origin.
         * There are many other portions of the code that need to be
         * updated to account for this.
         * A future iteration of the code will account for this.
         */
        r      = x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]];
        r_soft = sqrt(r * r + alpha_2);
        /* x is a matrix, so it is applied in ``GetDipoleAcceration'' */
        x = 1.0;
      }
      else
      {
        r_soft = SoftCoreDistance(location[nuclei_idx], idx_array, elec_idx);
        r      = EuclideanDistance(location[nuclei_idx], idx_array, elec_idx);
        x      = x_value[dim_idx][idx_array[elec_idx * num_dims + dim_idx]];
      }

      /* Coulomb term */
      ret_val -= dcomp(x * z[nuclei_idx] / (r_soft * r_soft * r_soft), 0.0);

      /* Gaussian Donuts */
      for (PetscInt i = 0; i < gaussian_size[nuclei_idx]; ++i)
      {
        tmp = gaussian_decay_rate[nuclei_idx][i] *
              (r - gaussian_r_0[nuclei_idx][i]);
        ret_val -= dcomp(x * gaussian_decay_rate[nuclei_idx][i] * tmp *
                             gaussian_amplitude[nuclei_idx][i] *
                             exp(-0.5 * (tmp * tmp)) / r,
                         0.0);
      }

      /* Exponential Donuts */
      for (PetscInt i = 0; i < exponential_size[nuclei_idx]; ++i)
      {
        tmp = exponential_decay_rate[nuclei_idx][i] *
              std::abs(r - exponential_r_0[nuclei_idx][i]);
        ret_val -=
            dcomp(x * exponential_amplitude[nuclei_idx][i] *
                      exponential_decay_rate[nuclei_idx][i] * exp(-tmp) / r,
                  0.0);
      }

      /* Square Well Donuts */
      for (PetscInt i = 0; i < square_well_size[nuclei_idx]; ++i)
      {
        if (std::abs(square_well_amplitude[nuclei_idx][i]) > 1e-14 and
            world.rank() == 0)
        {
          std::cout << "WARNING: square well potential don't work with dipole "
                       "acceleration.\n";
        }
      }

      /* Yukawa Donuts */
      for (PetscInt i = 0; i < yukawa_size[nuclei_idx]; ++i)
      {
        tmp = yukawa_decay_rate[nuclei_idx][i] *
              std::abs(r - yukawa_r_0[nuclei_idx][i]);
        tmp_soft = std::abs(r_soft - yukawa_r_0[nuclei_idx][i]);
        ret_val -= dcomp(x * yukawa_amplitude[nuclei_idx][i] * exp(-tmp) /
                                 (tmp_soft * tmp_soft * r) +
                             x * yukawa_decay_rate[nuclei_idx][i] *
                                 yukawa_amplitude[nuclei_idx][i] * exp(-tmp) /
                                 (tmp_soft * r),
                         0.0);
      }
    }
  }

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
void Wavefunction::Normalize() { Normalize(psi, 0.0); }

/* normalizes the array provided */
void Wavefunction::Normalize(Vec& data, double dv)
{
  PetscReal total = Norm(data, dv);
  VecScale(data, 1.0 / total);
}

/* returns norm of psi */
double Wavefunction::Norm() { return Norm(psi, 0.0); }

/* returns norm of array using trapezoidal rule */
double Wavefunction::Norm(Vec& data, double dv)
{
  PetscLogEventBegin(time_norm, 0, 0, 0, 0);
  dcomp dot_product;
  double total = 0;
  VecPointwiseMult(psi_tmp, jacobian, data);
  VecDot(data, psi_tmp, &dot_product);
  total = sqrt(dot_product.real());
  PetscLogEventEnd(time_norm, 0, 0, 0, 0);
  return total;
}

double Wavefunction::GetEnergy(Mat* h) { return GetEnergy(h, psi); }

double Wavefunction::GetEnergy(Mat* h, Vec& p)
{
  PetscLogEventBegin(time_energy, 0, 0, 0, 0);
  PetscComplex energy;
  MatMult(*h, p, psi_tmp_cyl);
  VecPointwiseMult(psi_tmp_cyl, jacobian, psi_tmp_cyl);
  VecDot(p, psi_tmp_cyl, &energy);
  PetscLogEventEnd(time_energy, 0, 0, 0, 0);
  return energy.real();
}

/* returns position expectation value <psi|x_{elec_idx,dim_idx}|psi> */
double Wavefunction::GetPosition(PetscInt elec_idx, PetscInt dim_idx)
{
  PetscLogEventBegin(time_position, 0, 0, 0, 0);
  PetscComplex expectation;
  if (coordinate_system_idx == 3)
  {
    if (position_mat_alloc)
    {
      /* apply position matrix */
      MatMult(position_mat[dim_idx], psi, psi_tmp_cyl);
      /* apply Jacobian after (doesn't commute with matrix) */
      VecPointwiseMult(psi_tmp, jacobian, psi_tmp_cyl);
      VecDot(psi, psi_tmp, &expectation);
    }
    else /* return zero if the position expectation matrix has not been built
          */
    {
      expectation = 0.0;
    }
  }
  else if (coordinate_system_idx == 4)
  {
    if (position_mat_alloc)
    {
      /* apply position matrix */
      MatMult(position_mat[dim_idx], psi, psi_tmp_cyl);
      /* apply Jacobian after (doesn't commute with matrix) */
      VecPointwiseMult(psi_tmp, jacobian, psi_tmp_cyl);
      VecDot(psi, psi_tmp, &expectation);
    }
    else /* return zero if the position expectation matrix has not been built
          */
    {
      expectation = 0.0;
    }
  }
  else
  {
    /* apply Jacobian */
    VecPointwiseMult(psi_tmp_cyl, jacobian, psi);
    /* scale by location */
    VecPointwiseMult(psi_tmp,
                     position_expectation[elec_idx * num_dims + dim_idx],
                     psi_tmp_cyl);
    VecDot(psi, psi_tmp, &expectation);
  }
  PetscLogEventEnd(time_position, 0, 0, 0, 0);
  return expectation.real();
}

/* returns dipole acceleration value <psi|x_{elec_idx,dim_idx}/r^3|psi> */
double Wavefunction::GetDipoleAcceration(PetscInt elec_idx, PetscInt dim_idx)
{
  PetscLogEventBegin(time_dipole_acceration, 0, 0, 0, 0);
  PetscComplex expectation;
  if (coordinate_system_idx == 3)
  {
    if (position_mat_alloc)
    {
      /* apply position matrix */
      MatMult(position_mat[dim_idx], psi, psi_tmp_cyl);
      /* apply 1/r^3 (doesn't commute with matrix) */
      VecPointwiseMult(psi_tmp_cyl,
                       dipole_acceleration[elec_idx * num_dims + dim_idx],
                       psi_tmp_cyl);
      /* apply Jacobian after (doesn't commute with matrix) */
      VecPointwiseMult(psi_tmp, jacobian, psi_tmp_cyl);
      VecDot(psi, psi_tmp, &expectation);
    }
    else /* return zero if the position expectation matrix has not been built
          */
    {
      expectation = 0.0;
    }
  }
  else
  {
    VecPointwiseMult(psi_tmp_cyl, jacobian, psi);
    VecPointwiseMult(psi_tmp,
                     dipole_acceleration[elec_idx * num_dims + dim_idx],
                     psi_tmp_cyl);
    VecDot(psi, psi_tmp, &expectation);
  }
  PetscLogEventEnd(time_dipole_acceration, 0, 0, 0, 0);
  return expectation.real();
}

double Wavefunction::GetGobbler()
{
  PetscLogEventBegin(time_gobbler, 0, 0, 0, 0);
  PetscComplex expectation;
  VecPointwiseMult(psi_tmp_cyl, jacobian, psi);
  VecPointwiseMult(psi_tmp, ECS, psi_tmp_cyl);
  VecDot(psi, psi_tmp, &expectation);
  PetscLogEventEnd(time_gobbler, 0, 0, 0, 0);
  return expectation.real();
}

PetscInt Wavefunction::GetProjectionSize()
{
  if (coordinate_system_idx == 3)
  {
    PetscInt projection_size = 0;
    for (int n_index = 1; n_index <= num_states; ++n_index)
    {
      for (int l_idx = 0; l_idx < fmin(n_index, l_max + 1); ++l_idx)
      {
        for (int m_idx = -1. * std::min(m_max, l_idx);
             m_idx < std::min(m_max, l_idx) + 1; ++m_idx)
        {
          projection_size++;
        }
      }
    }
    return projection_size;
  }
  else
  {
    return num_states;
  }
}

void Wavefunction::ResetPsi()
{
  CreatePsi();
  CleanUp();
}

void Wavefunction::ZeroPhasePsiSmall()
{
  dcomp val;
  double phase;
  /* Get phase from master */
  if (world.rank() == 0)
  {
    /* use the first entry in the array to set the phase */
    PetscInt idx = 0;
    VecGetValues(psi_small, 1, &idx, &val);
    phase = std::arg(val);
  }
  /* Share phase with all processors*/
  broadcast(world, phase, 0);
  /* Remove phase from wavefunction */
  VecScale(psi_small, std::exp(-imag * phase));
}

void Wavefunction::RadialHGroundPsiSmall()
{
  PetscInt low, high;
  PetscComplex val;
  PetscReal r;

  VecGetOwnershipRange(psi_small, &low, &high);
  for (PetscInt idx = low; idx < high; idx++)
  {
    r = x_value[2][idx];
    if (r > 0)
    {
      val = dcomp(2.0 * std::exp(-r) / r, 0.0);
    }
    else
    {
      val = 0.0;
    }
    VecSetValues(psi_small, 1, &idx, &val, INSERT_VALUES);
  }
  VecAssemblyBegin(psi_small);
  VecAssemblyEnd(psi_small);
}

void Wavefunction::SetPositionMat(Mat* input_mat)
{
  position_mat = new Mat[num_dims];
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    if (position_mat_alloc == false)
    {
      MatConvert(input_mat[dim_idx], MATSAME, MAT_INITIAL_MATRIX,
                 &position_mat[dim_idx]);
    }
  }
  position_mat_alloc = true;
}

double Wavefunction::SoftCoreDistance(double* location, PetscInt idx)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] - x_value[dim_idx][idx];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Wavefunction::SoftCoreDistance(double* location,
                                      std::vector< PetscInt >& idx_array,
                                      PetscInt elec_idx)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx * num_dims)]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Wavefunction::SoftCoreDistance(std::vector< PetscInt >& idx_array,
                                      PetscInt elec_idx_1, PetscInt elec_idx_2)
{
  double distance = ee_soft_core_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_1 * num_dims)]] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_2 * num_dims)]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Wavefunction::EuclideanDistance(double* location, PetscInt idx)
{
  double distance = 0.0;
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] - x_value[dim_idx][idx];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Wavefunction::EuclideanDistance(double* location,
                                       std::vector< PetscInt >& idx_array,
                                       PetscInt elec_idx)
{
  double distance = 0.0;
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx * num_dims)]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Wavefunction::EuclideanDistance(std::vector< PetscInt >& idx_array,
                                       PetscInt elec_idx_1, PetscInt elec_idx_2)
{
  double distance = 0.0;
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_1 * num_dims)]] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_2 * num_dims)]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

/* Returns the an array of alternating i,j components of the local matrix */
std::vector< PetscInt > Wavefunction::GetIndexArray(PetscInt idx_i)
{
  PetscInt idx_j      = idx_i;
  PetscInt total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector< PetscInt > num(total_dims);
  /* idx for return */
  std::vector< PetscInt > idx_array(total_dims * 2);
  /* used for convenience. Could/should be optimized */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      num[elec_idx * num_dims + dim_idx] = num_x[dim_idx];
    }
  }
  /* loop over dimensions backwards so psi becomes
   * psi[x_1,y_1,z_1,x_2,y_2,z_2,...]
   * where x_1 is the first dimension of the first electron and x_2 is the
   * first dimension of the second electron and so on */
  for (PetscInt i = total_dims - 1; i >= 0; --i)
  {
    idx_array[2 * i] = idx_i % num[i];
    idx_i /= num[i];
    idx_array[2 * i + 1] = idx_j % num[i];
    idx_j /= num[i];
  }
  return idx_array;
}

PetscInt* Wavefunction::GetNumX() { return num_x; }

PetscInt Wavefunction::GetNumPsi() { return num_psi; }

PetscInt Wavefunction::GetNumPsiBuild() { return num_psi_build; }

Vec* Wavefunction::GetPsi() { return &psi; }

Vec* Wavefunction::GetPsiSmall() { return &psi_small; }

double** Wavefunction::GetXValue() { return x_value; }

PetscInt* Wavefunction::GetLValues() { return l_values; }

PetscInt* Wavefunction::GetMValues() { return m_values; }
PetscInt** Wavefunction::GetEigenValues() { return eigen_values; }

PetscInt* Wavefunction::GetLBlockSize() { return l_block_size; }
PetscInt Wavefunction::GetMaxBlockSize() { return max_block_size; }

PetscInt** Wavefunction::GetGobblerIdx() { return gobbler_idx; }

PetscInt Wavefunction::GetWriteCounterCheckpoint()
{
  return write_counter_checkpoint;
}

/* destructor */
Wavefunction::~Wavefunction()
{
  if (world.rank() == 0) std::cout << "Deleting Wavefunction\n";
  /* do not delete dim_size or delta_x since they belong to the Parameter
   * class and will be freed there*/
  for (PetscInt i = 0; i < num_dims; i++)
  {
    delete x_value[i];
  }
  delete[] x_value;
  CleanUp();
  VecDestroy(&psi);
  VecDestroy(&psi_tmp);
  if (coordinate_system_idx == 3)
  {
    VecDestroy(&psi_small);
    delete l_values;
    delete m_values;
  }
  if (coordinate_system_idx == 4)
  {
    for (int i = 0; i < num_x[1]; ++i)
    {
      delete eigen_values[i];
    }
    delete[] eigen_values;
    delete l_block_size;
  }
  if (position_mat_alloc)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatDestroy(&position_mat[dim_idx]);
    }
  }
  delete num_x;
}
