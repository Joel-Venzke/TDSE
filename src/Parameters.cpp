#include "Parameters.h"

/* Constructor */
Parameters::Parameters(std::string file_name) { Setup(file_name); }

/* basically the default constructor */
void Parameters::Setup(std::string file_name)
{
  if (world.rank() == 0)
  {
    std::cout << "Simulation running on " << world.size() << " Processors\n"
              << "Reading input file: " << file_name << "\n"
              << std::flush;
  }

  double polar_norm    = 0.0; /* the norm for the polarization vector */
  double poynting_norm = 0.0; /* the norm for the poynting vector */
  double intensity     = 0.0; /* the norm for the poynting vector */

  /* read data from file */
  json data = FileToJson(file_name);

  /* get numeric information */
  delta_t           = data["delta_t"];
  num_dims          = data["dimensions"].size();
  num_electrons     = data["num_electrons"];
  coordinate_system = data["coordinate_system"];

  if (coordinate_system == "Cartesian")
  {
    coordinate_system_idx = 0;
  }
  else if (coordinate_system == "Cylindrical")
  {
    coordinate_system_idx = 1;
  }
  else
  {
    coordinate_system_idx = -1;
  }

  dim_size          = std::make_unique< double[] >(num_dims);
  delta_x_min       = std::make_unique< double[] >(num_dims);
  delta_x_min_end   = std::make_unique< double[] >(num_dims);
  delta_x_max       = std::make_unique< double[] >(num_dims);
  delta_x_max_start = std::make_unique< double[] >(num_dims);

  for (PetscInt i = 0; i < num_dims; ++i)
  {
    dim_size[i]          = data["dimensions"][i]["dim_size"];
    delta_x_min[i]       = data["dimensions"][i]["delta_x_min"];
    delta_x_min_end[i]   = data["dimensions"][i]["delta_x_min_end"];
    delta_x_max[i]       = data["dimensions"][i]["delta_x_max"];
    delta_x_max_start[i] = data["dimensions"][i]["delta_x_max_start"];
  }

  /* get simulation behavior */
  restart                     = data["restart"];
  target                      = data["target"]["name"];
  num_nuclei                  = data["target"]["nuclei"].size();
  alpha                       = data["alpha"];
  write_frequency_checkpoint  = data["write_frequency_checkpoint"];
  write_frequency_observables = data["write_frequency_observables"];
  write_frequency_eigin_state = data["write_frequency_eigin_state"];
  gobbler                     = data["gobbler"];
  order                       = data["order"];
  sigma                       = data["sigma"];
  tol                         = data["tol"];
  state_solver                = data["state_solver"];
  if (state_solver == "File")
  {
    state_solver_idx = 0;
  }
  else if (state_solver == "ITP")
  {
    state_solver_idx = 1;
  }
  else if (state_solver == "Power")
  {
    state_solver_idx = 2;
  }
  else if (state_solver == "SLEPC")
  {
    state_solver_idx = 3;
  }
  start_state = data["start_state"];
  if (state_solver_idx != 2)
  {
    num_states = data["states"];
  }
  else
  {
    num_states = data["states"].size();

    state_energy = std::make_unique< double[] >(num_states);

    for (PetscInt i = 0; i < num_states; i++)
    {
      state_energy[i] = data["states"][i]["energy"];
    }
  }

  z        = std::make_unique< double[] >(num_nuclei);
  z_c      = std::make_unique< double[] >(num_nuclei);
  c0       = std::make_unique< double[] >(num_nuclei);
  r0       = std::make_unique< double[] >(num_nuclei);
  sae_size = std::make_unique< PetscInt[] >(num_nuclei);
  a        = new double*[num_nuclei];
  b        = new double*[num_nuclei];
  location = new double*[num_nuclei];
  for (PetscInt i = 0; i < num_nuclei; ++i)
  {
    z[i] = data["target"]["nuclei"][i]["z"];

    if (z[i] == 0.0)
    {
      z_c[i] = data["target"]["nuclei"][i]["SAE"]["z_c"];
      c0[i]  = data["target"]["nuclei"][i]["SAE"]["c0"];
      r0[i]  = data["target"]["nuclei"][i]["SAE"]["r0"];
      if (data["target"]["nuclei"][i]["SAE"]["a"].size() !=
          data["target"]["nuclei"][i]["SAE"]["b"].size())
      {
        EndRun("Nuclei " + std::to_string(i) +
               " SAE potential a and b must have the same size");
      }
      sae_size[i] = data["target"]["nuclei"][i]["SAE"]["a"].size();

      a[i] = new double[sae_size[i]];
      b[i] = new double[sae_size[i]];
      for (PetscInt j = 0; j < sae_size[i]; ++j)
      {
        a[i][j] = data["target"]["nuclei"][i]["SAE"]["a"][j];
        b[i][j] = data["target"]["nuclei"][i]["SAE"]["b"][j];
      }
    }
    else
    {
      /* So we don't have issues deleting them*/
      a[i] = new double[1];
      b[i] = new double[1];

      /* So they write nicely */
      z_c[i]      = 0.0;
      c0[i]       = 0.0;
      r0[i]       = 0.0;
      sae_size[i] = 0;
      a[i][0]     = 0.0;
      b[i][0]     = 0.0;
    }

    location[i] = new double[num_dims];
    if (data["target"]["nuclei"][i]["location"].size() < num_dims)
    {
      EndRun("Nuclei " + std::to_string(i) +
             " location has to small of a dimension");
    }
    for (PetscInt j = 0; j < num_dims; ++j)
    {
      location[i][j] = data["target"]["nuclei"][i]["location"][j];
    }
  }

  /* index is used throughout code for efficiency */
  /* and ease of writing to hdf5 */
  if (target == "He")
  {
    target_idx = 0;
  }
  else if (target == "H")
  {
    target_idx = 1;
  }
  else
  {
    target_idx = -1;
  }

  propagate      = data["propagate"];
  free_propagate = data["free_propagate"];

  /* get pulse information */
  num_pulses      = data["laser"]["pulses"].size();
  experiment_type = data["laser"]["experiment_type"];

  /* allocate memory */
  pulse_shape         = std::make_unique< std::string[] >(num_pulses);
  pulse_shape_idx     = std::make_unique< PetscInt[] >(num_pulses);
  power_on            = std::make_unique< PetscInt[] >(num_pulses);
  power_off           = std::make_unique< PetscInt[] >(num_pulses);
  cycles_on           = std::make_unique< double[] >(num_pulses);
  cycles_plateau      = std::make_unique< double[] >(num_pulses);
  cycles_off          = std::make_unique< double[] >(num_pulses);
  cycles_delay        = std::make_unique< double[] >(num_pulses);
  cep                 = std::make_unique< double[] >(num_pulses);
  energy              = std::make_unique< double[] >(num_pulses);
  field_max           = std::make_unique< double[] >(num_pulses);
  ellipticity         = std::make_unique< double[] >(num_pulses);
  helicity            = std::make_unique< std::string[] >(num_pulses);
  helicity_idx        = std::make_unique< PetscInt[] >(num_pulses);
  polarization_vector = new double*[num_pulses];
  if (num_dims == 3) poynting_vector= new double*[num_pulses];

  /* set pulses up by experiment type */

  if (experiment_type == "File")
  {
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      /* Keep from breaking other things */
      pulse_shape_idx[pulse_idx] = 0;
      power_on[pulse_idx]        = 2.0;
      power_off[pulse_idx]       = 2.0;
      cycles_on[pulse_idx]       = 1.0;
      cycles_plateau[pulse_idx]  = 0.0;
      cycles_off[pulse_idx]      = 1.0;
      cycles_delay[pulse_idx]    = 0.0;
      cep[pulse_idx]             = 0.0;
      energy[pulse_idx]          = 1.0;
      field_max[pulse_idx]       = 1.0;
      ellipticity[pulse_idx]     = 0.0;
      helicity_idx[pulse_idx]    = 0;

      if (num_dims == 3)
      {
        poynting_vector[pulse_idx]    = new double[num_dims];
        poynting_vector[pulse_idx][0] = 0.0;
        poynting_vector[pulse_idx][1] = 0.0;
        poynting_vector[pulse_idx][2] = 1.0;
      }

      /* Get polarization */
      polarization_vector[pulse_idx] = new double[num_dims];
      polar_norm                     = 0.0;
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        polarization_vector[pulse_idx][dim_idx] =
            data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx];
        polar_norm += polarization_vector[pulse_idx][dim_idx] *
                      polarization_vector[pulse_idx][dim_idx];
      }
      /* normalize the polarization vector*/
      if (polar_norm < 1e-10)
      {
        EndRun("Polarization Vector has Norm of Zero");
      }
      polar_norm = sqrt(polar_norm);
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        polarization_vector[pulse_idx][dim_idx] /= polar_norm;
      }
    }
  }
  else
  {
    /* streaking */
    if (experiment_type == "streaking" and num_pulses != 2)
    {
      EndRun(" streaking only allows 2 pulses");
    }

    /* read in IR and XUV parameters */
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      power_on[pulse_idx]  = 1.0;
      power_off[pulse_idx] = 1.0;
      pulse_shape[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["pulse_shape"];

      /* index used similar target_idx */
      if (pulse_shape[pulse_idx] == "sin2")
      {
        EndRun(
            "\"sin2\" is no longer an option. Please change to \"sin\" and set "
            "\"power_on\" and \"power_off\" to \"2\"\n");
      }
      else if (pulse_shape[pulse_idx] == "sin")
      {
        pulse_shape_idx[pulse_idx] = 0;
        power_on[pulse_idx]  = data["laser"]["pulses"][pulse_idx]["power_on"];
        power_off[pulse_idx] = data["laser"]["pulses"][pulse_idx]["power_off"];
      }
      else if (pulse_shape[pulse_idx] == "gaussian")
      {
        pulse_shape_idx[pulse_idx] = 1;
      }
      else
      {
        pulse_shape_idx[pulse_idx] = -1;
      }

      if (data["laser"]["pulses"][pulse_idx]["polarization_vector"].size() <
          num_dims)
      {
        EndRun("Polarization vector dimension is to small for pulse " +
               std::to_string(pulse_idx));
      }

      polarization_vector[pulse_idx] = new double[num_dims];
      polar_norm                     = 0.0;
      if (num_dims == 3)
      {
        poynting_vector[pulse_idx] = new double[num_dims];
        poynting_norm              = 0.0;
      }
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        polarization_vector[pulse_idx][dim_idx] =
            data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx];
        polar_norm += polarization_vector[pulse_idx][dim_idx] *
                      polarization_vector[pulse_idx][dim_idx];

        if (num_dims == 3)
        {
          poynting_vector[pulse_idx][dim_idx] =
              data["laser"]["pulses"][pulse_idx]["poynting_vector"][dim_idx];
          poynting_norm += poynting_vector[pulse_idx][dim_idx] *
                           poynting_vector[pulse_idx][dim_idx];
        }
      }
      /* normalize the polarization vector*/
      if (polar_norm < 1e-10)
      {
        EndRun("Polarization Vector has Norm of Zero");
      }
      polar_norm = sqrt(polar_norm);
      if (num_dims == 3)
      {
        if (poynting_norm < 1e-10)
        {
          EndRun("Poynting Vector has Norm of Zero");
        }
        poynting_norm = sqrt(poynting_norm);
      }
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        polarization_vector[pulse_idx][dim_idx] /= polar_norm;
        if (num_dims == 3 and poynting_norm > 1e-10)
        {
          poynting_vector[pulse_idx][dim_idx] /= poynting_norm;
        }
      }

      cep[pulse_idx]    = data["laser"]["pulses"][pulse_idx]["cep"];
      energy[pulse_idx] = data["laser"]["pulses"][pulse_idx]["energy"];
      ellipticity[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["ellipticity"];
      helicity[pulse_idx] = data["laser"]["pulses"][pulse_idx]["helicity"];

      intensity = data["laser"]["pulses"][pulse_idx]["intensity"];
      field_max[pulse_idx] =
          std::sqrt(intensity / 3.51e16) * c / energy[pulse_idx];

      if (helicity[pulse_idx] == "right")
      {
        helicity_idx[pulse_idx] = 0;
      }
      else if (helicity[pulse_idx] == "left")
      {
        helicity_idx[pulse_idx] = 1;
      }
      else
      {
        helicity_idx[pulse_idx] = -1;
      }

      cycles_on[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_on"];
      cycles_plateau[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["cycles_plateau"];
      cycles_off[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_off"];

      /* IR specific */
      if (experiment_type == "default" or
          (experiment_type == "streaking" and pulse_idx == 0))
      {
        cycles_delay[pulse_idx] =
            data["laser"]["pulses"][pulse_idx]["cycles_delay"];
      }
      /* XUV specific */
      else if (experiment_type == "streaking" and pulse_idx == 1)
      {
        tau_delay = data["laser"]["pulses"][pulse_idx]["tau_delay"];

        double center_XUV_cycles =
            energy[pulse_idx] *
            ((2 * pi *
              (cycles_delay[pulse_idx - 1] + cycles_on[pulse_idx - 1]) /
              energy[pulse_idx - 1]) +
             tau_delay) /
            (2 * pi);

        if (pulse_shape_idx[pulse_idx] == 0)
          cycles_delay[pulse_idx] = center_XUV_cycles - cycles_on[pulse_idx];
        else if (pulse_shape_idx[pulse_idx] == 1)
          cycles_delay[pulse_idx] =
              center_XUV_cycles - 6 * cycles_on[pulse_idx];
        else
          EndRun(
              "Streaking simulation: XUV does not have a valid "
              "pulse_shape");

        if (cycles_delay[pulse_idx] < 0)
        {
          cycles_delay[pulse_idx - 1] -= cycles_delay[pulse_idx] *
                                         energy[pulse_idx - 1] /
                                         energy[pulse_idx];
          cycles_delay[pulse_idx] = 0;
        }
      }
      /* transient absorption spectroscopy */
      else if (experiment_type == "transient")
      {
        EndRun("\ntransient absorption not supported yet\n");
      }
      else
      {
        EndRun("Unsupported experiment_type");
      }
    }
  }
  /* ensure input is good */
  Validate();

  if (world.rank() == 0)
  {
    std::cout << "Reading input complete\n" << std::flush;
  }
}

Parameters::~Parameters()
{
  for (PetscInt i = 0; i < num_nuclei; ++i)
  {
    delete location[i];
    delete a[i];
    delete b[i];
  }
  delete[] location;
  delete[] a;
  delete[] b;
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
  {
    delete polarization_vector[pulse_idx];
    if (num_dims == 3) delete poynting_vector[pulse_idx];
  }
  delete[] polarization_vector;
  if (num_dims == 3) delete[] poynting_vector;
}

/* checks important input parameters for errors */
void Parameters::Validate()
{
  if (world.rank() == 0)
  {
    std::cout << "Validating input\n";
  }
  std::string err_str;
  bool error_found = false; /* set to true if error is found */
  double total;

  /* coordinate system checks */
  if (coordinate_system_idx == -1)
  {
    error_found = true;
    err_str += "\nUnsupported coordinate system: ";
    err_str += coordinate_system;
    err_str += "\nSupported coordinate systems are:\n";
    err_str += "\"Cartesian\" and \"Cylindrical\"\n";
  }
  if (coordinate_system_idx == 1)
  {
    if (num_dims != 2)
    {
      error_found = true;
      err_str +=
          "\nCylindrical coordinate systems only supports 2D simulations\n";
    }
    if (num_electrons != 1)
    {
      error_found = true;
      err_str +=
          "\nCylindrical coordinate systems only supports single electron "
          "simulations\n";
    }
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
    {
      if (polarization_vector[pulse_idx][0] > 1e-14)
      {
        std::cout << polarization_vector[pulse_idx][0] << "\n";
        error_found = true;
        err_str +=
            "\nCylindrical coordinate systems only supports polarization "
            "vectors\npointing in the z direction (i.e. [0.0,1.0])\nPulse " +
            std::to_string(pulse_idx) + " does not meet this requirement\n";
      }
      if (ellipticity[pulse_idx] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nCylindrical coordinate systems only supports linear polarized "
            "light\nPulse " +
            std::to_string(pulse_idx) + " has a non zero ellipticity\n";
      }
    }
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (location[nuclei_idx][0] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nCylindrical coordinate systems only supports nuclei on the z "
            "axis (i.e. [0.0, z_value])\nNuclei " +
            std::to_string(nuclei_idx) + " has a non zero radial coordinate\n";
      }
    }
  }
  /* Check pulses */
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
  {
    /* Check pulse shapes */
    if (pulse_shape[pulse_idx] != "sin" and
        pulse_shape[pulse_idx] != "gaussian" and experiment_type != "File")
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has unsupported pulse shape: \"";
      err_str += pulse_shape[pulse_idx] + "\"\n";
      err_str += "Current support includes: ";
      err_str += "\"sin\" and \"gaussian\"\n";
    }

    /* check cycles_on */
    if (cycles_on[pulse_idx] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has cycles_on: \"";
      err_str += std::to_string(cycles_on[pulse_idx]) + "\"\n";
      err_str += "cycles_on should be >= 0\n";
    }

    /* check cycles_plateau */
    if (cycles_plateau[pulse_idx] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has cycles_plateau: \"";
      err_str += std::to_string(cycles_plateau[pulse_idx]) + "\"\n";
      err_str += "cycles_plateau should be >= 0\n";
    }

    /* check cycles_off */
    if (cycles_off[pulse_idx] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has cycles_off: \"";
      err_str += std::to_string(cycles_off[pulse_idx]) + "\"\n";
      err_str += "cycles_off should be >= 0\n";
    }

    /* exclude delay because it is zero anyways */
    /* pulses must exist so we don't run supper long time scales */
    double p_length = cycles_on[pulse_idx] + cycles_off[pulse_idx] +
                      cycles_plateau[pulse_idx];
    if (p_length <= 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has length: \"";
      err_str += std::to_string(p_length) + "\"\n";
      err_str += "the length should be > 0\n";
    }
    if (num_dims == 3)
    {
      total = 0.0;
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        total += polarization_vector[pulse_idx][dim_idx] *
                 poynting_vector[pulse_idx][dim_idx];
      }
      if (total > 1e-10)
      {
        error_found = true;
        err_str += "\nPulse ";
        err_str += std::to_string(pulse_idx);
        err_str += " has non orthogonal poynting_vector and";
        err_str += "polarization_vector\n";
        err_str += "current dot product is: ";
        err_str += std::to_string(total);
        err_str += "\n";
      }
    }
  }

  if (num_electrons > 3)
  {
    error_found = true;
    err_str +=
        "\nOne does not simply calculate more than 3 electrons exactly\n";
  }

  if (num_dims > 3)
  {
    error_found = true;
    err_str += "\nCome on we live in a 3D world!\n";
    err_str += "You provided more than 3 spacial dimensions\n";
  }

  /* state_solver issues*/
  if (state_solver == "ITP")
  {
    error_found = true;
    err_str += "\nInvalid state solver: \"";
    err_str += state_solver;
    err_str += "\"\nITP sucks, so I dropped support for it\n";
    err_str += "\nvalid solvers are \"File\", \"SLEPC\", and \"Power\"\n";
  }
  else if (state_solver != "Power" and state_solver != "SLEPC" and
           state_solver != "File")
  {
    error_found = true;
    err_str += "\nInvalid state solver: \"";
    err_str += state_solver;
    err_str += "\"\nvalid solvers are \"File\", \"SLEPC\", and \"Power\"\n";
  }

  if (start_state >= num_states)
  {
    error_found = true;
    err_str +=
        "\nThe start_state must be less than the total number of states you "
        "wish to calculate\n";
  }

  /* exit here to get all errors in one run */
  if (error_found)
  {
    EndRun(err_str);
  }
  else
  {
    if (world.rank() == 0)
    {
      std::cout << "Input valid\n";
    }
  }
}

/* getters */
double Parameters::GetDeltaT() { return delta_t; }

PetscInt Parameters::GetNumDims() { return num_dims; }

PetscInt Parameters::GetNumElectrons() { return num_electrons; }

PetscInt Parameters::GetCoordinateSystemIdx() { return coordinate_system_idx; }

PetscInt Parameters::GetRestart() { return restart; }

std::string Parameters::GetTarget() { return target; }

PetscInt Parameters::GetTargetIdx() { return target_idx; }

PetscInt Parameters::GetNumNuclei() { return num_nuclei; }

double** Parameters::GetLocation() { return location; }

double** Parameters::GetA() { return a; }

double** Parameters::GetB() { return b; }

double Parameters::GetAlpha() { return alpha; }

PetscInt Parameters::GetWriteFrequencyCheckpoint()
{
  return write_frequency_checkpoint;
}

PetscInt Parameters::GetWriteFrequencyObservables()
{
  return write_frequency_observables;
}

PetscInt Parameters::GetWriteFrequencyEigenState()
{
  return write_frequency_eigin_state;
}

double Parameters::GetGobbler() { return gobbler; }

PetscInt Parameters::GetOrder() { return order; }

double Parameters::GetSigma() { return sigma; }

PetscInt Parameters::GetNumStates() { return num_states; }
PetscInt Parameters::GetStartState() { return start_state; }

double Parameters::GetTol() { return tol; }

PetscInt Parameters::GetStateSolverIdx() { return state_solver_idx; }

std::string Parameters::GetStateSolver() { return state_solver; }

PetscInt Parameters::GetPropagate() { return propagate; }

PetscInt Parameters::GetFreePropagate() { return free_propagate; }

PetscInt Parameters::GetNumPulses() { return num_pulses; }

double** Parameters::GetPolarizationVector() { return polarization_vector; }

double** Parameters::GetPoyntingVector()
{
  if (num_dims == 3)
  {
    return poynting_vector;
  }
  else
  {
    return NULL;
  }
}
