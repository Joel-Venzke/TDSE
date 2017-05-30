#include "Parameters.h"

/*
 * Reads file and returns a string
 */
std::string FileToString(std::string file_name)
{
  // TODO(jove7731): check if file exists
  std::ifstream t(file_name);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  return str;
}

/*
 * Reads file and returns a json object
 */
json FileToJson(std::string file_name)
{
  auto j = json::parse(FileToString(file_name));
  return j;
}

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

  /* read data from file */
  json data = FileToJson(file_name);

  /* get numeric information */
  delta_t       = data["delta_t"];
  num_dims      = data["dimensions"].size();
  num_electrons = data["num_electrons"];
  dim_size      = std::make_unique<double[]>(num_dims);
  delta_x       = std::make_unique<double[]>(num_dims);
  delta_x_2     = std::make_unique<double[]>(num_dims);

  for (PetscInt i = 0; i < num_dims; ++i)
  {
    dim_size[i]  = data["dimensions"][i]["dim_size"];
    delta_x[i]   = data["dimensions"][i]["delta_x"];
    delta_x_2[i] = delta_x[i] * delta_x[i];
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
  sigma                       = data["sigma"];
  tol                         = data["tol"];
  state_solver                = data["state_solver"];
  num_states                  = data["states"].size();

  state_energy = std::make_unique<double[]>(num_states);

  for (PetscInt i = 0; i < num_states; i++)
  {
    state_energy[i] = data["states"][i]["energy"];
  }

  z        = std::make_unique<double[]>(num_nuclei);
  z_c      = std::make_unique<double[]>(num_nuclei);
  c0       = std::make_unique<double[]>(num_nuclei);
  r0       = std::make_unique<double[]>(num_nuclei);
  sae_size = std::make_unique<PetscInt[]>(num_nuclei);
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

  propagate      = data["propagate"];
  free_propagate = data["free_propagate"];

  /* get pulse information */
  num_pulses = data["laser"]["pulses"].size();

  /* allocate memory */
  pulse_shape         = std::make_unique<std::string[]>(num_pulses);
  pulse_shape_idx     = std::make_unique<PetscInt[]>(num_pulses);
  cycles_on           = std::make_unique<double[]>(num_pulses);
  cycles_plateau      = std::make_unique<double[]>(num_pulses);
  cycles_off          = std::make_unique<double[]>(num_pulses);
  cycles_delay        = std::make_unique<double[]>(num_pulses);
  cep                 = std::make_unique<double[]>(num_pulses);
  energy              = std::make_unique<double[]>(num_pulses);
  field_max           = std::make_unique<double[]>(num_pulses);
  ellipticity         = std::make_unique<double[]>(num_pulses);
  helicity            = std::make_unique<std::string[]>(num_pulses);
  helicity_idx        = std::make_unique<PetscInt[]>(num_pulses);
  polarization_vector = new double*[num_pulses];
  if (num_dims == 3) poynting_vector = new double*[num_pulses];

  /* read data */
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
  {
    pulse_shape[pulse_idx] = data["laser"]["pulses"][pulse_idx]["pulse_shape"];

    /* index used similar target_idx */
    if (pulse_shape[pulse_idx] == "sin2")
    {
      pulse_shape_idx[pulse_idx] = 0;
    }
    else if (pulse_shape[pulse_idx] == "linear")
    {
      pulse_shape_idx[pulse_idx] = 1;
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
    polar_norm                       = sqrt(polar_norm);
    if (num_dims == 3) poynting_norm = sqrt(poynting_norm);
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      polarization_vector[pulse_idx][dim_idx] /= polar_norm;
      if (num_dims == 3 and poynting_norm > 1e-10)
      {
        poynting_vector[pulse_idx][dim_idx] /= poynting_norm;
      }
    }

    cycles_on[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_on"];
    cycles_plateau[pulse_idx] =
        data["laser"]["pulses"][pulse_idx]["cycles_plateau"];
    cycles_off[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_off"];
    cycles_delay[pulse_idx] =
        data["laser"]["pulses"][pulse_idx]["cycles_delay"];
    cep[pulse_idx]         = data["laser"]["pulses"][pulse_idx]["cep"];
    energy[pulse_idx]      = data["laser"]["pulses"][pulse_idx]["energy"];
    field_max[pulse_idx]   = data["laser"]["pulses"][pulse_idx]["field_max"];
    ellipticity[pulse_idx] = data["laser"]["pulses"][pulse_idx]["ellipticity"];
    helicity[pulse_idx]    = data["laser"]["pulses"][pulse_idx]["helicity"];

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

  /* Check pulses */
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
  {
    /* Check pulse shapes */
    if (pulse_shape[pulse_idx] != "sin2")
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has unsupported pulse shape: \"";
      err_str += pulse_shape[pulse_idx] + "\"\n";
      err_str += "Current support includes: ";
      err_str += "\"sin2\"\n";
    }

    /* check field_max */
    if (field_max[pulse_idx] <= 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(pulse_idx);
      err_str += " has unsupported field_max: \"";
      err_str += std::to_string(field_max[pulse_idx]) + "\"\n";
      err_str += "field_max should be greater than 0\n";
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
  if (state_solver == "file")
  {
    error_found = true;
    err_str += "\nInvalid state solver: \"";
    err_str += "States from file is not supported yet\"";
  }
  else if (state_solver != "ITP" && state_solver != "Power")
  {
    error_found = true;
    err_str += "\nInvalid state solver: \"";
    err_str += state_solver;
    err_str += "\"\nvalid solvers are \"ITP\" and \"Power\"\n";
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

double Parameters::GetSigma() { return sigma; }

PetscInt Parameters::GetNumStates() { return num_states; }

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
