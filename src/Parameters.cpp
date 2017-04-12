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

/* prints error message, kills code and returns -1 */
void Parameters::EndRun(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  exit(-1);
}

/* prints error message, kills code and returns exit_val */
void Parameters::EndRun(std::string str, int exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  exit(exit_val);
}

/* Constructor */
Parameters::Parameters(std::string file_name) { Setup(file_name); }

/* basically the default constructor */
void Parameters::Setup(std::string file_name)
{
  if (world.rank() == 0)
  {
    std::cout << "Reading input file: " << file_name << "\n" << std::flush;
  }

  /* read data from file */
  json data = FileToJson(file_name);

  /* get numeric information */
  delta_t       = data["delta_t"];
  num_dims      = data["dimensions"].size();
  num_electrons = data["num_electrons"];
  dim_size      = std::make_unique<double[]>(num_dims);
  delta_x       = std::make_unique<double[]>(num_dims);

  for (int i = 0; i < num_dims; ++i)
  {
    dim_size[i] = data["dimensions"][i]["dim_size"];
    delta_x[i]  = data["dimensions"][i]["delta_x"];
  }

  /* get simulation behavior */
  restart                     = data["restart"];
  target                      = data["target"];
  alpha                       = data["alpha"];
  write_frequency_propagation = data["write_frequency_propagation"];
  write_frequency_eigin_state = data["write_frequency_eigin_state"];
  gobbler                     = data["gobbler"];
  sigma                       = data["sigma"];
  tol                         = data["tol"];
  state_solver                = data["state_solver"];
  num_states                  = data["states"].size();

  this->state_energy = std::make_unique<double[]>(num_states);

  for (int i = 0; i < num_states; i++)
  {
    state_energy[i] = data["states"][i]["energy"];
  }

  /* index is used throughout code for efficiency */
  /* and ease of writing to hdf5 */
  if (target == "He")
  {
    target_idx = 0;
    z          = 2.0; /* He atomic number */
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

  propagate = data["propagate"];

  /* get pulse information */
  num_pulses = data["pulses"].size();

  /* allocate memory */
  pulse_shape     = std::make_unique<std::string[]>(num_pulses);
  pulse_shape_idx = std::make_unique<int[]>(num_pulses);
  cycles_on       = std::make_unique<double[]>(num_pulses);
  cycles_plateau  = std::make_unique<double[]>(num_pulses);
  cycles_off      = std::make_unique<double[]>(num_pulses);
  cycles_delay    = std::make_unique<double[]>(num_pulses);
  cep             = std::make_unique<double[]>(num_pulses);
  energy          = std::make_unique<double[]>(num_pulses);
  field_max       = std::make_unique<double[]>(num_pulses);

  /* read data */
  for (int i = 0; i < num_pulses; ++i)
  {
    pulse_shape[i] = data["pulses"][i]["pulse_shape"];

    /* index used similar target_idx */
    if (pulse_shape[i] == "sin2")
    {
      pulse_shape_idx[i] = 0;
    }
    else if (pulse_shape[i] == "linear")
    {
      pulse_shape_idx[i] = 1;
    }

    cycles_on[i]      = data["pulses"][i]["cycles_on"];
    cycles_plateau[i] = data["pulses"][i]["cycles_plateau"];
    cycles_off[i]     = data["pulses"][i]["cycles_off"];
    cycles_delay[i]   = data["pulses"][i]["cycles_delay"];
    cep[i]            = data["pulses"][i]["cep"];
    energy[i]         = data["pulses"][i]["energy"];
    field_max[i]      = data["pulses"][i]["field_max"];
  }

  /* ensure input is good */
  if (world.rank() == 0)
  {
    Validate();
  }

  if (world.rank() == 0)
  {
    std::cout << "Reading input complete\n" << std::flush;
  }
}

Parameters::~Parameters() {}

/* checks important input parameters for errors */
void Parameters::Validate()
{
  std::cout << "Validating input\n";
  std::string err_str;
  bool error_found = false; /* set to true if error is found */

  /* Check pulses */
  for (int i = 0; i < num_pulses; i++)
  {
    /* Check pulse shapes */
    if (pulse_shape[i] != "sin2")
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has unsupported pulse shape: \"";
      err_str += pulse_shape[i] + "\"\n";
      err_str += "Current support includes: ";
      err_str += "\"sin2\"\n";
    }

    /* check field_max */
    if (field_max[i] <= 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has unsupported field_max: \"";
      err_str += std::to_string(field_max[i]) + "\"\n";
      err_str += "field_max should be greater than 0\n";
    }

    /* check cycles_on */
    if (cycles_on[i] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has cycles_on: \"";
      err_str += std::to_string(cycles_on[i]) + "\"\n";
      err_str += "cycles_on should be >= 0\n";
    }

    /* check cycles_plateau */
    if (cycles_plateau[i] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has cycles_plateau: \"";
      err_str += std::to_string(cycles_plateau[i]) + "\"\n";
      err_str += "cycles_plateau should be >= 0\n";
    }

    /* check cycles_off */
    if (cycles_off[i] < 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has cycles_off: \"";
      err_str += std::to_string(cycles_off[i]) + "\"\n";
      err_str += "cycles_off should be >= 0\n";
    }

    /* exclude delay because it is zero anyways */
    /* pulses must exist so we don't run supper long time scales */
    double p_length = cycles_on[i] + cycles_off[i] + cycles_plateau[i];
    if (p_length <= 0)
    {
      error_found = true;
      err_str += "\nPulse ";
      err_str += std::to_string(i);
      err_str += " has length: \"";
      err_str += std::to_string(p_length) + "\"\n";
      err_str += "the length should be > 0\n";
    }
  }

  if (num_dims > 1 and num_electrons == 2)
  {
    error_found = true;
    err_str += "\n2 active electron calculations only work in 1D \"";
  }

  if (num_electrons > 2)
  {
    error_found = true;
    err_str +=
        "\nOne does not simply calculate more than 2 electrons exactly \"";
  }

  if (num_electrons != 2)
  {
    error_found = true;
    err_str += "\nOnly 2 electron simulation are supported currently\"";
  }

  /* target */
  if (target != "He")
  {
    error_found = true;
    err_str += "\nInvalid target: \"";
    err_str += target;
    err_str += "\" \nvalid targets are \"He\"";
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
    std::cout << "Input valid\n";
  }
}

/* getters */
double Parameters::GetDeltaT() { return delta_t; }

int Parameters::GetNumDims() { return num_dims; }

int Parameters::GetNumElectrons() { return num_electrons; }

int Parameters::GetRestart() { return restart; }

std::string Parameters::GetTarget() { return target; }

int Parameters::GetTargetIdx() { return target_idx; }

double Parameters::GetZ() { return z; }

double Parameters::GetAlpha() { return alpha; }

int Parameters::GetWriteFrequencyPropagation()
{
  return write_frequency_propagation;
}

int Parameters::GetWriteFrequencyEigenState()
{
  return write_frequency_eigin_state;
}

double Parameters::GetGobbler() { return gobbler; }

double Parameters::GetSigma() { return sigma; }

int Parameters::GetNumStates() { return num_states; }

double Parameters::GetTol() { return tol; }

int Parameters::GetStateSolverIdx() { return state_solver_idx; }

std::string Parameters::GetStateSolver() { return state_solver; }

int Parameters::GetPropagate() { return propagate; }

int Parameters::GetNumPulses() { return num_pulses; }