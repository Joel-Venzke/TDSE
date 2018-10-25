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
  coordinate_system = data["coordinate_system"];

  if (coordinate_system == "Cartesian")
  {
    coordinate_system_idx = 0;
  }
  else if (coordinate_system == "Cylindrical")
  {
    coordinate_system_idx = 1;
  }
  else if (coordinate_system == "RBF")
  {
    coordinate_system_idx = 2;
  }
  else
  {
    coordinate_system_idx = -1;
  }

  delta_t       = data["delta_t"];
  num_electrons = data["num_electrons"];

  if (coordinate_system_idx == 2)
  {
    num_dims          = 3;
    dim_size          = std::make_unique< double[] >(num_dims);
    delta_x_min       = std::make_unique< double[] >(num_dims);
    delta_x_min_end   = std::make_unique< double[] >(num_dims);
    delta_x_max       = std::make_unique< double[] >(num_dims);
    delta_x_max_start = std::make_unique< double[] >(num_dims);

    for (PetscInt i = 0; i < num_dims; ++i)
    {
      dim_size[i]          = 0.0;
      delta_x_min[i]       = 0.0;
      delta_x_min_end[i]   = 0.0;
      delta_x_max[i]       = 0.0;
      delta_x_max_start[i] = 0.0;
    }

    gobbler = 0.0;
    order   = 0;
  }
  else
  {
    num_dims          = data["dimensions"].size();
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

    gobbler = data["gobbler"];
    order   = data["order"];
  }

  /* get simulation behavior */
  restart    = data["restart"];
  target     = data["target"]["name"];
  num_nuclei = data["target"]["nuclei"].size();
  alpha      = data["alpha"];
  if (num_electrons > 1)
    ee_soft_core = data["ee_soft_core"];
  else
    ee_soft_core = 0.0;
  write_frequency_checkpoint  = data["write_frequency_checkpoint"];
  write_frequency_observables = data["write_frequency_observables"];
  write_frequency_eigin_state = data["write_frequency_eigin_state"];
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

  gauge = data["gauge"];
  if (gauge == "Velocity")
  {
    gauge_idx = 0;
  }
  else if (gauge == "Length")
  {
    gauge_idx = 1;
  }
  else
  {
    gauge_idx = -1;
  }

  num_start_state = data["start_state"]["index"].size();
  if (data["start_state"]["amplitude"].size() != num_start_state)
  {
    EndRun(
        "Start state amplitude and index sizes do not match. Double check "
        "input file.");
  }
  if (data["start_state"]["phase"].size() != num_start_state)
  {
    EndRun(
        "Start state phase and index sizes do not match. Double check "
        "input file.");
  }
  start_state_idx       = new PetscInt[num_start_state];
  start_state_amplitude = new double[num_start_state];
  start_state_phase     = new double[num_start_state];
  for (PetscInt i = 0; i < num_start_state; i++)
  {
    start_state_idx[i]       = data["start_state"]["index"][i];
    start_state_amplitude[i] = data["start_state"]["amplitude"][i];
    start_state_phase[i]     = data["start_state"]["phase"][i];
  }

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

  z                      = std::make_unique< double[] >(num_nuclei);
  exponential_size       = std::make_unique< PetscInt[] >(num_nuclei);
  exponential_r_0        = new double*[num_nuclei];
  exponential_amplitude  = new double*[num_nuclei];
  exponential_decay_rate = new double*[num_nuclei];
  gaussian_size          = std::make_unique< PetscInt[] >(num_nuclei);
  gaussian_r_0           = new double*[num_nuclei];
  gaussian_amplitude     = new double*[num_nuclei];
  gaussian_decay_rate    = new double*[num_nuclei];
  yukawa_size            = std::make_unique< PetscInt[] >(num_nuclei);
  yukawa_r_0             = new double*[num_nuclei];
  yukawa_amplitude       = new double*[num_nuclei];
  yukawa_decay_rate      = new double*[num_nuclei];
  location               = new double*[num_nuclei];
  for (PetscInt i = 0; i < num_nuclei; ++i)
  {
    /* Coulomb term */
    z[i] = data["target"]["nuclei"][i]["z"];

    /* Gaussian Donuts */
    gaussian_size[i]       = data["target"]["nuclei"][i]["gaussian_r_0"].size();
    gaussian_r_0[i]        = new double[gaussian_size[i]];
    gaussian_amplitude[i]  = new double[gaussian_size[i]];
    gaussian_decay_rate[i] = new double[gaussian_size[i]];
    if (data["target"]["nuclei"][i]["gaussian_r_0"].size() !=
            data["target"]["nuclei"][i]["gaussian_amplitude"].size() or
        data["target"]["nuclei"][i]["gaussian_r_0"].size() !=
            data["target"]["nuclei"][i]["gaussian_decay_rate"].size())
    {
      EndRun("Nuclei " + std::to_string(i) +
             " all Gaussian terms must have the same size");
    }
    for (PetscInt j = 0; j < gaussian_size[i]; ++j)
    {
      gaussian_r_0[i][j] = data["target"]["nuclei"][i]["gaussian_r_0"][j];
      gaussian_amplitude[i][j] =
          data["target"]["nuclei"][i]["gaussian_amplitude"][j];
      gaussian_decay_rate[i][j] =
          data["target"]["nuclei"][i]["gaussian_decay_rate"][j];
    }

    /* exponential Donuts */
    exponential_size[i] = data["target"]["nuclei"][i]["exponential_r_0"].size();
    exponential_r_0[i]  = new double[exponential_size[i]];
    exponential_amplitude[i]  = new double[exponential_size[i]];
    exponential_decay_rate[i] = new double[exponential_size[i]];
    if (data["target"]["nuclei"][i]["exponential_r_0"].size() !=
            data["target"]["nuclei"][i]["exponential_amplitude"].size() or
        data["target"]["nuclei"][i]["exponential_r_0"].size() !=
            data["target"]["nuclei"][i]["exponential_decay_rate"].size())
    {
      EndRun("Nuclei " + std::to_string(i) +
             " all exponential terms must have the same size");
    }
    for (PetscInt j = 0; j < exponential_size[i]; ++j)
    {
      exponential_r_0[i][j] = data["target"]["nuclei"][i]["exponential_r_0"][j];
      exponential_amplitude[i][j] =
          data["target"]["nuclei"][i]["exponential_amplitude"][j];
      exponential_decay_rate[i][j] =
          data["target"]["nuclei"][i]["exponential_decay_rate"][j];
    }

    /* yukawa Donuts */
    yukawa_size[i]       = data["target"]["nuclei"][i]["yukawa_r_0"].size();
    yukawa_r_0[i]        = new double[yukawa_size[i]];
    yukawa_amplitude[i]  = new double[yukawa_size[i]];
    yukawa_decay_rate[i] = new double[yukawa_size[i]];
    if (data["target"]["nuclei"][i]["yukawa_r_0"].size() !=
            data["target"]["nuclei"][i]["yukawa_amplitude"].size() or
        data["target"]["nuclei"][i]["yukawa_r_0"].size() !=
            data["target"]["nuclei"][i]["yukawa_decay_rate"].size())
    {
      EndRun("Nuclei " + std::to_string(i) +
             " all yukawa terms must have the same size");
    }
    for (PetscInt j = 0; j < yukawa_size[i]; ++j)
    {
      yukawa_r_0[i][j] = data["target"]["nuclei"][i]["yukawa_r_0"][j];
      yukawa_amplitude[i][j] =
          data["target"]["nuclei"][i]["yukawa_amplitude"][j];
      yukawa_decay_rate[i][j] =
          data["target"]["nuclei"][i]["yukawa_decay_rate"][j];
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

  field_max_states = data["field_max_states"];

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
  gaussian_length     = new double[num_pulses];
  polarization_vector = new double*[num_pulses];
  if (num_dims == 3) poynting_vector = new double*[num_pulses];

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
      gaussian_length[pulse_idx] = 5.0;

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
      /* set to zero for IO reasons and to ensure it is not used if it is not
       * read from the input file */
      power_on[pulse_idx]        = 0.0;
      power_off[pulse_idx]       = 0.0;
      gaussian_length[pulse_idx] = 1.0; /* the factor for non Gaussian pulses */
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
        gaussian_length[pulse_idx] =
            data["laser"]["pulses"][pulse_idx]["gaussian_length"];
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
          std::sqrt(intensity / 3.51e16) * c /
          (energy[pulse_idx] *
           std::sqrt(1 + ellipticity[pulse_idx] * ellipticity[pulse_idx]));

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
              (cycles_delay[pulse_idx - 1] +
               gaussian_length[pulse_idx - 1] * cycles_on[pulse_idx - 1]) /
              energy[pulse_idx - 1]) +
             tau_delay) /
            (2 * pi);

        cycles_delay[pulse_idx] =
            center_XUV_cycles -
            gaussian_length[pulse_idx] * cycles_on[pulse_idx];

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
    delete gaussian_r_0[i];
    delete gaussian_amplitude[i];
    delete gaussian_decay_rate[i];
    delete exponential_r_0[i];
    delete exponential_amplitude[i];
    delete exponential_decay_rate[i];
    delete yukawa_r_0[i];
    delete yukawa_amplitude[i];
    delete yukawa_decay_rate[i];
  }
  delete[] location;
  delete[] gaussian_r_0;
  delete[] gaussian_amplitude;
  delete[] gaussian_decay_rate;
  delete[] exponential_r_0;
  delete[] exponential_amplitude;
  delete[] exponential_decay_rate;
  delete[] yukawa_r_0;
  delete[] yukawa_amplitude;
  delete[] yukawa_decay_rate;
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
  {
    delete polarization_vector[pulse_idx];
    if (num_dims == 3) delete poynting_vector[pulse_idx];
  }
  delete[] polarization_vector;
  if (num_dims == 3) delete[] poynting_vector;
  delete gaussian_length;
  delete start_state_idx;        ///< index of states in super position
  delete start_state_amplitude;  ///< amplitude of states in super position
  delete start_state_phase;
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
    err_str += "\"RBF\", \"Cartesian\", and \"Cylindrical\"\n";
  }
  if (coordinate_system_idx == 2 and num_electrons != 1)
  {
    error_found = true;
    err_str += "\nRBF only supports 1 electron\n";
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

  if (field_max_states != 0 and propagate == 1)
  {
    error_found = true;
    err_str += "\nYou're not allowed to propagate a field max state.\n";
    err_str += "Set propagate or field_max_states to 0\n";
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

  if (num_electrons > 2)
  {
    error_found = true;
    err_str +=
        "\nOne does not simply calculate more than 2 electrons exactly\n";
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

  if (gauge_idx == -1)
  {
    error_found = true;
    err_str += "\nInvalid gauge: \"";
    err_str += gauge;
    err_str += "\"\nvalid gauges are \"Velocity\", and \"Length\"\n";
  }

  for (int idx = 0; idx < num_start_state; ++idx)
  {
    if (start_state_idx[idx] >= num_states)
    {
      error_found = true;
      err_str +=
          "\nThe start_state must be less than the total number of states you "
          "wish to calculate\n";
    }
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

double** Parameters::GetExponentialR0() { return exponential_r_0; }

double** Parameters::GetExponentialAmplitude() { return exponential_amplitude; }

double** Parameters::GetExponentialDecayRate()
{
  return exponential_decay_rate;
}

double** Parameters::GetGaussianR0() { return gaussian_r_0; }

double** Parameters::GetGaussianAmplitude() { return gaussian_amplitude; }

double** Parameters::GetGaussianDecayRate() { return gaussian_decay_rate; }

double** Parameters::GetYukawaR0() { return yukawa_r_0; }

double** Parameters::GetYukawaAmplitude() { return yukawa_amplitude; }

double** Parameters::GetYukawaDecayRate() { return yukawa_decay_rate; }

double Parameters::GetAlpha() { return alpha; }

double Parameters::GetEESoftCore() { return ee_soft_core; }

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

PetscInt Parameters::GetNumStartState() { return num_start_state; }

PetscInt* Parameters::GetStartStateIdx() { return start_state_idx; }

double* Parameters::GetStartStateAmplitude() { return start_state_amplitude; }

double* Parameters::GetStartStatePhase() { return start_state_phase; }

double Parameters::GetTol() { return tol; }

PetscInt Parameters::GetStateSolverIdx() { return state_solver_idx; }

std::string Parameters::GetStateSolver() { return state_solver; }

PetscInt Parameters::GetGaugeIdx() { return gauge_idx; }

PetscInt Parameters::GetPropagate() { return propagate; }

PetscInt Parameters::GetFreePropagate() { return free_propagate; }

PetscInt Parameters::GetFieldMaxStates() { return field_max_states; }

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

double* Parameters::GetGaussianLength() { return gaussian_length; }
