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

  /* read data from file */
  json data = FileToJson(file_name);

  /* get numeric information */
  CheckParameter(data["coordinate_system"].size(), "coordinate_system");
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
  else if (coordinate_system == "Spherical")
  {
    coordinate_system_idx = 3;
  }
  else if (coordinate_system == "HypersphericalRRC")
  {
    coordinate_system_idx = 4;
  }
  else if (coordinate_system == "Hyperspherical")
  {
    coordinate_system_idx = 5;
  }
  else
  {
    coordinate_system_idx = -1;
  }
  ReadData(data);

  /* ensure input is good */
  Validate();

  if (world.rank() == 0)
  {
    std::cout << "Reading input complete\n" << std::flush;
  }
}

void Parameters::ReadGridCylindrical(json data)
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

void Parameters::ReadGridSpherical(json data)
{
  CheckParameter(data["dimensions"].size(), "dimensions");
  if (data["dimensions"].size() != 1)
  {
    EndRun(
        "The Spherical code take 1 dimension parameter with 'l_max' and "
        "'m_max' as additional parameter");
  }
  num_dims          = 3;
  dim_size          = std::make_unique< double[] >(num_dims);
  delta_x_min       = std::make_unique< double[] >(num_dims);
  delta_x_min_end   = std::make_unique< double[] >(num_dims);
  delta_x_max       = std::make_unique< double[] >(num_dims);
  delta_x_max_start = std::make_unique< double[] >(num_dims);

  /* The last dimension (diagonal of matrix) is r and we use Finite
   * difference for this */
  int dim_input_idx = 0;
  int dim_idx       = 2;
  CheckParameter(data["dimensions"][dim_input_idx]["dim_size"].size(),
                 "dimensions - dim_size");
  dim_size[dim_idx] = data["dimensions"][dim_input_idx]["dim_size"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min"].size(),
                 "dimensions - delta_x_min");
  delta_x_min[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_min"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min_end"].size(),
                 "dimensions - delta_x_min_end");
  delta_x_min_end[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_min_end"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max"].size(),
                 "dimensions - delta_x_max");
  delta_x_max[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_max"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max_start"].size(),
                 "dimensions - delta_x_max_start");
  delta_x_max_start[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_max_start"];

  /* The next two dimensions are expanded in spherical harmonics with indexes
   * l and m */
  for (dim_idx = 0; dim_idx < 2; ++dim_idx)
  {
    delta_x_min[dim_idx]     = 1.0;
    delta_x_min_end[dim_idx] = 0.0;
    delta_x_max[dim_idx]     = 1.0;
  }

  /* get m_max note m goes from -m to m so 2m+1 terms*/

  dim_size[0] = 1;

  /* get l_max */
  CheckParameter(data["dimensions"][0]["l_max"].size(), "dimensions - l_max");
  l_max = data["dimensions"][0]["l_max"];

  /* get m_max */
  CheckParameter(data["dimensions"][0]["m_max"].size(), "dimensions - m_max");
  m_max = data["dimensions"][0]["m_max"];

  /* this dimension combines l and m to avoid tensor grid issues */
  dim_size[1] = GetIdxFromLM(l_max, m_max, m_max) + 1;

  CheckParameter(data["gobbler"].size(), "gobbler");
  gobbler = data["gobbler"];

  CheckParameter(data["order"].size(), "order");
  order = data["order"];
}

void Parameters::ReadGridHypersphericalRRC(json data)
{
  CheckParameter(data["dimensions"].size(), "dimensions");
  if (data["dimensions"].size() != 1)
  {
    EndRun(
        "The Hyperspherical code take 1 dimension parameter with 'k_max', "
        "'l_max' and 'm_max' as additional parameter");
  }
  /* The number of electrons is set to one since the 6D space is not
   *  a tensor product of 2 - 3D spaces. If once codes up the bi-spherical
   * code then this tensor product will be useful
   */
  num_electrons     = 1;
  num_dims          = 3;
  dim_size          = std::make_unique< double[] >(num_dims);
  delta_x_min       = std::make_unique< double[] >(num_dims);
  delta_x_min_end   = std::make_unique< double[] >(num_dims);
  delta_x_max       = std::make_unique< double[] >(num_dims);
  delta_x_max_start = std::make_unique< double[] >(num_dims);

  /* The last dimension (diagonal of matrix) is r and we use Finite
   * difference for this */
  int dim_input_idx = 0;
  int dim_idx       = 2;
  CheckParameter(data["dimensions"][dim_input_idx]["dim_size"].size(),
                 "dimensions - dim_size");
  dim_size[dim_idx] = data["dimensions"][dim_input_idx]["dim_size"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min"].size(),
                 "dimensions - delta_x_min");
  delta_x_min[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_min"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min_end"].size(),
                 "dimensions - delta_x_min_end");
  delta_x_min_end[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_min_end"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max"].size(),
                 "dimensions - delta_x_max");
  delta_x_max[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_max"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max_start"].size(),
                 "dimensions - delta_x_max_start");
  delta_x_max_start[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_max_start"];

  /* avoiding issues with lack of tensor product */
  for (dim_idx = 0; dim_idx < 2; ++dim_idx)
  {
    delta_x_min[dim_idx]     = 1.0;
    delta_x_min_end[dim_idx] = 0.0;
    delta_x_max[dim_idx]     = 1.0;
  }

  /* dim 0 is set to 1 since hyperspherical harmonics cannot be written
   *  as a tensor product
   */
  dim_size[0] = 1;

  /* get k_max */
  CheckParameter(data["dimensions"][0]["k_max"].size(), "dimensions - k_max");
  k_max = data["dimensions"][0]["k_max"];

  /* get l_max */
  CheckParameter(data["dimensions"][0]["l_max"].size(), "dimensions - l_max");
  l_max = data["dimensions"][0]["l_max"];

  /* get m_max */
  CheckParameter(data["dimensions"][0]["m_max"].size(), "dimensions - m_max");
  m_max = data["dimensions"][0]["m_max"];

  /* this dimension combines l and m to avoid tensor grid issues */
  dim_size[1] = GetHypersphereSizeRRC(k_max, l_max);

  CheckParameter(data["gobbler"].size(), "gobbler");
  gobbler = data["gobbler"];

  CheckParameter(data["order"].size(), "order");
  order = data["order"];
}

void Parameters::ReadGridHyperspherical(json data)
{
  CheckParameter(data["dimensions"].size(), "dimensions");
  if (data["dimensions"].size() != 1)
  {
    EndRun(
        "The Hyperspherical code take 1 dimension parameter with 'k_max', "
        "'l_max' and 'm_max' as additional parameter");
  }
  /* The number of electrons is set to one since the 6D space is not
   *  a tensor product of 2 - 3D spaces. If once codes up the bi-spherical
   * code then this tensor product will be useful
   */
  num_electrons     = 1;
  num_dims          = 3;
  dim_size          = std::make_unique< double[] >(num_dims);
  delta_x_min       = std::make_unique< double[] >(num_dims);
  delta_x_min_end   = std::make_unique< double[] >(num_dims);
  delta_x_max       = std::make_unique< double[] >(num_dims);
  delta_x_max_start = std::make_unique< double[] >(num_dims);

  /* The last dimension (diagonal of matrix) is r and we use Finite
   * difference for this */
  int dim_input_idx = 0;
  int dim_idx       = 2;
  CheckParameter(data["dimensions"][dim_input_idx]["dim_size"].size(),
                 "dimensions - dim_size");
  dim_size[dim_idx] = data["dimensions"][dim_input_idx]["dim_size"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min"].size(),
                 "dimensions - delta_x_min");
  delta_x_min[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_min"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_min_end"].size(),
                 "dimensions - delta_x_min_end");
  delta_x_min_end[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_min_end"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max"].size(),
                 "dimensions - delta_x_max");
  delta_x_max[dim_idx] = data["dimensions"][dim_input_idx]["delta_x_max"];

  CheckParameter(data["dimensions"][dim_input_idx]["delta_x_max_start"].size(),
                 "dimensions - delta_x_max_start");
  delta_x_max_start[dim_idx] =
      data["dimensions"][dim_input_idx]["delta_x_max_start"];

  /* avoiding issues with lack of tensor product */
  for (dim_idx = 0; dim_idx < 2; ++dim_idx)
  {
    delta_x_min[dim_idx]     = 1.0;
    delta_x_min_end[dim_idx] = 0.0;
    delta_x_max[dim_idx]     = 1.0;
  }

  /* dim 0 is set to 1 since hyperspherical harmonics cannot be written
   *  as a tensor product
   */
  dim_size[0] = 1;

  /* get k_max */
  CheckParameter(data["dimensions"][0]["k_max"].size(), "dimensions - k_max");
  k_max = data["dimensions"][0]["k_max"];

  /* get l_max */
  CheckParameter(data["dimensions"][0]["l_max"].size(), "dimensions - l_max");
  l_max = data["dimensions"][0]["l_max"];

  /* get m_max */
  CheckParameter(data["dimensions"][0]["m_max"].size(), "dimensions - m_max");
  m_max = data["dimensions"][0]["m_max"];

  /* this dimension combines l and m to avoid tensor grid issues */
  dim_size[1] = GetHypersphereSizeNonRRC(k_max, l_max);

  CheckParameter(data["gobbler"].size(), "gobbler");
  gobbler = data["gobbler"];

  CheckParameter(data["order"].size(), "order");
  order = data["order"];
}

void Parameters::ReadGridNormal(json data)
{
  CheckParameter(data["dimensions"].size(), "dimensions");
  num_dims          = data["dimensions"].size();
  dim_size          = std::make_unique< double[] >(num_dims);
  delta_x_min       = std::make_unique< double[] >(num_dims);
  delta_x_min_end   = std::make_unique< double[] >(num_dims);
  delta_x_max       = std::make_unique< double[] >(num_dims);
  delta_x_max_start = std::make_unique< double[] >(num_dims);

  for (PetscInt i = 0; i < num_dims; ++i)
  {
    CheckParameter(data["dimensions"][i]["dim_size"].size(),
                   "dimensions - dim_size");
    dim_size[i] = data["dimensions"][i]["dim_size"];

    CheckParameter(data["dimensions"][i]["delta_x_min"].size(),
                   "dimensions - delta_x_min");
    delta_x_min[i] = data["dimensions"][i]["delta_x_min"];

    CheckParameter(data["dimensions"][i]["delta_x_min_end"].size(),
                   "dimensions - delta_x_min_end");
    delta_x_min_end[i] = data["dimensions"][i]["delta_x_min_end"];

    CheckParameter(data["dimensions"][i]["delta_x_max"].size(),
                   "dimensions - delta_x_max");
    delta_x_max[i] = data["dimensions"][i]["delta_x_max"];

    CheckParameter(data["dimensions"][i]["delta_x_max_start"].size(),
                   "dimensions - delta_x_max_start");
    delta_x_max_start[i] = data["dimensions"][i]["delta_x_max_start"];
  }

  CheckParameter(data["gobbler"].size(), "gobbler");
  gobbler = data["gobbler"];

  CheckParameter(data["order"].size(), "order");
  order = data["order"];
}

void Parameters::ReadNumerics(json data)
{
  CheckParameter(data["delta_t"].size(), "delta_t");
  delta_t = data["delta_t"];

  CheckParameter(data["num_electrons"].size(), "num_electrons");
  num_electrons = data["num_electrons"];

  if (coordinate_system_idx == 2)
  {
    ReadGridCylindrical(data);
  }
  else if (coordinate_system_idx == 3)
  {
    ReadGridSpherical(data);
  }
  else if (coordinate_system_idx == 4)
  {
    ReadGridHypersphericalRRC(data);
  }
  else if (coordinate_system_idx == 5)
  {
    ReadGridHyperspherical(data);
  }
  else
  {
    ReadGridNormal(data);
  }
}

void Parameters::ReadSolver(json data)
{
  CheckParameter(data["state_solver"].size(), "state_solver");
  state_solver = data["state_solver"];
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
  else
  {
    state_solver_idx = -1;
  }

  if (state_solver_idx != 2)
  {
    CheckParameter(data["states"].size(), "states");
    num_states = data["states"];
  }
  else
  {
    CheckParameter(data["states"].size(), "states");
    num_states = data["states"].size();

    state_energy = std::make_unique< double[] >(num_states);

    for (PetscInt i = 0; i < num_states; i++)
    {
      CheckParameter(data["states"][i]["energy"].size(), "states - energy");
      state_energy[i] = data["states"][i]["energy"];
    }
  }
}

void Parameters::ReadExponentialPot(json data, PetscInt i)
{
  /* exponential Donuts */
  exponential_size[i] = data["target"]["nuclei"][i]["exponential_r_0"].size();
  exponential_r_0[i]  = new double[exponential_size[i]];
  exponential_amplitude[i]  = new double[exponential_size[i]];
  exponential_decay_rate[i] = new double[exponential_size[i]];
  if (exponential_size[i] == 0 and world.rank() == 0)
  {
    std::cout << "WARNING: No exponential potential for nuclei " << i << "\n";
  }
  else if (data["target"]["nuclei"][i]["exponential_r_0"].size() !=
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
}

void Parameters::ReadGaussianPot(json data, PetscInt i)
{ /* Gaussian Donuts */
  gaussian_size[i]       = data["target"]["nuclei"][i]["gaussian_r_0"].size();
  gaussian_r_0[i]        = new double[gaussian_size[i]];
  gaussian_amplitude[i]  = new double[gaussian_size[i]];
  gaussian_decay_rate[i] = new double[gaussian_size[i]];
  if (gaussian_size[i] == 0 and world.rank() == 0)
  {
    std::cout << "WARNING: No Gaussian potential for nuclei " << i << "\n";
  }
  else if (data["target"]["nuclei"][i]["gaussian_r_0"].size() !=
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
}

void Parameters::ReadSquareWellPot(json data, PetscInt i)
{ /* Square well Donuts */
  square_well_size[i] = data["target"]["nuclei"][i]["square_well_r_0"].size();
  square_well_r_0[i]  = new double[square_well_size[i]];
  square_well_amplitude[i] = new double[square_well_size[i]];
  square_well_width[i]     = new double[square_well_size[i]];
  if (square_well_size[i] == 0 and world.rank() == 0)
  {
    std::cout << "WARNING: No square well potential for nuclei " << i << "\n";
  }
  else if (data["target"]["nuclei"][i]["square_well_r_0"].size() !=
               data["target"]["nuclei"][i]["square_well_amplitude"].size() or
           data["target"]["nuclei"][i]["square_well_r_0"].size() !=
               data["target"]["nuclei"][i]["square_well_width"].size())
  {
    EndRun("Nuclei " + std::to_string(i) +
           " all square_well terms must have the same size");
  }
  for (PetscInt j = 0; j < square_well_size[i]; ++j)
  {
    square_well_r_0[i][j] = data["target"]["nuclei"][i]["square_well_r_0"][j];
    square_well_amplitude[i][j] =
        data["target"]["nuclei"][i]["square_well_amplitude"][j];
    square_well_width[i][j] =
        data["target"]["nuclei"][i]["square_well_width"][j];
  }
}

void Parameters::ReadYukawaPot(json data, PetscInt i)
{ /* yukawa Donuts */
  yukawa_size[i]       = data["target"]["nuclei"][i]["yukawa_r_0"].size();
  yukawa_r_0[i]        = new double[yukawa_size[i]];
  yukawa_amplitude[i]  = new double[yukawa_size[i]];
  yukawa_decay_rate[i] = new double[yukawa_size[i]];
  if (yukawa_size[i] == 0 and world.rank() == 0)
  {
    std::cout << "WARNING: No Yukawa potential for nuclei " << i << "\n";
  }
  else if (data["target"]["nuclei"][i]["yukawa_r_0"].size() !=
               data["target"]["nuclei"][i]["yukawa_amplitude"].size() or
           data["target"]["nuclei"][i]["yukawa_r_0"].size() !=
               data["target"]["nuclei"][i]["yukawa_decay_rate"].size())
  {
    EndRun("Nuclei " + std::to_string(i) +
           " all yukawa terms must have the same size");
  }
  for (PetscInt j = 0; j < yukawa_size[i]; ++j)
  {
    yukawa_r_0[i][j]       = data["target"]["nuclei"][i]["yukawa_r_0"][j];
    yukawa_amplitude[i][j] = data["target"]["nuclei"][i]["yukawa_amplitude"][j];
    yukawa_decay_rate[i][j] =
        data["target"]["nuclei"][i]["yukawa_decay_rate"][j];
  }
}

void Parameters::ReadTarget(json data)
{
  CheckParameter(data["target"]["name"].size(), "target - name");
  target = data["target"]["name"];

  CheckParameter(data["target"]["nuclei"].size(), "target - nuclei");
  num_nuclei = data["target"]["nuclei"].size();

  CheckParameter(data["alpha"].size(), "alpha");
  alpha = data["alpha"];

  if (num_electrons > 1 or coordinate_system_idx == 4 or
      coordinate_system_idx == 5)
  {
    CheckParameter(data["ee_soft_core"].size(), "ee_soft_core");
    ee_soft_core = data["ee_soft_core"];
  }
  else
  {
    ee_soft_core = 0;
  }

  CheckParameter(data["tol"].size(), "tol");
  tol = data["tol"];

  ReadSolver(data);

  z                      = std::make_unique< double[] >(num_nuclei);
  exponential_size       = std::make_unique< PetscInt[] >(num_nuclei);
  exponential_r_0        = new double*[num_nuclei];
  exponential_amplitude  = new double*[num_nuclei];
  exponential_decay_rate = new double*[num_nuclei];
  gaussian_size          = std::make_unique< PetscInt[] >(num_nuclei);
  gaussian_r_0           = new double*[num_nuclei];
  gaussian_amplitude     = new double*[num_nuclei];
  gaussian_decay_rate    = new double*[num_nuclei];
  square_well_size       = std::make_unique< PetscInt[] >(num_nuclei);
  square_well_r_0        = new double*[num_nuclei];
  square_well_amplitude  = new double*[num_nuclei];
  square_well_width      = new double*[num_nuclei];
  yukawa_size            = std::make_unique< PetscInt[] >(num_nuclei);
  yukawa_r_0             = new double*[num_nuclei];
  yukawa_amplitude       = new double*[num_nuclei];
  yukawa_decay_rate      = new double*[num_nuclei];
  location               = new double*[num_nuclei];
  for (PetscInt i = 0; i < num_nuclei; ++i)
  {
    /* Coulomb term */
    CheckParameter(data["target"]["nuclei"][i]["z"].size(),
                   "target - nuclei - z");
    z[i] = data["target"]["nuclei"][i]["z"];

    ReadExponentialPot(data, i);
    ReadGaussianPot(data, i);
    ReadSquareWellPot(data, i);
    ReadYukawaPot(data, i);

    location[i] = new double[num_dims];
    CheckParameter(data["target"]["nuclei"][i]["location"].size(),
                   "target - nuclei - location");
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
}

void Parameters::ReadInitialStateSpherical(json data)
{
  CheckParameter(data["start_state"]["n_index"].size(),
                 "start_state - n_index");
  num_start_state   = data["start_state"]["n_index"].size();
  start_state_l_idx = new PetscInt[num_start_state];
  start_state_m_idx = new PetscInt[num_start_state];
  if (data["start_state"]["amplitude"].size() != num_start_state)
  {
    EndRun(
        "'start_state - amplitude' and 'start_state - n_index' sizes do not "
        "match. Double check input file.");
  }
  if (data["start_state"]["phase"].size() != num_start_state)
  {
    EndRun(
        "'start_state - phase' and 'start_state - n_index' sizes do not "
        "match. Double check input file.");
  }
  if (data["start_state"]["l_index"].size() != num_start_state)
  {
    EndRun(
        "'start_state - l_index' and 'start_state - n_index' sizes do not "
        "match. Double check input file.");
  }
  if (data["start_state"]["m_index"].size() != num_start_state)
  {
    EndRun(
        "'start_state - m_index' and 'start_state - n_index' sizes do not "
        "match. Double check input file.");
  }
  start_state_idx       = new PetscInt[num_start_state];
  start_state_amplitude = new double[num_start_state];
  start_state_phase     = new double[num_start_state];
  for (PetscInt i = 0; i < num_start_state; i++)
  {
    CheckParameter(data["start_state"]["n_index"][i].size(),
                   "start_state - n_index");
    start_state_idx[i] = data["start_state"]["n_index"][i];
    CheckParameter(data["start_state"]["l_index"][i].size(),
                   "start_state - l_index");
    start_state_l_idx[i] = data["start_state"]["l_index"][i];
    CheckParameter(data["start_state"]["m_index"][i].size(),
                   "start_state - m_index");
    start_state_m_idx[i] = data["start_state"]["m_index"][i];
    CheckParameter(data["start_state"]["amplitude"][i].size(),
                   "start_state - amplitude");
    start_state_amplitude[i] = data["start_state"]["amplitude"][i];
    CheckParameter(data["start_state"]["phase"][i].size(),
                   "start_state - phase");
    start_state_phase[i] = data["start_state"]["phase"][i];
  }
}

void Parameters::ReadInitialStateDefault(json data)
{
  CheckParameter(data["start_state"]["index"].size(), "start_state - index");
  num_start_state = data["start_state"]["index"].size();

  if (data["start_state"]["amplitude"].size() != num_start_state)
  {
    EndRun(
        "'start_state - amplitude' and 'start_state - index' sizes do not "
        "match. Double check input file.");
  }
  if (data["start_state"]["phase"].size() != num_start_state)
  {
    EndRun(
        "'start_state - phase' and 'start_state - index' sizes do not match. "
        "Double check input file.");
  }
  start_state_idx       = new PetscInt[num_start_state];
  start_state_amplitude = new double[num_start_state];
  start_state_phase     = new double[num_start_state];
  for (PetscInt i = 0; i < num_start_state; i++)
  {
    CheckParameter(data["start_state"]["index"][i].size(),
                   "start_state - index");
    start_state_idx[i] = data["start_state"]["index"][i];

    CheckParameter(data["start_state"]["amplitude"][i].size(),
                   "start_state - amplitude");
    start_state_amplitude[i] = data["start_state"]["amplitude"][i];
    CheckParameter(data["start_state"]["phase"][i].size(),
                   "start_state - phase");
    start_state_phase[i] = data["start_state"]["phase"][i];
  }
}

void Parameters::ReadInitialState(json data)
{
  if (coordinate_system_idx == 3)
  {
    ReadInitialStateSpherical(data);
  }
  else
  {
    ReadInitialStateDefault(data);
  }
}

void Parameters::ReadExperimentFile(json data)
{
  double polar_norm    = 0.0; /* the norm for the polarization vector */
  double poynting_norm = 0.0; /* the norm for the poynting vector */

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

    /* Get polarization */
    polarization_vector[pulse_idx] = new double[num_dims];
    polar_norm                     = 0.0;
    if (num_dims == 3)
    {
      poynting_vector[pulse_idx] = new double[num_dims];
      poynting_norm              = 0.0;
    }
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      CheckParameter(
          data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx]
              .size(),
          "laser - pulses - polarization_vector");
      polarization_vector[pulse_idx][dim_idx] =
          data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx];
      polar_norm += polarization_vector[pulse_idx][dim_idx] *
                    polarization_vector[pulse_idx][dim_idx];

      if (num_dims == 3)
      {
        CheckParameter(
            data["laser"]["pulses"][pulse_idx]["poynting_vector"][dim_idx]
                .size(),
            "laser - pulses - poynting_vector");
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
  }
}

void Parameters::ReadPulse(json data)
{
  double polar_norm    = 0.0; /* the norm for the polarization vector */
  double poynting_norm = 0.0; /* the norm for the poynting vector */
  double intensity     = 0.0; /* the norm for the poynting vector */

  CheckParameter(data["laser"]["pulses"].size(), "laser - pulses");
  num_pulses = data["laser"]["pulses"].size();

  CheckParameter(data["laser"]["experiment_type"].size(),
                 "laser - experiment_type");
  experiment_type = data["laser"]["experiment_type"];

  CheckParameter(data["laser"]["frequency_shift"].size(),
                 "laser - frequency_shift");
  frequency_shift = data["laser"]["frequency_shift"];

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
    ReadExperimentFile(data);
  }
  else
  {
    /* streaking */
    if (experiment_type == "streaking" and num_pulses < 2)
    {
      EndRun(" streaking requires 2 or more pulses");
    }

    /* read in IR and XUV parameters */
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      /* set to zero for IO reasons and to ensure it is not used if it is not
       * read from the input file */
      power_on[pulse_idx]        = 0.0;
      power_off[pulse_idx]       = 0.0;
      gaussian_length[pulse_idx] = 1.0; /* the factor for non Gaussian pulses */
      CheckParameter(data["laser"]["pulses"][pulse_idx]["pulse_shape"].size(),
                     "laser - pulses - pulse_shape");
      pulse_shape[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["pulse_shape"];

      /* index used similar target_idx */
      if (pulse_shape[pulse_idx] == "sin2")
      {
        EndRun(
            "\"sin2\" is no longer an option. Please change to \"sin\" and "
            "set "
            "\"power_on\" and \"power_off\" to \"2\"\n");
      }
      else if (pulse_shape[pulse_idx] == "sin")
      {
        pulse_shape_idx[pulse_idx] = 0;
        CheckParameter(data["laser"]["pulses"][pulse_idx]["power_on"].size(),
                       "laser - pulses - power_on");
        power_on[pulse_idx] = data["laser"]["pulses"][pulse_idx]["power_on"];
        CheckParameter(data["laser"]["pulses"][pulse_idx]["power_off"].size(),
                       "laser - pulses - power_off");
        power_off[pulse_idx] = data["laser"]["pulses"][pulse_idx]["power_off"];
      }
      else if (pulse_shape[pulse_idx] == "gaussian")
      {
        pulse_shape_idx[pulse_idx] = 1;
        CheckParameter(
            data["laser"]["pulses"][pulse_idx]["gaussian_length"].size(),
            "laser - pulses - gaussian_length");
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
        std::cout
            << "ERROR: Polarization vector dimension is to small for pulse " +
                   std::to_string(pulse_idx) + "\n";
        CheckParameter(
            data["laser"]["pulses"][pulse_idx]["polarization_vector"].size(),
            "laser - pulses - polarization_vector");
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
        CheckParameter(
            data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx]
                .size(),
            "laser - pulses - polarization_vector");
        polarization_vector[pulse_idx][dim_idx] =
            data["laser"]["pulses"][pulse_idx]["polarization_vector"][dim_idx];
        polar_norm += polarization_vector[pulse_idx][dim_idx] *
                      polarization_vector[pulse_idx][dim_idx];

        if (num_dims == 3)
        {
          CheckParameter(
              data["laser"]["pulses"][pulse_idx]["poynting_vector"][dim_idx]
                  .size(),
              "laser - pulses - poynting_vector");
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

      CheckParameter(data["laser"]["pulses"][pulse_idx]["cep"].size(),
                     "laser - pulses - cep");
      cep[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cep"];

      CheckParameter(data["laser"]["pulses"][pulse_idx]["energy"].size(),
                     "laser - pulses - energy");
      energy[pulse_idx] = data["laser"]["pulses"][pulse_idx]["energy"];

      CheckParameter(data["laser"]["pulses"][pulse_idx]["ellipticity"].size(),
                     "laser - pulses - ellipticity");
      ellipticity[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["ellipticity"];

      CheckParameter(data["laser"]["pulses"][pulse_idx]["helicity"].size(),
                     "laser - pulses - helicity");
      helicity[pulse_idx] = data["laser"]["pulses"][pulse_idx]["helicity"];

      CheckParameter(data["laser"]["pulses"][pulse_idx]["intensity"].size(),
                     "laser - pulses - intensity");
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

      CheckParameter(data["laser"]["pulses"][pulse_idx]["cycles_on"].size(),
                     "laser - pulses - cycles_on");
      cycles_on[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_on"];

      CheckParameter(
          data["laser"]["pulses"][pulse_idx]["cycles_plateau"].size(),
          "laser - pulses - cycles_plateau");
      cycles_plateau[pulse_idx] =
          data["laser"]["pulses"][pulse_idx]["cycles_plateau"];

      CheckParameter(data["laser"]["pulses"][pulse_idx]["cycles_off"].size(),
                     "laser - pulses - cycles_off");
      cycles_off[pulse_idx] = data["laser"]["pulses"][pulse_idx]["cycles_off"];

      /* IR specific */
      if (experiment_type == "default" or
          (experiment_type == "streaking" and pulse_idx == 0))
      {
        CheckParameter(
            data["laser"]["pulses"][pulse_idx]["cycles_delay"].size(),
            "laser - pulses - cycles_delay");
        cycles_delay[pulse_idx] =
            data["laser"]["pulses"][pulse_idx]["cycles_delay"];
      }
      /* XUV specific */
      else if (experiment_type == "streaking" and pulse_idx > 0)
      {
        if (data["laser"]["pulses"][pulse_idx]["tau_delay"].size() == 0)
        {
          std::cout
              << "\n\nERROR: You are using the streaking 'experiment_type'\n "
                 "  "
                 "    For every pulse after the first one you need to "
                 "replace 'cycles_delay' with 'tau_delay'.\n       Use the "
                 "default 'experiment_type' if you wish to uses "
                 "'cycles_delay'";
          EndRun(
              "You must provide a 'laser - pulses - tau_delay' parameter in "
              "the input.json file.\n       See the Input section of the "
              "documentation for "
              "details.\n");
        }
        tau_delay = data["laser"]["pulses"][pulse_idx]["tau_delay"];

        double center_XUV_cycles =
            energy[pulse_idx] *
            ((2 * pi * (cycles_delay[0] + gaussian_length[0] * cycles_on[0]) /
              energy[0]) +
             tau_delay) /
            (2 * pi);

        cycles_delay[pulse_idx] =
            center_XUV_cycles -
            gaussian_length[pulse_idx] * cycles_on[pulse_idx];

        if (cycles_delay[pulse_idx] < 0)
        {
          for (PetscInt prev_pulse_idx = 0; prev_pulse_idx < pulse_idx;
               ++prev_pulse_idx)
          {
            /* delay all other pulses */
            cycles_delay[prev_pulse_idx] -= cycles_delay[pulse_idx] *
                                            energy[prev_pulse_idx] /
                                            energy[pulse_idx];
          }

          /* set the pulse to zero delay */
          cycles_delay[pulse_idx] = 0;
        }
      }
      else
      {
        EndRun("Unsupported experiment_type");
      }
    }
  }

  /* implement frequency shift */
  if (frequency_shift == 1)
  {
    if (experiment_type == "streaking" or experiment_type == "transient")
    {
      /* could be fixed at end of next for loop
       * This will require re-calculating cycles_delay from tau_delay since
       * the time of the peak of each laser pulse is subject to change with
       * the frequency shift */
      EndRun(
          "\nFrequency shift is not supported for streaking or transient "
          "type "
          "experiments. "
          "\nAll of the cycles_delay value needs to be corrected. "
          "\nParameters.cpp file for notes on how to make this fix\n");
    }

    /* loop over all pulses */
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      /* set default to no shift */
      double shift = 1.0;
      if (pulse_shape_idx[pulse_idx] == 0)
      {
        /* make sure the pulse is a purely sin^2 envelop */
        if (cycles_on[pulse_idx] == cycles_off[pulse_idx] and
            cycles_plateau[pulse_idx] == 0 and power_on[pulse_idx] == 2 and
            power_off[pulse_idx] == 2)
        {
          /* calculate shift */
          double mu_sin = 4.0 * asin(exp(-1.0 / 4.0)) * asin(exp(-1.0 / 4.0));
          shift =
              (1.0 +
               sqrt(1 + mu_sin /
                            ((cycles_on[pulse_idx] + cycles_off[pulse_idx]) *
                             (cycles_on[pulse_idx] + cycles_off[pulse_idx])))) /
              2.0;
        }
        else
        {
          EndRun(
              "\nFrequency shift only supports true sin^2 or Gaussian like "
              "pulse.\ncycles_on = cycles_off, power_on = power_off = 2, and "
              "cycles_plateau=0\n");
        }
      }
      else if (pulse_shape_idx[pulse_idx] == 1)
      {
        /* make sure the pulse is a purely gauss envelop */
        if (cycles_on[pulse_idx] == cycles_off[pulse_idx] and
            cycles_plateau[pulse_idx] == 0)
        {
          /* calculate shift */
          double mu_gaus = 4 * 2 * log(2.0) / (pi * pi);
          shift =
              (1.0 +
               sqrt(1 + mu_gaus /
                            ((cycles_on[pulse_idx] + cycles_off[pulse_idx]) *
                             (cycles_on[pulse_idx] + cycles_off[pulse_idx])))) /
              2.0;
        }
        else
        {
          EndRun(
              "\nFrequency shift only supports true sin^2 or Gaussian like "
              "pulse.\ncycles_on = cycles_off, and "
              "cycles_plateau=0\n");
        }
      }
      else /* non gauss or sin pulse shapes */
      {
        EndRun(
            "\nFrequency shift only supports true sin^2 or Gaussian like "
            "pulse.\n");
      }

      /* shift the energy */
      energy[pulse_idx] /= shift;
      /* updated cycles_delay by using calculating tau_delay and converting
       * between shifted and unshifted energies */
      cycles_delay[pulse_idx] /= shift;
      /* The peak of the A field is central frequency dependent*/
      field_max[pulse_idx] *= shift;
    }
  }
}

void Parameters::ReadData(json data)
{
  ReadNumerics(data);

  /* get simulation behavior */
  CheckParameter(data["restart"].size(), "restart");
  restart = data["restart"];

  CheckParameter(data["write_frequency_checkpoint"].size(),
                 "write_frequency_checkpoint");
  write_frequency_checkpoint = data["write_frequency_checkpoint"];

  CheckParameter(data["write_frequency_observables"].size(),
                 "write_frequency_observables");
  write_frequency_observables = data["write_frequency_observables"];

  CheckParameter(data["write_frequency_eigin_state"].size(),
                 "write_frequency_eigin_state");
  write_frequency_eigin_state = data["write_frequency_eigin_state"];

  CheckParameter(data["sigma"].size(), "sigma");
  sigma = data["sigma"];

  CheckParameter(data["gauge"].size(), "gauge");
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

  CheckParameter(data["propagate"].size(), "propagate");
  propagate = data["propagate"];

  CheckParameter(data["free_propagate"].size(), "free_propagate");
  free_propagate = data["free_propagate"];

  CheckParameter(data["field_max_states"].size(), "field_max_states");
  field_max_states = data["field_max_states"];

  ReadTarget(data);

  ReadInitialState(data);

  ReadPulse(data);
}

/**
 * @brief Ensures the parameter exists in the input.json file
 * @details If the size is zero, an error message is printed and the
 * simulation is aborted
 *
 * @param size the size of the json object (use the .size() function)
 * @param doc_string the subsubsection of the Input documentation
 */
void Parameters::CheckParameter(int size, std::string doc_string)
{
  if (size == 0)
  {
    EndRun("You must provide a '" + doc_string +
           "' parameter in the "
           "input.json file.\n"
           "       See the Input section of the documentation for details.\n");
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
    delete square_well_r_0[i];
    delete square_well_amplitude[i];
    delete square_well_width[i];
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
  delete[] square_well_r_0;
  delete[] square_well_amplitude;
  delete[] square_well_width;
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
  delete start_state_idx;  ///< index of states in super position
  if (coordinate_system_idx == 3)
  {
    delete start_state_l_idx;
    delete start_state_m_idx;
  }                              ///< index of states in super position
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
  if (coordinate_system_idx == 3) /* Spherical code */
  {
    if (num_electrons != 1)
    {
      error_found = true;
      err_str += "\nSpherical only supports 1 electron currently\n";
    }
    if (gauge_idx != 1)
    {
      error_found = true;
      err_str += "\nSpherical only supports \"Length\" gauge currently\n";
    }
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (location[nuclei_idx][0] > 1e-14 or location[nuclei_idx][1] > 1e-14 or
          location[nuclei_idx][2] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nSpherical coordinate systems only supports nuclei on the z "
            "axis (i.e. [0.0, 0.0, 0.0])\nNuclei " +
            std::to_string(nuclei_idx) + " has a non zero radial coordinate\n";
      }
    }
    if (m_max < 1)
    {
      for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
      {
        if ((polarization_vector[pulse_idx][0] > 1e-14 or
             polarization_vector[pulse_idx][1] > 1e-14) and
            m_max < 1)
        {
          error_found = true;
          err_str +=
              "\nSpherical coordinate systems requires m_max>0 for laser "
              "that "
              "are not z - polarized\nPulse" +
              std::to_string(pulse_idx) + " does not meet this requirement\n ";
        }
        if (ellipticity[pulse_idx] > 1e-14)
        {
          error_found = true;
          err_str +=
              "\nSpherical coordinate systems requires m_max>0 for pulses "
              "with "
              "nonzero ellipticity\nPulse " +
              std::to_string(pulse_idx) + " has a non zero ellipticity\n";
        }
      }
    }
  }
  if (coordinate_system_idx == 4) /* Hyperspherical code */
  {
    if (gauge_idx != 1)
    {
      error_found = true;
      err_str += "\nSpherical only supports \"Length\" gauge currently\n";
    }
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (location[nuclei_idx][0] > 1e-14 or location[nuclei_idx][1] > 1e-14 or
          location[nuclei_idx][2] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nSpherical coordinate systems only supports nuclei on the z "
            "axis "
            "(i.e. [0.0, 0.0, 0.0])\nNuclei " +
            std::to_string(nuclei_idx) +=
            " has a non zero radial coordinate\n ";
      }
    }
    if (m_max > 0)
    {
      error_found = true;
      err_str += "\nHyperspherical only supports m = 0\n";
    }
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
    {
      if (polarization_vector[pulse_idx][0] > 1e-14 or
          polarization_vector[pulse_idx][1] > 1e-14)
      {
        error_found = true;
        err_str += "\nHyperspherical only supports z-polarized lasers\nPulse" +
                   std::to_string(pulse_idx) +
                   " does not meet this requirement\n";
      }
      if (ellipticity[pulse_idx] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nHyperspherical coordinate systems only supports linear "
            "polarized light\nPulse " +
            std::to_string(pulse_idx) + " has a non zero ellipticity\n";
      }
    }
    if (abs(alpha) > 0 or abs(ee_soft_core) > 0)
    {
      error_found = true;
      err_str +=
          "\nHyperspherical coordinate systems does not support soft cores\n";
    }
  }
  if (coordinate_system_idx == 5) /* Hyperspherical code */
  {
    if (gauge_idx != 1)
    {
      error_found = true;
      err_str += "\nSpherical only supports \"Length\" gauge currently\n";
    }
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (location[nuclei_idx][0] > 1e-14 or location[nuclei_idx][1] > 1e-14 or
          location[nuclei_idx][2] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nSpherical coordinate systems only supports nuclei on the z "
            "axis "
            "(i.e. [0.0, 0.0, 0.0])\nNuclei " +
            std::to_string(nuclei_idx) +=
            " has a non zero radial coordinate\n ";
      }
    }
    if (m_max > 0)
    {
      error_found = true;
      err_str += "\nHyperspherical only supports m = 0\n";
    }
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; pulse_idx++)
    {
      if (polarization_vector[pulse_idx][0] > 1e-14 or
          polarization_vector[pulse_idx][1] > 1e-14)
      {
        error_found = true;
        err_str += "\nHyperspherical only supports z-polarized lasers\nPulse" +
                   std::to_string(pulse_idx) +
                   " does not meet this requirement\n";
      }
      if (ellipticity[pulse_idx] > 1e-14)
      {
        error_found = true;
        err_str +=
            "\nHyperspherical coordinate systems only supports linear "
            "polarized light\nPulse " +
            std::to_string(pulse_idx) + " has a non zero ellipticity\n";
      }
    }
    if (abs(alpha) > 0 or abs(ee_soft_core) > 0)
    {
      error_found = true;
      err_str +=
          "\nHyperspherical coordinate systems does not support soft cores\n";
    }
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

    if (pulse_shape[pulse_idx] == "sin")
    {
      if ((power_on[pulse_idx] % 2) != 0)
      {
        error_found = true;
        err_str += "\nPulse ";
        err_str += std::to_string(pulse_idx);
        err_str += " has power_on: \"";
        err_str += std::to_string(power_on[pulse_idx]) + "\"\n";
        err_str += "power_on should be even\n";
      }
      if ((power_off[pulse_idx] % 2) != 0)
      {
        error_found = true;
        err_str += "\nPulse ";
        err_str += std::to_string(pulse_idx);
        err_str += " has power_off: \"";
        err_str += std::to_string(power_off[pulse_idx]) + "\"\n";
        err_str += "power_off should be even\n";
      }
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
  else if (state_solver_idx == -1)
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
    if (coordinate_system_idx == 3)
    {
      if (start_state_idx[idx] - 1 >= num_states)
      {
        error_found = true;
        err_str +=
            "\nThe start_state - n_index must be less than the total number "
            "of "
            "states you wish to calculate\n";
      }
      if (start_state_l_idx[idx] >= start_state_idx[idx])
      {
        error_found = true;
        err_str +=
            "\nThe start_state - l_index must be less than start_state - "
            "n_index\n";
      }
      if (start_state_l_idx[idx] > l_max)
      {
        error_found = true;
        err_str += "\nThe start_state - l_index must be less than l_max\n";
      }
      if (std::abs(start_state_m_idx[idx]) > start_state_l_idx[idx])
      {
        error_found = true;
        err_str +=
            "\nThe magnitude of start_state - m_index must be less than or "
            "equal to start_state - l_index\n";
      }
      if (std::abs(start_state_m_idx[idx]) > m_max)
      {
        error_found = true;
        err_str += "\nThe start_state - m_index must be less than m_max\n";
      }
    }
    else if (start_state_idx[idx] >= num_states)
    {
      error_found = true;
      err_str +=
          "\nThe start_state must be less than the total number of states "
          "you wish to calculate\n";
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

PetscInt Parameters::GetMMax() { return m_max; }
PetscInt Parameters::GetLMax() { return l_max; }
PetscInt Parameters::GetKMax() { return k_max; }
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

double** Parameters::GetSquareWellR0() { return square_well_r_0; }
double** Parameters::GetSquareWellAmplitude() { return square_well_amplitude; }
double** Parameters::GetSquareWellWidth() { return square_well_width; }

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

PetscInt* Parameters::GetStartStateLIdx() { return start_state_l_idx; }

PetscInt* Parameters::GetStartStateMIdx() { return start_state_m_idx; }

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

PetscInt Parameters::GetFrequencyShift() { return frequency_shift; }

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
