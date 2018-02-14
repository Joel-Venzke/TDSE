#include "Pulse.h"

Pulse::Pulse(HDF5Wrapper& data_file, Parameters& p)
{
  if (world.rank() == 0)
  {
    std::cout << "Creating pulses\n" << std::flush;
  }

  /* get number of pulses and dt from Parameters */
  pulse_alloc               = false;
  num_pulses                = p.GetNumPulses();
  num_dims                  = p.GetNumDims();
  delta_t                   = p.GetDeltaT();
  polarization_vector_major = p.GetPolarizationVector();
  if (num_dims == 3)
  {
    poynting_vector = p.GetPoyntingVector();
  }
  ellipticity      = p.ellipticity.get();
  helicity_idx     = p.helicity_idx.get();
  max_pulse_length = 0; /* stores longest pulse */

  pulse_shape_idx = p.pulse_shape_idx.get();
  gaussian_sigma  = 5.0;
  gauge_idx       = p.GetGaugeIdx();
  power_on        = p.power_on.get();
  power_off       = p.power_off.get();
  cycles_on       = p.cycles_on.get();
  cycles_plateau  = p.cycles_plateau.get();
  cycles_off      = p.cycles_off.get();
  cycles_delay    = p.cycles_delay.get();
  cep             = p.cep.get();
  energy          = p.energy.get();
  field_max       = p.field_max.get();
  experiment_type = p.experiment_type;

  InitializePolarization();
  InitializePulseLength();
  InitializeTime();
  InitializePulse();
  InitializeField();

  Checkpoint();

  DeallocatePulses();

  if (world.rank() == 0)
  {
    std::cout << "Pulses created\n" << std::flush;
  }
}

Pulse::~Pulse()
{
  if (world.rank() == 0)
  {
    std::cout << "Deleting Pulse\n" << std::flush;
  }
  delete cycles_total;
  delete time;
  DeallocatePulses();
  for (PetscInt i = 0; i < num_dims; ++i)
  {
    delete field[i];
  }
  delete[] field;
}

void Pulse::InitializePolarization()
{
  double polar_norm = 0.0;
  /* allocate arrays */
  polarization_vector_minor = new double*[num_pulses];
  /* get data from Parameters */
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
  {
    polarization_vector_minor[pulse_idx] = new double[num_dims];
    /* Take cross product */
    if (num_dims == 1)
    {
      polarization_vector_minor[pulse_idx][0] = 0.0;
    }
    else if (num_dims == 2)
    {
      polarization_vector_minor[pulse_idx][0] =
          -1.0 * polarization_vector_major[pulse_idx][1];
      polarization_vector_minor[pulse_idx][1] =
          polarization_vector_major[pulse_idx][0];
    }
    else if (num_dims == 3)
    {
      polarization_vector_minor[pulse_idx][0] =
          poynting_vector[pulse_idx][1] *
              polarization_vector_major[pulse_idx][2] -
          poynting_vector[pulse_idx][2] *
              polarization_vector_major[pulse_idx][1];

      polarization_vector_minor[pulse_idx][1] =
          poynting_vector[pulse_idx][2] *
              polarization_vector_major[pulse_idx][0] -
          poynting_vector[pulse_idx][0] *
              polarization_vector_major[pulse_idx][2];

      polarization_vector_minor[pulse_idx][2] =
          poynting_vector[pulse_idx][0] *
              polarization_vector_major[pulse_idx][1] -
          poynting_vector[pulse_idx][1] *
              polarization_vector_major[pulse_idx][0];
    }
    else
    {
      EndRun("How many dimensions are you using?");
    }

    /* normalize and scale with ellipticity */
    polar_norm = 0.0;
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      polar_norm += polarization_vector_minor[pulse_idx][dim_idx] *
                    polarization_vector_minor[pulse_idx][dim_idx];
    }
    /* normalize the polarization vector*/
    polar_norm = sqrt(polar_norm);
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (polar_norm > 1e-10)
      {
        polarization_vector_minor[pulse_idx][dim_idx] /= polar_norm;
      }
      /* scale with ellipticity */
      polarization_vector_minor[pulse_idx][dim_idx] *= ellipticity[pulse_idx];
    }
  }
}

void Pulse::InitializePulseLength()
{
  PetscInt pulse_length = 0;
  cycles_total          = new double[num_pulses];

  if (experiment_type == "File")
  {
    ReadPulseFromFile();
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      pulse_length = cycles_total[pulse_idx];
      /* find the largest */
      if (pulse_length > max_pulse_length)
      {
        max_pulse_length = pulse_length;
      }
    }
  }
  else
  {
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      if (pulse_shape_idx[pulse_idx] == 0)
      {
        cycles_total[pulse_idx] =
            cycles_delay[pulse_idx] + cycles_on[pulse_idx] +
            cycles_plateau[pulse_idx] + cycles_off[pulse_idx];
      }
      else if (pulse_shape_idx[pulse_idx] ==
               1) /* Gaussian needs 6 sigma tails */
      {
        cycles_total[pulse_idx] =
            cycles_delay[pulse_idx] + gaussian_sigma * cycles_on[pulse_idx] +
            cycles_plateau[pulse_idx] + gaussian_sigma * cycles_off[pulse_idx];
      }

      /* calculate length (number of array cells) of each pulse */
      pulse_length = ceil(2.0 * pi * cycles_total[pulse_idx] /
                          (energy[pulse_idx] * delta_t));

      /* find the largest */
      if (pulse_length > max_pulse_length)
      {
        max_pulse_length = pulse_length;
      }
    }
  }
}

void Pulse::ReadPulseFromFile()
{
  PetscInt file_time_size  = 0;
  PetscInt file_pulse_size = 0;

  json data  = FileToJson("pulses.json");
  file_time  = new double*[num_pulses];
  file_pulse = new double*[num_pulses];
  file_size  = new PetscInt[num_pulses];
  for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
  {
    file_time_size  = data["pulses"][pulse_idx]["time"].size();
    file_pulse_size = data["pulses"][pulse_idx]["field"].size();
    if (file_time_size != file_pulse_size)
    {
      EndRun("Size of Time and Field of pulse " + std::to_string(pulse_idx) +
             " do not match");
    }
    file_time[pulse_idx]  = new double[file_time_size];
    file_pulse[pulse_idx] = new double[file_time_size];
    for (int time_idx = 0; time_idx < file_time_size; ++time_idx)
    {
      file_time[pulse_idx][time_idx] =
          data["pulses"][pulse_idx]["time"][time_idx];
      file_pulse[pulse_idx][time_idx] =
          data["pulses"][pulse_idx]["field"][time_idx];
    }

    /* store size of pulse */
    file_size[pulse_idx] = file_time_size;

    /* figure out how many time steps we need */
    cycles_total[pulse_idx] =
        std::ceil(file_time[pulse_idx][file_time_size - 1] / delta_t) + 1;
  }
}

/* Build array with time in au */
void Pulse::InitializeTime()
{
  time = new double[max_pulse_length];
  for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
  {
    time[time_idx] = time_idx * delta_t;
  }
}

/**
 * @brief Linear interpolation of file pulse
 *
 * @param pulse_idx index of pulse in file
 * @param time time you want the interpolation at
 *
 * @return value of the pulse at the interpolated time
 */
double Pulse::Interpolate(PetscInt pulse_idx, double time)
{
  if (time > file_time[pulse_idx][file_size[pulse_idx] - 1]) return 0.0;
  PetscInt time_idx = 1;
  double slope;
  while (time >= file_time[pulse_idx][time_idx]) time_idx++;
  slope =
      (file_pulse[pulse_idx][time_idx] - file_pulse[pulse_idx][time_idx - 1]) /
      (file_time[pulse_idx][time_idx] - file_time[pulse_idx][time_idx - 1]);
  return file_pulse[pulse_idx][time_idx - 1] +
         slope * (time - file_time[pulse_idx][time_idx - 1]);
}

/* build the nth pulse */
void Pulse::InitializePulse(PetscInt n)
{
  if (experiment_type == "File")
  {
    if (!pulse_alloc)
    {
      pulse_envelope[n] = new double[max_pulse_length];
      pulse_value[n]    = new double*[num_dims];
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        pulse_value[n][dim_idx] = new double[max_pulse_length];
      }
    }

    for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
    {
      /* Get E-Field from file */
      pulse_envelope[n][time_idx] = Interpolate(n, time[time_idx]);
    }

    for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
    {
      /* Time integral */
      if (time_idx > 0)
      {
        pulse_envelope[n][time_idx] = pulse_envelope[n][time_idx - 1] -
                                      pulse_envelope[n][time_idx] * c * delta_t;
      }
      else
      {
        pulse_envelope[n][time_idx] *= -1.0 * c * delta_t;
      }
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        pulse_value[n][dim_idx][time_idx] =
            polarization_vector_major[n][dim_idx] * pulse_envelope[n][time_idx];
      }
    }
  }
  else
  {
    PetscInt on_start      = 0;
    PetscInt plateau_start = 0;
    PetscInt off_start     = 0;
    PetscInt off_end       = 0;
    double period          = 2.0 * pi / energy[n];
    double s1;  // value of sin (x for Gaussian)
    double sn;  // sin to the nth power
    double current_cep = cep[n] + (((int)cycles_on[n]) - cycles_on[n]);

    /* index that turns pulse on */
    on_start = ceil(period * cycles_delay[n] / (delta_t));

    if (pulse_shape_idx[n] == 0)
    {
      /* index that holds pulse at max */
      plateau_start =
          ceil(period * (cycles_on[n] + cycles_delay[n]) / (delta_t));
      /* index that turns pulse off */
      off_start =
          ceil(period * (cycles_plateau[n] + cycles_on[n] + cycles_delay[n]) /
               (delta_t));

      /* index that holds pulse at 0 */
      off_end = ceil(
          period *
          (cycles_off[n] + cycles_plateau[n] + cycles_on[n] + cycles_delay[n]) /
          (delta_t));
    }
    else if (pulse_shape_idx[n] == 1) /* Gaussian needs 6 sigma tails */
    {
      /* index that holds pulse at max */
      plateau_start =
          ceil(period * (gaussian_sigma * cycles_on[n] + cycles_delay[n]) /
               (delta_t));
      /* index that turns pulse off */
      off_start = ceil(period *
                       (cycles_plateau[n] + gaussian_sigma * cycles_on[n] +
                        cycles_delay[n]) /
                       (delta_t));

      /* index that holds pulse at 0 */
      off_end = ceil(period *
                     (gaussian_sigma * cycles_off[n] + cycles_plateau[n] +
                      gaussian_sigma * cycles_on[n] + cycles_delay[n]) /
                     (delta_t));
    }

    if (!pulse_alloc)
    {
      pulse_envelope[n] = new double[max_pulse_length];
    }

    double fac = 2 * sqrt(log(2));

    for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
    {
      if (time_idx < on_start)
      { /* pulse still off */
        pulse_envelope[n][time_idx] = 0.0;
      }
      else if (time_idx < plateau_start)
      { /* pulse ramping on */
        if (pulse_shape_idx[n] == 0)
        {
          s1 = sin(energy[n] * delta_t * (time_idx - on_start) /
                   (4.0 * cycles_on[n]));
          sn = 1.0;
          for (int power = 0; power < power_on[n]; ++power)
          {
            sn *= s1;
          }
          pulse_envelope[n][time_idx] = field_max[n] * sn;
        }
        else if (pulse_shape_idx[n] == 1)
        {
          s1 = (energy[n] * delta_t * (plateau_start - time_idx)) /
               (2.0 * pi * fac * cycles_on[n]);
          pulse_envelope[n][time_idx] = field_max[n] * exp(-1.0 * s1 * s1);
          // pulse_envelope[n][time_idx] = -1.0 * s1 * s1;
        }
      }
      else if (time_idx < off_start)
      { /* pulse at max */
        pulse_envelope[n][time_idx] = field_max[n];
      }
      else if (time_idx < off_end)
      { /* pulse ramping off */
        if (pulse_shape_idx[n] == 0)
        {
          s1 = sin(energy[n] * delta_t * (time_idx - off_start) /
                       (4.0 * cycles_off[n]) +
                   pi / 2.0);
          sn = 1.0;
          for (int power = 0; power < power_off[n]; ++power)
          {
            sn *= s1;
          }
          pulse_envelope[n][time_idx] = field_max[n] * sn;
        }
        else if (pulse_shape_idx[n] == 1)
        {
          s1 = (energy[n] * delta_t * (time_idx - off_start)) /
               (2 * pi * fac * cycles_off[n]);
          pulse_envelope[n][time_idx] = field_max[n] * exp(-1.0 * s1 * s1);
          // pulse_envelope[n][time_idx] = -1.0 * s1 * s1;
        }
      }
      else
      { /* pulse is off */
        pulse_envelope[n][time_idx] = 0.0;
      }
    }

    if (!pulse_alloc)
    {
      pulse_value[n] = new double*[num_dims];
    }
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (!pulse_alloc)
      {
        pulse_value[n][dim_idx] = new double[max_pulse_length];
      }
      for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
      {
        /* calculate the actual pulse */
        pulse_value[n][dim_idx][time_idx] =
            polarization_vector_major[n][dim_idx] *
            pulse_envelope[n][time_idx] *
            sin(energy[n] * delta_t * (time_idx - on_start) +
                current_cep * 2 * pi);
        if (helicity_idx[n] == 0) /* right */
        {
          /* We want cos(...) */
          pulse_value[n][dim_idx][time_idx] +=
              polarization_vector_minor[n][dim_idx] *
              pulse_envelope[n][time_idx] *
              cos(energy[n] * delta_t * (time_idx - on_start) +
                  current_cep * 2 * pi);
        }
        else if (helicity_idx[n] == 1) /* left */
        {
          /* We want -1.0 * cos(...) */
          pulse_value[n][dim_idx][time_idx] -=
              polarization_vector_minor[n][dim_idx] *
              pulse_envelope[n][time_idx] *
              cos(energy[n] * delta_t * (time_idx - on_start) +
                  current_cep * 2 * pi);
        }
      }
    }
  }
}

/* sets up all pulses and calculates the field */
void Pulse::InitializePulse()
{
  /* set up the input pulses */
  if (!pulse_alloc)
  {
    pulse_value    = new double**[num_pulses];
    pulse_envelope = new double*[num_pulses];
  }
  for (PetscInt i = 0; i < num_pulses; ++i)
  {
    InitializePulse(i);
  }
  pulse_alloc = true;
}

void Pulse::InitializeField()
{
  /* calculate the field by summing each pulse */
  /* TODO(jove7731): add support for setting e_field */
  if (!pulse_alloc)
  {
    InitializePulse();
  }
  field = new double*[num_dims];
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    field[dim_idx] = new double[max_pulse_length];
    for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
    {
      field[dim_idx][time_idx] = 0.0;
      for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
      {
        field[dim_idx][time_idx] += pulse_value[pulse_idx][dim_idx][time_idx];
      }
    }
  }
  /* Calculate the E field from A by E= -1/c * (dA/dt)*/
  if (gauge_idx == 1)
  {
    /* Don't overwrite field until calculation is done */
    double* field_tmp = new double[max_pulse_length];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      /* Forward difference for first point (-1/c)  [-1, 1, 0] (1/dt) */
      field_tmp[0] = (field[dim_idx][1] - field[dim_idx][0]) / (delta_t * c);
      for (PetscInt time_idx = 1; time_idx < max_pulse_length - 1; ++time_idx)
      {
        /* calculate (-1/c)  [-1/2, 0, 1/2] (1/dt) */
        field_tmp[time_idx] =
            (field[dim_idx][time_idx - 1] - field[dim_idx][time_idx + 1]) /
            (2 * delta_t * c);
      }
      /* Backward difference for first point (-1/c)  [0, -1, 1] (1/dt) */
      field_tmp[max_pulse_length - 1] = (field[dim_idx][max_pulse_length - 2] -
                                         field[dim_idx][max_pulse_length - 1]) /
                                        (delta_t * c);

      /* overwrite field with the E field */
      for (PetscInt time_idx = 0; time_idx < max_pulse_length; ++time_idx)
      {
        field[dim_idx][time_idx] = field_tmp[time_idx];
      }
    }
    delete[] field_tmp;
  }
}

/* write out the state of the pulse */
void Pulse::Checkpoint()
{
  HDF5Wrapper data_file("Pulse.h5", "w");
  data_file.CreateGroup("/Pulse");
  /* write time, field, and field_envelope to hdf5 */
  data_file.WriteObject(time, max_pulse_length, "/Pulse/time",
                        "The time for each index of the pulse in a.u.");

  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    data_file.WriteObject(field[dim_idx], max_pulse_length,
                          "/Pulse/field_" + std::to_string(dim_idx),
                          "The value of the field in the " +
                              std::to_string(dim_idx) +
                              " dimension at each point in time in a.u.");
  }

  if (pulse_alloc)
  {
    /* write each pulse both value and envelope */
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      data_file.WriteObject(
          pulse_envelope[pulse_idx], max_pulse_length,
          "/Pulse/Pulse_envelope_" + std::to_string(pulse_idx),
          "The envelope function for the " + std::to_string(pulse_idx) +
              " pulse in the input file");
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        data_file.WriteObject(
            pulse_value[pulse_idx][dim_idx], max_pulse_length,
            "/Pulse/Pulse_value_" + std::to_string(pulse_idx) + "_" +
                std::to_string(dim_idx),
            "The pulse value for the " + std::to_string(pulse_idx) +
                " pulse's " + std::to_string(dim_idx) +
                " dimension in the input file");
      }
    }
  }
}

void Pulse::DeallocatePulses()
{
  if (pulse_alloc)
  {
    for (PetscInt pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
    {
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        delete pulse_value[pulse_idx][dim_idx];
      }
      delete[] pulse_value[pulse_idx];
      delete pulse_envelope[pulse_idx];
      if (experiment_type == "File")
      {
        delete file_pulse[pulse_idx];
        delete file_time[pulse_idx];
      }
    }
    delete[] pulse_value;
    delete[] pulse_envelope;
    if (experiment_type == "File")
    {
      delete[] file_pulse;
      delete[] file_time;
      delete file_size;
    }
    pulse_alloc = false;
  }
}

double** Pulse::GetField() { return field; }

double* Pulse::GetTime() { return time; }

PetscInt Pulse::GetMaxPulseLength() { return max_pulse_length; }

PetscInt Pulse::GetFieldMaxIdx()
{
  double pulse_max       = 0;
  double pulse_mag       = 0;
  PetscInt pulse_max_idx = 0;

  for (int time_idx = 0; time_idx < max_pulse_length; ++time_idx)
  {
    pulse_mag = 0;
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      pulse_mag += field[dim_idx][time_idx] * field[dim_idx][time_idx];
    }
    if (pulse_mag > pulse_max)
    {
      pulse_max     = pulse_mag;
      pulse_max_idx = time_idx;
    }
  }
  return pulse_max_idx;
}
