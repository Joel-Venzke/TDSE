#include "Pulse.h"

Pulse::Pulse(HDF5Wrapper& data_file, Parameters& p)
{
  int pulse_length = 0;

  if (world.rank() == 0)
  {
    std::cout << "Creating pulses\n" << std::flush;
  }

  /* get number of pulses and dt from Parameters */
  pulse_alloc         = false;
  num_pulses          = p.GetNumPulses();
  num_dims            = p.GetNumDims();
  delta_t             = p.GetDeltaT();
  polarization_vector = p.polarization_vector.get();
  max_pulse_length    = 0; /* stores longest pulse */

  pulse_shape_idx = p.pulse_shape_idx.get();
  cycles_on       = p.cycles_on.get();
  cycles_plateau  = p.cycles_plateau.get();
  cycles_off      = p.cycles_off.get();
  cycles_delay    = p.cycles_delay.get();
  cep             = p.cep.get();
  energy          = p.energy.get();
  field_max       = p.field_max.get();

  /* allocate arrays */
  cycles_total = new double[num_pulses];

  /* get data from Parameters */
  for (int i = 0; i < num_pulses; ++i)
  {
    cycles_total[i] =
        cycles_delay[i] + cycles_on[i] + cycles_plateau[i] + cycles_off[i];

    /* calculate length (number of array cells) of each pulse */
    pulse_length = ceil(2.0 * pi * cycles_total[i] / (energy[i] * delta_t)) + 1;

    /* find the largest */
    if (pulse_length > max_pulse_length)
    {
      max_pulse_length = pulse_length;
    }
  }

  InitializeTime();
  InitializePulse();
  InitializeField();

  Checkpoint(data_file);

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
  if (pulse_alloc)
  {
    for (int i = 0; i < num_pulses; ++i)
    {
      delete pulse_value[i];
      delete pulse_envelope[i];
    }
    delete[] pulse_value;
    delete[] pulse_envelope;
  }
  for (int i = 0; i < num_dims; ++i)
  {
    delete field[i];
  }
  delete[] field;
}

/* Build array with time in au */
void Pulse::InitializeTime()
{
  time = new double[max_pulse_length];
  for (int time_idx = 0; time_idx < max_pulse_length; ++time_idx)
  {
    time[time_idx] = time_idx * delta_t;
  }
}

/* build the nth pulse */
void Pulse::InitializePulse(int n)
{
  int on_start, plateau_start, off_start, off_end;
  double period = 2 * pi / energy[n];
  double s1;

  /* index that turns pulse on */
  on_start = ceil(period * cycles_delay[n] / (delta_t)) + 1;

  /* index that holds pulse at max */
  plateau_start =
      ceil(period * (cycles_on[n] + cycles_delay[n]) / (delta_t)) + 1;

  /* index that turns pulse off */
  off_start =
      ceil(period * (cycles_plateau[n] + cycles_on[n] + cycles_delay[n]) /
           (delta_t)) +
      1;

  /* index that holds pulse at 0 */
  off_end = ceil(period *
                 (cycles_off[n] + cycles_plateau[n] + cycles_on[n] +
                  cycles_delay[n]) /
                 (delta_t)) +
            1;
  if (!pulse_alloc)
  {
    pulse_envelope[n] = new double[max_pulse_length];
    pulse_value[n]    = new double[max_pulse_length];
  }
  for (int i = 0; i < max_pulse_length; ++i)
  {
    if (i < on_start)
    { /* pulse still off */
      pulse_envelope[n][i] = 0.0;
    }
    else if (i < plateau_start)
    { /* pulse ramping on */
      s1 = sin(energy[n] * delta_t * (i - on_start) / (4.0 * cycles_on[n]));
      pulse_envelope[n][i] = field_max[n] * s1 * s1;
    }
    else if (i < off_start)
    { /* pulse at max */
      pulse_envelope[n][i] = field_max[n];
    }
    else if (i < off_end)
    { /* pulse ramping off */
      s1 = sin(energy[n] * delta_t * (i - off_start) / (4.0 * cycles_off[n]));
      pulse_envelope[n][i] = field_max[n] * (1 - (s1 * s1));
    }
    else
    { /* pulse is off */
      pulse_envelope[n][i] = 0.0;
    }

    /* calculate the actual pulse */
    pulse_value[n][i] =
        pulse_envelope[n][i] *
        sin(energy[n] * delta_t * (i - on_start) + cep[n] * 2 * pi);
  }
}

/* sets up all pulses and calculates the field */
void Pulse::InitializePulse()
{
  /* set up the input pulses */
  if (!pulse_alloc)
  {
    pulse_value    = new double*[num_pulses];
    pulse_envelope = new double*[num_pulses];
  }
  for (int i = 0; i < num_pulses; ++i)
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
  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    field[dim_idx] = new double[max_pulse_length];
    for (int time_idx = 0; time_idx < max_pulse_length; ++time_idx)
    {
      field[dim_idx][time_idx] = 0;
      for (int pulse_idx = 0; pulse_idx < num_pulses; ++pulse_idx)
      {
        field[dim_idx][time_idx] +=
            polarization_vector[dim_idx] * pulse_value[pulse_idx][time_idx];
      }
    }
  }
}

/* write out the state of the pulse */
void Pulse::Checkpoint(HDF5Wrapper& data_file)
{
  data_file.CreateGroup("/Pulse");
  /* write time, field, and field_envelope to hdf5 */
  data_file.WriteObject(time, max_pulse_length, "/Pulse/time",
                        "The time for each index of the pulse in a.u.");

  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
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
    for (int i = 0; i < num_pulses; ++i)
    {
      data_file.WriteObject(pulse_envelope[i], max_pulse_length,
                            "/Pulse/Pulse_envelope_" + std::to_string(i),
                            "The envelope function for the " +
                                std::to_string(i) + " pulse in the input file");
      data_file.WriteObject(pulse_value[i], max_pulse_length,
                            "/Pulse/Pulse_value_" + std::to_string(i),
                            "The pulse value for the " + std::to_string(i) +
                                " pulse in the input file");
    }
  }
}

void Pulse::DeallocatePulses()
{
  if (pulse_alloc)
  {
    for (int i = 0; i < num_pulses; ++i)
    {
      delete pulse_value[i];
      delete pulse_envelope[i];
    }
    delete[] pulse_value;
    delete[] pulse_envelope;
    pulse_alloc = false;
  }
}

double** Pulse::GetField() { return field; }

double* Pulse::GetTime() { return time; }

int Pulse::GetMaxPulseLength() { return max_pulse_length; }
