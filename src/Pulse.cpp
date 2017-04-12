#include "Pulse.h"

Pulse::Pulse(HDF5Wrapper& data_file, Parameters& p)
{
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  int pulse_length = 0;

  if (rank == 0)
  {
    std::cout << "Creating pulses\n" << std::flush;
  }

  /* get number of pulses and dt from Parameters */
  pulse_alloc      = false;
  num_pulses       = p.GetNumPulses();
  delta_t          = p.GetDeltaT();
  max_pulse_length = 0; /* stores longest pulse */

  /* allocate arrays */
  pulse_shape_idx = new int[num_pulses];
  cycles_on       = new double[num_pulses];
  cycles_plateau  = new double[num_pulses];
  cycles_off      = new double[num_pulses];
  cycles_delay    = new double[num_pulses];
  cycles_total    = new double[num_pulses];
  cep             = new double[num_pulses];
  energy          = new double[num_pulses];
  field_max       = new double[num_pulses];

  /* get data from Parameters */
  for (int i = 0; i < num_pulses; ++i)
  {
    pulse_shape_idx[i] = p.pulse_shape_idx.get()[i];
    cycles_on[i]       = p.cycles_on.get()[i];
    cycles_plateau[i]  = p.cycles_plateau.get()[i];
    cycles_off[i]      = p.cycles_off.get()[i];
    cycles_delay[i]    = p.cycles_delay.get()[i];
    cycles_total[i] =
        cycles_delay[i] + cycles_on[i] + cycles_plateau[i] + cycles_off[i];
    cep[i]       = p.cep.get()[i];
    energy[i]    = p.energy.get()[i];
    field_max[i] = p.field_max.get()[i];

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
  InitializeAField();

  Checkpoint(data_file);

  DeallocatePulses();

  if (rank == 0)
  {
    std::cout << "Pulses created\n" << std::flush;
  }
}

Pulse::~Pulse()
{
  if (rank == 0)
  {
    std::cout << "Deleting Pulse\n" << std::flush;
  }
  delete pulse_shape_idx;
  delete cycles_on;
  delete cycles_plateau;
  delete cycles_off;
  delete cycles_delay;
  delete cycles_total;
  delete cep;
  delete energy;
  delete field_max;
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
  delete a_field;
}

/* Build array with time in au */
void Pulse::InitializeTime()
{
  time = new double[max_pulse_length];
  for (int i = 0; i < max_pulse_length; ++i)
  {
    time[i] = i * delta_t;
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

/* sets up all pulses and calculates the a_field */
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

void Pulse::InitializeAField()
{
  /* calculate the a_field by summing each pulse */
  /* TODO(jove7731): add support for setting e_field */
  if (!pulse_alloc)
  {
    InitializePulse();
  }
  a_field = new double[max_pulse_length];
  for (int i = 0; i < max_pulse_length; ++i)
  {
    a_field[i] = 0;
    for (int j = 0; j < num_pulses; ++j)
    {
      a_field[i] += pulse_value[j][i];
    }
  }
}

/* write out the state of the pulse */
void Pulse::Checkpoint(HDF5Wrapper& data_file)
{
  data_file.CreateGroup("/Pulse");
  /* write time, a_field, and a_field_envelope to hdf5 */
  data_file.WriteObject(time, max_pulse_length, "/Pulse/time",
                        "The time for each index of the pulse in a.u.");

  data_file.WriteObject(
      a_field, max_pulse_length, "/Pulse/a_field",
      "The value of the A field at each point in time in a.u.");

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

double* Pulse::GetAField() { return a_field; }

double* Pulse::GetTime() { return time; }

int Pulse::GetMaxPulseLength() { return max_pulse_length; }
