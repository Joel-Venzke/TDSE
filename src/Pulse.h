#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"

class Pulse : protected Utils
{
 private:
  int num_pulses;          /* number of pulses */
  double delta_t;          /* time step */
  int max_pulse_length;    /* length of longest pulse; */
  int *pulse_shape_idx;    /* index of pulse shape */
  double *cycles_on;       /* ramp on cycles */
  double *cycles_plateau;  /* plateau cycles */
  double *cycles_off;      /* ramp off cycles */
  double *cycles_delay;    /* cycles till it starts */
  double *cycles_total;    /* cycles in pulse */
  double *cep;             /* carrier envelope phase */
  double *energy;          /* photon energy */
  double *field_max;       /* max amplitude */
  double *time;            /* stores the time at each point */
  double **pulse_value;    /* pulse value */
  double **pulse_envelope; /* envelope function of pulse */
  double *a_field;         /* total vector potential */
  /* true if the individual pulses and envelopes are allocated */
  bool pulse_alloc;

  /* private to avoid unneeded allocation calls and to protect the */
  /* developer form accessing garbage arrays */
  void InitializePulse(int i);
  void InitializePulse();
  void InitializeTime();
  void DeallocatePulses();
  void InitializeAField();

 public:
  /* Constructor */
  Pulse(HDF5Wrapper &data_file, Parameters &p);

  /* Destructor */
  ~Pulse();

  /* write out data */
  void Checkpoint(HDF5Wrapper &data_file);

  /* accessors methods */
  double *GetAField();
  double *GetTime();
  int GetMaxPulseLength();
};
