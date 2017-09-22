#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"

class Pulse : protected Utils
{
 private:
  PetscInt num_pulses;       /* number of pulses */
  PetscInt num_dims;         /* number of dimensions in a simulation*/
  double delta_t;            /* time step */
  PetscInt max_pulse_length; /* length of longest pulse; */
  PetscInt *pulse_shape_idx; /* index of pulse shape */
  double gaussian_sigma;  /* number of std before you stop the gaussian pulse */
  PetscInt *power_on;     /* power of sin ramp on */
  PetscInt *power_off;    /* power of sin ramp off */
  double *cycles_on;      /* ramp on cycles */
  double *cycles_plateau; /* plateau cycles */
  double *cycles_off;     /* ramp off cycles */
  double *cycles_delay;   /* cycles till it starts */
  double *cycles_total;   /* cycles in pulse */
  double *cep;            /* carrier envelope phase */
  double *energy;         /* photon energy */
  double *field_max;      /* max amplitude */
  double *time;           /* stores the time at each point */
  double ***pulse_value;  /* pulse value */
  double **pulse_envelope; /* envelope function of pulse */
  /* polarization for major axis of the field */
  double **polarization_vector_major;
  /* polarization for minor axis of the field */
  double **polarization_vector_minor;
  double **poynting_vector; /* poynting vector of the field */
  double *ellipticity;      /* major_min/minor_max of the field */
  PetscInt *helicity_idx;   /* helicity of the field */
  double **field;           /* total vector potential */
  /* true if the individual pulses and envelopes are allocated */
  bool pulse_alloc;

  double **file_time;  /* stores the time at each point from the file */
  double **file_pulse; /* stores the time at each point from the file */
  PetscInt *file_size;
  std::string experiment_type;

  /* private to avoid unneeded allocation calls and to protect the */
  /* developer form accessing garbage arrays */
  void InitializePulse(PetscInt i);
  void InitializePolarization();
  void InitializePulseLength();
  void ReadPulseFromFile();
  void InitializePulse();
  void InitializeTime();
  void DeallocatePulses();
  void InitializeField();
  double Interpolate(PetscInt pulse_idx, double time);

 public:
  /* Constructor */
  Pulse(HDF5Wrapper &data_file, Parameters &p);

  /* Destructor */
  ~Pulse();

  /* write out data */
  void Checkpoint();

  /* accessors methods */
  double **GetField();
  double *GetTime();
  PetscInt GetMaxPulseLength();
};
