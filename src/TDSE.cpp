// #include "config.h"
#include <iostream>
#include "PETSCWrapper.h"
#include "ViewWrapper.h"
#include "Parameters.h"
#include "HDF5Wrapper.h"
#include "Pulse.h"
// #include "Wavefunction.h"
// #include "Hamiltonian.h"
// #include "Simulation.h"
#include <petsc.h>

int main(int argc, char** argv)
{
  PETSCWrapper p_wrap(argc,argv);
  p_wrap.PushStage("Set up");

  /* initialize all of the classes */
  ViewWrapper viewer_file("TDSE.h5");
  Parameters parameters(viewer_file, "input.json");
  HDF5Wrapper data_file(parameters);
  Pulse pulse(data_file, parameters);
  // Wavefunction wavefunction(data_file,parameters);
  // Hamiltonian hamiltonian(wavefunction,pulse,data_file,parameters);
  // Simulation s(hamiltonian,wavefunction,pulse,data_file,parameters);
  p_wrap.PopStage();

  p_wrap.PushStage("Eigen State");
  /* get ground states */
  switch (parameters.get_state_solver_idx()) {
    case 0: /* File */
      break;
    case 1: /* ITP */
      s.imag_time_prop(parameters.get_num_states());
      break;
    case 2: /* Power */
      s.power_method(parameters.get_num_states());
      break;
  }
  p_wrap.PopStage();

  p_wrap.PushStage("Propagation");
  if (parameters.get_propagate()==1) {
    s.propagate();
  }
  p_wrap.PopStage();
  return 0;
}
