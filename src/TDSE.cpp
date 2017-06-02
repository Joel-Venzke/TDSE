// #include "config.h"
#include <petsc.h>
#include <iostream>
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "PETSCWrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "ViewWrapper.h"
#include "Wavefunction.h"

int main(int argc, char** argv)
{
  PETSCWrapper p_wrap(argc, argv);
  p_wrap.Print(
      "\n******************* Setting up Simulation *******************\n\n");

  p_wrap.PushStage("Set up");

  /* initialize all of the classes */
  ViewWrapper viewer_file("TDSE.h5");
  Parameters parameters("input.json");
  HDF5Wrapper h5_file(parameters);
  Pulse pulse(h5_file, parameters);
  Wavefunction wavefunction(h5_file, viewer_file, parameters);
  Hamiltonian hamiltonian(wavefunction, pulse, h5_file, parameters);
  Simulation s(hamiltonian, wavefunction, pulse, h5_file, viewer_file,
               parameters);
  p_wrap.PopStage(); /* Set up */

  if (parameters.GetRestart() != 1)
  {
    p_wrap.Print(
        "\n****************** Eigen State Calculation ******************\n\n");

    p_wrap.PushStage("Eigen State");
    /* get ground states */
    switch (parameters.GetStateSolverIdx())
    {
      case 0: /* File */
        break;
      case 2: /* Power */
        s.PowerMethod(parameters.GetNumStates());
        break;
    }
    p_wrap.PopStage(); /* Eigen State */
  }

  p_wrap.PushStage("Propagation");
  if (parameters.GetPropagate() == 1)
  {
    p_wrap.Print(
        "\n************************ Propagation ************************\n\n");
    s.Propagate();
  }
  p_wrap.PopStage(); /* Propagation */

  p_wrap.Print(
      "\n******************** Simulation Complete ********************\n\n");

  return 0;
}
