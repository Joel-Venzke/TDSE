/**
 * @file TDSE.cpp
 * @brief Solves the Time-Dependent Schrodinger Equation for ultrafast laser
 * pulses
 * @author Joel Venzke
 * @date 06/13/2017
 */

#include <slepc.h>
#include <iostream>
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "PETSCWrapper.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "ViewWrapper.h"
#include "Wavefunction.h"

/**
 * @brief Solves the Time-Dependent Schrodinger Equation for ultrafast laser
 * pulses
 * @details Creates the needed objects, controls the structure of each
 * calculation using the Parameter class and controls the petsc timing stages
 */
int main(int argc, char** argv)
{
  mpi::communicator world;
  clock_t sim_time = clock();
  std::ofstream timing_file;

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
        s.FromFile(parameters.GetNumStates());
        break;
      case 2: /* Power */
        s.PowerMethod(parameters.GetNumStates());
        break;
      case 3: /* SLEPC */
        s.EigenSolve(parameters.GetNumStates());
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

  if (world.rank() == 0)
  {
    timing_file.open("total_run_time.log");
    timing_file << "Total run time: "
                << ((float)clock() - sim_time) / CLOCKS_PER_SEC << " (sec)\n";
    timing_file.close();
  }
  return 0;
}
