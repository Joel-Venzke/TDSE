// #include "config.h"
#include <iostream>
// #include "Hamiltonian.h"
#include "Parameters.h"
#include "Pulse.h"
// #include "Simulation.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"
#include <petsc.h>

int main(int argc, char** argv) {
    // initialize all of the classes
    PetscErrorCode ierr;
    PetscLogStage stage;
    ierr = PetscInitialize(&argc,&argv,(char*)0,"TDSE");
    CHKERRQ(ierr);
    ierr = PetscInitializeFortran();
    CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Set up", &stage);
    CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);
    CHKERRQ(ierr);

    Parameters parameters("input.json");
    HDF5Wrapper data_file(parameters);
    Pulse pulse(data_file, parameters);
    Wavefunction wavefunction(data_file,parameters);
    // Hamiltonian hamiltonian(wavefunction,pulse,data_file,parameters);
    // // Simulation s(hamiltonian,wavefunction,pulse,data_file,parameters);

    ierr = PetscLogStagePop();
    CHKERRQ(ierr);
    ierr = PetscBarrier(NULL); CHKERRQ(ierr);
    CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Eigen State", &stage);
    CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);
    CHKERRQ(ierr);
    // // get ground states
    // switch (parameters.get_state_solver_idx()) {
    //     case 0: // File
    //         break;
    //     case 1: // ITP
    //         s.imag_time_prop(parameters.get_num_states());
    //         break;
    //     case 2: // Power
    //         s.power_method(parameters.get_num_states());
    //         break;
    // }
    ierr = PetscLogStagePop();
    CHKERRQ(ierr);
    ierr = PetscBarrier(NULL); CHKERRQ(ierr);
    CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Propagation", &stage);
    CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);
    CHKERRQ(ierr);
    // if (parameters.get_propagate()==1) {
    //     s.propagate();
    // }
    ierr = PetscLogStagePop();
    CHKERRQ(ierr);
    ierr = PetscBarrier(NULL); CHKERRQ(ierr);
    CHKERRQ(ierr);

    ierr = PetscFinalize(); CHKERRQ(ierr);
    CHKERRQ(ierr);
    return 0;
}
