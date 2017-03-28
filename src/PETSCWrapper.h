#pragma once
#include <petsc.h>

class PETSCWrapper {
private:
    PetscInt       rank;
    PetscInt       ierr;
    PetscLogStage  stage;
public:
    // Constructor
    PETSCWrapper(int argc, char** argv);

    // Destructor
    ~PETSCWrapper();

    void PushStage(std::string stage_name);
    void PopStage();
};