#include "PETSCWrapper.h"

PETSCWrapper::PETSCWrapper(int argc, char** argv)
{
  ierr = PetscInitialize(&argc, &argv, (char*)0, "TDSE");
  ierr = PetscInitializeFortran();
}

// Destructor
PETSCWrapper::~PETSCWrapper() { ierr = PetscFinalize(); }

void PETSCWrapper::PushStage(std::string stage_name)
{
  ierr = PetscLogStageRegister(stage_name.c_str(), &stage);
  ierr = PetscLogStagePush(stage);
}

void PETSCWrapper::PopStage()
{
  ierr = PetscLogStagePop();
  ierr = PetscBarrier(NULL);
}
