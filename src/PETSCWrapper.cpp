#include "PETSCWrapper.h"

PETSCWrapper::PETSCWrapper(int argc, char** argv)
{
  ierr  = PetscInitialize(&argc, &argv, (char*)0, "TDSE");
  ierr  = PetscInitializeFortran();
  setup = true;
}

PETSCWrapper::PETSCWrapper() { setup = false; }

// Destructor
PETSCWrapper::~PETSCWrapper()
{
  if (setup)
  {
    ierr = PetscFinalize();
  }
}

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

void PETSCWrapper::Print(std::string message)
{
  PetscPrintf(PETSC_COMM_WORLD, message.c_str());
}
