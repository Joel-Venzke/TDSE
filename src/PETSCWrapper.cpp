#include "PETSCWrapper.h"

PETSCWrapper::PETSCWrapper(int argc, char** argv)
{
  ierr  = PetscInitialize(&argc, &argv, (char*)0, "TDSE");
  ierr  = PetscInitializeFortran();
  ierr  = SlepcInitialize(&argc, &argv, (char*)0, "SLEPC");
  setup = true;
}

PETSCWrapper::PETSCWrapper() { setup = false; }

// Destructor
PETSCWrapper::~PETSCWrapper()
{
  if (setup)
  {
    ierr = SlepcFinalize();
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
