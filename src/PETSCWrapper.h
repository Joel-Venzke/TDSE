#pragma once
#include <petsc.h>

class PETSCWrapper
{
 private:
  PetscInt ierr;
  PetscLogStage stage;
  bool setup; /* keep from calling petsc finalize*/

 public:
  // Constructor
  PETSCWrapper();
  PETSCWrapper(int argc, char** argv);

  // Destructor
  ~PETSCWrapper();

  void PushStage(std::string stage_name);
  void PopStage();
};