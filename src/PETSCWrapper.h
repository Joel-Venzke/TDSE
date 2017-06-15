#pragma once
// #include <petsc.h>
#include <slepceps.h>

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
  void Print(std::string message);
};