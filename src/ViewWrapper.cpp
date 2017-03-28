#include "ViewWrapper.h"

/* Constructor */
/* Does not open file */
ViewWrapper::ViewWrapper(std::string f_name)
{
  /* set file name */
  SetFileName(f_name);

  /* not open */
  file_open = false;
}

/* no file set use open to set file */
ViewWrapper::ViewWrapper()
{
  /* initialize bools to false */
  file_name_set = false;
  file_open     = false;
}

/* destructor */
ViewWrapper::~ViewWrapper()
{
  /* close file if open */
  if (file_open)
  {
    Close();
  }
}

/* set the file name for this object */
void ViewWrapper::SetFileName(std::string f_name)
{
  if (file_name_set)
  {
    EndRun("File name already set to " + file_name);
  }

  file_name     = f_name;
  file_name_set = true;
}

/* Opens file in various modes
 * r: read only
 * w: write only (new file)
 * a: append write only
 * u: update read and write (new file)
 * ua: update read and write           */
void ViewWrapper::Open(std::string mode)
{
  if (!file_name_set)
  {
    EndRun("No file name set");
  }

  if (file_open)
  {
    EndRun("File already open");
  }

  if (mode == "w")
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_WRITE, &data_file);
  }
  else if (mode == "r")
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_READ, &data_file);
  }
  else if (mode == "a")
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_APPEND, &data_file);
  }
  else if (mode == "u")
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_UPDATE, &data_file);
  }
  else if (mode == "ua")
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_APPEND_UPDATE, &data_file);
  }
  else
  {
    EndRun("Invalid file mode " + mode);
  }
  PetscViewerSetFromOptions(data_file);
  file_open = true;
}

/* closes file */
void ViewWrapper::Close()
{
  if (file_open)
  {
    ierr = PetscViewerDestroy(&data_file);
  }
}

/* push group */
void ViewWrapper::PushGroup(std::string group_path)
{
  PetscViewerHDF5PushGroup(data_file, group_path.c_str());
}

const char *ViewWrapper::GetGroup()
{
  const char *name = nullptr;
  ierr             = PetscViewerHDF5GetGroup(data_file, &name);
  return name;
}

/* pop group */
void ViewWrapper::PopGroup()
{
  const char *name = nullptr;
  ierr             = PetscViewerHDF5GetGroup(data_file, &name);
  if (name == NULL)
  {
    EndRun("No group left to pop");
  }
  ierr = PetscViewerHDF5PopGroup(data_file);
}

/* sets the time step for the file */
void ViewWrapper::SetTime(PetscInt time_step)
{
  PetscViewerHDF5SetTimestep(data_file, time_step);
}

/* writes and attribute */
void ViewWrapper::WriteAttribute(std::string var_name, std::string name,
                                 PetscReal value)
{
  std::string group_path = GetGroup();
  std::cout << group_path << "\n";
  PetscViewerHDF5WriteAttribute(data_file, (group_path + var_name).c_str(),
                                name.c_str(), PETSC_DOUBLE, &value);
}

void ViewWrapper::WriteAttribute(std::string var_name, std::string name,
                                 std::string value)
{
  std::string group_path = GetGroup();
  std::cout << group_path << "\n";
  PetscViewerHDF5WriteAttribute(data_file, (group_path + var_name).c_str(),
                                name.c_str(), PETSC_STRING, value.c_str());
}

/* write a frame */
void ViewWrapper::WriteObject(PetscObject data)
{
  PetscObjectView(data, data_file);
}

/* reads a frame */
void ViewWrapper::ReadObject(PetscObject data) {}

/* end run after printing error string with exit value -1 */
void ViewWrapper::EndRun(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  exit(-1);
}

/* end run after printing error string with exit_val */
void ViewWrapper::EndRun(std::string str, int exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  exit(exit_val);
}
