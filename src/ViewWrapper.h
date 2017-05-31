#pragma once
#include <petscviewerhdf5.h>
#include "Utils.h"

class ViewWrapper : protected Utils
{
 private:
  PetscViewer data_file; /* viewer object */
  bool file_name_set;    /* true if file name has been set */
  bool file_open;        /* true if file is open */
  PetscInt ierr;         /* error code */
 public:
  std::string file_name; /* file name */
  /* Constructor
   * does not open file */
  ViewWrapper(std::string f_name);

  /* no file set use open to set file */
  ViewWrapper();

  /* destructor */
  ~ViewWrapper();

  /* set the file name for this object */
  void SetFileName(std::string f_name);

  /* Opens file in various modes
   * r: read only
   * w: write only (new file)
   * a: append write only        */
  void Open(std::string mode = "w");

  /* closes file */
  void Close();

  /* push group */
  void PushGroup(std::string group_path);

  const char* GetGroup();

  /* pop group */
  void PopGroup();

  /* sets the time step for the file */
  void SetTime(PetscInt time_step);

  /* writes and attribute */
  void WriteAttribute(std::string path, std::string name, PetscReal value);
  void WriteAttribute(std::string path, std::string name, std::string value);

  /* write a frame */
  void WriteObject(PetscObject data);

  /* reads a frame */
  void ReadObject(Vec data);
  void ReadObject(Mat data);
};
