#pragma once
#include <petsc.h>
#include <complex>
#include <vector>
#include "H5Cpp.h"
#include "Parameters.h"
#include "Utils.h"

class HDF5Wrapper : protected Utils
{
 private:
  H5::H5File *data_file;
  std::string file_name;
  bool file_open;
  bool header;
  H5::CompType *complex_data_type;

  hsize_t *GetHsizeT(PetscInt size, PetscInt *dims);
  void DefineComplex();

 public:
  /* Constructor */
  HDF5Wrapper(Parameters &p);
  HDF5Wrapper(std::string f_name, Parameters &p);
  HDF5Wrapper(std::string f_name);

  /* destructor */
  ~HDF5Wrapper();

  void Open();
  void Close();

  void SetHeader(bool h);

  void CreateGroup(H5std_string group_path);

  // write single entry
  void WriteObject(PetscInt data, H5std_string var_path);
  void WriteObject(PetscInt data, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(double data, H5std_string var_path);
  void WriteObject(double data, H5std_string var_path, H5std_string attribute);

  void WriteObject(PetscInt *data, PetscInt size, H5std_string var_path);
  void WriteObject(PetscInt *data, PetscInt size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(PetscInt *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path);
  void WriteObject(PetscInt *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path, H5std_string attribute);

  void WriteObject(double *data, PetscInt size, H5std_string var_path);
  void WriteObject(double *data, PetscInt size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(double *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path);
  void WriteObject(double *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path, H5std_string attribute);

  void WriteObject(dcomp *data, PetscInt size, H5std_string var_path);
  void WriteObject(dcomp *data, PetscInt size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(dcomp *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path);
  void WriteObject(dcomp *data, PetscInt size, PetscInt *dims,
                   H5std_string var_path, H5std_string attribute);

  /* time series writes */
  void WriteObject(dcomp *data, PetscInt size, H5std_string var_path,
                   PetscInt write_idx);
  void WriteObject(dcomp *data, PetscInt size, H5std_string var_path,
                   H5std_string attribute, PetscInt write_idx);

  void WriteObject(double data, H5std_string var_path, PetscInt write_idx);
  void WriteObject(double data, H5std_string var_path, H5std_string attribute,
                   PetscInt write_idx);

  /* write for parameters */
  void WriteHeader(Parameters &p);

  /* reads restart and validates file */
  void ReadRestart(Parameters &p);
  void ReadRestart(Parameters &p, std::string f_name);
};