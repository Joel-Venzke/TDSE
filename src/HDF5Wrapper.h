#pragma once
#include <petsc.h>
#include <complex>
#include <memory>
#include <vector>
#include "H5Cpp.h"
#include "Parameters.h"
#include "Utils.h"
#include "hdf5.h"

class HDF5Wrapper : protected Utils
{
 private:
  mpi::communicator world;
  std::shared_ptr<H5::H5File> data_file;
  std::string file_name;
  bool file_open;
  bool header;

  std::unique_ptr<hsize_t[]> GetHsizeT(int &size, int *dims);

 public:
  /* ructor */
  HDF5Wrapper(Parameters &p);
  HDF5Wrapper(std::string f_name, Parameters &p);
  HDF5Wrapper(std::string f_name);

  /* destructor */
  ~HDF5Wrapper();

  void Open();
  void Close();

  template <typename T>
  H5::PredType getter(T data);
  template <typename T>
  H5::PredType getter(std::unique_ptr<T[]> &data);

  void SetHeader(bool h);

  void CreateGroup(H5std_string group_path);
  /* begin template try */
  template <typename T>
  void WriteObject(T data, H5std_string var_path);
  template <typename T>
  void WriteObject(T data, H5std_string var_path, H5std_string attribute);

  template <typename T>
  void WriteObject(T *data, int size, H5std_string var_path);
  template <typename T>
  void WriteObject(T *data, int size, H5std_string var_path,
                   H5std_string attribute);
  template <typename T>
  void WriteObject(T *data, int size, int *dims, H5std_string var_path);
  template <typename T>
  void WriteObject(T *data, int size, int *dims, H5std_string var_path,
                   H5std_string attribute);

  /**/
  /* no Templates for anything involving dcomps for now */
  /**/

  /* write single entry */
  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  H5std_string var_path);
  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  H5std_string var_path, H5std_string attribute);

  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  std::unique_ptr<int[]> &dims, H5std_string var_path);
  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  std::unique_ptr<int[]> &dims, H5std_string var_path,
  //                  H5std_string attribute);

  // /* time series writes */
  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  H5std_string var_path, int write_idx);
  // void WriteObject(std::unique_ptr<dcomp[]> &data, int size,
  //                  H5std_string var_path, H5std_string attribute,
  //                  int write_idx);

  void WriteObject(double data, H5std_string var_path, int write_idx);
  void WriteObject(double data, H5std_string var_path, H5std_string attribute,
                   int write_idx);

  /* write for parameters */
  void WriteHeader(Parameters &p);

  /* reads restart and validates file */
  void ReadRestart(Parameters &p);
  void ReadRestart(Parameters &p, std::string f_name);

  /* kill run */
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};