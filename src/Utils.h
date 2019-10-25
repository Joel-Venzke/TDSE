#pragma once
#include <gsl/gsl_sf_coupling.h>
#include <math.h>
#include <slepc.h>
#include <stdlib.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "json.hpp"

using json = nlohmann::json;

namespace mpi = boost::mpi;

typedef std::complex< double > dcomp;

class Utils
{
 protected:
  mpi::communicator world;
  const double pi  = 3.1415926535897932384626433832795;
  const double c   = 1 / 7.2973525664e-3;
  const dcomp imag = dcomp(0.0, 1.0);

 public:
  void EndRun(std::string str);
  void EndRun(std::string str, PetscInt exit_val);

  std::string FileToString(std::string file_name);
  json FileToJson(std::string file_name);
  double ClebschGordanCoef(int l1, int l2, int l3, int m1, int m2, int m3);
  PetscInt GetIdxFromLM(PetscInt l_val, PetscInt m_val, PetscInt m_max);
};