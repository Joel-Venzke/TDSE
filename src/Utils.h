#pragma once
#include <math.h>
#include <petsc.h>
#include <stdlib.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/group.hpp>
#include <boost/optional/optional_io.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

namespace mpi = boost::mpi;

typedef std::complex<double> dcomp;

class Utils
{
 protected:
  mpi::communicator world;
  const double pi = 3.1415926535897;
  const double c  = 1 / 7.2973525664e-3;

 public:
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};