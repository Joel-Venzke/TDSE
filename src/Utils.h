#pragma once
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_gamma.h>
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
#include <unordered_map>
#include <vector>
#include "jacobi_polynomial.hpp"
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
  double C(int a, int b, int c);
  std::unordered_map< std::string, double > rrc_lookup;
  PetscLogEvent sphere_harm;
  PetscLogEvent jacobi_poly;
  PetscLogEvent rrc_time;
  PetscLogEvent rrc_load_time;

 public:
  Utils();
  void EndRun(std::string str);
  void EndRun(std::string str, PetscInt exit_val);

  std::string FileToString(std::string file_name);
  json FileToJson(std::string file_name);
  double ClebschGordanCoef(int l1, int l2, int l3, int m1, int m2, int m3);
  double Wigner9j(int a, int b, int c, int d, int e, int f, int g, int h,
                  int i);
  PetscInt GetIdxFromLM(PetscInt l_val, PetscInt m_val, PetscInt m_max);
  PetscInt GetHypersphereSize(PetscInt k_max, PetscInt l_max);
  void LoadRRC();
  double RRC(int total_angular_momentum, int L, int l_xi, int l_yi, int l_xk,
             int l_yk, int parity);
  double Factorial(double n);
  double DoubleFactorial(double n);
  double Sign(double num);
  double Normilization(int K, int l1, int l2);
  void SpherHarm(int K, int n, int lx, int ly, int L, int M,
                 std::vector< double > &angles, std::vector< double > &sphere,
                 std::vector< double > &x_vals);
};