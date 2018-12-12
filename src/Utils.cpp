#include "Utils.h"

/* prints error message, kills code and returns -1 */
void Utils::EndRun(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  mpi::environment::abort(-1);
}

/* prints error message, kills code and returns exit_val */
void Utils::EndRun(std::string str, PetscInt exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  mpi::environment::abort(exit_val);
}

/*
 * Reads file and returns a string
 */
std::string Utils::FileToString(std::string file_name)
{
  // TODO(jove7731): check if file exists
  std::ifstream t(file_name);
  std::string str((std::istreambuf_iterator< char >(t)),
                  std::istreambuf_iterator< char >());
  return str;
}

/*
 * Reads file and returns a json object
 */
json Utils::FileToJson(std::string file_name)
{
  auto j = json::parse(FileToString(file_name));
  return j;
}

/**
 * @brief Calculates Clebsch-Gordan Coefficients
 * @details Calculates Clebsch-Gordan Coefficients in the form
 * <l1,m1,l2,m2|l3,m3>. Note for dipole we get non zero values for
 * <l,m,1,0|(l+-1),m>
 *
 * @param l1 value in <l1,m1,l2,m2|l3,m3>
 * @param l2 value in <l1,m1,l2,m2|l3,m3>
 * @param l3 value in <l1,m1,l2,m2|l3,m3>
 * @param m1 value in <l1,m1,l2,m2|l3,m3>
 * @param m2 value in <l1,m1,l2,m2|l3,m3>
 * @param m3 value in <l1,m1,l2,m2|l3,m3>
 */
double Utils::ClebschGordanCoef(int l1, int l2, int l3, int m1, int m2, int m3)
{
  int neg_power = l1 - l2 + m3;

  /* include */
  if (neg_power % 2 == 0)
  {
    return sqrt(2 * l3 + 1) *
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
  }
  else
  {
    return -1.0 * sqrt(2 * l3 + 1) *
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
  }
}

PetscInt Utils::GetIdxFromLM(PetscInt l_val, PetscInt m_val, PetscInt m_max)
{
  /* Indexing for m increases faster then l so each element can be labeled by */
  /* (l,m) in the order  (0,0), (1,-1), (1,0), (1,1), (2,-2), ect. */

  /* if l is less than m_max we get the normal l^2 spherical harmonics */
  if (l_val <= m_max)
  {
    return l_val * l_val + l_val + m_val;
  }
  else /* Once we hit m_max, then each l has a fixed size of 2 * m_max + 1 */
  {
    return m_max * m_max + (l_val - m_max) * (2 * m_max + 1) + m_max + m_val;
  }
}
