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

/**
 * @brief Calculates Wigner9j symbol
 * @details Calculates Wigner9j symbol in the form
 * < a b c >
 * < d e f >
 * < g h i >
 */
double Utils::Wigner9j(int a, int b, int c, int d, int e, int f, int g, int h, int i)
{
  return gsl_sf_coupling_9j(2 * a, 2 * b, 2 * c, 2 * d, 2 * e, 2 * f, 2 * g, 2 * h, 2 * i);
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

double Utils::Factorial(double n)
{
  return (n == 1 || n == 0) ? 1 : Factorial(n - 1) * n;
}

double Utils::DoubleFactorial(double n)
{
  return (n == 1 || n == 0) ? 1 : DoubleFactorial(n - 2) * n;
}

/* Utility function for RRC routine */
double Utils::C(int a, int b, int c)
{
  return Factorial(2*a+b+c+1)/Factorial(a+b+c+1)/Factorial(a)/DoubleFactorial(2*(a+b)+1)/DoubleFactorial(2*(a+c)+1);
}

double Utils::Sign(double num)
{
  if (num < 0)
  {
    return -1.;
  }
  else
  {
    return 1.;
  }
}

double Utils::RRC(int total_angular_momentum, int L, int l_xi, int l_yi, int l_xk, int l_yk, int parity)
{
  double phi; /* angle of jacobi rotation */
  double a_11, a_12, a_21, a_22; /* rotation matrix elements */
  int n_i, n_k; /* order of jacobi polynomials */
  int n_i2, n_k2; /* n_i and n_k times 2*/
  dcomp product, mu_nu_sum, lambda_product, lambda_sum;

  /* Making the infinite mass approximation 
  *  Full problem is arctan((-1)**p*sqrt((M*m_2/(m_1*m_3)))
  *  with m_3 as nucleus mass and M = m_1 + m_2 + m_3
  */
  if (parity % 2 == 0)
  {
      phi = atan(1);
  }
  else 
  {
      phi = atan(-1);
  }

  a_11 = cos(phi);
  a_22 = cos(phi);
  a_12 = sin(phi);
  a_21 = -sin(phi);

  n_i2 = (total_angular_momentum-l_xi-l_yi);
  n_k2 = (total_angular_momentum-l_xk-l_yk);
  if (n_i2 % 2 == 1 or n_k2 % 2 == 1 or n_i2 < 0 or n_k2 < 0)
  {
    return 0.0;
  }
  n_i = n_i2 / 2;
  n_k = n_k2 / 2;

  lambda_sum = dcomp(0.,0.);
  for (int lambda_1 = 0; lambda_1 < total_angular_momentum + 1; ++lambda_1)
  {
    for (int lambda_2 = 0; lambda_2 < total_angular_momentum - lambda_1 + 1; ++lambda_2)
    {
      for (int lambda_3 = 0; lambda_3 < total_angular_momentum - lambda_1 - lambda_2 + 1 ; ++lambda_3)
      {
        for (int lambda_4 = 0; lambda_4 < total_angular_momentum - lambda_1 - lambda_2 - lambda_3 + 1; ++lambda_4)
        {
          mu_nu_sum = dcomp(0.,0.);
          for (int mu = 0; mu < (total_angular_momentum - lambda_1 - lambda_2 - lambda_3 - lambda_4) / 2 + 1; ++mu)
          {
            for (int nu = 0; nu < (total_angular_momentum - lambda_1 - lambda_2 - lambda_3 - lambda_4 - 2 * mu) / 2 + 1; ++nu)
            {
              if ((2 * nu + 2 * mu + lambda_4 + lambda_3 + lambda_2 + lambda_1) == total_angular_momentum)
              {
                product = 1;
                ((mu)%2==0) ? product *= 1 : product *= -1;
                product *= pow(abs(a_12), 2 * mu + lambda_1 + lambda_2);
                product *= pow(abs(a_11), 2 * nu + lambda_3 + lambda_4);
                product *= C(mu,lambda_1,lambda_2);
                product *= C(nu,lambda_3,lambda_4);
                mu_nu_sum += product;
              }
            }
          }
          lambda_product = 1;
          lambda_product *= mu_nu_sum;
          lambda_product *= pow(dcomp(0.,1), (lambda_1 - lambda_2 + l_yk - l_yi));
          lambda_product *= 2*lambda_1+1;
          lambda_product *= 2*lambda_2+1;
          lambda_product *= 2*lambda_3+1;
          lambda_product *= 2*lambda_4+1;
          lambda_product *= ClebschGordanCoef(lambda_1,lambda_3,l_xi,0,0,0);
          lambda_product *= ClebschGordanCoef(lambda_2,lambda_3,l_xk,0,0,0);
          lambda_product *= ClebschGordanCoef(lambda_2,lambda_4,l_yi,0,0,0);
          lambda_product *= ClebschGordanCoef(lambda_1,lambda_4,l_yk,0,0,0);
          lambda_product *= pow(Sign(a_12), lambda_1);
          lambda_product *= pow(Sign(a_21), lambda_2);
          lambda_product *= pow(Sign(a_11), lambda_3);
          lambda_product *= pow(Sign(a_22), lambda_4);
          lambda_product *= Wigner9j(lambda_3, lambda_1, l_xi,
                                     lambda_2, lambda_4, l_yi,
                                     l_xk,     l_yk,     L);

          lambda_sum += lambda_product;
        }
      }
    }
  }


  ((n_i+n_k)%2==0) ? lambda_sum *= 1 : lambda_sum *= -1;
  lambda_sum /= sqrt(C(n_i,l_xi,l_yi));
  lambda_sum /= sqrt(C(n_k,l_xk,l_yk));
  return lambda_sum.real();
}
