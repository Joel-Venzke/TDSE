#include "Utils.h"

Utils::Utils()
{
  PetscLogEventRegister("Sphere Harm", PETSC_VIEWER_CLASSID, &sphere_harm);
  PetscLogEventRegister("jacobi_poly", PETSC_VIEWER_CLASSID, &jacobi_poly);
  PetscLogEventRegister("RRC", PETSC_VIEWER_CLASSID, &rrc_time);
  PetscLogEventRegister("RRC_LOAD", PETSC_VIEWER_CLASSID, &rrc_load_time);
}
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
double Utils::Wigner9j(int a, int b, int c, int d, int e, int f, int g, int h,
                       int i)
{
  return gsl_sf_coupling_9j(2 * a, 2 * b, 2 * c, 2 * d, 2 * e, 2 * f, 2 * g,
                            2 * h, 2 * i);
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

PetscInt Utils::GetHypersphereSizeRRC(PetscInt k_max, PetscInt l_max)
{
  /* These loops ranges could be optimized though this is only called a few
   *  times and therefore the simple implementation is uses
   */
  PetscInt count = 0;
  for (int L_val = 0; L_val < l_max + 1; ++L_val)
  {
    for (int k_val = 0; k_val < k_max + 1; ++k_val)
    {
      for (int l_1 = 0; l_1 < k_val + 1;
           l_1 += 2) /* only even values couple to ground state */
      {
        for (int l_2 = 0; l_2 < k_val + 1; ++l_2)
        {
          for (int n = 0; n < k_val / 2 + 1; ++n)
          {
            if (k_val == l_1 + l_2 + 2 * n)
            {
              for (int L = abs(l_1 - l_2); L < l_1 + l_2 + 1; ++L)
              {
                if (L == L_val and k_val % 2 == L_val % 2)
                {
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  return count;
}

PetscInt Utils::GetHypersphereSizeNonRRC(PetscInt k_max, PetscInt l_max)
{
  /* These loops ranges could be optimized though this is only called a few
   *  times and therefore the simple implementation is uses
   */
  PetscInt count = 0;
  for (int L_val = 0; L_val < l_max + 1; ++L_val)
  {
    for (int k_val = 0; k_val < k_max + 1; ++k_val)
    {
      for (int l_1 = 0; l_1 < k_val + 1; l_1 += 1)
      {
        for (int l_2 = 0; l_2 < k_val + 1; ++l_2)
        {
          for (int n = 0; n < k_val / 2 + 1; ++n)
          {
            if (k_val == l_1 + l_2 + 2 * n)
            {
              for (int L = abs(l_1 - l_2); L < l_1 + l_2 + 1; ++L)
              {
                if (L == L_val)
                {
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  return count;
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
  return Factorial(2 * a + b + c + 1) / Factorial(a + b + c + 1) /
         Factorial(a) / DoubleFactorial(2 * (a + b) + 1) /
         DoubleFactorial(2 * (a + c) + 1);
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

void Utils::LoadRRC()
{
  PetscLogEventBegin(rrc_load_time, 0, 0, 0, 0);
  std::ifstream file("RRC.txt");

  std::string line;
  int count = 0;

  while (std::getline(file, line))
  {  // '\n' is the default delimiter

    std::istringstream iss(line);
    std::string token;
    std::string key;
    while (std::getline(iss, token, '\t'))
    {
      if (count % 2 == 0)
      {
        key = token;
      }
      else
      {
        rrc_lookup[key] = std::stod(token);
      }
      count++;
    }
  }
  file.close();
  PetscLogEventEnd(rrc_load_time, 0, 0, 0, 0);
}

double Utils::RRC(int total_angular_momentum, int L, int l_xi, int l_yi,
                  int l_xk, int l_yk, int parity)
{
  double phi;                    /* angle of jacobi rotation */
  double a_11, a_12, a_21, a_22; /* rotation matrix elements */
  int n_i, n_k;                  /* order of jacobi polynomials */
  int n_i2, n_k2;                /* n_i and n_k times 2*/
  dcomp product, mu_nu_sum, lambda_product, lambda_sum;
  std::string key;
  PetscLogEventBegin(rrc_time, 0, 0, 0, 0);
  key = to_string(total_angular_momentum) + "_" + to_string(L) + "_" +
        to_string(l_xi) + "_" + to_string(l_yi) + "_" + to_string(l_xk) + "_" +
        to_string(l_yk) + "_" + to_string(parity);
  if (rrc_lookup.count(key) == 1)
  {
    PetscLogEventEnd(rrc_time, 0, 0, 0, 0);
    return rrc_lookup[key];
  }

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

  n_i2 = (total_angular_momentum - l_xi - l_yi);
  n_k2 = (total_angular_momentum - l_xk - l_yk);
  if (n_i2 % 2 == 1 or n_k2 % 2 == 1 or n_i2 < 0 or n_k2 < 0)
  {
    PetscLogEventEnd(rrc_time, 0, 0, 0, 0);
    return 0.0;
  }

  n_i = n_i2 / 2;
  n_k = n_k2 / 2;

  lambda_sum = dcomp(0., 0.);
  for (int lambda_1 = 0; lambda_1 < total_angular_momentum + 1; ++lambda_1)
  {
    for (int lambda_2 = 0; lambda_2 < total_angular_momentum - lambda_1 + 1;
         ++lambda_2)
    {
      for (int lambda_3 = 0;
           lambda_3 < total_angular_momentum - lambda_1 - lambda_2 + 1;
           ++lambda_3)
      {
        for (int lambda_4 = 0; lambda_4 < total_angular_momentum - lambda_1 -
                                              lambda_2 - lambda_3 + 1;
             ++lambda_4)
        {
          mu_nu_sum = dcomp(0., 0.);
          for (int mu = 0; mu < (total_angular_momentum - lambda_1 - lambda_2 -
                                 lambda_3 - lambda_4) /
                                        2 +
                                    1;
               ++mu)
          {
            for (int nu = 0; nu < (total_angular_momentum - lambda_1 -
                                   lambda_2 - lambda_3 - lambda_4 - 2 * mu) /
                                          2 +
                                      1;
                 ++nu)
            {
              if ((2 * nu + 2 * mu + lambda_4 + lambda_3 + lambda_2 +
                   lambda_1) == total_angular_momentum)
              {
                product = 1;
                ((mu) % 2 == 0) ? product *= 1 : product *= -1;
                product *= pow(abs(a_12), 2 * mu + lambda_1 + lambda_2);
                product *= pow(abs(a_11), 2 * nu + lambda_3 + lambda_4);
                product *= C(mu, lambda_1, lambda_2);
                product *= C(nu, lambda_3, lambda_4);
                mu_nu_sum += product;
              }
            }
          }
          lambda_product = 1;
          lambda_product *= mu_nu_sum;
          lambda_product *=
              pow(dcomp(0., 1), (lambda_1 - lambda_2 + l_yk - l_yi));
          lambda_product *= 2 * lambda_1 + 1;
          lambda_product *= 2 * lambda_2 + 1;
          lambda_product *= 2 * lambda_3 + 1;
          lambda_product *= 2 * lambda_4 + 1;
          lambda_product *=
              ClebschGordanCoef(lambda_1, lambda_3, l_xi, 0, 0, 0);
          lambda_product *=
              ClebschGordanCoef(lambda_2, lambda_3, l_xk, 0, 0, 0);
          lambda_product *=
              ClebschGordanCoef(lambda_2, lambda_4, l_yi, 0, 0, 0);
          lambda_product *=
              ClebschGordanCoef(lambda_1, lambda_4, l_yk, 0, 0, 0);
          lambda_product *= pow(Sign(a_12), lambda_1);
          lambda_product *= pow(Sign(a_21), lambda_2);
          lambda_product *= pow(Sign(a_11), lambda_3);
          lambda_product *= pow(Sign(a_22), lambda_4);
          lambda_product *= Wigner9j(lambda_3, lambda_1, l_xi, lambda_2,
                                     lambda_4, l_yi, l_xk, l_yk, L);

          lambda_sum += lambda_product;
        }
      }
    }
  }

  ((n_i + n_k) % 2 == 0) ? lambda_sum *= 1 : lambda_sum *= -1;
  lambda_sum /= sqrt(C(n_i, l_xi, l_yi));
  lambda_sum /= sqrt(C(n_k, l_xk, l_yk));

  rrc_lookup[key] = lambda_sum.real();
  PetscLogEventEnd(rrc_time, 0, 0, 0, 0);
  return lambda_sum.real();
}

double Utils::Normilization(int K, int l1, int l2)
{
  double n = (K - l1 - l2) / 2.;
  return sqrt(2. * Factorial(n) * (K + 2) / gsl_sf_gamma(n + l1 + 3 / 2.) /
              gsl_sf_gamma(n + l2 + 3 / 2.) * Factorial(n + l1 + l2 + 1));
}

/* K, n, lx, ly, L, M are spherical harmonic eigen values
 * angles are the angles you wish to evaluate the spherical harmonic at
 * sphere is the return variable
 * x_vals = cos(2. * angle[idx])
 */
void Utils::SpherHarm(int K, int n, int lx, int ly, int L, int M,
                      std::vector< double > &angle,
                      std::vector< double > &sphere,
                      std::vector< double > &x_vals)
{
  double *j_vals;
  double norm = Normilization(K, ly, lx);
  PetscLogEventBegin(sphere_harm, 0, 0, 0, 0);
  PetscLogEventBegin(jacobi_poly, 0, 0, 0, 0);

  j_vals = j_polynomial(angle.size(), n, ly + 0.5, lx + 0.5, &x_vals[0]);
  PetscLogEventEnd(jacobi_poly, 0, 0, 0, 0);

  /* calculate spherical harmonics */
  for (int idx = 0; idx < angle.size(); ++idx)
  {
    sphere[idx] = norm * j_vals[n * angle.size() + idx] *
                  pow(cos(angle[idx]), lx) * pow(sin(angle[idx]), ly);
  }
  PetscLogEventEnd(sphere_harm, 0, 0, 0, 0);
  delete j_vals;
}
