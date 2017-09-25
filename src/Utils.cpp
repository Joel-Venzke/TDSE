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
