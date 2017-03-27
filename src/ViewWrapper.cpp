#include "ViewWrapper.h"

/* constructor */
/* file_name needs ending ".h5" */
/* overwrites existing file */
ViewWrapper::ViewWrapper(std::string f_name) {
}

/* constructor */
/* file_name needs ending ".h5" */
ViewWrapper::ViewWrapper() {
}

void ViewWrapper::open(std::string file_name, std::string mode = 'w') {
}

void ViewWrapper::close() {
}

/* Writes int to HDF5 */
/* takes int and var_path as inputs */
void ViewWrapper::write_object(
    int data,
    H5std_string var_path) {
}

void ViewWrapper::create_group(H5std_string group_path) {
}

void ViewWrapper::close() {
}

/* destructor */
ViewWrapper::~ViewWrapper(){
    std::cout << "Deleting ViewWrapper: " << file_name << "\n";
    close();
}

/* end run after printing error string with exit value -1 */
void ViewWrapper::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

/* end run after printing error string with exit_val */
void ViewWrapper::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}