#pragma once
#include <iostream>
#include "Parameters.h"
#include <complex>
#include "HDF5Wrapper.h"
#include <petsc.h>
#include <petscviewerhdf5.h>

#define dcomp std::complex<double>

class Wavefunction {
private:
    const double pi = 3.1415926535897;
    int    num_dims;        // number of dimensions
    double *dim_size;       // sizes of each dimension in a.u.
    double *delta_x;        // step sizes of each dimension in a.u.
    int    *num_x;          // number of grid points in each dimension
    int    num_psi_12;      // number of points in psi_1 and psi_2
    int    num_psi;         // number of points in psi
    double **x_value;       // location of grid point in each dimension
    dcomp  *psi_1;          // wavefunction for electron 1
    dcomp  *psi_1_gobbler;  // boundary for electron 1
    dcomp  *psi_2;          // wavefunction for electron 2
    dcomp  *psi_2_gobbler;  // boundary for electron 2
    Vec    psi;            // wavefunction for 2 electron system
    Vec    psi_gobbler;    // boundary for 2 electron system
    // true if psi_1 and psi_2 are allocated
    bool   psi_12_alloc;
    bool   psi_alloc;
    // false if its not the first time checkpointing the wavefunction
    bool   first_pass;
    double sigma;  // std of gaussian guess
    double offset; // distance that starts gobbler
    double width;  // width of gobbler

    int    write_counter;

    // hidden from user for safety
    void create_grid();
    void create_psi();
    void cleanup();
public:
    // Constructor
    Wavefunction(HDF5Wrapper& data_file, Parameters& p);

    // destructor
    ~Wavefunction();

    // IO
    void checkpoint(HDF5Wrapper& data_file, double time);
    void checkpoint_psi(HDF5Wrapper& data_file,
        H5std_string var_path, int write_idx);

    // tools
    void normalize();
    void normalize(dcomp *data, int length, double dx);
    double norm();
    double norm(dcomp *data, int length, double dx);
    // double get_energy(Eigen::SparseMatrix<dcomp> *h);
    void reset_psi();
    void gobble_psi();

    int* get_num_x();
    int  get_num_psi();
    int  get_num_psi_12();
    // Eigen::VectorXcd* get_psi();
    double*  get_delta_x();
    double** get_x_value();

    // error handling
    void end_run(std::string str);
    void end_run(std::string str, int exit_val);
};
