#include <iostream>
#include "Wavefunction.h"
#include <complex>
#include <math.h>    // ceil()

#define dcomp std::complex<double>

// prints error message, kills code and returns -1
void Wavefunction::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

// prints error message, kills code and returns exit_val
void Wavefunction::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}

Wavefunction::Wavefunction(HDF5Wrapper& data_file, Parameters & p) {
    std::cout << "Creating Wavefunction\n";

    // initialize values
    psi_12_alloc = false;
    first_pass   = true;
    num_dims     = p.get_num_dims();
    dim_size     = p.get_dim_size();
    delta_x      = p.get_delta_x();
    num_psi_12   = 1;

    // validation
    if (num_dims>1) {
        end_run("Only 1D is currently supported");
    }

    // allocate grid
    create_grid();

    // allocate psi_1, psi_2, and psi
    create_psi();

    // write out data
    checkpoint(data_file, 0, 0.0);

    for (int i=1; i<100; i++) {
        // allocate psi_1, psi_2, and psi
        create_psi(i*.1);

        // write out data
        checkpoint(data_file, i, i*p.get_delta_t());
    }

    // delete psi_1 and psi_2
    cleanup();

    std::cout << "Wavefunction created\n";
}

void Wavefunction::checkpoint(HDF5Wrapper& data_file, int write_idx,
    double time) {
    std::cout << "Checkpointing Wavefunction: " << write_idx << "\n";
    std::string str;

    // only write out at start
    if (first_pass) {
        // size of each dim
        data_file.write_object(num_x, num_dims, "/Wavefunction/num_x",
            "The number of physical dimension in the simulation");

        // write each dims x values
        for (int i=0; i<num_dims; i++) {
            str = "x_value_";
            str += std::to_string(i);
            data_file.write_object(x_value[i], num_x[i],
                "/Wavefunction/"+str,
                "The coordinates of the "+std::to_string(i)+
                "dimension");
        }

        // write psi_1 and psi_2 if still allocated
        if (psi_12_alloc) {
            data_file.write_object(psi_1, num_psi_12,
                "/Wavefunction/psi_1",
                "Wavefunction of first electron");
            data_file.write_object(psi_2, num_psi_12,
                "/Wavefunction/psi_2",
                "Wavefunction of second electron");
        }

        // write data and attribute
        data_file.write_object(psi, num_psi,
            "/Wavefunction/psi",
            "Wavefunction for the two electron system", write_idx);

        // write time and attribute
        data_file.write_object(time, "/Wavefunction/psi_time",
            "Time step that psi was written to disk", write_idx);

        // allow for future passes to write psi only
        first_pass = false;
    } else {
        // write whenever this function is called
        data_file.write_object(psi, num_psi,
            "/Wavefunction/psi", write_idx);

        // write time
        data_file.write_object(time, "/Wavefunction/psi_time",
            write_idx);
    }
}

void Wavefunction::create_grid() {
    int    center;    // idx of the 0.0 in the grid
    double current_x; // used for setting grid

    // allocation
    num_x    = new int[num_dims];
    x_value  = new double*[num_dims];

    // initialize for loop
    num_psi_12  = 1.0;

    // build grid
    for (int i=0; i<num_dims; i++) {
        num_x[i] =  ceil(dim_size[i]/delta_x[i]);

        // odd number so it is even on both sides
        if (num_x[i]%2==0) num_x[i]++;

        // size of 1d array for psi
        num_psi_12  *= num_x[i];

        // find center of grid
        center = num_x[i]/2+1;

        // allocate grid
        x_value[i] = new double[num_x[i]];

        // store center
        x_value[i][center] = 0.0;

        // loop over all others
        for (int j=center-1; j>0; j--) {
            // get x value
            current_x = (j-center)*delta_x[i];

            // double checking index
            if (j-1<0 || num_x[i]-j>=num_x[i]) {
                end_run("Allocation error in grid");
            }

            // set negative side
            x_value[i][j-1] = current_x;
            // set positive side
            x_value[i][num_x[i]-j] = -1*current_x;
        }
    }
}

// builds psi from 2 Gaussian psi (one for each electron)
void Wavefunction::create_psi() {
    double sigma;     // variance for Gaussian in psi
    double sigma2;    // variance squared for Gaussian in psi
    double x2;        // x value squared

    // allocate data
    psi_1 = new dcomp[num_psi_12];
    psi_2 = new dcomp[num_psi_12];

    sigma  = 0.50;

    sigma2 = sigma*sigma;
    // TODO: needs to be changed for more than one dim
    for (int i=0; i<num_psi_12; i++) {
        // get x value squared
        x2 = x_value[0][i];
        x2 *= x2;

        // Gaussian centered around 0.0 with variation sigma
        psi_1[i] = dcomp(exp(-1*x2/(2*sigma2)),0.0);
        psi_2[i] = dcomp(exp(-1*x2/(2*sigma2)),0.0);
    }

    // psi_1 and psi_2 are allocated
    psi_12_alloc = true;

    // get size of psi
    num_psi = num_psi_12*num_psi_12;

    // allocate psi
    psi = new dcomp[num_psi];

    // tensor product of psi_1 and psi_2
    for (int i=0; i<num_psi_12; i++) { // e_2 dim
        for (int j=0; j<num_psi_12; j++) { // e_1 dim
            psi[i*num_psi_12+j] = psi_1[i]*psi_2[j];
        }
    }

    // normalize all psi
    normalize();
}

void Wavefunction::create_psi(double offset) {
    double sigma;     // variance for Gaussian in psi
    double sigma2;    // variance squared for Gaussian in psi
    double x1;        // x value squared
    double x2;        // x value squared

    // allocate data
    if (! psi_12_alloc) {
        psi_1 = new dcomp[num_psi_12];
        psi_2 = new dcomp[num_psi_12];
    }

    sigma  = 0.50;

    sigma2 = sigma*sigma;
    // TODO: needs to be changed for more than one dim
    for (int i=0; i<num_psi_12; i++) {
        // get x value squared
        x1 = x_value[0][i];
        x1 *= x1;
        x2 = (x_value[0][i]-offset);
        x2 *= x2;

        // Gaussian centered around 0.0 with variation sigma
        psi_1[i] = dcomp(exp(-1*x1/(2*sigma2)),0.0);
        psi_2[i] = dcomp(exp(-1*x2/(2*sigma2)),0.0);
    }
    if (! psi_12_alloc) {
        // psi_1 and psi_2 are allocated
        psi_12_alloc = true;

        // get size of psi
        num_psi = num_psi_12*num_psi_12;

        // allocate psi
        psi = new dcomp[num_psi];
    }

    // tensor product of psi_1 and psi_2
    for (int i=0; i<num_psi_12; i++) { // e_2 dim
        for (int j=0; j<num_psi_12; j++) { // e_1 dim
            psi[i*num_psi_12+j] = psi_1[i]*psi_2[j];
        }
    }

    // normalize all psi
    normalize();
}

// delete psi_1 and psi_2 since they are not used later
void Wavefunction::cleanup() {
    delete psi_1;
    delete psi_2;
    psi_12_alloc = false;
}

// normalize psi_1, psi_2, and psi
void Wavefunction::normalize() {
    if (psi_12_alloc) {
        normalize(psi_1, num_psi_12, delta_x[0]);
        normalize(psi_2, num_psi_12, delta_x[0]);
    }
    normalize(psi, num_psi, delta_x[0]);
}

// normalizes the array provided
void Wavefunction::normalize(dcomp *data, int length, double dx) {
    double total = 0;

    // trapezoidal rule of conj(data)*data
    total += (std::conj(data[0])*data[0]).real();
    total += (std::conj(data[length-1])*data[length-1]).real();
    for (int i=1; i<length-1; i++) {
        total += 2.0*(std::conj(data[i])*data[i]).real();
    }
    total  *= dx/2.0;

    // square root to get normalization factor
    total = sqrt(total);

    // normalize data
    for (int i=0; i<length; i++) {
        data[i].real(data[i].real()/total);
        data[i].imag(data[i].imag()/total);
    }
}

// destructor
Wavefunction::~Wavefunction(){
    std::cout << "Deleting Wavefunction\n";
    // do not delete dim_size or delta_x
    // since they belong to the Parameter class
    // and will be freed there
    delete num_x;
    for (int i=0; i<num_dims; i++) {
        delete x_value[i];
    }
    delete[] x_value;
    if (psi_12_alloc) {
        delete psi_1;
        delete psi_2;
    }
    delete psi;
}
