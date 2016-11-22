#include "config.h"
#include <iostream>
#include "Hamiltonian.h"
#include "Observables.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefuction.h"
#include <netcdfcpp.h>


int main() {
	Parameters p("input.json");
    if (p.get_restart()) {
        std::cout << "Restart not implemented!\n";
        return 2;
    }
    else {
		// Open NetCDF file
		NcFile data_file("TDSE.nc", NcFile::Replace);
	    if (!data_file.is_valid()){
	        std::cout << "Couldn't open file!\n";
	        return 2;
	    }
	    p.write_header(&data_file);
	}
    Hamiltonian h;
    Observables o;
    Pulse pulse;
    Simulation s;
    Wavefuction w;


    int num_pulses;
    int num_dims;
    double  *dim_size;  
	double  *delta_x;   
	int     *size_of_x;
    std::string *pulse_shape;
    double      *cycles_on;
    double      *cycles_plateau;
    double      *cycles_off;
    double      *cep;
    double      *energy;
    double      *e_max;
    double      **grid;
    std::string str;

    std::cout << "\n\ndelta t: " << p.get_delta_t() << "\n";
	num_dims   = p.get_num_dims();
    std::cout << "num dims: " << num_dims << "\n";
    dim_size   = p.get_dim_size();
    delta_x    = p.get_delta_x();
    size_of_x  = p.get_size_of_x();
    grid       = new double*[size_of_x[0]];

    
    // NcVar** vars = new NcVar*[num_dims];
    for (int i = 0; i < num_dims; ++i) {
    	std::cout << "dim[" << i << "] ";
        std::cout << dim_size[i] << " ";
        std::cout << delta_x[i] << " ";
        std::cout << size_of_x[i] << "\n";
        grid[i] = new double[size_of_x[i]];
        for (int j = 0; j < size_of_x[i]; ++j)
        {
            grid[i][j] = j*delta_x[i];
        }
        // std::cout << "grid_"+std::to_string(i);
        // str = "grid_"+std::to_string(i);
        // dims[i] = dataFile.add_dim(str.c_str(), size_of_x[i]);
        // vars[i] = dataFile.add_var(str.c_str(), ncDouble, dims[i]);
        // vars[i]->put(&grid[i][0], size_of_x[i]);
    }

    // NcFile dataFile("TDSE.nc", NcFile::Replace);
    // if (!dataFile.is_valid()){
    //     std::cout << "Couldn't open file!\n";
    //     return NC_ERR;
    // }
    // NcDim* xDim = dataFile.add_dim("grid_x", size_of_x[0]);
    // NcDim* yDim = dataFile.add_dim("grid_y", size_of_x[1]);
    // NcVar* data = dataFile.add_var("grid", ncDouble, xDim, yDim);
    // data->put(&grid[0][0], size_of_x[0], size_of_x[1]);


    std::cout << "restart: " << p.get_restart() << "\n";
    std::cout << "target: " << p.get_target() << "\n";
    num_pulses = p.get_num_pulses();
    pulse_shape = p.get_pulse_shape();
    cycles_on = p.get_cycles_on();
    cycles_plateau = p.get_cycles_plateau();
    cycles_off = p.get_cycles_off();
    cep = p.get_cep();
    energy = p.get_energy();
    e_max = p.get_e_max();
    for (int i = 0; i < num_pulses; ++i)
    {
        std::cout << "Pulse[" << i << "]: ";
        std::cout << pulse_shape[i] << " ";
        std::cout << cycles_on[i] << " ";
        std::cout << cycles_plateau[i] << " ";
        std::cout << cycles_off[i] << " ";
        std::cout << cep[i] << " ";
        std::cout << energy[i] << " ";
        std::cout << e_max[i] << "\n";
    }
    return 0;
}
