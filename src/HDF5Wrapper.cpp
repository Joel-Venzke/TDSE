#include "HDF5Wrapper.h"

void HDF5Wrapper::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

void HDF5Wrapper::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}

void HDF5Wrapper::write_object(
    int data,
    H5std_string var_path) {

    hsize_t h5_size[1];
    h5_size[0]     = 1;

    DataSpace h5_space(1,h5_size);

    DataSet data_set = data_file->createDataSet(
                            var_path, 
                            PredType::NATIVE_INT, h5_space);
    data_set.write(&data, PredType::NATIVE_INT);
}

void HDF5Wrapper::write_object(
    double data,
    H5std_string var_path) {

    hsize_t h5_size[1];
    h5_size[0]     = 1;

    DataSpace h5_space(1,h5_size);

    DataSet data_set = data_file->createDataSet(
                            var_path, 
                            PredType::NATIVE_DOUBLE, h5_space);
    data_set.write(&data, PredType::NATIVE_DOUBLE);
}

void HDF5Wrapper::write_object(
    int *data, 
    int size,
    H5std_string var_path) {

    hsize_t h5_size[1];
    h5_size[0]     = size;

    DataSpace h5_space(1,h5_size);

    DataSet data_set = data_file->createDataSet(
                            var_path, 
                            PredType::NATIVE_INT, h5_space);
    data_set.write(data, PredType::NATIVE_INT);
}

void HDF5Wrapper::write_object(
    double *data, 
    int size,
    H5std_string var_path) {

    hsize_t h5_size[1];
    h5_size[0]     = size;

    DataSpace h5_space(1,h5_size);

    DataSet data_set = data_file->createDataSet(
                            var_path, 
                            PredType::NATIVE_DOUBLE, h5_space);

    data_set.write(data, PredType::NATIVE_DOUBLE);
}

void HDF5Wrapper::write_header(Parameters & p){
	int num_dims = p.get_num_dims();
	int num_pulses = p.get_num_pulses();

    // set up group
    Group param_group( data_file->createGroup( "/Parameters" ));
    Group pulse_group( data_file->createGroup( "/Pulse" ));

    // write out header values
    write_object(num_dims,"/Parameters/num_dims");
    write_object(p.get_dim_size(),num_dims,"/Parameters/dim_size");
    write_object(p.get_delta_x(),num_dims,"/Parameters/delta_x");
    write_object(p.get_delta_t(),"/Parameters/delta_t");
    write_object(p.get_target_idx(),"/Parameters/target_idx");
    write_object(num_pulses,"/Parameters/num_pulses");
    write_object(p.get_pulse_shape_idx(), num_pulses,
    	"/Parameters/pulse_shape_idx");
    write_object(p.get_cycles_on(), num_pulses,
    	"/Parameters/cycles_on");
    write_object(p.get_cycles_plateau(), num_pulses,
    	"/Parameters/cycles_plateau");
    write_object(p.get_cycles_off(), num_pulses,
    	"/Parameters/cycles_off");
    write_object(p.get_cycles_delay(), num_pulses, 
    	"/Parameters/cycles_delay");
    write_object(p.get_cep(), num_pulses, "/Parameters/cep");
    write_object(p.get_energy(), num_pulses, "/Parameters/energy");
    write_object(p.get_e_max(), num_pulses, "/Parameters/e_max");
}

// TDOD: set up restart
void HDF5Wrapper::read_restart(Parameters & p) {
    std::string err_str;
    // Check if restart can use current file
    if (p.get_restart() == 1 ) {
        err_str += "Restart not implemented yet \n";
    }
    end_run(err_str);
}

void HDF5Wrapper::read_restart(Parameters & p, std::string file_name) {
    std::string err_str;
    // Check if restart can use current file
    if (p.get_restart() == 1 ) {
        err_str += "Restart not implemented yet \n";
    }
    end_run(err_str);
}

HDF5Wrapper::HDF5Wrapper(std::string file_name, Parameters & p) {
	if (p.get_restart() == 1) {
        read_restart(p,file_name);
    } else {
        data_file = new H5File( file_name, H5F_ACC_TRUNC);
        write_header(p);
    }
}

HDF5Wrapper::HDF5Wrapper( Parameters & p) {
	if (p.get_restart() == 1) {
        read_restart(p);
    } else {
        data_file = new H5File( "TDSE.h5", H5F_ACC_TRUNC);
        write_header(p);
    }
}

HDF5Wrapper::~HDF5Wrapper(){
	std::cout << "Deleting HDF5Wrapper\n";
	delete data_file;
}
