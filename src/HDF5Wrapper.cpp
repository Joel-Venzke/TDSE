#include "HDF5Wrapper.h"
#include <complex>

#define dcomp std::complex<double>

// end run after printing error string with exit value -1
void HDF5Wrapper::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

// end run after printing error string with exit_val
void HDF5Wrapper::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}

hsize_t* HDF5Wrapper::get_hsize_t(int size, int *dims) {
    // size of array
    hsize_t *h5_size = new hsize_t[size];
    for (int i=0; i<size; i++) {
        h5_size[i] = dims[i];
    }
    return h5_size;
}

// Writes int to HDF5
// takes int and var_path as inputs
void HDF5Wrapper::write_object(
    int data,
    H5std_string var_path) {
    reopen();
    // size of "array"
    hsize_t h5_size[1];
    h5_size[0]     = 1;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(&data, PredType::NATIVE_INT);

    close();
}

// Writes double to HDF5
// takes double and var_path as inputs
void HDF5Wrapper::write_object(
    double data,
    H5std_string var_path) {
    reopen();
    // size of "array"
    hsize_t h5_size[1];
    h5_size[0]     = 1;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(&data, PredType::NATIVE_DOUBLE);

    close();
}

// Writes 1D int array to HDF5
// takes 1D int array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    int *data,
    int size,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0]     = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_INT);

    close();
}

// Writes N-D int array to HDF5
// takes N-D int array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    int *data,
    int size,
    int *dims,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);

    // make DataSpace for array
    DataSpace h5_space(size,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_INT);

    delete[] h5_size;

    close();
}

// Writes 1D double array to HDF5
// takes 1D double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    double *data,
    int size,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0]     = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_DOUBLE);

    close();
}

// Writes N-Dim double array to HDF5
// takes N-Dim double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    double *data,
    int size,
    int* dims,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);

    // make DataSpace for array
    DataSpace h5_space(size,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_DOUBLE);

    delete[] h5_size;

    close();
}

// Writes 1D complex double array to HDF5
// takes 1D complex double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0] = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            complex_data_type[0], h5_space);

    // write data to file
    data_set.write(data, complex_data_type[0]);

    close();
}

void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    int *dims,
    H5std_string var_path) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            complex_data_type[0], h5_space);

    // write data to file
    data_set.write(data, complex_data_type[0]);

    delete[] h5_size;

    close();
}

// Writes int to HDF5
// takes int and var_path as inputs
void HDF5Wrapper::write_object(
    int data,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of "array"
    hsize_t h5_size[1];
    h5_size[0]     = 1;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(&data, PredType::NATIVE_INT);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    close();
}

// Writes double to HDF5
// takes double and var_path as inputs
void HDF5Wrapper::write_object(
    double data,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of "array"
    hsize_t h5_size[1];
    h5_size[0]     = 1;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(&data, PredType::NATIVE_DOUBLE);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    close();
}

// Writes 1D int array to HDF5
// takes 1D int array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    int *data,
    int size,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0]     = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_INT);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    close();
}

// Writes N-D int array to HDF5
// takes N-D int array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    int *data,
    int size,
    int *dims,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_INT, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_INT);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    delete h5_size;

    close();
}

// Writes 1D double array to HDF5
// takes 1D double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    double *data,
    int size,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0]     = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_DOUBLE);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    close();
}

// Writes N-D double array to HDF5
// takes N-D double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    double *data,
    int size,
    int *dims,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            PredType::NATIVE_DOUBLE, h5_space);

    // write data to file
    data_set.write(data, PredType::NATIVE_DOUBLE);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    delete h5_size;

    close();
}

// Writes 1D complex double array to HDF5
// takes 1D complex double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t h5_size[1];
    h5_size[0] = size;

    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            complex_data_type[0], h5_space);

    // write data to file
    data_set.write(data, complex_data_type[0]);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    close();
}

// Writes N-D complex double array to HDF5
// takes N-D complex double array, array size, and var_path as inputs
void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    int * dims,
    H5std_string var_path,
    H5std_string attribute) {
    reopen();
    // size of array
    hsize_t *h5_size = get_hsize_t(size, dims);


    // make DataSpace for array
    DataSpace h5_space(1,h5_size);

    // build the header for the data entry
    DataSet data_set = data_file->createDataSet(
                            var_path,
                            complex_data_type[0], h5_space);

    // write data to file
    data_set.write(data, complex_data_type[0]);

    // write attribute
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = data_set.createAttribute( "Attribute", str_type,
        att_space );
    att.write( str_type, attribute );

    delete h5_size;

    close();
}

void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    H5std_string var_path,
    int write_idx) {
    reopen();
    // create object
    if (write_idx==0) {
        // size of array
        hsize_t h5_size[2];
        h5_size[0] = 1;
        h5_size[1] = size;

        // max size of data set.
        // Built this way to make time the first dimension
        hsize_t h5_max_size[2];
        h5_max_size[0] = H5S_UNLIMITED; // time dim
        h5_max_size[1] = H5S_UNLIMITED; // spacial dim

        hsize_t h5_chunk[2];
        h5_chunk[0] = 1;
        h5_chunk[1] = size;

        // set up data space
        DataSpace h5_space(2,h5_size, h5_max_size);

        // Modify dataset creation property to enable chunking
        DSetCreatPropList prop;
        prop.setChunk(2, h5_chunk);

        DataSet *data_set = new DataSet(data_file->createDataSet(
                                        var_path,
                                        complex_data_type[0],
                                        h5_space, prop));

        // write data
        data_set->write(data, complex_data_type[0]);

        // save for later use
        // TODO: make into hash table
        extendable_dataset_complex.push_back(data_set);
        extendable_string_complex.push_back(var_path);
    } else {
        // find the index of the dataset
        // TODO: make into hash table
        int idx = 0;
        int idx_max = extendable_string_complex.size();
        while (extendable_string_complex[idx]!=var_path) {
            idx++;

            // throw error if dataset is not created yet
            if (idx >= idx_max) {
                std::string str = var_path;
                str += " not found when writing time step ";
                str += std::to_string(write_idx);
                end_run(str);
            }
        }

        // get data set pointer from array
        DataSet *data_set = extendable_dataset_complex[idx];

        // new dimension of dataset
        hsize_t h5_size[2];
        h5_size[0] = write_idx+1;
        h5_size[1] = size;

        // sizes of extension
        hsize_t h5_extend[2];
        h5_extend[0] = 1;
        h5_extend[1] = size;

        // sizes of extension
        hsize_t h5_offset[2];
        h5_offset[0] = write_idx;
        h5_offset[1] = 0;

        // extend data set
        data_set->extend(h5_size);

        // get hyperslab
        DataSpace *filespace = new DataSpace(data_set->getSpace ());
        filespace->selectHyperslab(H5S_SELECT_SET,
                                   h5_extend, h5_offset);

        // get memory space
        DataSpace *memspace = new DataSpace(2, h5_extend, NULL);

        // write data
        data_set->write(data,complex_data_type[0],
                        *memspace, *filespace);

        // clean up
        delete filespace;
        delete memspace;
    }

    close();
}

void HDF5Wrapper::write_object(
    dcomp *data,
    int size,
    H5std_string var_path,
    H5std_string attribute,
    int write_idx) {
    reopen();
    // create object
    if (write_idx==0) {
        // size of array
        hsize_t h5_size[2];
        h5_size[0] = 1;
        h5_size[1] = size;

        // max size of data set.
        // Built this way to make time the first dimension
        hsize_t h5_max_size[2];
        h5_max_size[0] = H5S_UNLIMITED; // time dim
        h5_max_size[1] = H5S_UNLIMITED; // spacial dim

        hsize_t h5_chunk[2];
        h5_chunk[0] = 1;
        h5_chunk[1] = size;

        // set up data space
        DataSpace h5_space(2,h5_size, h5_max_size);

        // Modify dataset creation property to enable chunking
        DSetCreatPropList prop;
        prop.setChunk(2, h5_chunk);

        DataSet *data_set = new DataSet(data_file->createDataSet(
                                        var_path,
                                        complex_data_type[0],
                                        h5_space, prop));

        // write data
        data_set->write(data, complex_data_type[0]);

        // save for later use
        // TODO: make into hash table
        extendable_dataset_complex.push_back(data_set);
        extendable_string_complex.push_back(var_path);

        // write attribute
        StrType str_type(0, H5T_VARIABLE);
        DataSpace att_space(H5S_SCALAR);
        Attribute att = data_set->createAttribute( "Attribute",
            str_type, att_space );
        att.write( str_type, attribute );
    } else {
        // find the index of the dataset
        // TODO: make into hash table
        int idx = 0;
        int idx_max = extendable_string_complex.size();
        while (extendable_string_complex[idx]!=var_path) {
            idx++;

            // throw error if dataset is not created yet
            if (idx >= idx_max) {
                std::string str = var_path;
                str += " not found when writing time step ";
                str += std::to_string(write_idx);
                end_run(str);
            }
        }

        // get data set pointer from array
        DataSet *data_set = extendable_dataset_complex[idx];

        // new dimension of dataset
        hsize_t h5_size[2];
        h5_size[0] = write_idx+1;
        h5_size[1] = size;

        // sizes of extension
        hsize_t h5_extend[2];
        h5_extend[0] = 1;
        h5_extend[1] = size;

        // sizes of extension
        hsize_t h5_offset[2];
        h5_offset[0] = write_idx;
        h5_offset[1] = 0;

        // extend data set
        data_set->extend(h5_size);

        // get hyperslab
        DataSpace *filespace = new DataSpace(data_set->getSpace ());
        filespace->selectHyperslab(H5S_SELECT_SET,
                                   h5_extend, h5_offset);

        // get memory space
        DataSpace *memspace = new DataSpace(2, h5_extend, NULL);

        // write data
        data_set->write(data,complex_data_type[0],
                        *memspace, *filespace);

        // clean up
        delete filespace;
        delete memspace;
    }

    close();
}


void HDF5Wrapper::write_object(
    double data,
    H5std_string var_path,
    int write_idx) {
    reopen();
    // create object
    if (write_idx==0) {
        // size of array
        hsize_t h5_size[1];
        h5_size[0] = 1;

        // max size of data set.
        // Built this way to make time the first dimension
        hsize_t h5_max_size[1];
        h5_max_size[0] = H5S_UNLIMITED; // time dim

        hsize_t h5_chunk[1];
        h5_chunk[0] = 1;

        // set up data space
        DataSpace h5_space(1,h5_size, h5_max_size);

        // Modify dataset creation property to enable chunking
        DSetCreatPropList prop;
        prop.setChunk(1, h5_chunk);

        DataSet *data_set = new DataSet(data_file->createDataSet(
                                        var_path,
                                        PredType::NATIVE_DOUBLE,
                                        h5_space, prop));

        // write data
        data_set->write(&data, PredType::NATIVE_DOUBLE);

        // save for later use
        // TODO: make into hash table
        extendable_dataset_double.push_back(data_set);
        extendable_string_double.push_back(var_path);
    } else {
        // find the index of the dataset
        // TODO: make into hash table
        int idx = 0;
        int idx_max = extendable_string_double.size();
        while (extendable_string_double[idx]!=var_path) {
            idx++;

            // throw error if dataset is not created yet
            if (idx >= idx_max) {
                std::string str = var_path;
                str += " not found when writing time step ";
                str += std::to_string(write_idx);
                end_run(str);
            }
        }

        // get data set pointer from array
        DataSet *data_set = extendable_dataset_double[idx];

        // new dimension of dataset
        hsize_t h5_size[1];
        h5_size[0] = write_idx+1;

        // sizes of extension
        hsize_t h5_extend[1];
        h5_extend[0] = 1;

        // sizes of extension
        hsize_t h5_offset[1];
        h5_offset[0] = write_idx;

        // extend data set
        data_set->extend(h5_size);

        // get hyperslab
        DataSpace *filespace = new DataSpace(data_set->getSpace ());
        filespace->selectHyperslab(H5S_SELECT_SET,
                                   h5_extend, h5_offset);

        // get memory space
        DataSpace *memspace = new DataSpace(1, h5_extend, NULL);

        // write data
        data_set->write(&data,PredType::NATIVE_DOUBLE,
                        *memspace, *filespace);

        // clean up
        delete filespace;
        delete memspace;
    }

    close();
}

void HDF5Wrapper::write_object(
    double data,
    H5std_string var_path,
    H5std_string attribute,
    int write_idx) {
    reopen();
    // create object
    if (write_idx==0) {
        // size of array
        hsize_t h5_size[1];
        h5_size[0] = 1;

        // max size of data set.
        // Built this way to make time the first dimension
        hsize_t h5_max_size[1];
        h5_max_size[0] = H5S_UNLIMITED;

        hsize_t h5_chunk[1];
        h5_chunk[0] = 1;

        // set up data space
        DataSpace h5_space(1,h5_size, h5_max_size);

        // Modify dataset creation property to enable chunking
        DSetCreatPropList prop;
        prop.setChunk(1, h5_chunk);

        DataSet *data_set = new DataSet(data_file->createDataSet(
                                        var_path,
                                        PredType::NATIVE_DOUBLE,
                                        h5_space, prop));

        // write data
        data_set->write(&data, PredType::NATIVE_DOUBLE);

        // save for later use
        // TODO: make into hash table
        extendable_dataset_double.push_back(data_set);
        extendable_string_double.push_back(var_path);

        // write attribute
        StrType str_type(0, H5T_VARIABLE);
        DataSpace att_space(H5S_SCALAR);
        Attribute att = data_set->createAttribute( "Attribute",
            str_type, att_space );
        att.write( str_type, attribute );
    } else {
        // find the index of the dataset
        // TODO: make into hash table
        int idx = 0;
        int idx_max = extendable_string_double.size();
        while (extendable_string_double[idx]!=var_path) {
            idx++;

            // throw error if dataset is not created yet
            if (idx >= idx_max) {
                std::string str = var_path;
                str += " not found when writing time step ";
                str += std::to_string(write_idx);
                end_run(str);
            }
        }

        // get data set pointer from array
        DataSet *data_set = extendable_dataset_double[idx];

        // new dimension of dataset
        hsize_t h5_size[1];
        h5_size[0] = write_idx+1;

        // sizes of extension
        hsize_t h5_extend[1];
        h5_extend[0] = 1;

        // sizes of extension
        hsize_t h5_offset[1];
        h5_offset[0] = write_idx;

        // extend data set
        data_set->extend(h5_size);

        // get hyperslab
        DataSpace *filespace = new DataSpace(data_set->getSpace ());
        filespace->selectHyperslab(H5S_SELECT_SET,
                                   h5_extend, h5_offset);

        // get memory space
        DataSpace *memspace = new DataSpace(1, h5_extend, NULL);

        // write data
        data_set->write(&data,PredType::NATIVE_DOUBLE,
                        *memspace, *filespace);

        // clean up
        delete filespace;
        delete memspace;
    }

    close();
}

void HDF5Wrapper::create_group(H5std_string group_path) {
    reopen();
    Group new_group(data_file->createGroup(group_path));

    close();
}

// writes out header for Parameters and builds the various
// groups that will be used by other classes
void HDF5Wrapper::write_header(Parameters & p){
    int num_dims = p.get_num_dims();
    int num_pulses = p.get_num_pulses();
    header = true;

    // set up group
    create_group("/Parameters");
    create_group("/Pulse" );
    create_group("/Wavefunction" );

    // write out header values
    write_object(num_dims,"/Parameters/num_dims",
        "Number of dimension in simulation");
    write_object(p.get_dim_size(),num_dims, "/Parameters/dim_size",
        "The length of that dimension in atomic units.");
    write_object(p.get_delta_x(),num_dims, "/Parameters/delta_x",
        "The step sizes in that dimension in atomic units.");
    write_object(p.get_delta_t(), "/Parameters/delta_t",
        "The size of the time step in atomic units.");
    write_object(p.get_target_idx(), "/Parameters/target_idx",
        "The index of the target. He:0 ");
    write_object(num_pulses, "/Parameters/num_pulses",
        "The number of pulses from the input file");
    write_object(p.get_pulse_shape_idx(), num_pulses,
        "/Parameters/pulse_shape_idx",
        "The index of the pulse shape. Sin2:0");
    write_object(p.get_cycles_on(), num_pulses,
        "/Parameters/cycles_on",
        "Number of cycles the pulse ramps on for");
    write_object(p.get_cycles_plateau(), num_pulses,
        "/Parameters/cycles_plateau",
        "Number of cycles the pulse stays at max amplitude for");
    write_object(p.get_cycles_off(), num_pulses,
        "/Parameters/cycles_off",
        "Number of cycles the pulse ramps off for");
    write_object(p.get_cycles_delay(), num_pulses,
        "/Parameters/cycles_delay",
        "Number of cycles before the pulse starts");
    write_object(p.get_cep(), num_pulses, "/Parameters/cep",
        "The carrying phase envelope of the pulse. It is defined at the time the pulse starts to turn on.");
    write_object(p.get_energy(), num_pulses, "/Parameters/energy",
        "The fundamental angular frequency of the pulse. Corresponds to the energy of the photons in atomic units.");
    write_object(p.get_e_max(), num_pulses, "/Parameters/e_max",
        "The maximum amplitude of the pulse in atomic units.");

    header = false;
    close();
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

// TODO: more of this restart stuff
void HDF5Wrapper::read_restart(Parameters & p, std::string f_name){
    std::string err_str;
    // Check if restart can use current file
    if (p.get_restart() == 1 ) {
        err_str += "Restart not implemented yet \n";
    }
    end_run(err_str);
}

// constructor
// file_name needs ending ".h5"
HDF5Wrapper::HDF5Wrapper(std::string f_name, Parameters & p) {
    file_name = f_name;
    header    = false;
    if (p.get_restart() == 1) {
        read_restart(p,file_name);
    } else {
        data_file = new H5File( file_name, H5F_ACC_TRUNC);
        file_open = true;
        write_header(p);
    }
    define_complex();
    close();
}

// constructor
// file_name needs ending ".h5"
HDF5Wrapper::HDF5Wrapper(std::string f_name) {
    file_name = f_name;
    header    = false;
    data_file = new H5File( file_name, H5F_ACC_TRUNC);
    file_open = true;
    define_complex();
    close();
}

// constructor
// file_name needs ending ".h5"
HDF5Wrapper::HDF5Wrapper( Parameters & p) {
    header    = false;
    file_name = "TDSE.h5";
    if (p.get_restart() == 1) {
        read_restart(p);
    } else {
        data_file = new H5File( file_name, H5F_ACC_TRUNC);
        file_open = true;
        write_header(p);
    }
    define_complex();
    close();
}

void HDF5Wrapper::define_complex() {
    complex_data_type = new CompType(sizeof(dcomp(1.0,1.0)));
    complex_data_type->insertMember( "r", 0, PredType::NATIVE_DOUBLE);
    complex_data_type->insertMember( "i", sizeof(double), PredType::NATIVE_DOUBLE);
}

void HDF5Wrapper::reopen() {
    if (!header && !file_open) {
        data_file = new H5File( file_name, H5F_ACC_RDWR);
        file_open = true;
    }
}

void HDF5Wrapper::close() {
    if (!header && file_open) {
        data_file->close();
        delete data_file;
        file_open = false;
    }
}

void HDF5Wrapper::set_header(bool h) {
    header = h;
}

// destructor
HDF5Wrapper::~HDF5Wrapper(){
    std::cout << "Deleting HDF5Wrapper\n";
    close();
    for (int i=0; i<extendable_dataset_double.size(); i++) {
        delete extendable_dataset_double[i];
    }
    for (int i=0; i<extendable_dataset_complex.size(); i++) {
        delete extendable_dataset_complex[i];
    }
}
