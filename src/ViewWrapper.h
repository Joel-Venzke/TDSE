#pragma once

class ViewWrapper {
private:
    PetscViewer    data_file;
    std::string    file_name;
    bool           file_name_set;
    bool           file_open;
    PetscInt       ierr;
public:
    /* Constructor */
    /* overwrites existing file */
    ViewWrapper(std::string f_name);

    /* no file set use open to set file */
    ViewWrapper();

    /* destructor */
    ~ViewWrapper();

    /* Opens file in various modes */
    /* r: read only                */
    /* w: write only (new file)    */
    /* a: append write only        */
    void open(std::string file_name, std::string mode = 'w');
    void open(std::string mode = 'w');

    /* closes file */
    void close();

    /* push group */
    void push_group(H5std_string group_path);

    /* pop group */
    void pop_group(H5std_string group_path);

    /* sets the time step for the file */
    void set_time(H5std_string group_path);

    /* writes and attribute */
    void set_attribute();

    /* write a frame */
    void write_object(int data, H5std_string var_path);

    /* reads a frame */
    void read_object(int data, H5std_string var_path);

    /* kill run */
    void end_run(std::string str);
    void end_run(std::string str, int exit_val);
};
