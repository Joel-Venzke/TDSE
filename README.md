# TDSE
This code was developed by Joel Venzke during his PHD. It solves the Time Dependent Schroedinger Equation.

# INSTALL 
Copy a `build.${SYSTEM}` file that is similar to your system to `build`. Then make the needed changes to paths and run `.\build`. If the `prefix` is in your path, the `TDSE` command will run the code anywhere on your system.

## Manual install
To install this code run the following commands in the root directory replacing `${INSTALL_DIR}` with the path to where you want the binary installed at.

```
./configure --prefix=${INSTALL_DIR}
make
make install
```

#DEVELOPING
If changes are make to the anything relating to autoconf tools, you need to run `autoreconf -i` before installing 

#USAGE
The code can be used by using `TDSE` as a command assuming the ${INSTALL_DIR} is in your path


#NOTES
HDF5 complex
http://stackoverflow.com/questions/24937785/best-way-to-save-an-array-of-complex-numbers-with-hdf5-and-c
