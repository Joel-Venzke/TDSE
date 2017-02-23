# TDSE
This code was developed by Joel Venzke during his PHD. It solves the Time Dependent Schroedinger Equation.

# INSTALL
Copy a `build.${SYSTEM}` file that is similar to your system to `build`. Then make the needed changes to paths and run `.\build`. If the `prefix` is in your path, the `TDSE` command will run the code anywhere on your system. You may also need to copy Makefile.${System} to Makefile in the ${TDSE_DIR}/src/ directory

## Manual install
To install this code run the following commands in the root directory replacing `${INSTALL_DIR}` with the path to where you want the binary installed at.

```
cd ${TDSE_DIR}/bin
make
```

#USAGE
The code can be used by using `TDSE` in the ${TDSE_DIR}/bin/


#NOTES
anaconda may cause issues with install
