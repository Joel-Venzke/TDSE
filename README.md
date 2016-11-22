# TDSE
This code was developed by Joel Venzke during his PHD. It solves the Time Dependent Schroedinger Equation.

# INSTALL 
To install this code run the following commands in the root directory replacing `${INSTALL_DIR}` with the path to where you want the binary installed at.

```
./configure --prefix=${INSTALL_DIR}
make
make install
```

#DEVELOPING
If changes are make to the anything relating to autoconf tools, you need to run `autoreconf -i` before installing 

#USAGE
The code can be used by using `TDSE` as a command
