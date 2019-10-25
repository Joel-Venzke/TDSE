import numpy as np
import h5py

f = None
p = None

try:
    f = h5py.File("TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
except:
    f = h5py.File("Observables.h5", "r")
    p = f

data = f["Wavefunction"]["projections"][:, :, :]
# get square of projections
data = np.absolute(data[:, :, 0] + 1j * data[:, :, 1])
data *= data

print("Ionization:", 1.-data[-1, :].sum())