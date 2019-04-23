import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.signal import argrelmin, argrelmax
from matplotlib.colors import LogNorm


# return list of states up to state_number
def state_name(state_number, l_max, m_max):

    name_list = []
    state_idx = 0
    n_val = 1
    while state_idx < state_number:
        for l_val in np.arange(0, min(n_val, l_max + 1)):
            m_range = min(l_val, m_max)
            for m_val in np.arange(-m_range, m_range + 1):
                name_list.append("(" + str(n_val) + "," + str(l_val) + "," +
                                 str(m_val) + ")")
                state_idx += 1
        n_val += 1

    return name_list


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