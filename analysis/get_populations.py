import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.signal import argrelmin, argrelmax
from matplotlib.colors import LogNorm


def get_shells(state_number):
    shells = [0]
    while (state_number > shells[-1]):
        shells.append(shells[-1] + len(shells))
    return np.array(shells)


def state_single_name(state_number, shells):
    # find the n value for this state
    n_value = 0
    for n, shell in enumerate(shells):
        if state_number > shell:
            n_value = n + 1

    # calculate quantum number l
    l_value = state_number - shells[n_value - 1]

    # create label
    if l_value == 1:
        ret_val = str(n_value) + "s"
    elif l_value == 2:
        ret_val = str(n_value) + "p"
    elif l_value == 3:
        ret_val = str(n_value) + "d"
    elif l_value == 4:
        ret_val = str(n_value) + "f"
    else:  # anything greater that f is just a number
        ret_val = str(n_value) + ",l=" + str(l_value - 1)

    return ret_val


# return list of states up to state_number
def state_name(state_number):
    # get size of each shell
    shells = get_shells(state_number)

    # get list of names
    name_list = []
    for state in range(1, state_number + 1):
        name_list.append(state_single_name(state, shells))

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

state_labels = state_name(data.shape[1])

for state_number in range(data.shape[1]):
    print state_labels[state_number] + "\t" + str(
        data[1, state_number]) + "\t" + str(data[-1, state_number])