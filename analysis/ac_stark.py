import numpy as np
import h5py
import json
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpi4py import MPI


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
    elif l_value > 24:  # anything greater that z is just a number
        ret_val = str(n_value) + ",l=" + str(l_value - 1)
    else:  # any
        ret_val = str(n_value) + chr(ord('g') + l_value - 5)

    return ret_val


# return list of states up to state_number
def state_name(state_number):
    # get size of each shell
    shells = [0]
    while (state_number > shells[-1]):
        shells.append(shells[-1] + len(shells))

    # get list of names
    name_list = []
    for state in range(1, state_number + 1):
        name_list.append(state_single_name(state, shells))

    return name_list


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

with open('input.json', 'r') as data_file:
    data = json.load(data_file)

target_name = data["target"]["name"]
# read data
target = h5py.File(target_name + ".h5", "r")
file = h5py.File("TDSE.h5", "r")
psi_value = target["psi"]
energy = target["Energy"][:]
psi_time = file["Wavefunction"]["time"][:]
shape = file["Wavefunction"]["num_x"][:]
x = file["Wavefunction"]["x_value_0"][:]
w_0 = file["Parameters"]["energy"][0]
w_0 = 0.057
w_0_square = w_0 * w_0

if len(shape) > 1:
    y = file["Wavefunction"]["x_value_1"][:]

name_list = state_name(len(psi_value))

dipole_list = np.zeros(psi_value.shape[0])
for i in range(psi_value.shape[0]):
    if i % size == rank:
        psi_i = psi_value[i]
        if rank == 0 and i > 0:
            for idx, d in enumerate(dipole_list):
                if np.abs(d) > 0:
                    print name_list[idx], str(d) + ",",
            print

        psi_i = psi_i[:, 0] + 1j * psi_i[:, 1]
        psi_i.shape = tuple(shape)

        psi_tmp = np.array(psi_i)
        # normalize
        if file["Parameters"]["coordinate_system_idx"][0] == 1:
            psi_tmp = np.multiply(x, psi_tmp.transpose()).transpose()

        psi_i /= np.sqrt(np.abs(np.vdot(psi_i, psi_tmp)))

        # jacobian
        if file["Parameters"]["coordinate_system_idx"][0] == 1:
            psi_i = np.multiply(x, psi_i.transpose()).transpose()
        # dipole
        psi_i = np.multiply(y, psi_i)

        for f, psi_f in enumerate(psi_value):
            if i != f:
                w_fi = energy[f] - energy[i]
                psi_f = psi_f[:, 0] + 1j * psi_f[:, 1]
                psi_f.shape = tuple(shape)

                psi_tmp = np.array(psi_f)
                # normalize
                if file["Parameters"]["coordinate_system_idx"][0] == 1:
                    psi_tmp = np.multiply(x, psi_tmp.transpose()).transpose()

                psi_f /= np.sqrt(np.abs(np.vdot(psi_f, psi_tmp)))

                # magnitude squared
                dipole = np.abs(np.vdot(psi_f, psi_i))
                dipole *= dipole

                # laser dependence
                dipole *= w_fi / (w_fi * w_fi - w_0_square)

                # leading minus sign
                dipole *= -1.0
                dipole_list[i] += dipole

        # send data to master
        if rank == 0:
            for task in range(1, size):
                if i + task < dipole_list.shape[0]:
                    dipole_list[i + task] = comm.recv(source=task)
        else:
            comm.send(dipole_list[i], dest=0)

if rank == 0:
    for idx, d in enumerate(dipole_list):
        if np.abs(d) > 0:
            print name_list[idx], str(d) + ",",
    print
