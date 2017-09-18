import numpy as np
import matplotlib
import find_peaks_2d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

taus, moms, folders = [], [], []

for tau in ["-20", "-10", "0", "10", "20"]:
    key = "tau_" + tau + "/"
    folders.append(key)

# for each folder containing a 
# sim with a different delay
for fold in folders:
    #read data
    f = h5py.File(fold + "TDSE.h5", "r")
    psi_value = f["Wavefunction"]["psi"][-1]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    tau_delay  = f["Parameters"]["tau_delay"][0]

    taus.append(tau_delay)

    # calc color bounds
    for i, psi in enumerate(psi_value):
        if i > 0 and i < 2:
            psi = psi[:, 0] + 1j * psi[:, 1]
            max_val_tmp = np.max(np.absolute(psi))
            if (max_val_tmp > max_val):
                max_val = max_val_tmp