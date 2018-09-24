import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import pylab as plb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
folders = ["00p5/", "06/"]

# font = {'size': 18}
# matplotlib.rc('font', **font)
fig = plt.figure()
for fold in folders:
    print "Plotting", fold
    # read data
    f = h5py.File(fold + "TDSE.h5", "r")
    psi_value = f["Projections"]["psi"]
    psi_time = f["Projections"]["time"][:]
    shape = f["Wavefunction"]["num_x"][:]
    gobbler = f["Parameters"]["gobbler"][0]
    dt = f["Parameters"]["delta_t"][0]
    upper_idx = (shape * gobbler - 1).astype(int)
    lower_idx = shape - upper_idx
    # calculate location for time to be printed

    x = f["Wavefunction"]["x_value_0"][:]
    kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
    if f["Parameters"]["coordinate_system_idx"][0] != 1:
        exit("Code only supports cylindrical currently")

    if len(shape) == 2:
        y = f["Wavefunction"]["x_value_1"][:]
        ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
        time_x = np.min(ky[lower_idx[1]:upper_idx[1]]) * 0.95
        time_y = np.max(kx[lower_idx[0]:upper_idx[0]]) * 0.9
    else:
        exit("2D versions are supported currently")

    delta_k = (kx[1] - kx[0])
    spectrum_k = np.arange(0, max(ky.max(), kx.max()), delta_k)
    spectrum_w = 0.5 * spectrum_k * spectrum_k + 0.5  # Ip
    spectrum = np.zeros(spectrum_k.shape)

    psi = psi_value[-1]
    # reshape data
    psi = psi[:, 0] + 1j * psi[:, 1]
    psi.shape = tuple(shape)
    data = None

    if len(shape) == 2:
        if f["Parameters"]["coordinate_system_idx"][0] == 1:

            psi = np.pad(psi, ((psi.shape[0], 0), (0, 0)), 'symmetric')
            # create k space
            psi = np.abs(np.fft.fftshift(np.fft.fft2(psi)))

            # delete the -rho (y-axis) data to avoid double counts
            psi = psi[psi.shape[0] / 2:]

            # integrate over |k|
            for i, line in enumerate(psi):
                for j, point in enumerate(line):
                    cur_k = np.sqrt(ky[j] * ky[j] + kx[i] * kx[i])
                    if cur_k < spectrum_k[-1]:
                        index = np.argmin(np.abs(cur_k - spectrum_k))
                        spectrum[index] += point

            # print max
            index = np.argmax(np.abs(spectrum))
            print "Max k:", spectrum_k[index], "Max w:", spectrum_w[
                index], spectrum_w[index - 1], spectrum_w[index - 2]

            # plot
            plt.plot(spectrum_w, spectrum, label=fold)
            plt.axvline(x=spectrum_w[index], ls='--', c='r')
plt.xlim(0, 3)
plt.legend()
fig.savefig("spec.png")
