import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import pylab as plb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
from scipy.special import sph_harm
from scipy.special import gamma
from scipy.special import hyp1f1
from scipy.misc import factorial
from scipy.special import spherical_jn
from scipy.interpolate import RectBivariateSpline
folders = ["02/", "10/"]


# note pho = r*k
# see nist website for definition https://dlmf.nist.gov/33.2
# top sign used for +- or -+ portions
# eta is Z/k (and minus for electrons, hydrogen atom -1/k)
def coulomb_wave_function(l, eta, rho):
    # calculate C_l(eta) coef using alternate def
    # norm_factor = 2**l * np.exp(-np.pi * eta / 2) * np.abs(
    #     gamma(l + 1 + 1.0j * eta)) / factorial(2 * l + 1)
    product = np.arange(1.0, l + 1.0)
    if l == 0:
        product = np.array([1.0])
    product = np.add.outer(eta**2, product**2)
    product = np.prod(product, axis=1)
    norm_factor = 2**l * np.sqrt(2 * np.pi * eta / (
        np.exp(np.pi * eta / 2) - 1) * product) / factorial(2 * l + 1)
    # return the f_l function
    return_array = np.zeros(norm_factor.shape)
    for idx in np.arange(norm_factor.shape[0]):
        print idx, l + 1.0 - 1.0j * eta[idx], 2.0 * l + 2.0, 1.0j * 2.0 * rho[
            idx]
        return_array[idx] = norm_factor[idx] * rho[idx]**(l + 1) * np.exp(
            -1.0j * rho[idx]) * hyp1f1(l + 1.0 - 1.0j * eta[idx],
                                       2.0 * l + 2.0, 1.0j * 2.0 * rho[idx])
    return return_array


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

    delta_k = (kx[1] - kx[0]) * 2
    spectrum_k = np.arange(kx[0], max(ky.max(), kx.max()), delta_k)
    spectrum_w = 0.5 * spectrum_k * spectrum_k + 0.51088  # Ip
    spectrum = np.zeros(spectrum_k.shape)

    psi = psi_value[12]
    # reshape data
    psi = psi[:, 0] + 1j * psi[:, 1]
    psi.shape = tuple(shape)
    data = None

    if len(shape) == 2:
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            x = np.arange(psi.shape[0]) * (x[1] - x[0])
            y = np.arange(-1.0 * np.floor(psi.shape[1] / 2),
                          np.ceil(psi.shape[1] / 2)) * (y[1] - y[0])
            kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
            ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
            print psi.shape, x.shape, y.shape, kx.shape, ky.shape

            w_max = 8.0
            k_max = np.sqrt(w_max * 2.0)
            delta_k = (ky[1] - ky[0]) / 2
            k_values = np.arange(0.5, k_max + delta_k, delta_k)
            delta_theta = 0.01
            theta = np.arange(0, np.pi, delta_theta)
            delta_r = 0.3
            r = np.arange(delta_r / 2, min(y.max(), x.max()), delta_r)
            l_max = 5
            l_values = np.arange(0, l_max + 1)
            f_psi_real = RectBivariateSpline(x, y, psi.real)
            f_psi_imag = RectBivariateSpline(x, y, psi.imag)

            energy_spectrum = np.zeros(
                (l_values.shape[0], k_values.shape[0]), dtype=complex)

            print l_values.shape, k_values.shape, theta.shape, r.shape
            # integration goes here
            for l_idx, l_val in enumerate(l_values):
                print l_val
                for theta_idx, theta_val in enumerate(theta):
                    for r_idx, r_val in enumerate(r):
                        energy_spectrum[l_idx, :] += (
                            delta_theta * delta_r * r_val * r_val *
                            np.sin(theta_val) *
                            (f_psi_real(r_val * np.sin(theta_val),
                                        r_val * np.cos(theta_val)) +
                             1.0j * f_psi_imag(r_val * np.sin(theta_val),
                                               r_val * np.cos(theta_val))) *
                            k_values * sph_harm(0, l_val, 0, theta_val)
                            .conjugate() * coulomb_wave_function(
                                l_val, -1.0 / k_values,
                                k_values * r_val).conjugate())[0]

            # plot
            #for l_idx, l_data in enumerate(np.abs(energy_spectrum)):
            #    plt.plot(k_values*k_values/2.0, l_data, label=fold+str(l_idx))
            e_0 = 0.51
            w = k_values * k_values / 2.0 + e_0
            gamma = 1.0 / np.sqrt(2 * np.pi * w / e_0 - 1)
            cross = (e_0 / w)**4 * np.exp(-4 * gamma / np.arctan(gamma)) / (
                1 - np.exp(-2 * np.pi * gamma))
            c = np.array(
                zip(k_values * k_values / 2.0, (np.abs(energy_spectrum)**2
                                                ).sum(axis=0)))
            print fold.split("/")[0] + "_data_coulomb.csv"
            np.savetxt(
                fold.split("/")[0] + "_data_coulomb.csv",
                c,
                delimiter=',',
                header="w,E",
                comments="")
            plt.plot(
                k_values * k_values / 2.0, (np.abs(energy_spectrum)**2).sum(
                    axis=0),
                label=fold)
            # plt.axvline(x=spectrum_w[index], ls='--', c='r')
plt.axvline(2.0 - 0.51088, c='k', lw=1)
plt.axvline(4.0 - 0.51088, c='k', lw=1)
plt.axvline(6.0 - 0.51088, c='k', lw=1)
plt.xlim(0, 4)
#plt.ylim(0, 1e7)
#plt.legend()
fig.savefig("spec_no_cross_lin.png")