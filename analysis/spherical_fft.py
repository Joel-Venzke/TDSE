import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import sph_harm
import pylab as plb
font = {'size': 22}
matplotlib.rc('font', **font)


# Produces the index that is closest to the needed r value
def get_r_vals(psi_cooridnate_values, r):
    shape = r.shape
    r = r.flatten()
    data = np.zeros(r.shape, dtype='int')
    for idx, r_val in enumerate(r):
        r_idx = np.argmin(np.abs(psi_cooridnate_values[2] - r_val))
        data[idx] = r_idx
    data.shape = shape
    return data


# Returns the wave function at the coordinates provided
def get_data(psi,
             psi_cooridnate_values,
             r,
             theta,
             phi,
             r_vals,
             l_values,
             m_values,
             r_cut=100.,
             alpha=0.075):
    l = 0
    m = 0
    # Handle l=0 to initialize data
    data = np.zeros(
        (psi[0, l, r_vals] * sph_harm(m, l, phi, theta)).shape, dtype=complex)
    # loop over l>0
    for lm_idx in np.arange(0, psi_cooridnate_values[1].shape[0]):
        l = l_values[lm_idx]
        m = m_values[lm_idx]
        data += ((
            -1.0)**m) * psi[0, lm_idx, r_vals] * sph_harm(m, l, phi, theta)
    data[r > psi_cooridnate_values[2].max()] = 1e-100
    data[r < r_cut] *= np.exp(-alpha * (r[r < r_cut] - r_cut)**2)
    return data


# creates an xz plan and returns the inputs needed for get_data
def cacluate_xz_plane(psi_cooridnate_values, x_size, z_size, resolution):
    x, y, z = np.meshgrid(
        np.linspace(-x_size, x_size, resolution),
        np.array([0.0]), np.linspace(-z_size, z_size, resolution))
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros(r.shape, dtype='float')
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    phi = np.arctan2(y, x)
    r_vals = get_r_vals(psi_cooridnate_values, r)
    return x, y, z, r, theta, phi, r_vals


# creates an yz plan and returns the inputs needed for get_data
def cacluate_yz_plane(psi_cooridnate_values, y_size, z_size, resolution):
    x, y, z = np.meshgrid(
        np.array([0.0]),
        np.linspace(-y_size, y_size, resolution),
        np.linspace(-z_size, z_size, resolution))
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros(r.shape, dtype='float')
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    phi = np.arctan2(y, x)
    r_vals = get_r_vals(psi_cooridnate_values, r)
    return x, y, z, r, theta, phi, r_vals


# creates an xy plan and returns the inputs needed for get_data
def cacluate_xy_plane(psi_cooridnate_values, x_size, y_size, resolution):
    x, y, z = np.meshgrid(
        np.linspace(-x_size, x_size, resolution),
        np.linspace(-y_size, y_size, resolution), np.array([0.0]))
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros(r.shape, dtype='float')
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    phi = np.arctan2(y, x)
    r_vals = get_r_vals(psi_cooridnate_values, r)
    return x, y, z, r, theta, phi, r_vals


resolution = 500

f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(
        f["Wavefunction"]["x_value_" + str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r_max = psi_cooridnate_values[2].max()
# pre-calculate grid so the plotting can be vectorized
print "Calculating index set for xy plane"
x, y, z, r, theta, phi, r_vals = cacluate_xy_plane(psi_cooridnate_values,
                                                   r_max, r_max, resolution)
x_values = x[0, :, 0]
y_values = y[:, 0, 0]
kx = x_values * 2.0 * np.pi / (x_values.shape[0] *
                               (x_values[1] - x_values[0]) *
                               (x_values[1] - x_values[0]))
ky = y_values * 2.0 * np.pi / (y_values.shape[0] *
                               (y_values[1] - y_values[0]) *
                               (y_values[1] - y_values[0]))
fig = plt.figure(figsize=(24, 18), dpi=80)
for i, psi in enumerate(psi_value):
    if i > 0:  # the 0th index is garbage
        print "plotting", i
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = shape
        psi_norm = np.sqrt((psi * psi.conjugate()).sum())
        plane_data = get_data(psi, psi_cooridnate_values, r, theta, phi,
                              r_vals, l_values, m_values)[:, :, 0]
        cs = plt.imshow(
            np.abs(plane_data)**2,
            norm=LogNorm(1e-10),
            extent=[-r_max, r_max, -r_max, r_max])
        plt.colorbar(cs)
        plt.xlabel("y-axis (a.u.)")
        plt.ylabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("wave_cut_xy_" + str(i).zfill(8) + ".png")
        plt.clf()

        fft_data = np.abs(np.fft.fftshift(np.fft.fft2(plane_data)))**2
        cs = plt.imshow(
            fft_data,
            # norm=LogNorm(1e-10),
            extent=[ky.min(), ky.max(), kx.min(),
                    kx.max()])
        plt.colorbar(cs)
        plt.xlabel("y-axis (a.u.)")
        plt.ylabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("momentum_xy_" + str(i).zfill(8) + ".png")
        plb.xlim([-1.0, 1.0])
        plb.ylim([-1.0, 1.0])
        plt.savefig("momentum_zoom_xy_" + str(i).zfill(8) + ".png")
        plt.clf()
