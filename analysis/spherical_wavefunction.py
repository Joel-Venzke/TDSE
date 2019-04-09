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
def get_data(psi, psi_cooridnate_values, r, theta, phi, r_vals, l_values,
             m_values):
    l = 0
    m = 0
    # Handle l=0 to initialize data
    data = np.zeros(
        (psi[0, l, r_vals] * sph_harm(m, l, phi, theta)).shape, dtype=complex)
    # loop over l>0
    for lm_idx in np.arange(0, psi_cooridnate_values[1].shape[0]):
        l = l_values[lm_idx]
        m = m_values[lm_idx]
        print("(l,m)", l, m, (np.sqrt(
            (psi[0, lm_idx, :] * psi[0, lm_idx, :].conjugate()).sum())
                              / psi_norm).real, sph_harm(
                                  m, l, phi[0, 0, 0], theta[0, 0, 0]))
        data += ((
            -1.0)**m) * psi[0, lm_idx, r_vals] * sph_harm(m, l, phi, theta)
    data[r > psi_cooridnate_values[2].max()] = 1e-100
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


f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(
        f["Wavefunction"]["x_value_" + str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

# r_max = 15
r_max = psi_cooridnate_values[2].max()
# pre-calculate grid so the plotting can be vectorized
print("Calculating index set for xy plane")
x, y, z, r, theta, phi, r_vals = cacluate_xy_plane(psi_cooridnate_values,
                                                   r_max, r_max, 500)

cmaps = ['cividis']
fig = plt.figure(figsize=(12, 9), dpi=80)
for i, psi in enumerate(psi_value):
    if i > 0:  # the 0th index is garbage
        print("plotting", i)
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = shape
        psi_norm = np.sqrt((psi * psi.conjugate()).sum())
        for l in np.arange(0, psi_cooridnate_values[1].shape[0]):
            print("Magnitude of (l,m) " + str(l_values[l]) + "," +
                  str(m_values[l]) + ": " + str((np.sqrt(
                      (psi[0, l, :] * psi[0, l, :].conjugate()).sum()) /
                                                 psi_norm).real))
        plane_data = get_data(psi, psi_cooridnate_values, r, theta, phi,
                              r_vals, l_values, m_values)[:, :, 0]
        cs = plt.imshow(
            (np.abs(plane_data)**2),
            norm=LogNorm(1e-15),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap=cmaps[int(i / 50) % len(cmaps)])
        plt.colorbar(cs)
        plt.ylabel("y-axis (a.u.)")
        plt.xlabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_xy_" + str(i).zfill(8) + ".png")
        plt.clf()

        cs = plt.imshow(
            (np.angle(plane_data)),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap='twilight',
            vmin=-np.pi,
            vmax=np.pi)
        plt.colorbar(cs)
        plt.ylabel("y-axis (a.u.)")
        plt.xlabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_xy_phase_" + str(i).zfill(8) + ".png")
        plt.clf()

print("Calculating index set for xz plane")
x, y, z, r, theta, phi, r_vals = cacluate_xz_plane(psi_cooridnate_values,
                                                   r_max, r_max, 500)

fig = plt.figure(figsize=(12, 9), dpi=80)
for i, psi in enumerate(psi_value):
    if i > 0:  # the 0th index is garbage
        print("plotting", i)
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = shape
        psi_norm = np.sqrt((psi * psi.conjugate()).sum())
        for l in np.arange(0, psi_cooridnate_values[1].shape[0]):
            print("Magnitude of (l,m) " + str(l_values[l]) + "," +
                  str(m_values[l]) + ": " + str((np.sqrt(
                      (psi[0, l, :] * psi[0, l, :].conjugate()).sum()) /
                                                 psi_norm).real))
        plane_data = get_data(psi, psi_cooridnate_values, r, theta, phi,
                              r_vals, l_values, m_values)[0]
        cs = plt.imshow(
            np.transpose(np.abs(plane_data)**2),
            norm=LogNorm(1e-15),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap=cmaps[int(i / 50) % len(cmaps)])
        plt.colorbar(cs)
        plt.ylabel("z-axis (a.u.)")
        plt.xlabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_xz_" + str(i).zfill(8) + ".png")
        plt.clf()

        cs = plt.imshow(
            np.transpose(np.angle(plane_data)),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap='twilight',
            vmin=-np.pi,
            vmax=np.pi)
        plt.colorbar(cs)
        plt.ylabel("z-axis (a.u.)")
        plt.xlabel("x-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_xz_phase_" + str(i).zfill(8) + ".png")
        plt.clf()

print("Calculating index set for yz plane")
x, y, z, r, theta, phi, r_vals = cacluate_yz_plane(psi_cooridnate_values,
                                                   r_max, r_max, 500)

fig = plt.figure(figsize=(12, 9), dpi=80)
for i, psi in enumerate(psi_value):
    if i > 0:  # the 0th index is garbage
        print("plotting", i)
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = shape
        psi_norm = np.sqrt((psi * psi.conjugate()).sum())
        for l in np.arange(0, psi_cooridnate_values[1].shape[0]):
            print("Magnitude of (l,m) " + str(l_values[l]) + "," +
                  str(m_values[l]) + ": " + str((np.sqrt(
                      (psi[0, l, :] * psi[0, l, :].conjugate()).sum()) /
                                                 psi_norm).real))
        plane_data = get_data(psi, psi_cooridnate_values, r, theta, phi,
                              r_vals, l_values, m_values)[:, 0, :]
        print(r.shape)
        cs = plt.imshow(
            np.transpose(np.abs(plane_data)**2),
            norm=LogNorm(1e-15),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap=cmaps[int(i / 50) % len(cmaps)])
        plt.colorbar(cs)
        plt.ylabel("z-axis (a.u.)")
        plt.xlabel("y-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_yz_" + str(i).zfill(8) + ".png")
        plt.clf()

        cs = plt.imshow(
            np.transpose(np.angle(plane_data)),
            extent=[-r_max, r_max, -r_max, r_max],
            cmap='twilight',
            vmin=-np.pi,
            vmax=np.pi)
        plt.colorbar(cs)
        plt.ylabel("z-axis (a.u.)")
        plt.xlabel("y-axis (a.u.)")
        plt.tight_layout()
        plt.savefig("figs/wave_yz_phase_" + str(i).zfill(8) + ".png")
        plt.clf()
