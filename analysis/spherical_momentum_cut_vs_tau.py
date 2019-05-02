import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import sph_harm
import pylab as plb
from scipy.interpolate import interp2d
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


folders = [
    'cycles_delay_0.00', 'cycles_delay_0.10', 'cycles_delay_0.20',
    'cycles_delay_0.30', 'cycles_delay_0.40', 'cycles_delay_0.50',
    'cycles_delay_0.60', 'cycles_delay_0.70', 'cycles_delay_0.80',
    'cycles_delay_0.90', 'cycles_delay_1.00'
]
folders.sort()
print(folders)
factor = 2
resolution = 1024 * factor
zoom_size = 1.5

f = h5py.File(folders[0] + "/TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(
        f["Wavefunction"]["x_value_" + str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r_max = psi_cooridnate_values[2].max() * factor
# pre-calculate grid so the plotting can be vectorized
print("Calculating index set for xy plane")
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

# w_res = 0.8623956
w_res = 0.790472
tau = 2. * np.pi / w_res
w_las = 0.8 * w_res

delta_theta = 0.1
theta_max = 180
theta_ring = np.arange(-theta_max * np.pi / 180.,
                       theta_max * np.pi / 180 + delta_theta, delta_theta)
ring_data_las = np.zeros((theta_ring.shape[0], len(folders)))
ring_data_res = np.zeros((theta_ring.shape[0], len(folders)))
ring_data_las_res = np.zeros((theta_ring.shape[0], len(folders)))
for fold_idx, fold in enumerate(folders):
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi_value = f["Wavefunction"]["psi"]
    i = len(psi_value) - 1
    print("plotting", fold, i)
    psi = psi_value[i]
    psi = psi[:, 0] + 1j * psi[:, 1]
    psi.shape = shape
    psi_norm = np.sqrt((psi * psi.conjugate()).sum())
    plane_data = get_data(
        psi,
        psi_cooridnate_values,
        r,
        theta,
        phi,
        r_vals,
        l_values,
        m_values,
        r_cut=0.)[:, :, 0]
    print(np.sqrt((plane_data * plane_data.conjugate()).sum()))
    plane_data = get_data(
        psi,
        psi_cooridnate_values,
        r,
        theta,
        phi,
        r_vals,
        l_values,
        m_values,
        r_cut=220.)[:, :, 0]
    print(np.sqrt((plane_data * plane_data.conjugate()).sum()))

    cs = plt.imshow(
        np.transpose(np.abs(plane_data)**2),
        norm=LogNorm(1e-10),
        extent=[-r_max, r_max, -r_max, r_max])
    plt.colorbar(cs)
    plt.xlabel("x-axis (a.u.)")
    plt.ylabel("y-axis (a.u.)")
    plt.tight_layout()
    plt.savefig("wave_cut_xy_" + fold + ".png")
    plt.clf()

    fft_data = np.abs(np.fft.fftshift(np.fft.fft2(plane_data)))**2

    cs = plt.imshow(
        np.transpose(fft_data),
        extent=[ky.min(), ky.max(), kx.min(),
                kx.max()])
    plt.colorbar(cs)
    plt.xlabel("x-axis (a.u.)")
    plt.ylabel("y-axis (a.u.)")
    plt.tight_layout()
    plb.xlim([-zoom_size, zoom_size])
    plb.ylim([-zoom_size, zoom_size])
    plt.savefig("momentum_xy_" + fold + ".png")
    plt.clf()

    interp = interp2d(kx, ky, fft_data, kind='cubic')

    k_max = np.sqrt(2. * (w_las * 2 - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_las[theta_idx, fold_idx] = interp(x_interp[theta_idx],
                                                    y_interp[theta_idx])[0]

    k_max = np.sqrt(2. * (w_res * 2 - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_res[theta_idx, fold_idx] = interp(x_interp[theta_idx],
                                                    y_interp[theta_idx])[0]

    k_max = np.sqrt(2. * (w_res + w_las - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_las_res[theta_idx, fold_idx] = interp(
            x_interp[theta_idx], y_interp[theta_idx])[0]

np.savetxt("xy_data_las.txt",ring_data_las)
np.savetxt("xy_data_res.txt",ring_data_res)
np.savetxt("xy_data_las_res.txt",ring_data_las_res)
np.savetxt("theta.txt", theta_ring)

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{las}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_las_xy.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{res}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_res_xy.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$\\omega_{res}+\\omega_{las}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_las_res_xy.png")
plt.clf()

for fold_idx, fold in enumerate(folders):
    ring_data_las[:, fold_idx] /= ring_data_las[:, fold_idx].max()
    ring_data_res[:, fold_idx] /= ring_data_res[:, fold_idx].max()
    ring_data_las_res[:, fold_idx] /= ring_data_las_res[:, fold_idx].max()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{las}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_las_xy.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{res}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_res_xy.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$\\omega_{res}+\\omega_{las}$ - xy plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_las_res_xy.png")
plt.clf()

# pre-calculate grid so the plotting can be vectorized
print("Calculating index set for xz plane")
x, y, z, r, theta, phi, r_vals = cacluate_xz_plane(psi_cooridnate_values,
                                                   r_max, r_max, resolution)
x_values = x[0, :, 0]
z_values = z[0, 0, :]
kx = x_values * 2.0 * np.pi / (x_values.shape[0] *
                               (x_values[1] - x_values[0]) *
                               (x_values[1] - x_values[0]))
kz = z_values * 2.0 * np.pi / (z_values.shape[0] *
                               (z_values[1] - z_values[0]) *
                               (z_values[1] - z_values[0]))

w_res = 0.790472
tau = 2. * np.pi / w_res
w_las = 0.8 * w_res

delta_theta = 0.1
theta_max = 180
theta_ring = np.arange(-theta_max * np.pi / 180.,
                       theta_max * np.pi / 180 + delta_theta, delta_theta)
ring_data_las = np.zeros((theta_ring.shape[0], len(folders)))
ring_data_res = np.zeros((theta_ring.shape[0], len(folders)))
ring_data_las_res = np.zeros((theta_ring.shape[0], len(folders)))
for fold_idx, fold in enumerate(folders):
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi_value = f["Wavefunction"]["psi"]
    i = len(psi_value) - 1
    print("plotting", fold, i)
    psi = psi_value[i]
    psi = psi[:, 0] + 1j * psi[:, 1]
    psi.shape = shape
    psi_norm = np.sqrt((psi * psi.conjugate()).sum())
    plane_data = get_data(
        psi,
        psi_cooridnate_values,
        r,
        theta,
        phi,
        r_vals,
        l_values,
        m_values,
        r_cut=220.)[0, :, :]

    cs = plt.imshow(
        np.transpose(np.abs(plane_data)**2),
        norm=LogNorm(1e-10),
        extent=[-r_max, r_max, -r_max, r_max])
    plt.colorbar(cs)
    plt.xlabel("x-axis (a.u.)")
    plt.ylabel("z-axis (a.u.)")
    plt.tight_layout()
    plt.savefig("wave_cut_xz_" + fold + ".png")
    plt.clf()

    fft_data = np.abs(np.fft.fftshift(np.fft.fft2(plane_data)))**2

    cs = plt.imshow(
        np.transpose(fft_data),
        extent=[kz.min(), kz.max(), kx.min(),
                kx.max()])
    plt.colorbar(cs)
    plt.xlabel("x-axis (a.u.)")
    plt.ylabel("z-axis (a.u.)")
    plt.tight_layout()
    plb.xlim([-zoom_size, zoom_size])
    plb.ylim([-zoom_size, zoom_size])
    plt.savefig("momentum_xz_" + fold + ".png")
    plt.clf()

    interp = interp2d(kx, kz, fft_data, kind='cubic')

    k_max = np.sqrt(2. * (w_las * 2 - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_las[theta_idx, fold_idx] = interp(x_interp[theta_idx],
                                                    y_interp[theta_idx])[0]

    k_max = np.sqrt(2. * (w_res * 2 - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_res[theta_idx, fold_idx] = interp(x_interp[theta_idx],
                                                    y_interp[theta_idx])[0]

    k_max = np.sqrt(2. * (w_res + w_las - 0.917965))
    x_interp = k_max * np.cos(theta_ring)
    y_interp = k_max * np.sin(theta_ring)
    for theta_idx in range(x_interp.shape[0]):
        ring_data_las_res[theta_idx, fold_idx] = interp(
            x_interp[theta_idx], y_interp[theta_idx])[0]

np.savetxt("xz_data_las.txt",ring_data_las)
np.savetxt("xz_data_res.txt",ring_data_res)
np.savetxt("xz_data_las_res.txt",ring_data_las_res)

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{las}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_las_xz.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{res}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_res_xz.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$\\omega_{res}+\\omega_{las}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_las_res_xz.png")
plt.clf()

for fold_idx, fold in enumerate(folders):
    ring_data_las[:, fold_idx] /= ring_data_las[:, fold_idx].max()
    ring_data_res[:, fold_idx] /= ring_data_res[:, fold_idx].max()
    ring_data_las_res[:, fold_idx] /= ring_data_las_res[:, fold_idx].max()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{las}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_las_xz.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$2*\\omega_{res}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_res_xz.png")
plt.clf()

fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("$\\omega_{res}+\\omega_{las}$ - xz plane")
plt.contourf(np.arange(0, 1.1, 0.1), theta_ring, ring_data_las_res, 30)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("Delay ($2\\pi/\\omega_{res}$)")
plt.ylabel("Angle from z-axis ($\\theta$)")
plt.axhline(y=np.pi / 2.0, color='r')
plt.axhline(y=0, color='r')
plt.axhline(y=-np.pi / 2.0, color='r')
plt.savefig("ring_norm_las_res_xz.png")
plt.clf()
