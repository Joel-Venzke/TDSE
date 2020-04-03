import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import sph_harm
import pylab as plb
from scipy.interpolate import interp2d
font = {'size': 22}
matplotlib.rc('font', **font)


def helium_sae(r):
    return -1.0 / r - np.exp(-2.0329 * r) / r - 0.3953 * np.exp(-6.1805 * r)


def get_state(r, energy, l, potential, z=1.0, min_r=500.):
    """ 
    Returns the continuum state and phase shift for a given
    potential, radius, energy, and angular momentum eigenvalue
    """
    k = np.sqrt(2 * energy)
    dr = r[1] - r[0]

    # make sure we go to large r where the boundary conditions apply
    new_r = None
    if r.max() < min_r:
        new_r = np.arange(dr, min_r + dr, dr)
    else:
        new_r = r
    # pre compute r^2
    r2 = new_r * new_r
    psi = np.zeros(new_r.shape)
    # assume r[0] is at dr
    # we can use anything and normalize later
    psi[0] = 1.0

    psi[1] = psi[0] * (dr * dr * (
        (l * (l + 1) / r2[0]) + 2 * potential(new_r[0]) + 2 * energy) + 2)

    for idx in np.arange(2, new_r.shape[0]):
        psi[idx] = psi[idx - 1] * (dr * dr * (
            (l * (l + 1) / r2[idx - 1]) + 2 * potential(new_r[idx - 1]) -
            2 * energy) + 2) - psi[idx - 2]
    r_val = new_r[-2]
    psi_r = psi[-2]
    dpsi_r = (psi[-1] - psi[-3]) / (2 * dr)

    norm = np.sqrt(
        np.abs(psi_r)**2 + (np.abs(dpsi_r) / (k + z / (k * r_val)))**2)
    psi /= norm

    phase = np.angle(
        (1.j * psi_r + dpsi_r / (k + z / (k * r_val))) /
        (2 * k * r_val)**(1.j * z / k)) - k * r_val + l * np.pi / 2
    # phase = 0.0
    # make sure to reshape psi to the right size
    return phase, psi[:r.shape[0]]


# gets energy of state
def get_energy(psi_1, target):
    return target["Energy_l_" + str(psi_1[1])][psi_1[0] - 1 - psi_1[1], 0, 0]


def get_k_slice(energy,
                psi,
                r,
                l_max,
                m_max,
                potential,
                target,
                plane='xy',
                d_phi=0.001):
    phi = np.arange(-np.pi, np.pi + d_phi, d_phi)
    np.savetxt("theta.txt", phi)
    return_psi = np.zeros(phi.shape[0], dtype=complex)
    r_length = r.shape[0]
    lm_idx = 0
    for l_val in np.arange(0, l_max + 1):
        phase_shift, k_vec = get_state(r, energy, l_val, potential)
        m_range = min(l_val, m_max)
        for m_val in np.arange(-m_range, m_range + 1):
            lower_idx = lm_idx * r_length
            upper_idx = (lm_idx + 1) * r_length
            # project out bound states
            try:
                for cur_psi_energy, psi_bound in zip(
                        target["/Energy_l_" + str(l_val)],
                        target["/psi_l_" + str(l_val) + "/psi"]):
                    if cur_psi_energy[0, 0] < 0:
                        psi_bound = psi_bound[:, 0] + 1.j * psi_bound[:, 1]
                        psi[lower_idx:upper_idx] -= np.sum(
                            psi_bound.conj() *
                            psi[lower_idx:upper_idx]) * psi_bound
            except:
                pass
            coef = np.exp(-1.j * phase_shift) * 1.j**l_val * np.sum(
                k_vec.conj() * psi[lower_idx:upper_idx])
            if plane == 'xz':
                return_psi += coef * sph_harm(m_val, l_val, 0, phi)
            elif plane == 'xy':
                return_psi += coef * sph_harm(m_val, l_val, phi, np.pi / 2)
            lm_idx += 1
    max_angles = [np.argmax(np.abs(return_psi)**2)]
    return (phi[max_angles[np.abs(phi[max_angles]).argmin()]] +
            np.pi / 3) % np.pi - np.pi / 3, return_psi


folders = [
    "cycles_delay_0.00", "cycles_delay_0.10", "cycles_delay_0.20",
    "cycles_delay_0.30", "cycles_delay_0.40", "cycles_delay_0.50",
    "cycles_delay_0.60", "cycles_delay_0.70", "cycles_delay_0.80",
    "cycles_delay_0.90", "cycles_delay_0.05", "cycles_delay_0.15",
    "cycles_delay_0.25", "cycles_delay_0.35", "cycles_delay_0.45",
    "cycles_delay_0.55", "cycles_delay_0.65", "cycles_delay_0.75",
    "cycles_delay_0.85", "cycles_delay_0.95"
]
folders.sort()
print(folders)

target = h5py.File(folders[0] + "/H.h5", "r")
f = h5py.File(folders[0] + "/TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(f["Wavefunction"]["x_value_" +
                                                   str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r = psi_cooridnate_values[2]

idx = -1

e_ground = get_energy([1, 0, 0, 1], target)
e_excited = get_energy([2, 1, 1, 1], target)
e_final = 0.7233645
d_phi = np.pi / 5
angle = []
data = []
for fold in folders:
    print(fold)
    delay = float(fold.split("_")[-1])
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi = f["Wavefunction"]["psi"][idx]
    psi = psi[:, 0] + 1.j * psi[:, 1]
    cur_angle, cur_psi = get_k_slice(e_final,
                                     psi,
                                     r,
                                     l_values.max(),
                                     m_values.max(),
                                     helium_sae,
                                     target,
                                     plane='xy')
    data.append(cur_psi)
    angle.append([delay, cur_angle])
data = np.array(data)
angle = np.array(angle)
print(angle)

np.savetxt("xy_data_2p_signal.txt", np.abs(data.transpose())**2)
fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("xy_data_2p_signal")
d_ang = 0.001
plt.contourf(angle[:, 0], np.arange(-np.pi, np.pi + d_ang, d_ang),
             np.abs(data.transpose())**2, 20)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("delay")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.savefig("xy_data_2p_signal.png")
plt.clf()

e_ground = get_energy([1, 0, 0, 1], target)
e_excited = get_energy([2, 1, 1, 1], target)
e_final = 0.7951555
d_phi = np.pi / 5
angle = []
data = []
for fold in folders:
    print(fold)
    delay = float(fold.split("_")[-1])
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi = f["Wavefunction"]["psi"][idx]
    psi = psi[:, 0] + 1.j * psi[:, 1]
    cur_angle, cur_psi = get_k_slice(e_final,
                                     psi,
                                     r,
                                     l_values.max(),
                                     m_values.max(),
                                     helium_sae,
                                     target,
                                     plane='xy')
    data.append(cur_psi)
    angle.append([delay, cur_angle])
data = np.array(data)
angle = np.array(angle)
print(angle)

np.savetxt("xy_data_3p_signal.txt", np.abs(data.transpose())**2)
fig = plt.figure(figsize=(12, 9), dpi=80)
plt.title("xy_data_3p_signal")
d_ang = 0.001
plt.contourf(angle[:, 0], np.arange(-np.pi, np.pi + d_ang, d_ang),
             np.abs(data.transpose())**2, 20)
plt.colorbar()
plt.xlim([0.0, 1.0])
plt.xlabel("delay")
plt.ylabel("Angle from x-axis ($\\phi$)")
plt.savefig("xy_data_3p_signal.png")
plt.clf()

# e_ground = get_energy([1,0,0,1],target)
# e_excited = get_energy([2,1,1,1],target)
# e_final = (2.*0.8666666667*np.abs(0.815939)) - np.abs(e_ground)
# d_phi = np.pi/5
# angle = []
# data = []
# for fold in folders:
#     print(fold)
#     delay = float(fold.split("_")[-1])
#     f = h5py.File(fold + "/TDSE.h5", "r")
#     psi = f["Wavefunction"]["psi"][idx]
#     psi = psi[:,0]+1.j*psi[:,1]
#     cur_angle, cur_psi = get_k_slice(e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target, plane='xz')
#     data.append(cur_psi)
#     angle.append([delay,cur_angle])
# data = np.array(data)
# angle = np.array(angle)
# print(angle)

# np.savetxt("xz_data_las.txt",np.abs(data.transpose())**2)
# fig = plt.figure(figsize=(12, 9), dpi=80)
# plt.title("xz_data_las")
# d_ang = 0.001
# plt.contourf(angle[:,0], np.arange(-np.pi,np.pi+d_ang,d_ang), np.abs(data.transpose())**2, 20)
# plt.colorbar()
# plt.xlim([0.0, 1.0])
# plt.xlabel("delay")
# plt.ylabel("Angle from x-axis ($\\phi$)")
# plt.savefig("xz_data_las.png")
# plt.clf()

# e_ground = get_energy([1,0,0,1],target)
# e_excited = get_energy([2,1,1,1],target)
# e_final = (2.*np.abs(0.815939)) - np.abs(e_ground)
# d_phi = np.pi/5
# angle = []
# data = []
# for fold in folders:
#     print(fold)
#     delay = float(fold.split("_")[-1])
#     f = h5py.File(fold + "/TDSE.h5", "r")
#     psi = f["Wavefunction"]["psi"][idx]
#     psi = psi[:,0]+1.j*psi[:,1]
#     cur_angle, cur_psi = get_k_slice(e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target, plane='xz')
#     data.append(cur_psi)
#     angle.append([delay,cur_angle])
# data = np.array(data)
# angle = np.array(angle)
# print(angle)

# np.savetxt("xz_data_res.txt",np.abs(data.transpose())**2)
# fig = plt.figure(figsize=(12, 9), dpi=80)
# plt.title("xz_data_res")
# d_ang = 0.001
# plt.contourf(angle[:,0], np.arange(-np.pi,np.pi+d_ang,d_ang), np.abs(data.transpose())**2, 20)
# plt.colorbar()
# plt.xlim([0.0, 1.0])
# plt.xlabel("delay")
# plt.ylabel("Angle from x-axis ($\\phi$)")
# plt.savefig("xz_data_res.png")
# plt.clf()

# e_ground = get_energy([1,0,0,1],target)
# e_excited = get_energy([2,1,1,1],target)
# e_final = ((1+0.8666666667)*np.abs(0.815939)) - np.abs(e_ground)
# d_phi = np.pi/5
# angle = []
# data = []
# for fold in folders:
#     print(fold)
#     delay = float(fold.split("_")[-1])
#     f = h5py.File(fold + "/TDSE.h5", "r")
#     psi = f["Wavefunction"]["psi"][idx]
#     psi = psi[:,0]+1.j*psi[:,1]
#     cur_angle, cur_psi = get_k_slice(e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target, plane='xy')
#     data.append(cur_psi)
#     angle.append([delay,cur_angle])
# data = np.array(data)
# angle = np.array(angle)
# print(angle)

# np.savetxt("xy_data_las_res.txt",np.abs(data.transpose())**2)
# fig = plt.figure(figsize=(12, 9), dpi=80)
# plt.title("xy_data_las_res")
# d_ang = 0.001
# plt.contourf(angle[:,0], np.arange(-np.pi,np.pi+d_ang,d_ang), np.abs(data.transpose())**2, 20)
# plt.colorbar()
# plt.xlim([0.0, 1.0])
# plt.xlabel("delay")
# plt.ylabel("Angle from x-axis ($\\phi$)")
# plt.savefig("xy_data_las_res.png")
# plt.clf()

# e_ground = get_energy([1,0,0,1],target)
# e_excited = get_energy([2,1,1,1],target)
# e_final = (2.*0.8666666667*np.abs(0.815939)) - np.abs(e_ground)
# d_phi = np.pi/5
# angle = []
# data = []
# for fold in folders:
#     print(fold)
#     delay = float(fold.split("_")[-1])
#     f = h5py.File(fold + "/TDSE.h5", "r")
#     psi = f["Wavefunction"]["psi"][idx]
#     psi = psi[:,0]+1.j*psi[:,1]
#     cur_angle, cur_psi = get_k_slice(e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target, plane='xy')
#     data.append(cur_psi)
#     angle.append([delay,cur_angle])
# data = np.array(data)
# angle = np.array(angle)
# print(angle)

# np.savetxt("xy_data_las.txt",np.abs(data.transpose())**2)
# fig = plt.figure(figsize=(12, 9), dpi=80)
# plt.title("xy_data_las")
# d_ang = 0.001
# plt.contourf(angle[:,0], np.arange(-np.pi,np.pi+d_ang,d_ang), np.abs(data.transpose())**2, 20)
# plt.colorbar()
# plt.xlim([0.0, 1.0])
# plt.xlabel("delay")
# plt.ylabel("Angle from x-axis ($\\phi$)")
# plt.savefig("xy_data_las.png")
# plt.clf()

# e_ground = get_energy([1,0,0,1],target)
# e_excited = get_energy([2,1,1,1],target)
# e_final = (2.*np.abs(0.815939)) - np.abs(e_ground)
# d_phi = np.pi/5
# angle = []
# data = []
# for fold in folders:
#     print(fold)
#     delay = float(fold.split("_")[-1])
#     f = h5py.File(fold + "/TDSE.h5", "r")
#     psi = f["Wavefunction"]["psi"][idx]
#     psi = psi[:,0]+1.j*psi[:,1]
#     cur_angle, cur_psi = get_k_slice(e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target, plane='xy')
#     data.append(cur_psi)
#     angle.append([delay,cur_angle])
# data = np.array(data)
# angle = np.array(angle)
# print(angle)

# np.savetxt("xy_data_res.txt",np.abs(data.transpose())**2)
# fig = plt.figure(figsize=(12, 9), dpi=80)
# plt.title("xy_data_res")
# d_ang = 0.001
# plt.contourf(angle[:,0], np.arange(-np.pi,np.pi+d_ang,d_ang), np.abs(data.transpose())**2, 20)
# plt.colorbar()
# plt.xlim([0.0, 1.0])
# plt.xlabel("delay")
# plt.ylabel("Angle from x-axis ($\\phi$)")
# plt.savefig("xy_data_res.png")
# plt.clf()
