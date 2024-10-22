from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import json
from scipy.special import legendre
font = {'size': 22}
matplotlib.rc('font', **font)


def helium_sae(r):
    return -1.0 / r - np.exp(-2.0329 * r) / r - 0.3953 * np.exp(-6.1805 * r)


def helium_sae_soft(r):
    alpha = 3.70e-02
    r_soft = np.sqrt(r**2 + alpha**2)
    return -1.0 / r_soft - np.exp(-2.0329 * r) / r_soft - 0.3953 * np.exp(
        -6.1805 * r)


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


def get_k_sphere(energy, psi, r, l_max, m_max, potential, target,
                 d_angle=0.01):
    phi_angles = np.arange(-np.pi, np.pi, d_angle)
    phi = np.arange(-np.pi, np.pi, d_angle)
    theta = np.arange(0, np.pi + d_angle, d_angle)
    np.savetxt("phi.txt", phi)
    np.savetxt("theta.txt", theta)
    theta, phi = np.meshgrid(theta, phi)
    return_psi = np.zeros(phi.shape, dtype=complex)
    r_length = r.shape[0]
    lm_idx = 0
    with open("spherical_harm_TDSE_" + str(energy) + ".txt", "w") as f:
        f.write("# l, m, |Y_lm|, Arg(Y_lm)\n")
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
                coef = np.exp(-1.j*phase_shift)*1.j**l_val * \
                    np.sum(k_vec.conj()*psi[lower_idx:upper_idx])
                f.write(
                    str(l_val) + ", " + str(m_val) + ", " + str(np.abs(coef)) +
                    ", " + str(np.angle(coef)) + "\n")
                return_psi += coef * sph_harm(m_val, l_val, phi, theta)
                lm_idx += 1
    max_angles = [np.argmax(np.abs(return_psi)**2)]
    return phi, theta, return_psi, phi_angles, d_angle


beta_max = 10
target = h5py.File("H.h5", "r")
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(f["Wavefunction"]["x_value_" +
                                                   str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r = psi_cooridnate_values[2]

e_ground = get_energy([1, 0, 0, 1], target)
e_excited = get_energy([2, 1, 1, 1], target)
# e_final_list = np.arange(0.01,1.0,0.1)
laser_energy = 0
with open('input.json') as json_file:
    input_file = json.load(json_file)
    laser_energy = input_file["laser"]["pulses"][0]["energy"]
e_final_list = [(1. * np.abs(laser_energy)) - np.abs(e_excited), 0.01928205187,
                0.1346748821, 0.224546544, 0.3399393742]
de = 0.01
e_final_list = np.arange(de, 1.5 + de / 2, de)
# e_final_list = [0.09375897933, 0.1836306412, 0.2990234714]

pad_yield = []
phi_angles = None
psi = f["Wavefunction"]["psi"][-1]
psi = psi[:, 0] + 1.j * psi[:, 1]
for e_final in e_final_list:
    print(e_final)
    phi, theta, cur_psi, phi_angles, d_angle = get_k_sphere(
        e_final, psi, r, l_values.max(), m_values.max(), helium_sae_soft,
        target)
    pad_yield.append(np.abs(cur_psi)**2)

pad_yield = np.array(pad_yield)
max_val = pad_yield.max()

with open("Beta.txt", "w") as f:
    f.write("# Energy sigma(ish)")
    for l in np.arange(1, beta_max + 1):
        f.write(" beta_%02d" % l)
    f.write("\n")
    for idx, e_final in enumerate(e_final_list):
        cur_data = pad_yield[idx]
        f.write(str(e_final) + " ")
        beta_list = np.zeros([beta_max + 1])
        for l in np.arange(0, beta_max + 1):
            beta_list[l] = np.sum((l + 0.5) * legendre(l)(np.cos(theta)) *
                                  cur_data * d_angle * d_angle * np.sin(theta))
        beta_list[1:] /= beta_list[0]
        for l in np.arange(0, beta_max + 1):
            f.write(str(beta_list[l]) + " ")
        f.write("\n")

# for idx, e_final in enumerate(e_final_list):
#     cur_data = pad_yield[idx]/max_val
#     with open("3D_Beta_"+fold+".txt", "w") as f:
#         f.write("#")
#         for l in np.arange(0,beta_max+1):
#             f.write(" l_%2d" % l)
#         f.write("\n")
#         for l in np.arange(0,beta_max+1):
#             for m in np.arange(-beta_max,beta_max+1):
#                 if np.abs(m)>l:
#                     f.write("(0.0+0.0j) ")
#                 else:
#                     beta_val = np.sum(sph_harm(m, l, phi, theta)*cur_data*d_angle*d_angle*np.sin(theta))
#                     f.write(str(beta_val)+" ")
#             f.write("\n")
