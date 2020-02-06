from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import json
from scipy.special import legendre
font = {'size': 22}
matplotlib.rc('font', **font)


def helium_sae(r):
    return -1.0/r - np.exp(-2.0329*r)/r - 0.3953 * np.exp(-6.1805*r)


def get_state(r, energy, l, potential, z=1.0, min_r=500.):
    """ 
    Returns the continuum state and phase shift for a given
    potential, radius, energy, and angular momentum eigenvalue
    """
    k = np.sqrt(2*energy)
    dr = r[1] - r[0]

    # make sure we go to large r where the boundary conditions apply
    new_r = None
    if r.max() < min_r:
        new_r = np.arange(dr, min_r+dr, dr)
    else:
        new_r = r
    # pre compute r^2
    r2 = new_r*new_r
    psi = np.zeros(new_r.shape)
    # assume r[0] is at dr
    # we can use anything and normalize later
    psi[0] = 1.0

    psi[1] = psi[0]*(dr*dr*((l*(l+1)/r2[0])+2*potential(new_r[0])+2*energy)+2)

    for idx in np.arange(2, new_r.shape[0]):
        psi[idx] = psi[idx-1]*(dr*dr*((l*(l+1)/r2[idx-1]) +
                                      2*potential(new_r[idx-1])-2*energy)+2) - psi[idx-2]
    r_val = new_r[-2]
    psi_r = psi[-2]
    dpsi_r = (psi[-1]-psi[-3])/(2*dr)

    norm = np.sqrt(np.abs(psi_r)**2+(np.abs(dpsi_r)/(k+z/(k*r_val)))**2)
    psi /= norm

    phase = np.angle((1.j*psi_r + dpsi_r/(k+z/(k*r_val))) /
                     (2*k*r_val)**(1.j*z/k)) - k*r_val + l*np.pi/2
    # phase = 0.0
    # make sure to reshape psi to the right size
    return phase, psi[:r.shape[0]]

# gets energy of state


def get_energy(psi_1, target):
    return target["Energy_l_"+str(psi_1[1])][psi_1[0]-1-psi_1[1], 0, 0]


def get_k_sphere(energy, psi, r, l_max, m_max, potential, target, d_angle=0.01):
    phi_angles = np.arange(-np.pi, np.pi, d_angle)
    phi = np.arange(-np.pi, np.pi, d_angle)
    theta = np.arange(0, np.pi+d_angle, d_angle)
    np.savetxt("phi.txt", phi)
    np.savetxt("theta.txt", theta)
    theta, phi = np.meshgrid(theta, phi)
    return_psi = np.zeros(phi.shape, dtype=complex)
    r_length = r.shape[0]
    lm_idx = 0
    with open("spherical_harm_TDSE_"+fold+".txt", "w") as f:
        f.write("# l, m, |Y_lm|, Arg(Y_lm)\n")
        for l_val in np.arange(0, l_max + 1):
            phase_shift, k_vec = get_state(r, energy, l_val, potential)
            m_range = min(l_val, m_max)
            for m_val in np.arange(-m_range, m_range + 1):
                lower_idx = lm_idx*r_length
                upper_idx = (lm_idx+1)*r_length
                # project out bound states
                try:
                    for psi_bound in target["/psi_l_"+str(l_val)+"/psi"]:
                        psi_bound = psi_bound[:, 0] + 1.j*psi_bound[:, 1]
                        psi[lower_idx:upper_idx] -= np.sum(
                            psi_bound.conj()*psi[lower_idx:upper_idx])*psi_bound
                except:
                    pass
                coef = np.exp(-1.j*phase_shift)*1.j**l_val * \
                    np.sum(k_vec.conj()*psi[lower_idx:upper_idx])
                f.write(str(l_val)+", "+str(m_val)+", " +
                        str(np.abs(coef))+", "+str(np.angle(coef))+"\n")
                return_psi += coef*sph_harm(m_val, l_val, phi, theta)
                lm_idx += 1
    max_angles = [np.argmax(np.abs(return_psi)**2)]
    return phi, theta, return_psi, phi_angles, d_angle

beta_max = 10
folders = ["cycles_delay_0.00"]
# folders = ["cycles_delay_0.45","cycles_delay_0.60"]
# folders = ["cycles_delay_0.00", "cycles_delay_0.10", "cycles_delay_0.20", "cycles_delay_0.30", "cycles_delay_0.40", "cycles_delay_0.50", "cycles_delay_0.60", "cycles_delay_0.70", "cycles_delay_0.80", "cycles_delay_0.90",
#            "cycles_delay_1.00", "cycles_delay_0.05", "cycles_delay_0.15", "cycles_delay_0.25", "cycles_delay_0.35", "cycles_delay_0.45", "cycles_delay_0.55", "cycles_delay_0.65", "cycles_delay_0.75", "cycles_delay_0.85", "cycles_delay_0.95"]
# folders = ["cycles_delay_0.00", "cycles_delay_0.10", "cycles_delay_0.20", "cycles_delay_0.30", "cycles_delay_0.40", "cycles_delay_0.50", "cycles_delay_0.60", "cycles_delay_0.70", "cycles_delay_0.80", "cycles_delay_0.90",
#             "cycles_delay_1.00"]
folders.sort()
target = h5py.File(folders[0]+"/H.h5", "r")
f = h5py.File(folders[0]+"/TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(
        f["Wavefunction"]["x_value_" + str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r = psi_cooridnate_values[2]
laser_energy = 0
with open(folders[0]+'/input.json') as json_file:
    input_file = json.load(json_file)
    laser_energy = input_file["laser"]["pulses"][0]["energy"]

laser_energy = 0
with open(folders[0]+'/input.json') as input_file:
    input_json = json.load(input_file)
    laser_energy = input_json["laser"]["pulses"][0]["energy"]

e_ground = get_energy([1, 0, 0, 1], target)
e_excited = get_energy([2, 1, 1, 1], target)
e_final = (2*np.abs(laser_energy)) - np.abs(e_ground)

pad_yield = []
phi_angles = None
for fold in folders:
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi = f["Wavefunction"]["psi"][-1]
    psi = psi[:, 0]+1.j*psi[:, 1]
    # print(e_final)
    phi, theta, cur_psi, phi_angles, d_angle = get_k_sphere(
        e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target)
    pad_yield.append(np.abs(cur_psi)**2)

pad_yield = np.array(pad_yield)
max_val = pad_yield.max()
print("# delay theta phi")
fig = plt.figure()
for idx, fold in enumerate(folders):
    with open(fold + '/input.json') as json_file:
        input_file = json.load(json_file)
        delay = float(fold.split("_")[-1])
        cur_data = pad_yield[idx]/max_val
        max_idx = np.unravel_index(np.argmax(cur_data), cur_data.shape)
        intensity = input_file["laser"]["pulses"][0]["intensity"]
        cycles = input_file["laser"]["pulses"][0]["cycles_on"] + \
            input_file["laser"]["pulses"][0]["cycles_off"]
        print(delay, theta[max_idx], phi[max_idx])
        X, Y, Z = cur_data*np.sin(theta)*np.cos(phi), cur_data * \
            np.sin(theta)*np.sin(phi), cur_data*np.cos(theta)
        ax = fig.add_subplot(111, projection='3d')
        cmap = cm.get_cmap("viridis")
        ax.plot_surface(X, Y, Z, facecolors=cmap(cur_data))
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.axis('off')
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # plt.title("TDSE - Delay: %.2f \n Cycles: %3.0f    Intensity: %.0e" %
        #           (delay, cycles, intensity))
        # plt.show()
        if np.abs(cycles - 10) < 1e-5 and np.abs(intensity-1e14)< 1e-5:
            ax.text(1.15, -0.5, 0, "(b)")
        if np.abs(cycles - 10) < 1e-5 and np.abs(intensity-1e10)< 1e-5:
            ax.text(1.15, -0.5, 0, "(d)")
        if np.abs(cycles - 2) < 1e-5 and np.abs(intensity-1e10)< 1e-5:
            ax.text(1.15, -0.5, 0, "(c)")
        if np.abs(cycles - 2) < 1e-5 and np.abs(intensity-1e14)< 1e-5:
            ax.text(1.15, -0.5, 0, "(a)")
        # plt.savefig("PAD_%02.0f_%.1e_%.2f.png" % (cycles, intensity, delay))
        plt.savefig("PAD_%02.0f_%.1e_%.2f_0p4.png" % (cycles, intensity, delay))
        # plt.savefig("PAD_%02.0f_%.1e_%.2f_1p0.png" % (cycles, intensity, delay))
        plt.clf()

for idx, fold in enumerate(folders):
    with open("Asym_from_x_"+fold+".txt", "w") as f:
        f.write("# angle, front_back, flip_about_xy\n")
        with open(fold + '/input.json') as json_file:
            input_file = json.load(json_file)
            delay = float(fold.split("_")[-1])
            cur_data = pad_yield[idx]/max_val
            for angle in phi_angles:
                idx_left = np.abs(phi-angle)
                idx_left[idx_left > np.pi] = np.pi*2-idx_left[idx_left > np.pi]
                idx_left = idx_left < np.pi/2
                idx_right = np.logical_not(idx_left)
                left_top_sum = np.sum(
                    (cur_data[idx_left]*d_angle*d_angle*np.sin(theta[idx_left]))[theta[idx_left] < (np.pi/2)])
                left_bottom_sum = np.sum(
                    (cur_data[idx_left]*d_angle*d_angle*np.sin(theta[idx_left]))[theta[idx_left] >= (np.pi/2)])
                right_top_sum = np.sum(
                    (cur_data[idx_right]*d_angle*d_angle*np.sin(theta[idx_right]))[theta[idx_right] < (np.pi/2)])
                right_bottom_sum = np.sum(
                    (cur_data[idx_right]*d_angle*d_angle*np.sin(theta[idx_right]))[theta[idx_right] >= (np.pi/2)])

                X, Y, Z = np.sin(theta)*np.cos(phi), np.sin(theta) * \
                    np.sin(phi), np.cos(theta)

                f.write(str(angle)+", "+str((left_top_sum+left_bottom_sum-(right_top_sum+right_bottom_sum))/(left_top_sum+left_bottom_sum+right_top_sum+right_bottom_sum)
                                            )+", "+str((left_top_sum+right_bottom_sum-(right_top_sum+left_bottom_sum))/(left_top_sum+left_bottom_sum+right_top_sum+right_bottom_sum))+"\n")

for idx, fold in enumerate(folders):
    # do not normalize to get good cross section
    cur_data = pad_yield[idx]
    with open("Beta_"+fold+".txt", "w") as f:
        f.write("#")
        for l in np.arange(0,beta_max+1):
            f.write(" beta_%2d" % l)
        f.write("\n")
        for l in np.arange(0,beta_max+1):
            beta_val = np.sum((l+0.5)*legendre(l)(np.cos(theta))*cur_data*d_angle*d_angle*np.sin(theta))
            f.write(str(beta_val)+" ")
        f.write("\n")

X, Y, Z = cur_data*np.sin(theta)*np.cos(phi), cur_data * \
            np.sin(theta)*np.sin(phi), cur_data*np.cos(theta)

for idx, fold in enumerate(folders):
    cur_data = pad_yield[idx]/max_val
    with open("3D_Beta_"+fold+".txt", "w") as f:
        f.write("#")
        for l in np.arange(0,beta_max+1):
            f.write(" l_%2d" % l)
        f.write("\n")
        for l in np.arange(0,beta_max+1):
            for m in np.arange(-beta_max,beta_max+1):
                if np.abs(m)>l:
                    f.write("(0.0+0.0j) ")
                else:
                    beta_val = np.sum(sph_harm(m, l, phi, theta)*cur_data*d_angle*d_angle*np.sin(theta))
                    f.write(str(beta_val)+" ")
            f.write("\n")


beta_max = 10
folders = ["cycles_delay_0.00"]
# folders = ["cycles_delay_0.45","cycles_delay_0.60"]
# folders = ["cycles_delay_0.00", "cycles_delay_0.10", "cycles_delay_0.20", "cycles_delay_0.30", "cycles_delay_0.40", "cycles_delay_0.50", "cycles_delay_0.60", "cycles_delay_0.70", "cycles_delay_0.80", "cycles_delay_0.90",
#            "cycles_delay_1.00", "cycles_delay_0.05", "cycles_delay_0.15", "cycles_delay_0.25", "cycles_delay_0.35", "cycles_delay_0.45", "cycles_delay_0.55", "cycles_delay_0.65", "cycles_delay_0.75", "cycles_delay_0.85", "cycles_delay_0.95"]
# folders = ["cycles_delay_0.00", "cycles_delay_0.10", "cycles_delay_0.20", "cycles_delay_0.30", "cycles_delay_0.40", "cycles_delay_0.50", "cycles_delay_0.60", "cycles_delay_0.70", "cycles_delay_0.80", "cycles_delay_0.90",
#             "cycles_delay_1.00"]
folders.sort()
target = h5py.File(folders[0]+"/H.h5", "r")
f = h5py.File(folders[0]+"/TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
shape = f["Wavefunction"]["num_x"][:]
psi_cooridnate_values = []
for dim_idx in np.arange(shape.shape[0]):
    psi_cooridnate_values.append(
        f["Wavefunction"]["x_value_" + str(dim_idx)][:])
l_values = f["/Wavefunction/l_values"][:]
m_values = f["/Wavefunction/m_values"][:]

r = psi_cooridnate_values[2]
laser_energy = 0
with open(folders[0]+'/input.json') as json_file:
    input_file = json.load(json_file)
    laser_energy = input_file["laser"]["pulses"][0]["energy"]

laser_energy = 0
with open(folders[0]+'/input.json') as input_file:
    input_json = json.load(input_file)
    laser_energy = input_json["laser"]["pulses"][0]["energy"]

e_ground = get_energy([1, 0, 0, 1], target)
e_excited = get_energy([2, 1, 1, 1], target)
e_final = (1*np.abs(laser_energy)) - np.abs(e_excited)

pad_yield = []
phi_angles = None
for fold in folders:
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi = f["Wavefunction"]["psi"][-1]
    psi = psi[:, 0]+1.j*psi[:, 1]
    # print(e_final)
    phi, theta, cur_psi, phi_angles, d_angle = get_k_sphere(
        e_final, psi, r, l_values.max(), m_values.max(), helium_sae, target)
    pad_yield.append(np.abs(cur_psi)**2)

pad_yield = np.array(pad_yield)
max_val = pad_yield.max()
print("# delay theta phi")
fig = plt.figure()
for idx, fold in enumerate(folders):
    with open(fold + '/input.json') as json_file:
        input_file = json.load(json_file)
        delay = float(fold.split("_")[-1])
        cur_data = pad_yield[idx]/max_val
        max_idx = np.unravel_index(np.argmax(cur_data), cur_data.shape)
        intensity = input_file["laser"]["pulses"][0]["intensity"]
        cycles = input_file["laser"]["pulses"][0]["cycles_on"] + \
            input_file["laser"]["pulses"][0]["cycles_off"]
        print(delay, theta[max_idx], phi[max_idx])
        X, Y, Z = cur_data*np.sin(theta)*np.cos(phi), cur_data * \
            np.sin(theta)*np.sin(phi), cur_data*np.cos(theta)
        ax = fig.add_subplot(111, projection='3d')
        cmap = cm.get_cmap("viridis")
        ax.plot_surface(X, Y, Z, facecolors=cmap(cur_data))
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.axis('off')
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # plt.title("TDSE - Delay: %.2f \n Cycles: %3.0f    Intensity: %.0e" %
        #           (delay, cycles, intensity))
        # plt.show()
        if np.abs(cycles - 10) < 1e-5 and np.abs(intensity-1e14)< 1e-5:
            ax.text(1.15, -0.5, 0, "(b)")
        if np.abs(cycles - 10) < 1e-5 and np.abs(intensity-1e10)< 1e-5:
            ax.text(1.15, -0.5, 0, "(d)")
        if np.abs(cycles - 2) < 1e-5 and np.abs(intensity-1e10)< 1e-5:
            ax.text(1.15, -0.5, 0, "(c)")
        if np.abs(cycles - 2) < 1e-5 and np.abs(intensity-1e14)< 1e-5:
            ax.text(1.15, -0.5, 0, "(a)")
        # plt.savefig("PAD_%02.0f_%.1e_%.2f.png" % (cycles, intensity, delay))
        plt.savefig("PAD_%02.0f_%.1e_%.2f_0p4_excited.png" % (cycles, intensity, delay))
        # plt.savefig("PAD_%02.0f_%.1e_%.2f_1p0.png" % (cycles, intensity, delay))
        plt.clf()

for idx, fold in enumerate(folders):
    with open("Asym_excited_from_x_"+fold+".txt", "w") as f:
        f.write("# angle, front_back, flip_about_xy\n")
        with open(fold + '/input.json') as json_file:
            input_file = json.load(json_file)
            delay = float(fold.split("_")[-1])
            cur_data = pad_yield[idx]/max_val
            for angle in phi_angles:
                idx_left = np.abs(phi-angle)
                idx_left[idx_left > np.pi] = np.pi*2-idx_left[idx_left > np.pi]
                idx_left = idx_left < np.pi/2
                idx_right = np.logical_not(idx_left)
                left_top_sum = np.sum(
                    (cur_data[idx_left]*d_angle*d_angle*np.sin(theta[idx_left]))[theta[idx_left] < (np.pi/2)])
                left_bottom_sum = np.sum(
                    (cur_data[idx_left]*d_angle*d_angle*np.sin(theta[idx_left]))[theta[idx_left] >= (np.pi/2)])
                right_top_sum = np.sum(
                    (cur_data[idx_right]*d_angle*d_angle*np.sin(theta[idx_right]))[theta[idx_right] < (np.pi/2)])
                right_bottom_sum = np.sum(
                    (cur_data[idx_right]*d_angle*d_angle*np.sin(theta[idx_right]))[theta[idx_right] >= (np.pi/2)])

                X, Y, Z = np.sin(theta)*np.cos(phi), np.sin(theta) * \
                    np.sin(phi), np.cos(theta)

                f.write(str(angle)+", "+str((left_top_sum+left_bottom_sum-(right_top_sum+right_bottom_sum))/(left_top_sum+left_bottom_sum+right_top_sum+right_bottom_sum)
                                            )+", "+str((left_top_sum+right_bottom_sum-(right_top_sum+left_bottom_sum))/(left_top_sum+left_bottom_sum+right_top_sum+right_bottom_sum))+"\n")

for idx, fold in enumerate(folders):
    # do not normalize to get good cross section
    cur_data = pad_yield[idx]
    with open("Beta_excited_"+fold+".txt", "w") as f:
        f.write("#")
        for l in np.arange(0,beta_max+1):
            f.write(" beta_%2d" % l)
        f.write("\n")
        for l in np.arange(0,beta_max+1):
            beta_val = np.sum((l+0.5)*legendre(l)(np.cos(theta))*cur_data*d_angle*d_angle*np.sin(theta))
            f.write(str(beta_val)+" ")
        f.write("\n")

X, Y, Z = cur_data*np.sin(theta)*np.cos(phi), cur_data * \
            np.sin(theta)*np.sin(phi), cur_data*np.cos(theta)

for idx, fold in enumerate(folders):
    cur_data = pad_yield[idx]/max_val
    with open("3D_Beta_excited_"+fold+".txt", "w") as f:
        f.write("#")
        for l in np.arange(0,beta_max+1):
            f.write(" l_%2d" % l)
        f.write("\n")
        for l in np.arange(0,beta_max+1):
            for m in np.arange(-beta_max,beta_max+1):
                if np.abs(m)>l:
                    f.write("(0.0+0.0j) ")
                else:
                    beta_val = np.sum(sph_harm(m, l, phi, theta)*cur_data*d_angle*d_angle*np.sin(theta))
                    f.write(str(beta_val)+" ")
            f.write("\n")
