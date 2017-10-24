import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
observables = f["Observables"]
pulses = p["Pulse"]
p_time = pulses["time"][:]
time = observables["time"][1:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]

# Norm
print "Plotting Norm"
fig = plt.figure()
plt.plot(time, observables["norm"][1:])
plt.xlabel("Time (a.u.)")
plt.ylabel("Norm")
plt.title("Norm")
plt.tight_layout()
fig.savefig("figs/Norm.png")
plt.clf()
plt.close(fig)

# Ionization
print "Plotting Ionization"
fig = plt.figure()
plt.plot(time, 1.0 - observables["norm"][1:])
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization")
plt.title("Ionization")
plt.tight_layout()
fig.savefig("figs/Ionization.png")
plt.clf()
plt.close(fig)

# Ionization
print "Plotting Ionization Rate"
fig = plt.figure()
plt.plot(time, np.gradient(1.0 - observables["norm"][1:]))
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization Rate")
plt.title("Ionization Rate")
plt.tight_layout()
fig.savefig("figs/Ionization_rate.png")
plt.clf()
plt.close(fig)

# Gobbler
print "Plotting ECS Population"
fig = plt.figure()
if np.max(observables["gobbler"][1:]) > 1e-19:
    plt.semilogy(time, observables["gobbler"][1:])
    plt.xlabel("Time (a.u.)")
    plt.ylabel("ECS Population")
    plt.title("ECS Population")
    plt.tight_layout()
    fig.savefig("figs/ECS_population.png")
plt.clf()
plt.close(fig)

# Dipole
print "Plotting Dipole"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                time,
                -1.0 * observables["position_expectation_"
                                   + str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole (a.u.)")
plt.title("Dipole Moment")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole.png")
plt.clf()
plt.close(fig)

# dipole with ionization
print "Plotting Dipole with ionization"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                -1.0 * observables["position_expectation_"
                                   + str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
ax2.plot(time, 1.0 - observables["norm"][1:], 'r--', label="Ionization")
ax1.set_xlabel("Time a.u.")
ax1.set_ylabel("Dipole (a.u.)")
ax2.set_ylim(ymin=0)
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_with_ionization.png")

# Dipole with envelope
print "Plotting Dipole with field Envelope"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                np.abs(-1.0 *
                       observables["position_expectation_"
                                   + str(elec_idx) + "_" + str(dim_idx)][1:]),
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
for pulse_idx in range(num_pulses):
    ax2.plot(
        p_time,
        pulses["Pulse_envelope_" + str(pulse_idx)],
        'r--',
        label="Pulse " + str(pulse_idx))
ax1.set_xlabel("Time a.u.")
ax1.set_ylabel("Dipole (a.u.)")
ax2.set_ylabel("Field (a.u.)")
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_with_field_envelope.png")

# Dipole acceleration
print "Plotting Dipole Acceleration"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                time,
                observables["dipole_acceleration_"
                            + str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Dipole Acceleration")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole_acceleration.png")
plt.clf()
plt.close(fig)

# HHG Spectrum
print "Plotting HHG Spectrum"
fig = plt.figure()
energy = f["Parameters"]["energy"][0]
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            data = observables[
                "dipole_acceleration_" + str(elec_idx) + "_" + str(dim_idx)][
                    1:len(pulses["field_" + str(dim_idx)]
                          [checkpoint_frequency::checkpoint_frequency]) + 1]
            data = data * np.blackman(data.shape[0])
            padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
            paddT = np.max(time) * padd2 / data.shape[0]
            dH = 2 * np.pi / paddT / energy
            if np.max(data) > 1e-19:
                data = np.absolute(
                    np.fft.fft(
                        np.lib.pad(
                            data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                   int(np.ceil((padd2 - data.shape[0]) / 2))),
                            'constant',
                            constant_values=(0.0, 0.0))))
                plt.semilogy(
                    np.arange(data.shape[0]) * dH,
                    data,
                    label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.ylabel("HHG Spectrum (a.u.)")
plt.title("HHG Spectrum")
plt.legend()
plt.xlim([0, 100])
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum.png")
plt.clf()
plt.close(fig)

# Dipole acceleration with envelope
print "Plotting Dipole Acceleration with field Envelope"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                np.abs(observables["dipole_acceleration_" + str(elec_idx) + "_"
                                   + str(dim_idx)][1:]),
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
for pulse_idx in range(num_pulses):
    ax2.plot(
        p_time,
        pulses["Pulse_envelope_" + str(pulse_idx)],
        'r--',
        label="Pulse " + str(pulse_idx))
ax1.set_xlabel("Time (a.u.)")
ax1.set_ylabel("Dipole Acceleration (a.u.)")
ax2.set_ylabel("Field (a.u.)")
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_acceleration_with_field_envelope.png")

# Linearity
print "Plotting Linearity"
fig = plt.figure()
plt.plot(-1.0 * np.gradient(
    pulses["field_1"][checkpoint_frequency::checkpoint_frequency],
    f["Parameters"]["delta_t"][0] * checkpoint_frequency) * 7.2973525664e-3,
         np.gradient(
             pulses["field_1"][checkpoint_frequency::checkpoint_frequency],
             f["Parameters"]["delta_t"][0] * checkpoint_frequency) *
         7.2973525664e-3, "r-.")
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                observables["dipole_acceleration_"
                            + str(elec_idx) + "_" + str(dim_idx)]
                [1:len(pulses["field_" + str(dim_idx)]
                       [checkpoint_frequency::checkpoint_frequency]) + 1],
                -1.0 * np.gradient(pulses["field_1"][
                    checkpoint_frequency::checkpoint_frequency], f[
                        "Parameters"]["delta_t"][0] * checkpoint_frequency) *
                7.2973525664e-3,
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Field in (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Linearity")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Linearity.png")
plt.clf()
plt.close(fig)

# Projection
print "Plotting Projection"


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


def get_shells(state_number):
    shells = [0]
    while (state_number > shells[-1]):
        shells.append(shells[-1] + len(shells))
    return np.array(shells)


# return list of states up to state_number
def state_name(state_number):
    # get size of each shell
    shells = get_shells(state_number)

    # get list of names
    name_list = []
    for state in range(1, state_number + 1):
        name_list.append(state_single_name(state, shells))

    return name_list


#0field generator
def get_field_zeros(pulses):
    # field starts at zero
    field_zeros = [p_time[0]]

    # use the string to access what you want
    field = np.abs(pulses["field_1"])

    # loop over field with i as your index
    for i, p in enumerate(field):
        # avoid issues at ends
        if i > 0 and i < field.shape[0] - 1:
            if (p < field[i - 1] and p < field[i + 1]):
                field_zeros.append(p_time[i])

    # field ends at zero
    field_zeros.append(p_time[field.shape[0] - 1])

    return np.array(field_zeros)


#invoke projection and new time(for index to work)
data = f["Wavefunction"]["projections"][:, :, :]
w_time = f["Wavefunction"]["time"][:]

# get square of projections
data = np.absolute(data[:, :, 0] + 1j * data[:, :, 1])
data *= data

#plotting style/name set up
linestyles = ['-.', '-', '--', ':']
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
state_labels = state_name(data.shape[1])

# get projections nearest field zeros
plot_time = []
plot_data = []
for zero in get_field_zeros(pulses):
    idx = np.argmin(np.abs(w_time - zero))
    plot_time.append(w_time[idx])
    plot_data.append(data[idx])
plot_time = np.array(plot_time)
plot_data = np.array(plot_data)

fig = plt.figure()
for state_number in range(data.shape[1]):
    plt.semilogy(
        plot_time,
        plot_data[:, state_number],
        marker='o',
        label=state_labels[state_number],
        color=colors[state_number % len(colors)],
        linestyle=linestyles[(state_number / len(colors)) % len(linestyles)])

plt.ylabel("Population")
plt.xlabel("Time (a.u.)")
plt.ylim([1e-20, 10])
plt.legend(loc=2)
fig.savefig("figs/Projection_vs_time.png")
plt.clf()
plt.close(fig)

fig = plt.figure()
pulse_end_idx = np.argmin(np.abs(w_time - p_time[-1]))
for state_number in range(data.shape[1]):
    plt.plot(
        w_time[pulse_end_idx:],
        data[pulse_end_idx:, state_number],
        marker='o',
        label=state_labels[state_number],
        color=colors[state_number % len(colors)],
        linestyle=linestyles[(state_number / len(colors)) % len(linestyles)])

    plt.ylabel("Population")
    plt.xlabel("Time (a.u.)")
    plt.legend(loc=2)
    fig.savefig("figs/Projection_" + str(state_number).zfill(4) + "_" +
                state_labels[state_number] + ".png")
    plt.clf()
    plt.close(fig)

fig = plt.figure(figsize=(24, 18), dpi=80)
for idx in get_shells(plot_data.shape[1]):
    plt.axvline(x=idx, color='k')
plt.semilogy(range(data.shape[1]), data[-1, :], 'o-')

plt.ylabel("Population")
plt.xlabel("Bound State")
plt.title("A max: " + str(f["Parameters"]["field_max"][0]) + " Cycles: " + str(
    f["Parameters"]["cycles_on"][0] + f["Parameters"]["cycles_off"][0]))
plt.xlim([min(range(plot_data.shape[1])), max(range(plot_data.shape[1]))])
plt.xticks(range(plot_data.shape[1]), state_labels, rotation='vertical')
plt.ylim([1e-17, 10])
plt.grid()
fig.savefig("figs/Projection_at_end.png")
plt.clf()
plt.close(fig)


def sum_by_n(data):
    ret_val = []
    shells = get_shells(data.shape[1])
    ret_val = np.zeros((data.shape[0], len(shells) + 1))
    for n in range(len(shells)):
        for l in range(n):
            if n > 0:
                ret_val[:, n] += data[:, shells[n - 1] + l]
            else:
                ret_val[:, n] += data[:, l]
    return ret_val


fig = plt.figure()
by_n_value = sum_by_n(plot_data)
plt.semilogy(range(len(by_n_value[-1])), by_n_value[-1], 'o-')
plt.ylabel("Population")
plt.xlabel("N value")
plt.ylim([1e-20, 10])
plt.xlim([0, len(by_n_value[-1])])
plt.xticks(range(len(by_n_value[-1])))
fig.savefig("figs/Projection_at_end_by_n.png")
plt.clf()
plt.close(fig)

fig = plt.figure()
for n_value in range(1, by_n_value.shape[1]):
    plt.semilogy(
        plot_time,
        by_n_value[:, n_value],
        marker='o',
        label="n=" + str(n_value),
        color=colors[n_value % len(colors)],
        linestyle=linestyles[(n_value / len(colors)) % len(linestyles)])

plt.ylabel("Population")
plt.xlabel("Time (a.u.)")
plt.ylim([1e-20, 10])
plt.legend(loc=2)
fig.savefig("figs/Projection_vs_time_by_n.png")
plt.clf()
plt.close(fig)


def sum_by_l(data):
    ret_val = []
    shells = get_shells(data.shape[1])
    if len(shells) > 2:
        ret_val = np.zeros((data.shape[0], shells[-1] - shells[-2]))
    else:
        ret_val = np.zeros((data.shape[0], 3))
    for n in range(len(shells)):
        for l in range(n):
            if n > 0:
                ret_val[:, l] += data[:, shells[n - 1] + l]
            else:
                ret_val[:, l] += data[:, l]
    return ret_val


fig = plt.figure()
by_l_value = sum_by_l(plot_data)
plt.semilogy(range(len(by_l_value[-1])), by_l_value[-1], 'o-')
plt.ylabel("Population")
plt.xlabel("l value")
plt.ylim([1e-20, 10])
plt.xticks(range(len(by_l_value[-1])))
fig.savefig("figs/Projection_at_end_by_l.png")
plt.clf()
plt.close(fig)

fig = plt.figure()
for l_value in range(by_l_value.shape[1]):
    label = ""
    # create label
    if l_value == 0:
        label = "s"
    elif l_value == 1:
        label = "p"
    elif l_value == 2:
        label = "d"
    elif l_value == 3:
        label = "f"
    elif l_value > 23:  # anything greater that z is just a number
        label = ",l=" + str(l_value)
    else:  # any
        label = chr(ord('g') + l_value - 4)

    plt.semilogy(
        plot_time,
        by_l_value[:, l_value],
        marker='o',
        label=label,
        color=colors[l_value % len(colors)],
        linestyle=linestyles[(l_value / len(colors)) % len(linestyles)])

plt.ylabel("Population")
plt.xlabel("Time (a.u.)")
plt.ylim([1e-20, 10])
plt.legend(loc=2)
fig.savefig("figs/Projection_vs_time_by_l.png")
plt.clf()
plt.close(fig)


def grid_by_l_and_n(data):
    ret_val = []
    shells = get_shells(data.shape[1])
    l_dim = 0
    if len(shells) > 2:
        l_dim = shells[-1] - shells[-2]
    else:
        l_dim = 3
    n_dim = len(shells) + 1
    ret_val = np.zeros((n_dim, l_dim))
    for n in range(len(shells)):
        for l in range(n):
            if n > 0:
                ret_val[n, l] += data[-1, shells[n - 1] + l]
            else:
                ret_val[n, l] += data[-1, l]
    return ret_val


grid_data = grid_by_l_and_n(data)
fig = plt.figure()
plt.imshow(
    grid_data[1:],
    cmap='viridis',
    origin='lower',
    interpolation='none',
    norm=LogNorm(vmin=1e-15))
ax = plt.gca()
ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1))
ax.set_yticks(np.arange(.5, grid_data.shape[0], 1))
ax.set_xticklabels(np.arange(0, grid_data.shape[1], 1))
ax.set_yticklabels(np.arange(1, grid_data.shape[0], 1))
ax.grid(color='w', linestyle='-', linewidth=2)
plt.xlabel("l value")
plt.ylabel("n value")
plt.colorbar()
fig.savefig("figs/Projection_heat.png")
plt.clf()
plt.close(fig)

# plot populations by n
shells = get_shells(plot_data.shape[1])
n_max = len(shells)
font = {'family': 'normal', 'weight': 'bold', 'size': 22}

matplotlib.rc('font', **font)
for n_idx, idx in enumerate(shells):
    fig = plt.figure()
    plt.semilogy(range(data.shape[1]), data[-1, :], 'o-')

    plt.ylabel("Population")
    plt.xlabel("Bound State")
    plt.title(" Cycles: " + str(f["Parameters"]["cycles_on"][0] +
                                f["Parameters"]["cycles_off"][0]))
    plt.xlim([min(range(plot_data.shape[1])), max(range(plot_data.shape[1]))])
    plt.xticks(range(plot_data.shape[1]), state_labels, rotation='vertical')
    plt.ylim([1e-9, 1e-3])
    if n_idx == n_max - 1:
        plt.xlim([shells[n_idx], plot_data.shape[1] - 1])
    else:
        plt.xlim([shells[n_idx], shells[n_idx + 1] - 1])
    plt.grid()
    plt.tight_layout()
    fig.savefig("figs/Projection_at_end_n_" + str(n_idx + 1) + ".png")
    plt.clf()
    plt.close(fig)

for n_idx, idx in enumerate(shells):
    fig = plt.figure()
    plt.semilogy(range(data.shape[1]), data[-1, :], 'o-')

    plt.ylabel("Population")
    plt.xlabel("Bound State")
    if np.abs(f["Parameters"]["field_max"][0] - 93.7) < 1:
        plt.title("11 Photons")
    elif np.abs(f["Parameters"]["field_max"][0] - 114.39) < 1:
        plt.title("12 Photons")
    elif np.abs(f["Parameters"]["field_max"][0] - 131.79) < 1:
        plt.title("13 Photons")
    else:
        plt.title("14 Photons")
    plt.xlim([min(range(plot_data.shape[1])), max(range(plot_data.shape[1]))])
    plt.xticks(range(plot_data.shape[1]), state_labels, rotation='vertical')
    if n_idx < 10:
        plt.ylim([1e-9, 1e-3])
    else:
        plt.ylim([1e-14, 1e-3])
    if n_idx == n_max - 1:
        plt.xlim([shells[n_idx], plot_data.shape[1] - 1])
    else:
        plt.xlim([shells[n_idx], shells[n_idx + 1] - 1])
    plt.grid()
    plt.tight_layout()
    fig.savefig("figs/Projection_at_end_n_photons_" + str(n_idx + 1) + ".png")
    plt.clf()
    plt.close(fig)
