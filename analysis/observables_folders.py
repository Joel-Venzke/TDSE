import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

folders = []
plot_lines = {}

for dx in ["0.15", "0.2", "0.3", "0.4", "0.5", "0.6"]:
    for order in ["2"]:
        key = dx + "/" + order + "/"
        folders.append(key)

        line_style = ""
        if dx == "0.1":
            line_style += "r"
        if dx == "0.15":
            line_style += "b"
        if dx == "0.2":
            line_style += "g"
        if dx == "0.3":
            line_style += "m"
        if dx == "0.4":
            line_style += "c"
        if dx == "0.5":
            line_style += "k"
        if dx == "0.6":
            line_style += "y"

        if order == "2":
            line_style += "-"
        if order == "4":
            line_style += "-"
        if order == "6":
            line_style += "-"

        plot_lines[key] = line_style

for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
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
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    plt.plot(time, observables["norm"][1:], plot_lines[fold], label=fold)
plt.xlabel("Time (a.u.)")
plt.ylabel("Norm")
plt.title("Norm")
plt.tight_layout()
plt.legend()
fig.savefig("figs/Norm.png")
plt.clf()

# Ionization
print "Plotting Ionization"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    plt.plot(time, 1.0 - observables["norm"][1:], plot_lines[fold], label=fold)
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization")
plt.title("Ionization")
plt.tight_layout()
plt.legend()
fig.savefig("figs/Ionization.png")
plt.clf()

# Ionization
print "Plotting Ionization Rate"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    plt.plot(
        time,
        np.gradient(1.0 - observables["norm"][1:]),
        plot_lines[fold],
        label=fold)
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization Rate")
plt.title("Ionization Rate")
plt.tight_layout()
plt.legend()
fig.savefig("figs/Ionization_rate.png")
plt.clf()

# Gobbler
print "Plotting ECS Population"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    if np.max(observables["gobbler"][1:]) > 1e-19:
        plt.semilogy(
            time, observables["gobbler"][1:], plot_lines[fold], label=fold)
plt.xlabel("Time (a.u.)")
plt.ylabel("ECS Population")
plt.title("ECS Population")
plt.tight_layout()
plt.legend()
fig.savefig("figs/ECS_population.png")
plt.clf()

# Dipole
print "Plotting Dipole"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                     and f["Parameters"]["coordinate_system_idx"][0] == 1)):
                plt.plot(
                    time,
                    observables["position_expectation_"
                                + str(elec_idx) + "_" + str(dim_idx)][1:],
                    plot_lines[fold],
                    label=fold + "Electron " + str(elec_idx) + " Dim " +
                    str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole (a.u.)")
plt.title("Dipole Moment")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole.png")
plt.clf()

# Dipole acceleration
print "Plotting Dipole Acceleration"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                     and f["Parameters"]["coordinate_system_idx"][0] == 1)):
                plt.plot(
                    time,
                    observables["dipole_acceleration_"
                                + str(elec_idx) + "_" + str(dim_idx)][1:],
                    plot_lines[fold],
                    label=fold + "Electron " + str(elec_idx) + " Dim " +
                    str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Dipole Acceleration")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole_acceleration.png")
plt.clf()

# HHG Spectrum
print "Plotting HHG Spectrum"
fig = plt.figure()
count = 0
harm_value = 0
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    energy = f["Parameters"]["energy"][0]
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                     and f["Parameters"]["coordinate_system_idx"][0] == 1)):
                data = observables[
                    "dipole_acceleration_"
                    + str(elec_idx) + "_" + str(dim_idx)][1:len(
                        pulses["field_" + str(dim_idx)]
                        [checkpoint_frequency::checkpoint_frequency]) + 1]
                data = data * np.blackman(data.shape[0])
                padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
                paddT = np.max(time) * padd2 / data.shape[0]
                dH = 2 * np.pi / paddT / energy
                if np.max(data) > 1e-19:
                    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data,
                                (int(np.floor((padd2 - data.shape[0]) / 2)),
                                 int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
                    if count == 0:
                        count = 1
                        harm_value = data[np.argmin(
                            np.abs(np.arange(data.shape[0]) * dH - 11))]
                    else:
                        data *= harm_value / data[np.argmin(
                            np.abs(np.arange(data.shape[0]) * dH - 11))]
                    plt.semilogy(
                        np.arange(data.shape[0]) * dH,
                        data,
                        plot_lines[fold],
                        label=fold + "Electron " + str(elec_idx) + " Dim " +
                        str(dim_idx))
plt.ylabel("HHG Spectrum (a.u.)")
plt.title("HHG Spectrum")
plt.legend()
x_min = 0
x_max = 30
plt.xticks(np.arange(x_min, x_max + 1, 1.0))
plt.xlim([x_min, x_max])
plt.ylim([1e-3, 1e3])
plt.grid(True, which='both')
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum.png")
plt.clf()

# Linearity
print "Plotting Linearity"
fig = plt.figure()
for fold in folders:
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                     and f["Parameters"]["coordinate_system_idx"][0] == 1)):
                plt.plot(
                    observables["dipole_acceleration_"
                                + str(elec_idx) + "_" + str(dim_idx)]
                    [1:len(pulses["field_" + str(dim_idx)]
                           [checkpoint_frequency::checkpoint_frequency]) + 1],
                    pulses["field_" + str(dim_idx)][checkpoint_frequency::
                                                    checkpoint_frequency],
                    plot_lines[fold],
                    label=fold + "Electron " + str(elec_idx) + " Dim " +
                    str(dim_idx))
plt.xlabel("Field in (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Linearity")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Linearity.png")
plt.clf()
