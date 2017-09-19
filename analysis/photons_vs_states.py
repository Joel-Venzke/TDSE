import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import json

c = 1 / 7.2973525664e-3


def get_shells(state_number):
    shells = [0]
    while (state_number > shells[-1]):
        shells.append(shells[-1] + len(shells))
    return shells


with open('input.json', 'r') as data_file:
    data = json.load(data_file)
target_name = data["target"]["name"]
target = h5py.File(target_name + ".h5", "r")
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
energies = target["Energy"][:]
field_max = f["Parameters"]["field_max"][0]
u_p = field_max * field_max / (c * c * 4.0)

#H
energies = [
    -0.518536, -0.127738, -0.125544, -0.0564105, -0.0557645, -0.0556327,
    -0.03162, -0.031348, -0.0312945, -0.0312691, -0.0201923, -0.0200532,
    -0.0200265, -0.0200132, -0.0200069, -0.0140013, -0.0139208, -0.0139056,
    -0.0138979, -0.0138942, -0.013892, -0.0102754, -0.0102247, -0.0102153,
    -0.0102103, -0.010208, -0.0102066, -0.0102057, -0.00786051, -0.00782659,
    -0.0078203, -0.00781699, -0.00781544, -0.00781449, -0.00781385,
    -0.00781339, -0.00620669, -0.00618288, -0.00617849, -0.00617616,
    -0.00617507, -0.0061744, -0.00617395, -0.00617363, -0.00617338,
    -0.00502476, -0.0050074, -0.00500422, -0.00500252, -0.00500172,
    -0.00500123, -0.00500091, -0.00500067, -0.00500049, -0.00500035,
    -0.00415088, -0.00413784, -0.00413547, -0.00413418, -0.00413358,
    -0.00413322, -0.00413297, -0.00413279, -0.00413266, -0.00413255,
    -0.00413246, -0.00348662, -0.00347658, -0.00347475, -0.00347376,
    -0.0034733, -0.00347302, -0.00347283, -0.00347269, -0.00347259, -0.0034725,
    -0.00347244, -0.00347238, -0.00296993, -0.00296203, -0.0029606,
    -0.00295982, -0.00295945, -0.00295923, -0.00295908, -0.00295897,
    -0.00295889, -0.00295883, -0.00295877, -0.00295873, -0.0029587,
    -0.00256012, -0.00255379, -0.00255265, -0.00255203, -0.00255174,
    -0.00255156, -0.00255144, -0.00255135, -0.00255129, -0.00255123,
    -0.00255119, -0.00255116, -0.00255113, -0.00255111, -0.00222963,
    -0.00222449, -0.00222356
]
#He
# energies = [
#     -0.947675, -0.160426, -0.127575, -0.0652075, -0.0564096, -0.055578,
#     -0.0351759, -0.0316225, -0.031263, -0.0312522, -0.0219692, -0.0201935,
#     -0.0200076, -0.0200015, -0.0200008, -0.0150136, -0.0140017, -0.0138936,
#     -0.0138899, -0.0138895, -0.0138892, -0.0109059, -0.0102754, -0.0102072,
#     -0.0102048, -0.0102045, -0.0102044, -0.0102043, -0.00827949, -0.00786044,
#     -0.00781464, -0.00781302, -0.00781283, -0.00781272, -0.00781265,
#     -0.0078126, -0.00649913, -0.00620657, -0.00617438, -0.00617322,
#     -0.00617309, -0.00617301, -0.00617296, -0.00617293, -0.0061729,
#     -0.00523689, -0.00502462, -0.00500114, -0.00500029, -0.00500019,
#     -0.00500014, -0.0050001, -0.00500007, -0.00500005, -0.00500004,
#     -0.00430962, -0.00415075, -0.0041331, -0.00413246, -0.00413238,
#     -0.00413234, -0.00413231, -0.00413229, -0.00413228, -0.00413227,
#     -0.00413226, -0.00360848, -0.0034865, -0.0034729, -0.0034724, -0.00347234,
#     -0.00347231, -0.00347229, -0.00347227, -0.00347226, -0.00347225,
#     -0.00347225, -0.00347224, -0.0030655, -0.00296981, -0.00295911,
#     -0.00295872, -0.00295868, -0.00295865, -0.00295864, -0.00295862,
#     -0.00295861, -0.00295861, -0.0029586, -0.0029586, -0.00295859, -0.00263643,
#     -0.00255999, -0.00255135, -0.0025511, -0.00255106, -0.00255105,
#     -0.00255104, -0.00255104, -0.00255104, -0.00255103, -0.00255103,
#     -0.00255103, -0.00255103, -0.00255102, -0.00229006, -0.00222851
# ]

print "Plotting Spectrum"
fig = plt.figure(figsize=(24, 9), dpi=80)
font = {'size': 18}
matplotlib.rc('font', **font)
data = pulses["field_1"][:]
if np.max(data) > 1e-10:
    data_fft = np.absolute(
        np.fft.fft(
            np.lib.pad(
                data, (10 * data.shape[0], 10 * data.shape[0]),
                'constant',
                constant_values=(0.0, 0.0))))
    data_fft /= data_fft.max()
    # 2*pi/(dt*N)
    spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
        data_fft.shape[0] * (p_time[1] - p_time[0]))
    for n in range(1,
                   int(
                       abs((energies[0] - u_p) / spec_time[np.argmax(
                           data_fft[:data_fft.shape[0] / 2])])) + 2):
        plt.text(spec_time[data_fft[:data_fft.shape[0] / 2].argmax()] * n +
                 energies[0], 0.1, str(n))
        plt.semilogy(
            spec_time * n + energies[0], data_fft, label=str(n) + " photon")

shells = get_shells(19)
for i, energy in enumerate(energies):
    if i + 1 in shells:
        plt.text(energy, 1.5, "n=" + str(shells.index(i + 1)), rotation=90)
    plt.axvline(x=energy, color='k')
for i, energy in enumerate(energies):
    if i + 1 in shells:
        plt.text(
            energy + u_p,
            1.5,
            "n'=" + str(shells.index(i + 1)),
            rotation=90,
            color='b')
    plt.axvline(x=energy + u_p, color='b')
plt.axvline(x=0.0, color='r', linewidth=5)
plt.text(0.0, 1.5, "$I_p$", rotation=90, color='r')
plt.ylabel("Field Spectrum (arb)")
plt.xlabel("Energy $\omega$=" + str(f["Parameters"]["energy"][0]) + " (a.u.)")
plt.xlim([energies[0], energies[-1] + u_p])
plt.ylim([1e-3, 1.0])
# plt.legend()
fig.savefig("figs/Photons.png")
