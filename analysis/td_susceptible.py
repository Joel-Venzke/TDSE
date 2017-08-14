import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.signal import stft, chebwin
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
observables = f["Observables"]
o_time = observables["time"][:]
pulses = f["Pulse"]
p_time = pulses["time"][:]
energy = f["Parameters"]["energy"][0]
cycles = int(f["Parameters"]["cycles_on"][0] +
             f["Parameters"]["cycles_off"][0])
ts_per_cycle = 2 * np.pi / energy / (o_time[2] - o_time[1])
print cycles, ts_per_cycle, observables["position_expectation_0_1"][:].shape[0]

if f["Parameters"]["pulse_shape_idx"][0] == 1:
    idx_min = p_time.shape[0] / 2 - p_time.shape[0] / 6
    idx_max = p_time.shape[0] / 2 + p_time.shape[0] / 6
else:
    idx_min = 0
    idx_max = -1

max_data = []
min_data = []
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#ax2.plot(
#    o_time[1:] / (2 * np.pi / energy),
#    1.0 - observables["norm"][1:],
#    'r--',
#    label="Ionization")
for i in range(min(6, cycles)):
    if i != 0:
        window_size = int(ts_per_cycle * (i + 1))
        freq, t, dipole_fft = stft(
            observables["position_expectation_0_1"][:],
            fs=1.0 / (f["Parameters"]["delta_t"][0] * f["Parameters"][
                "write_frequency_observables"][0]),
            noverlap=int(window_size * 0.999),
            nperseg=window_size,
            window=chebwin(window_size, 80))
        freq, t, pulse_fft = stft(
            -1.0 * np.gradient(pulses["field_1"][::f["Parameters"][
                "write_frequency_observables"][0]], f["Parameters"]["delta_t"][
                    0] * f["Parameters"]["write_frequency_observables"][0]) *
            7.2973525664e-3,
            fs=1.0 / (f["Parameters"]["delta_t"][0] * f["Parameters"][
                "write_frequency_observables"][0]),
            noverlap=int(window_size * 0.999),
            nperseg=window_size,
            window=chebwin(window_size, 80))
        data = np.abs(dipole_fft[:, :pulses_fft.shape[1]] / pulse_fft)
        idx = dipole_fft.max(axis=1).argmax()
        if f["Parameters"]["pulse_shape_idx"][0] == 1:
            d_idx_min = data.shape[1] / 2 - data.shape[1] / 6
            d_idx_max = data.shape[1] / 2 + data.shape[1] / 6
        else:
            d_idx_min = 0
            d_idx_max = -1
        max_data.append(data[idx, d_idx_min:d_idx_max].max())
        min_data.append(data[idx, d_idx_min:d_idx_max].min())
        data = data * max_data[0] / max_data[i - 1]
        max_data[-1] = data[idx, d_idx_min:d_idx_max].max()
        min_data[-1] = data[idx, d_idx_min:d_idx_max].min()
        print i, window_size, t[1] - t[0], t[d_idx_min:d_idx_max][data[
            idx, d_idx_min:d_idx_max].argmax()], p_time[p_time.shape[0] /
                                                        2], max_data[-1]
        ax1.plot(
            t / (2 * np.pi / energy),
            np.abs(data[idx]),
            label=str((i + 1)) + " cycles")
        ax1.scatter(
            [t[d_idx_min:d_idx_max][data[idx, d_idx_min:d_idx_max].argmax()]] /
            (2 * np.pi / energy), [data[idx, d_idx_min:d_idx_max].max()])
plt.axvline(color='k', x=p_time[-1] / (2 * np.pi / energy) / 2.0)
ax1.legend(loc=2)
ax1.set_ylim([min_data[-1] - 0.05, max_data[-1] + 0.02])
fig.savefig("figs/td_susceptible.png")
