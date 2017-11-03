import numpy as np
import matplotlib.pyplot as plt

c = 1 / 7.2973525664e-3


def get_pulse(cycles_on, cycles_off, intensity, w, dt):
    time = np.arange(-int(
        ((cycles_on + cycles_off) * 2 * np.pi / w) / (2 * dt)),
                     int(((cycles_on + cycles_off) * 2 * np.pi / w) /
                         (2 * dt)) + 1) * dt
    c2 = np.sqrt(intensity / 3.51e16) * c / w * np.cos(np.pi * time / (
        2 * time[-1])) * np.cos(np.pi * time / (2 * time[-1]))
    return time, c2 * np.sin(w * time)


w_0 = 0.057 * 2
print w_0
w_min = w_0 / 2
w_max = w_0 * 2
dt = 0.045
plot_data = []
for cycles in range(1, 10):
    w_min = w_0 / 2
    w_max = w_0 * 2
    while (np.abs(w_max - w_min) > 1e-6):
        # print(w_max + w_min) / 2.0, " Error: ", np.abs(w_max - w_min)
        time, pulse = get_pulse(cycles / 2.0, cycles / 2.0, 5.31e13,
                                (w_max + w_min) / 2.0, dt)

        data = -1.0 * np.gradient(pulse, dt) * 7.2973525664e-3
        data_fft = np.absolute(
            np.fft.fft(
                np.lib.pad(
                    data, (10 * data.shape[0], 10 * data.shape[0]),
                    'constant',
                    constant_values=(0.0, 0.0))))
        spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
            data_fft.shape[0] * (dt))
        # plt.semilogy(spec_time, data_fft)
        w_new = spec_time[np.argmax(data_fft[:data_fft.shape[0] / 2])]
        if w_new > w_0:
            w_max = (w_max + w_min) / 2.0
        else:
            w_min = (w_max + w_min) / 2.0
    plot_data.append((w_max + w_min) / 2.0)
    print cycles, plot_data[-1], (w_0 - plot_data[-1]) / w_0
print plot_data

plt.plot(range(1, 5), plot_data, 'r-o', label="$\omega_A$")
plt.axhline(y=w_0, color='k', label="$\omega_E$")
plt.xlabel("Cycles")
plt.ylabel("Photon energy (au)")
plt.title("270nm")
plt.legend()
plt.savefig("270nm.png")
