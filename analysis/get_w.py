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


w_0 = 0.057
w_min = 0.02
w_max = w_0
dt = 0.005

while (np.abs(w_max - w_min) > 1e-4):
    print(w_max + w_min) / 2.0, " Error: ", np.abs(w_max - w_min)
    time, pulse = get_pulse(1, 1, 1e14, (w_max + w_min) / 2.0, dt)

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

print "w_A", (w_max + w_min) / 2.0