import matplotlib.pyplot as plt
import numpy as np

omega = 10.0
time = np.arange(0, 2.5 * np.pi, 0.01)
envelope = np.zeros(time.shape)
envelope[time >
         np.pi / 2.0] = np.sin(time[time > np.pi / 2.0] - np.pi / 2.0)**2
envelope[time > np.pi] = 1.0
envelope[time >
         3.0 * np.pi / 2.0] = np.sin(time[time > 3.0 * np.pi / 2.0] - np.pi)**2
envelope[time > 2.0 * np.pi] = 0.0
plt.plot(
    (time * omega / (2 * np.pi))[(time < np.pi / 2.0)],
    (envelope * np.sin(omega * time))[(time < np.pi / 2.0)],
    'C0-',
    label="cycles_delay")
plt.plot((time * omega / (2 * np.pi))[(time < np.pi / 2.0)],
         envelope[(time < np.pi / 2.0)], 'C0--')
plt.plot((time * omega / (2 * np.pi))[(time < np.pi / 2.0)],
         -envelope[(time < np.pi / 2.0)], 'C0--')

plt.plot(
    (time * omega / (2 * np.pi))[(time >= np.pi / 2.0)
                                 & (time < np.pi + 0.01)],
    (envelope * np.sin(omega * time))[(time >= np.pi / 2.0)
                                      & (time < np.pi + 0.01)],
    'C1-',
    label="cycles_on")
plt.plot((time * omega / (2 * np.pi))[(time >= np.pi / 2.0)
                                      & (time < np.pi + 0.01)],
         envelope[(time >= np.pi / 2.0) & (time < np.pi + 0.01)], 'C1--')
plt.plot((time * omega / (2 * np.pi))[(time >= np.pi / 2.0)
                                      & (time < np.pi + 0.01)],
         -envelope[(time >= np.pi / 2.0) & (time < np.pi + 0.01)], 'C1--')

plt.plot(
    (time * omega / (2 * np.pi))[(time >= np.pi)
                                 & (time < 3.0 * np.pi / 2.0 + 0.01)],
    (envelope * np.sin(omega * time))[(time >= np.pi)
                                      & (time < 3.0 * np.pi / 2.0 + 0.01)],
    'C2-',
    label="cycles_plateau")
plt.plot((time * omega / (2 * np.pi))[(time >= np.pi)
                                      & (time < 3.0 * np.pi / 2.0 + 0.01)],
         envelope[(time >= np.pi) & (time < 3.0 * np.pi / 2.0 + 0.01)], 'C2--')
plt.plot((time * omega / (2 * np.pi))[(time >= np.pi)
                                      & (time < 3.0 * np.pi / 2.0 + 0.01)],
         -envelope[(time >= np.pi) & (time < 3.0 * np.pi / 2.0 + 0.01)],
         'C2--')

plt.plot(
    (time * omega / (2 * np.pi))[(time >= 3.0 * np.pi / 2.0)
                                 & (time < 2.0 * np.pi + 0.01)],
    (envelope * np.sin(omega * time))[(time >= 3.0 * np.pi / 2.0)
                                      & (time < 2.0 * np.pi + 0.01)],
    'C3-',
    label="cycles_off")
plt.plot((time * omega / (2 * np.pi))[(time >= 3.0 * np.pi / 2.0)
                                      & (time < 2.0 * np.pi + 0.01)],
         envelope[(time >= 3.0 * np.pi / 2.0)
                  & (time < 2.0 * np.pi + 0.01)], 'C3--')
plt.plot((time * omega / (2 * np.pi))[(time >= 3.0 * np.pi / 2.0)
                                      & (time < 2.0 * np.pi + 0.01)],
         -envelope[(time >= 3.0 * np.pi / 2.0) & (time < 2.0 * np.pi + 0.01)],
         'C3--')
plt.xlabel("Optical Cycles")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("Pulse.png")