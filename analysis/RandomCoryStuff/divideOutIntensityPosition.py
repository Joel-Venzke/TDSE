#this script integrates over 2 pi for a given radius

import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as plb
from matplotlib.colors import LogNorm

print "start"

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]

gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx

x = f["Wavefunction"]["x_value_0"][:]
kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
# kxc = kx[lower_idx[0]:upper_idx[0]]

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
    # kyc = ky[lower_idx[1]:upper_idx[1]]

dkx = kx[1] - kx[0]
dky = ky[1] - ky[0]

fig = plt.figure()
print 'loading data'


# data = np.loadtxt('fft.txt', delimiter=',')
data = np.loadtxt('cutWave.txt')
data_adjusted = np.zeros(data.shape)
psi = np.zeros(data.shape)
psi = data

if f["Parameters"]["coordinate_system_idx"][0] == 1:
    x_min_idx = 0
else:
    x_min_idx = lower_idx[0]
x_max_idx = upper_idx[0]
x_min_idx = lower_idx[0]
y_min_idx = lower_idx[1]
y_max_idx = upper_idx[1]

xc = x[x_min_idx:x_max_idx]
yc = y[y_min_idx:y_max_idx]
kxc = xc * 2.0 * np.pi / (xc.shape[0] * (xc[1] - xc[0]) * (xc[1] - xc[0]))
kyc = yc * 2.0 * np.pi / (yc.shape[0] * (yc[1] - yc[0]) * (yc[1] - yc[0]))

saver = np.loadtxt('centralCycle.txt')

fieldTheta = np.arctan2(saver[:,2], saver[:, 1])
fieldIntensity = saver[:,1]**2 + saver[:,2]**2

# plt.plot(fieldTheta, fieldIntensity)
# plt.show()

tol = 0.02
r_current = 0
theta_current = 0

#for each point in data, divide 
#by field Intensity corresponding to correct angle
print "dividing out intensity...."
for j, val in enumerate(xc):
    for k, valy in enumerate(yc):
        theta_current = np.arctan2(valy, val)
        z = np.argmin(np.abs(fieldTheta - theta_current))
        data_adjusted[j][k] = \
        psi[j][k] / (saver[z][1]**2 + saver[z][2]**2)**2

psi = data_adjusted
# print "cutting again"
# r_critical = 30
# alpha = 25000000000000
# for j, val in enumerate(xc):
#     for k, valy in enumerate(yc):
#         r = np.sqrt(val**2 + valy**2)
#         if r <= r_critical:
#             data_adjusted[j][k] = data_adjusted[j][k] \
#             * np.exp(-alpha * (r - r_critical)**2)

i_vector = np.unravel_index(np.argmax(data_adjusted),
                    (data_adjusted.shape[0], data_adjusted.shape[1]))
print "max location is y = " + str(yc[i_vector[0]]) \
    + "a.u., " + "x = " + str(xc[i_vector[1]]) + "\n"
angle = np.arctan2(yc[i_vector[0]], xc[i_vector[1]])

print "angle of position max is " + str(angle * 180 / np.pi) +\
      " degrees."
# plt.plot(fieldTheta, fieldIntensity)
datapos = plt.imshow(
                    data_adjusted,
                    cmap='viridis',
                    origin='lower',
                    # vmin=10.0,vmax=20,
                    norm=LogNorm(vmin=1e-10),
                    extent=[yc.min(), yc.max(),
                            xc.min(), xc.max()])

plt.xlabel("$y$ (a.u.)")
plt.ylabel("$x$  (a.u.)")
# plb.xlim([-10, 10])
# plb.ylim([-10, 10])
plt.colorbar()
# fig.savefig("figs/adjusted_cut" + "_full.png")
# plb.xlim([-2, 2])
# plb.ylim([-2, 2])
fig.savefig("figs/adjusted_cut" + ".png")
plt.clf()
ft_full = None

print "FT adjusted cut Wavefunction"
ft_full = np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2
print ft_full
# np.savetxt('fftAdjusted.txt', ft_full, delimiter=',')
print "Calculating rotation angle..."
i_vector = np.unravel_index(np.argmax(ft_full),
    (ft_full.shape[0], ft_full.shape[1]))
print "max momentum is at k_y = " + str(kyc[i_vector[0]]) \
    + "a.u., " + "k_x = " + str(kxc[i_vector[1]]) + "\n"
angle = np.arctan2(kyc[i_vector[0]], kxc[i_vector[1]])

print " angle of momentum max is " + str(angle * 180 / np.pi) +\
      " degrees. Now plotting full spectrum"

dataft = plt.imshow(
# np.abs(np.fft.fftshift(np.fft.fft2(data_adjusted)))
np.sqrt(ft_full),
cmap='viridis',
origin='lower',
norm=LogNorm(vmin=1e-3),
# vmin=0.125,vmax=0.175,
extent=[kyc.min(), kyc.max(),
        kxc.min(), kxc.max()])

plt.xlabel("$k_y$ (a.u.)")
plt.ylabel("$k_x$  (a.u.)")
plb.xlim([-10, 10])
plb.ylim([-10, 10])
plt.colorbar()
fig.savefig("figs/adjusted_fftcutPos" + "_full.png")
plb.xlim([-2, 2])
plb.ylim([-2, 2])
fig.savefig("figs/adjusted_fftcutPos" + ".png")
plt.clf()
