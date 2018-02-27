#interpolation onto circular grid

import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
from scipy.interpolate import RectBivariateSpline, interp2d
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
dx = f["Parameters"]["delta_x_max"][0]

gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx

fig = plt.figure()
print 'loading data'

data = np.loadtxt('fft.txt', delimiter=',')
print "data shape is " + str(data.shape[0])\
 + ", " + str(data.shape[1])
data_adjusted = np.zeros(data.shape)

shape = data.shape[0]
x = np.arange(int(-0.5*shape)*dx, int(0.5*shape)*dx, dx)
print shape, x.shape[0]

upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx

kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
kxc = kx#[lower_idx:upper_idx]

y = np.arange(int(-0.5*shape)*dx, int(0.5*shape)*dx, dx)
print y.shape[0]
ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
kyc = ky#[lower_idx:upper_idx]

k = np.sqrt(kx**2 + ky**2)
dkx = kxc[1] - kxc[0]
dky = kyc[1] - kyc[0]
print k

i_vector = np.unravel_index(np.argmax(data),
                    (data.shape[0], data.shape[1]))
                #NOTE: for i_vector 0 is y, and 1 is x
print "max location is y = " + str(kyc[i_vector[1]]) \
            + "a.u., " + "x = " + str(kxc[i_vector[0]]) + "\n"
angle0 = np.arctan2(kyc[i_vector[1]], kxc[i_vector[0]])
rmax = np.sqrt(kyc[i_vector[1]]**2 + kxc[i_vector[0]]**2)
print "k_max is " + str(rmax)
saver = np.loadtxt('centralCycle.txt')
t_central = saver[0][:]
x_central = saver[1][:]
y_central = saver[2][:]
fieldTheta = np.arctan2(y_central, x_central)
strength = x_central**2 + y_central**2

dtheta = 0.0001
theta = np.arange(-np.pi, np.pi, dtheta)
dr = (2*k.max()) * dtheta / (2 * np.pi)
r = np.arange(-k.max(), k.max(), dr)

new_x = rmax * np.cos(theta)
new_y = rmax * np.sin(theta)

print new_x, new_y
f_mom = RectBivariateSpline(x, y, data, kx=3, ky=3)
# f_mom = interp2d(x, y, data)

newFFT = np.zeros(theta.shape[0])

for i, t in enumerate(theta):
    newFFT[i] = f_mom(new_x[i], new_y[i])

# i_vector = np.unravel_index(np.argmax(newFFT))
# print "max location is ky = " + str(new_y[i_vector[1]]) \
#     + "a.u., " + "kx = " + str(new_x[i_vector[0]]) + "\n"
thetai = np.argmax(newFFT)
print "max location is theta = " + str(180*theta[thetai]/np.pi)

dataft = plt.plot(theta, newFFT, fieldTheta, strength*newFFT.max()/strength.max(), '--')

# # color bar doesn't change during the video so only set it here
# if f["Parameters"]["coordinate_system_idx"][0] == 1:
#     plt.xlabel("$k_\\rho$ (a.u.)")
#     plt.ylabel("$k_z$ (a.u.)")
# else:
#     plt.xlabel("$k_x$ (a.u.)")
#     plt.ylabel("$k_y$  (a.u.)")
# plb.xlim([-10, 10])
# plb.ylim([-10, 10])
# plt.colorbar()
fig.savefig("figs/fft_cut_logInterp" + "_full.png")
# plb.xlim([-3, 3])
# plb.ylim([-3, 3])
# fig.savefig("figs/2d_fft_cut_logInterp" + ".png")
plt.clf()
