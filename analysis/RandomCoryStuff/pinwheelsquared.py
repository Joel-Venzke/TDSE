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

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))

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

dkx = kxc[1] - kxc[0]
print dkx
dky = kyc[1] - kyc[0]

fig = plt.figure()
print 'loading data'

data_adjusted = np.zeros([4798, 4798])
saver = np.loadtxt('centralCycle.txt')

fieldTheta = np.arctan2(saver[:,2], saver[:, 1])
fieldIntensity = saver[:,1]**2 + saver[:,2]**2

tol = 0.02
r_current = 0
theta_current = 0
z = 0

#for each point in data, divide 
#by field Intensity corresponding to correct angle
print "write intensity squared"
for j, val in enumerate(kxc):
    for k, valy in enumerate(kyc):
        theta_current = np.arctan2(valy, val)
        z = np.argmin(np.abs(fieldTheta - theta_current))
        data_adjusted[j][k] = (saver[z][1]**2 + saver[z][2]**2)**2
           
i_vector = np.unravel_index(np.argmax(data_adjusted),
                    (data_adjusted.shape[0], data_adjusted.shape[1]))
print "max location is ky = " + str(kyc[i_vector[0]]) \
    + "a.u., " + "kx = " + str(kxc[i_vector[1]]) + "\n"

print "value of " + str(data_adjusted[i_vector[0]][i_vector[1]])
angle = np.arctan2(kyc[i_vector[0]], kxc[i_vector[1]])

print "angle of momentum max is " + str(angle * 180 / np.pi) +\
      " degrees."
# plt.plot(fieldTheta, fieldIntensity)
dataft = plt.imshow(
                    data_adjusted,
                    cmap='viridis',
                    origin='lower',
                    vmin=0.0,vmax=0.61 ,
                    # norm=LogNorm(vmin=1e-5),
                    extent=[kyc.min(), kyc.max(),
                            kxc.min(), kxc.max()])

plt.xlabel("$k_y$ (a.u.)")
plt.ylabel("$k_x$  (a.u.)")
plt.colorbar()

plb.xlim([-2, 2])
plb.ylim([-2, 2])
fig.savefig("figs/adjusted_field_squared" + ".png")
plt.clf()

