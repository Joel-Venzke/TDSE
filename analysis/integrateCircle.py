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
kxc = kx[lower_idx[0]:upper_idx[0]]

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
    kyc = ky[lower_idx[1]:upper_idx[1]]

dkx = kxc[1] - kxc[0]
print dkx
dky = kyc[1] - kyc[0]

fig = plt.figure()
print 'loading data'
data = np.loadtxt('fft.txt', delimiter=',')

i_vector = np.unravel_index(np.argmax(data),
                    (data.shape[0], data.shape[1]))

#Get max momentum from FT, fix r to max
r_fix = np.sqrt(kyc[i_vector[0]]**2 +
                kxc[i_vector[1]]**2)
# r_fix = 0.763
print r_fix
dtheta = 0.02
# dr = 0.00001
# r = np.arange(0.0, 2.0, dr)
theta = np.arange(-np.pi, np.pi, dtheta)
theta_data = np.zeros(theta.shape[0])

# tol = np.sqrt(dky**2 + dkx**2)*10.0
# print tol
tol = 0.02
r_current = 0
theta_current = 0

#for every point, get radius and angle, interpolate
z = np.zeros(theta_data.shape[0])
for j, val in enumerate(kxc):
    for k, valy in enumerate(kyc):
        r_current = np.sqrt(val**2 + valy**2)
        theta_current = np.arctan2(valy, val)
        if(np.abs(r_current - r_fix) < tol):
           theta_data[np.argmin(np.abs(theta-theta_current))] += data[j][k]
           z[np.argmin(np.abs(theta-theta_current))] += 1

theta_data = theta_data / z
np.savetxt('theta_dataD.txt', theta_data, delimiter=',')
plt.plot(theta * 180 / np.pi, theta_data)
plb.xlim([-180, 180])
plb.xlabel('angle (degrees)')
plb.ylabel('population (arb. u)')
fig.savefig('figs/popVsTheta' + '.png')
plt.clf()