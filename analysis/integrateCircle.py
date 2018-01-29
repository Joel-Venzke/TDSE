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
# data = np.loadtxt('fftcomplex.txt').view(complex)
datanorm = np.loadtxt('fft.txt', delimiter=',')
i_vector = np.unravel_index(np.argmax(datanorm),
                    (datanorm.shape[0], datanorm.shape[1]))

#Get max momentum from FT, fix r to max
r_fix = np.sqrt(kyc[i_vector[0]]**2 +
                kxc[i_vector[1]]**2)
# r_fix = 0.763
print r_fix
dtheta = 0.01855
# dr = 0.00001
# r = np.arange(0.0, 2.0, dr)
theta = np.arange(-np.pi, np.pi, dtheta)
# theta_data = np.zeros(theta.shape[0], 'complex')
theta_data = np.zeros(theta.shape[0])

# tol = np.sqrt(dky**2 + dkx**2)*10.0
# print tol
tol = 0.0181
r_current = 0
theta_current = 0

#for every point, get radius and angle, interpolate
z = np.zeros(theta_data.shape[0])
for j, val in enumerate(kxc):
    for k, valy in enumerate(kyc):
        r_current = np.sqrt(val**2 + valy**2)
        theta_current = np.arctan2(valy, val)
        if(np.abs(r_current - r_fix) < tol):
           theta_data[np.argmin(np.abs(theta-theta_current))] += datanorm[j][k]
           z[np.argmin(np.abs(theta-theta_current))] += 1

central = np.loadtxt('centralCycle.txt')
t_central = central[0][:]
x_central = central[1][:]
y_central = central[2][:]
strength  = np.sqrt(x_central**2 + y_central**2)
angle     = np.arctan2(y_central, x_central)
theta_data = theta_data / z
norm_theta_data = np.abs(theta_data)**2

ft = np.fft.fft(theta_data)
ftnorm = np.abs(ft)**2
# np.savetxt('theta_dataComplex.txt', theta_data.view(float))
print ft[:5], ftnorm[:5]
# plt.plot(theta * 180 / np.pi, norm_theta_data, '--', 
#           angle * 180 / np.pi, strength 
#           * norm_theta_data.max() / strength.max())
plt.plot(theta * 180 / np.pi, theta_data, '--', 
          angle * 180 / np.pi, strength 
          * theta_data.max() / strength.max())

# plt.plot(ft)
plb.xlim([-180, 180])
#plb.ylim([11, 15])
#plb.ylim([2, 3])
plb.xlabel('angle (degrees)')
plb.ylabel('population (arb. u)')
# plt.show()
fig.savefig('figs/popVsTheta' + '.png')
plt.clf()
