import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as plb


dtheta = 0.01855
theta = np.arange(-np.pi, np.pi, dtheta)

data1 = np.loadtxt('theta_dataD.txt', delimiter=',')
# data1 -= np.min(data1)
# data1 = data1 / np.max(data1)
# data1 = data1 / np.sum(data1)
#os.chdir('/Users/cgoldsmith/Desktop/projects/angular/tempData/omega_1_04au_4cycles_elliptical')
os.chdir('/scratch/summit/cogo4490/streakingTDSE/angular/twoPhoton/softcoreStates/alpha0_2/circular/')
data2 = np.loadtxt('theta_dataD.txt', delimiter=',')
central = np.loadtxt('centralCycle.txt')
t_central = central[0][:]
x_central = central[1][:]
y_central = central[2][:]
strength  = np.sqrt(x_central**2 + y_central**2)
angle     = np.arctan2(y_central, x_central)
sumData = data1 + data2
# data2 -= data2.min()
# data2 = data2 / np.max(data2)
# data2 = data2 / np.sum(data2)
fig = plt.figure()
#plt.plot(angle * (180 / np.pi), strength * sumData.max() / strength.max(), theta * (180 / np.pi), data1, '--', theta * (180 / np.pi), data2, theta * (180 / np.pi), sumData)
plt.plot(theta * (180 / np.pi), sumData)#, '--', angle * (180 / np.pi), strength * sumData.max() / (2*strength.max()))
plb.xlim([-180, 180])
plb.xlabel('angle (degrees)')
plb.ylabel('population (arb. u)')
fig.savefig("figs/summedDistribution.png")
plt.clf()
#plt.show()
