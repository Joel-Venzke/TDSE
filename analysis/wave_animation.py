import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"]
psi_time  = f["Wavefunction"]["psi_time"][:]
x         = f["Wavefunction"]["x_value_0"][:]

# calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

# calculate color bounds
max_val = 0
for psi in psi_value:
    tmp = np.max(np.absolute(psi))
    if (tmp>max_val):
        max_val = tmp
print "max plot: ", max_val

# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value[0].shape[0])
print "dim size:", p_sqrt, "Should be integer"
p_sqrt = int(p_sqrt)

psi = psi_value[0]
psi.shape = (p_sqrt,p_sqrt)
# set up initial figure with color bar
fig = plt.figure()
plt.imshow(np.absolute(psi), cmap='viridis', origin='lower',
    extent=[x[0],x[-1],x[0],x[-1]],
    norm=LogNorm(vmin=1e-10, vmax=max_val))
# color bar doesn't change during the video so only set it here
plt.colorbar()
plt.xlabel("Electron 2 a.u.")
plt.ylabel("Electron 1 a.u.")
plt.title("Wave Function")

ims = []
i=0
# for psi, time in zip(psi_value[:4],psi_time[:4]):
for psi, time in zip(psi_value,psi_time):
    print i, time
    if i!=0:
        psi.shape = (p_sqrt,p_sqrt)
        # add frames
        ims.append((plt.imshow(np.absolute(psi), cmap='viridis', origin='lower',
                                extent=[x[0],x[-1],x[0],x[-1]],
                                norm=LogNorm(vmin=1e-10, vmax=max_val)),
            plt.text(time_x, time_y, "Time: "+str(time)+" a.u.",
                color='black'),))
    i+=1

print "Making animation"
# animate
im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000,
    blit=False)

print "Saving...this can take a while"
#save
im_ani.save('figs/Wavefunction.mp4', bitrate=-1, codec="libx264",
    extra_args=['-pix_fmt', 'yuv420p'])
