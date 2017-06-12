import numpy as np
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"]
psi_time  = f["Wavefunction"]["time"][:]
x         = f["Wavefunction"]["x_value_0"][:]
y         = f["Wavefunction"]["x_value_1"][:]
shape     = f["Wavefunction"]["num_x"][:]
gobbler   = f["Parameters"]["gobbler"][0]
upper_idx = (shape*gobbler-1).astype(int)
lower_idx = shape-upper_idx
print(shape, lower_idx, upper_idx)
# calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

max_val = 0
# calculate color bounds
for i,psi in enumerate(psi_value):
    if i>0 and i<2: # the zeroth wave function is the guess and not relevant
        psi = psi[:,0] + 1j*psi[:,1]
        max_val_tmp   = np.max(np.absolute(psi))
        if (max_val_tmp > max_val):
            max_val = max_val_tmp
max_val = 1


if len(shape) == 3: 
    from mayavi import mlab
    # mlab.options.offscreen = True
    z         = f["Wavefunction"]["x_value_2"][:]
    mlab.figure(bgcolor=(1.0,1.0,1.0),fgcolor=(0.0,0.0,0.0),size=(1000, 1000))
    for i, psi in enumerate(psi_value):
        if i>0: # the zeroth wave function is the guess and not relevant
            print("plotting", i)
            psi = psi[:,0]+1j*psi[:,1]
            psi.shape = tuple(shape)
            psi = psi[lower_idx[0]:upper_idx[0],lower_idx[1]:upper_idx[1],lower_idx[2]:upper_idx[2]]
            psi = np.log10(np.abs(psi))
            # mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(psi),
            #     vmin=-10,
            #     vmax=0.0,
            #     opacity=0.3,
            #     colormap="viridis",
            #     contours=11)
            # mlab.colorbar(nb_labels=11,orientation="vertical")
            # mlab.orientation_axes()
            # mlab.view(azimuth=0.0,distance='auto',elevation=90.0)
            # mlab.savefig("figs/Wave_iso_x_"+str(i).zfill(8)+".png")
            # mlab.view(azimuth=90.0,distance='auto',elevation=90.0)
            # mlab.savefig("figs/Wave_iso_y_"+str(i).zfill(8)+".png")
            # mlab.view(azimuth=0.0,distance='auto',elevation=0.0)
            # mlab.savefig("figs/Wave_iso_z_"+str(i).zfill(8)+".png")
            # mlab.view(azimuth=45.0,distance='auto',elevation=45.0)
            # mlab.savefig("figs/Wave_iso_"+str(i).zfill(8)+".png")

            mlab.clf()
            mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(psi),
                vmin=-10.0,
                vmax=0.0,
                opacity=0.3,
                colormap="viridis",
                contours=[1.0])
            mlab.pipeline.volume(mlab.pipeline.scalar_field(psi), 
               vmin=-10.0,
               vmax=0.0)
            mlab.colorbar(nb_labels=10,orientation="vertical")
            mlab.orientation_axes()
            mlab.view(azimuth=0.0,distance='auto',elevation=90.0)
            mlab.savefig("figs/Wave_density_10_cross_x_"+str(i).zfill(8)+".png")
            mlab.view(azimuth=90.0,distance='auto',elevation=90.0)
            mlab.savefig("figs/Wave_density_10_cross_y_"+str(i).zfill(8)+".png")
            mlab.view(azimuth=0.0,distance='auto',elevation=0.0)
            mlab.savefig("figs/Wave_density_10_cross_z_"+str(i).zfill(8)+".png")
            mlab.view(azimuth=45.0,distance='auto',elevation=45.0)
            mlab.savefig("figs/Wave_density_10_cross_"+str(i).zfill(8)+".png")
            mlab.clf()

elif len(shape) == 2:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # shape into a 3d array with time as the first axis
    fig = plt.figure()
    font = {'size'   : 18}
    matplotlib.rc('font', **font)
    for i, psi in enumerate(psi_value):
        if i>0: # the zeroth wave function is the guess and not relevant
            print("plotting", i)
            # set up initial figure with color bar
            psi = psi[:,0] + 1j*psi[:,1]
            psi.shape = tuple(shape)
            if f["Parameters"]["coordinate_system_idx"][0]==1:
                plt.imshow(np.absolute(np.multiply(np.conjugate(psi),np.multiply(x,psi.transpose()).transpose())), cmap='viridis', origin='lower',
                           extent=[y[0],y[-1],x[0],x[-1]],
                           norm=LogNorm(vmin=1e-18, vmax=np.max(np.absolute(np.vdot(psi,np.multiply(x,psi.transpose()).transpose())))))
                           #norm=LogNorm(vmin=1e-10, vmax=max_val))
            else:
                plt.imshow(np.absolute(psi), cmap='viridis', origin='lower',
                           extent=[y[0],y[-1],x[0],x[-1]],
                           norm=LogNorm(vmin=1e-10, vmax=max_val))
            plt.text(time_x, time_y, "Time: "+str(psi_time[i])+" a.u.",
                        color='white')
            # color bar doesn't change during the video so only set it here
            plt.colorbar()
            plt.xlabel("X-axis (a.u.)")
            plt.ylabel("Y-axis  (a.u.)")
            # plt.axis('off')
            fig.savefig("figs/Wave_"+str(i).zfill(8)+".png")
            plt.clf()
            plt.clf()
