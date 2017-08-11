import numpy as np
import h5py
target_name = "H"
# read data
target = h5py.File(target_name + ".h5", "r")
f = h5py.File("TDSE.h5", "r")
psi_value = target["psi"]
# psi_value = f["Wavefunction"]["psi"]
energy = target["Energy"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx
print shape
x = f["Wavefunction"]["x_value_0"][:]
y = f["Wavefunction"]["x_value_1"][:]

max_val = 0
# calculate color bounds
for i, psi in enumerate(psi_value):
    psi = psi[:, 0] + 1j * psi[:, 1]
    max_val_tmp = np.max(np.absolute(psi))
    if (max_val_tmp > max_val):
        max_val = max_val_tmp


def state_single_name(state_number, shells):
    # find the n value for this state
    n_value = 0
    for n, shell in enumerate(shells):
        if state_number > shell:
            n_value = n + 1

    # calculate quantum number l
    l_value = state_number - shells[n_value - 1]

    # create label
    if l_value == 1:
        ret_val = str(n_value) + "s"
    elif l_value == 2:
        ret_val = str(n_value) + "p"
    elif l_value == 3:
        ret_val = str(n_value) + "d"
    elif l_value == 4:
        ret_val = str(n_value) + "f"
    elif l_value > 24:  # anything greater that z is just a number
        ret_val = str(n_value) + ",l=" + str(l_value - 1)
    else:  # any
        ret_val = str(n_value) + chr(ord('g') + l_value - 5)

    return ret_val


# return list of states up to state_number
def state_name(state_number):
    # get size of each shell
    shells = [0]
    while (state_number > shells[-1]):
        shells.append(shells[-1] + len(shells))

    # get list of names
    name_list = []
    for state in range(1, state_number + 1):
        name_list.append(state_single_name(state, shells))

    return name_list


name_list = state_name(len(psi_value))

if len(shape) == 3:
    from mayavi import mlab
    z = f["Wavefunction"]["x_value_2"][:]
    mlab.figure(
        bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), size=(1000, 1000))
    for i, psi in enumerate(psi_value):
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        # print np.log10(np.abs(psi))
        mlab.pipeline.iso_surface(
            mlab.pipeline.scalar_field(np.log10(np.abs(psi))),
            vmin=-05,
            opacity=0.3,
            colormap="viridis",
            contours=[-3, -2, -1])
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(np.log10(np.abs(psi))),vmin=-15,
        #    plane_orientation='z_axes',
        #    transparent=True,colormap="viridis",
        #    slice_index=shape[2]/2)
        mlab.colorbar(nb_labels=4, orientation="vertical")
        # mlab.show()
        mlab.savefig("figs/target_" + str(i).zfill(8) + ".png")
        mlab.clf()

elif len(shape) == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    fig = plt.figure()
    for i, psi in enumerate(psi_value):
        print "plotting", i
        # set up initial figure with color bar
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            x_min_idx = 0
        else:
            x_min_idx = lower_idx[0]
        x_max_idx = upper_idx[0]
        y_min_idx = lower_idx[1]
        y_max_idx = upper_idx[1]
        x_max_idx = -1
        y_min_idx = 0
        y_max_idx = -1
        psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            psi = np.absolute(
                np.multiply(
                    np.conjugate(psi),
                    np.multiply(x[x_min_idx:x_max_idx], psi.transpose())
                    .transpose()))
            plt.imshow(
                psi,
                cmap='viridis',
                origin='lower',
                extent=[
                    y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                ],
                norm=LogNorm(vmin=1e-15, vmax=np.max(psi)))
        else:
            plt.imshow(
                np.absolute(psi),
                cmap='viridis',
                origin='lower',
                extent=[
                    y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                ],
                norm=LogNorm(vmin=1e-12, vmax=max_val))
        # color bar doesn't change during the video so only set it here
        plt.colorbar()
        plt.xlabel("X-axis (a.u.)")
        plt.ylabel("Y-axis  (a.u.)")
        plt.title(name_list[i] + " - Energy: " + str(energy[i]))
        fig.savefig("figs/" + target_name + "_log_state_" + str(i).zfill(3) +
                    ".jpg")
        plt.clf()
