import numpy as np
import h5py
import json


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


with open('input.json', 'r') as data_file:
    data = json.load(data_file)

target_name = data["target"]["name"]
# read data
target = h5py.File(target_name + ".h5", "r")
try:
    f = h5py.File("TDSE.h5", "r")
except:
    f = h5py.File("Observables.h5", "r")
psi_value = target["psi"]
energy = target["Energy"]
num_electrons = f["Parameters"]["num_electrons"][0]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
num_dims = len(shape)

if f["Parameters"]["coordinate_system_idx"][0]==4:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    import pylab as plb
    font = {'size': 22}
    matplotlib.rc('font', **font)
    for i, psi in enumerate(psi_value):
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi = np.abs(psi)
        r_vals = f["Wavefunction"]["x_value_2"][:]
        psi.shape = [psi.shape[0]//r_vals.shape[0],r_vals.shape[0]]
        psi = np.sum(psi, axis=0)
        plt.semilogy(r_vals,psi)
        plt.ylim([1e-10,1])
        plt.savefig("figs/state_"+str(i).zfill(5)+".png")
        plt.clf()


elif num_electrons == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    import pylab as plb
    font = {'size': 22}
    matplotlib.rc('font', **font)

    shape_2_elecron = np.concatenate([shape, shape])
    num_axis = num_dims * num_electrons
    for i, psi in enumerate(psi_value):
        psi_save = np.array(psi)
        print( "plotting", i)
        # set up initial figure with color bar
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi = (psi * np.conjugate(psi)).real
        psi.shape = tuple(shape_2_elecron)

        # plot projections onto each axis
        for electron_idx in np.arange(num_electrons):
            for dim_idx in np.arange(num_dims):
                plot_name = "figs/" + target_name + "_projection_on_dim_" + str(
                    dim_idx) + "_electron_" + str(
                        electron_idx) + "_state_" + str(i).zfill(8) + ".png"
                current_dim = electron_idx * num_dims + dim_idx
                axis_sum_list = []
                for this_dim in range(num_axis):
                    if this_dim != current_dim:
                        axis_sum_list.append(this_dim)
                axis_sum_list = tuple(axis_sum_list)
                data = np.sum(psi, axis=axis_sum_list)
                phase = np.angle(data)
                data = (data * np.conjugate(data)).real
                axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                fig = plt.figure()
                plt.plot(axis, data / data.max())
                if dim_idx == 0:
                    plt.axvline(-0.7, color='k')
                    plt.axvline(0.7, color='k')
                else:
                    plt.axvline(0.0, color='k')
                plt.xlabel("Dim:" + str(dim_idx) + "   Electron:" +
                           str(electron_idx))
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                plot_name = "figs/" + target_name + "_log_projection_on_dim_" + str(
                    dim_idx) + "_electron_" + str(
                        electron_idx) + "_state_" + str(i).zfill(8) + ".png"
                fig = plt.figure()
                plt.semilogy(axis, data / data.max())
                if dim_idx == 0:
                    plt.axvline(-0.7, color='k')
                    plt.axvline(0.7, color='k')
                else:
                    plt.axvline(0.0, color='k')
                plt.xlabel("Dim:" + str(dim_idx) + "   Electron:" +
                           str(electron_idx))
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)
        if num_dims == 2:
            dim_0 = f["Wavefunction"]["x_value_0"][:]
            dim_1 = f["Wavefunction"]["x_value_1"][:]
            # plot electron 0
            plot_name = "figs/" + target_name + "_2d_electron_0_state_" + str(
                i).zfill(8) + ".png"
            axis_sum_list = tuple([2, 3])
            fig = plt.figure()
            data = np.sum(psi, axis=axis_sum_list)
            axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]])
            plt.ylabel("Dim: 0")
            plt.xlabel("Dim: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            plot_name = "figs/" + target_name + "_log_2d_electron_0_state_" + str(
                i).zfill(8) + ".png"
            fig = plt.figure()
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]],
                norm=LogNorm(vmin=1e-15))
            plt.ylabel("Dim: 0")
            plt.xlabel("Dim: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            # plot electron 1
            plot_name = "figs/" + target_name + "_2d_electron_1_state_" + str(
                i).zfill(8) + ".png"
            axis_sum_list = tuple([0, 1])
            fig = plt.figure()
            data = np.sum(psi, axis=axis_sum_list)
            axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]])
            plt.ylabel("Dim: 0")
            plt.xlabel("Dim: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            plot_name = "figs/" + target_name + "_log_2d_electron_1_state_" + str(
                i).zfill(8) + ".png"
            fig = plt.figure()
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]],
                norm=LogNorm(vmin=1e-15))
            plt.ylabel("Dim: 0")
            plt.xlabel("Dim: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            # plot dim 0 for both electrons
            plot_name = "figs/" + target_name + "_2d_dim_0_state_" + str(
                i).zfill(8) + ".png"
            axis_sum_list = tuple([1, 3])
            fig = plt.figure()
            data = np.sum(psi, axis=axis_sum_list)
            axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]])
            plt.ylabel("Dim: 0    Electron: 0")
            plt.xlabel("Dim: 0    Electron: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            plot_name = "figs/" + target_name + "_log_2d_dim_0_state_" + str(
                i).zfill(8) + ".png"
            fig = plt.figure()
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]],
                norm=LogNorm(vmin=1e-15))
            plt.ylabel("Dim: 0    Electron: 0")
            plt.xlabel("Dim: 0    Electron: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            # plot dim 0 for both electrons
            plot_name = "figs/" + target_name + "_2d_dim_1_state_" + str(
                i).zfill(8) + ".png"
            axis_sum_list = tuple([0, 2])
            fig = plt.figure()
            data = np.sum(psi, axis=axis_sum_list)
            axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]])
            plt.ylabel("Dim: 1    Electron: 0")
            plt.xlabel("Dim: 1    Electron: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

            plot_name = "figs/" + target_name + "_log_2d_dim_1_state_" + str(
                i).zfill(8) + ".png"
            fig = plt.figure()
            plt.imshow(
                data / data.max(),
                cmap='viridis',
                extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]],
                norm=LogNorm(vmin=1e-15))
            plt.ylabel("Dim: 1    Electron: 0")
            plt.xlabel("Dim: 1    Electron: 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(plot_name)
            plt.close(fig)

elif num_electrons == 1:
    print( shape)
    x = f["Wavefunction"]["x_value_0"][:]

    if num_dims > 1:
        y = f["Wavefunction"]["x_value_1"][:]

    name_list = state_name(len(psi_value))

    if num_dims == 3:
        from mayavi import mlab
        z = f["Wavefunction"]["x_value_2"][:]
        mlab.figure(
            bgcolor=(1.0, 1.0, 1.0),
            fgcolor=(0.0, 0.0, 0.0),
            size=(1000, 1000))
        for i, psi in enumerate(psi_value):
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            # print np.log10(np.abs(psi))
            mlab.pipeline.iso_surface(
                mlab.pipeline.scalar_field(np.log10(np.abs(psi))),
                vmin=-0.5,
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

    elif num_dims == 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.colors import LogNorm
        import pylab as plb
        font = {'size': 22}

        matplotlib.rc('font', **font)
        fig = plt.figure()
        for i, psi in enumerate(psi_value):
            psi_save = np.array(psi)
            print( "plotting", i)
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            x_min_idx = 0
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
                    norm=LogNorm(vmin=1e-12))
            # color bar doesn't change during the video so only set it here
            plt.colorbar(pad=0.1)
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                plt.xlabel("z-axis (a.u.)")
                plt.ylabel("$\\rho$-axis  (a.u.)")
            else:
                plt.xlabel("X-axis (a.u.)")
                plt.ylabel("Y-axis  (a.u.)")
            plt.title(name_list[i])
            plt.tight_layout()
            fig.savefig("figs/" + target_name + "_log_state_" + str(i).zfill(3)
                        + ".jpg")
            plb.xlim([-30, 30])
            plb.ylim([0, 30])
            plt.tight_layout()
            fig.savefig("figs/" + target_name + "_log_state_small_" +
                        str(i).zfill(3) + ".jpg")
            plt.clf()
            psi = psi_save
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                print( psi.shape)
                print( np.angle(psi)[np.unravel_index(
                                    np.abs(psi).argmax(), psi.shape)])
                psi = psi * np.exp(-1.0j * np.angle(
                    np.angle(psi)[
                        np.unravel_index(np.abs(psi).argmax(), psi.shape)]))
                plt.imshow(
                    np.angle(psi),
                    cmap='hsv',
                    origin='lower',
                    extent=[
                        y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                    ])
            else:
                print( np.angle(psi)[np.unravel_index(
                                    np.abs(psi).argmax(), psi.shape)])
                # psi = psi * np.exp(-1.0j * np.angle(psi)[
                #     np.unravel_index(np.abs(psi).argmax(), psi.shape)])
                print( np.angle(psi)[psi.shape[0] // 2 - 15:
                                                    psi.shape[0] // 2 + 15, psi.shape[1] // 2])
                plt.imshow(
                    np.angle(psi),
                    cmap='hsv',
                    origin='lower',
                    extent=[
                        y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                    ])
                # print np.angle(psi)[psi.shape[0] / 2 + 15, psi.shape[1] / 2 -
            # 15], np.angle(psi)[psi.shape[0] / 2 + 10,
            #                    psi.shape[1] / 2 - 30]
            # color bar doesn't change during the video so only set it here
            plt.colorbar(pad=0.1)
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                plt.xlabel("z-axis (a.u.)")
                plt.ylabel("$\\rho$-axis  (a.u.)")
            else:
                plt.xlabel("X-axis (a.u.)")
                plt.ylabel("Y-axis  (a.u.)")
            plt.title(name_list[i])
            plt.tight_layout()
            fig.savefig("figs/" + target_name + "_log_state_phase_" +
                        str(i).zfill(3) + ".jpg")
            plb.xlim([-30, 30])
            plb.ylim([0, 30])
            plt.tight_layout()
            fig.savefig("figs/" + target_name + "_log_state_phase_small_" +
                        str(i).zfill(3) + ".jpg")
            plt.clf()

    elif num_dims == 1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pylab as plb
        import matplotlib.animation as animation
        from matplotlib.colors import LogNorm
        #    fig = plt.figure()
        for i, psi in enumerate(psi_value):
            print( "plotting", i)
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            plt.plot(x, (np.absolute(psi)))
            # color bar doesn't change during the video so only set it here
            plt.xlabel("z-axis (a.u.)")
            plt.ylabel("log10(psi)")
            # plb.xlim([-700, 700])
            #plb.ylim([-7, 0])
            plt.title(name_list[i] + " - Energy " + str(energy[i]))
            plt.savefig("figs/" + target_name + "_log_state_" + str(i).zfill(3)
                        + ".jpg")
            plt.clf()
else:
    print( "num_electrons:", num_electrons)
    print( "ERROR: number of electrons in this simulation is not supported")
