## Modular plotting (incl. 3D plotting) functions, directly carried over from
# thesis work
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import data_processing.data_processing_utilities as dputil
from matplotlib import cm, scale
from matplotlib import colors as mcolors
import matplotlib as mpl
# from matplotlib.collections.art3d import Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
def add_earth(fig,
                ax,
                R=6.371e6,
               pos_body=[0,0,0],
               labels='Earth',
               wires = 1,
               alpha = 0.05):    
    # Function used to plot a 3d earth
    # bm = np.array(bm.resize([int(d/2) for d in bm.size]))/256
    
    # lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
    # lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

    lons = np.linspace(-180, 180, 360) * np.pi/180 
    lats = np.linspace(-90, 90, 360) * np.pi/180 

    x = np.outer(np.cos(lons), np.cos(lats)).T * R
    y = np.outer(np.sin(lons), np.cos(lats)).T * R
    z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * R
    # ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)
    # ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha = 0.5)
    if wires:
        ax.plot_wireframe(x, y, z, color = 'green', rstride = 10, cstride = 10, alpha = alpha)
    else:
        ax.plot_surface(x, y, z, color = 'green', rstride = 10, cstride = 10, alpha = 0.9, zorder = 0.5)
    # ax.plot_wireframe(x, y, z, color = 'yellow', rstride = 10, cstride = 10, alpha = 1)
    return fig, ax
def make_3dplot(unit = 'm'):
    # function to make the figure object for 3d plots
    fig = plt.figure(figsize = (8,8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(f'x [{unit}]')
    ax.set_ylabel(f'y [{unit}]')
    ax.set_zlabel(f'z [{unit}]')
    return fig,ax
def add_orbit_basic(fig, ax,
                states, label,  c, alpha = 0.5, linewidth = 10):
    ax.plot(states[:,0], 
        states[:,1], 
        states[:,2], 
        c = c, 
        label = label, 
        linewidth = linewidth,
        alpha = alpha,
        zorder = 0.9)
    return fig, ax
def add_orbit(fig, ax,
                states_raw,
                sat_target, 
                indices_dict,
                ii_0,
                dt,
                c,                
                label = '',
                gm_e = 3.986e14
                ):
    # Function to plot the entire orbit
    # first calculates the orbit and proceeds to add it to the plot
    # if None is given for sat_target, then indices_dict must already be [ind_x, ind_y, ind_z]
    states_used = states_raw[:,1:] # skip time vector
    if sat_target != None: 
        ind_chosen = indices_dict[sat_target]['ind_pos']
    else:
        ind_chosen = indices_dict
    states_sat_0 = states_used[0,ind_chosen]
    r = np.sqrt(states_sat_0[0]**2 + states_sat_0[1]**2 + states_sat_0[2]**2) # semi-major axis [m]
    T_orbit = 2 * np.pi * np.sqrt( r**3 / gm_e)
    ii_e = ii_0 + int(T_orbit*1.1/dt)
    states_orbit = states_used[ii_0:ii_e,ind_chosen]

    ax.plot(states_orbit[:,0], 
        states_orbit[:,1], 
        states_orbit[:,2], 
        c = c, 
        label = label, 
        alpha = 0.5)
    
    return fig, ax
def add_glossary_link(fig, ax, link_dict, legend_loc = [0.9,0.2]):
    # adding legend and title to 3d plot
    link_type = link_dict['type'] 
    slant_range = [int(link_dict['slant_range_min']/1e3), int(link_dict['slant_range_max']/1e3)]
    time_window = link_dict['t_window'] if link_type=='permalink' else int(link_dict['t_window'])
    title = f'Type: {link_type}; slant_range: {slant_range} km, t_link: {time_window} s.'
    fig.suptitle(title, fontsize = 'large')
    fig.legend(bbox_to_anchor = legend_loc)
    return fig, ax


def add_glossary_pe(ax, ylabel = 'Pointing Error [urad]', ylim = [0,0]):
    ax.set_ylabel(ylabel, fontweight='bold', size=10)
    if ylim != [0,0]:
        ax.set_ylim((ylim[0], ylim[1]))
    ax.legend(loc = 1)
    ax.grid('on')
def add_single_los(fig, ax,
    state_h,
    state_t,
    color = 'r',
    label_used = 'LOS',
    normalize = 0,
    draw_at_origin = 0,
    len_normalized = 1,
    linewidth = 3,
    ):
    # plots a single LOS vector from the host to the target satellite
    if len(state_h.shape) > 1:
        state_h, state_t = state_h[0], state_t[0]
    if not draw_at_origin:
        los_plotted = [[state_h[0], state_t[0]],
        [state_h[1], state_t[1]],
        [state_h[2], state_t[2]],
        ]
        if normalize:
            los_plotted = [
                [state_h[0], state_h[0]+(state_t[0] - state_h[0])/np.linalg.norm(state_t)*len_normalized],
                [state_h[1], state_h[1]+(state_t[1] - state_h[1])/np.linalg.norm(state_t)*len_normalized],
                [state_h[2], state_h[2]+(state_t[2] - state_h[2])/np.linalg.norm(state_t)*len_normalized],
                            ]
    else:
        los = state_t[:3] - state_h[:3]
        if normalize:
            los_len = np.linalg.norm(los)
        else:
            los_len = len_normalized
        los_plotted = [
            [0, los[0]/los_len*len_normalized],
            [0, los[1]/los_len*len_normalized],
            [0, los[2]/los_len*len_normalized],
        ]
        # los 
    # ax.plot(
    #     los_plotted[0],
    #     los_plotted[1],
    #     los_plotted[2], color = color, linewidth = linewidth, label = label_used
    # )
    ax.quiver(
        X = los_plotted[0][0],
        Y = los_plotted[1][0],
        Z = los_plotted[2][0],
         U = los_plotted[0][1], 
         V = los_plotted[1][1], 
         W = los_plotted[2][1],
          color = color, linewidth = linewidth, label = label_used
    )
    return fig, ax    
def add_los(fig, ax,
    state_h,
    state_t,
    link_range = [0, 4000e3],
    split = None # every 100th value
    ):
    # plots the LOS vectors from host to target and colors them according to the link distance range
    # to avoid colored los, use input <link_range = None>

    # Make LineCollection object of all LOS vectors

    shape_los = np.shape(state_h)[0]
    if split != None:
        ii_used = range(0, shape_los, split)
    else:
        ii_used = range(0, shape_los, 1)
    shape_used = len(ii_used)
    all_los = np.zeros((shape_los,2,3))
    all_slant_range = np.zeros((shape_los,1))
    for ii, los in enumerate(all_los):
        los[0] = state_h[ii]
        los[1] = state_t[ii]
        all_los[ii] = los
        all_slant_range[ii] = np.sqrt((los[0][0] - los[1][0])**2 + 
                                    (los[0][1] - los[1][1])**2 +
                                    (los[0][2] - los[1][2])**2)
    los_used = np.zeros((shape_used,2,3))
    slant_range_used = all_slant_range[ii_used]
    # filter out most LOS for visibility sake
    if type(link_range) == type(None):
        link_range = [np.min(all_slant_range), np.max(all_slant_range)]
    for jj, ii in enumerate(ii_used):
        los_used[jj] = all_los[ii]
    if type(link_range) != type(None): # add scaled coloring
        # Prepare colormap for linecollection
        cmap = cm.rainbow
        # remap slant_ranges linearly to [0,1] range
        range_normalizer = mpl.colors.Normalize(vmin=link_range[0], vmax=link_range[1], clip = True)
        range_normalized = range_normalizer(slant_range_used).data.flatten()
        los_colors = [mcolors.to_rgba(c) for c in cmap(range_normalized)]
        all_los_lines = Line3DCollection(los_used, colors = los_colors, alpha = 1)
        # range_normalizer = mpl.colors.Normalize(vmin=link_range[0]/1e3, vmax=link_range[1]/1e3, clip = True)
        # range_normalized = range_normalizer(slant_range_used).data.flatten()/1e3
        # los_colors = [mcolors.to_rgba(c) for c in cmap(range_normalized)]
        # all_los_lines = Line3DCollection(los_used/1e3, colors = los_colors, alpha = 1)
        p = ax.add_collection3d(all_los_lines)
        # Add colorbar
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=range_normalizer, cmap=cmap),
        ax = ax,
        label = 'Slant range [m]',
        orientation = 'horizontal')
        cb.set_label(label = 'Slant range [m]', weight = 'bold')
    else: # if 
        all_los_lines = Line3DCollection(los_used, alpha = 0.5)
        p = ax.add_collection3d(all_los_lines)
    return fig, ax

def add_arc(fig, ax,
    c,
    state_i,
    label_f,
    s = 20,
    ind_r = [0,1,2]): # ToDo add LOS function
    X_f = state_i[:, ind_r[0]]
    Y_f = state_i[:, ind_r[1]]
    Z_f = state_i[:, ind_r[2]]
    ax.scatter(X_f, Y_f, Z_f, label=label_f, c = c, s = s)
    ax.scatter(X_f[0], Y_f[0], Z_f[0], c = c, s = s)
    # ax.scatter(X_f[-1], Y_f[-1], Z_f[-1], c = c, s = 1)
    return fig, ax    
def add_scatters_on_arc(fig, ax,
    c, 
    data_used,
    indices,
    label_f,
    s = 20,
    ind_taken = 0,
    ):
    ## Function used to add scatters in 3D of each satellite
    # plots each satellites state at the first time index\

    states_shell = np.zeros((len(indices),3))
    for ii, ind in enumerate(indices): # fill with states   
        ii_pos = ind['ind_pos'] 
        states_shell[ii,:] = data_used[ind_taken,ii_pos]
    ax.scatter(states_shell[:,0], states_shell[:,1], states_shell[:,2],
    label=label_f, c = c, s = s)
    return fig, ax
def add_scatters_simple(fig, ax,
    c, 
    data_used,
    label_f,
    s = 20,
    ind_used = [0]):
    ## Function used to add scatters in 3D for given satellite states

    ax.scatter(data_used[ind_used,0], data_used[ind_used,1], data_used[ind_used,2],
    label=label_f, c = c, s = s)
    return fig, ax
def add_ref_frame(fig, ax, colors = ['r', 'g', 'b'], chosen_setting = 0,
        all_settings = ['eci', 'gf', 'rsw'],
        axis_or = '',
        rot_gf = None,
        length = None,
        use_axis_labels = 1,
        origin = None):

    # optional inputs : 
    # 
    """  function to add reference frame axes
     possible settings include eci (J2000), rsw
     origin - XYZ of ref frame origin [m] in ECI
     use_axis_labels = 1 - ADD Axis label text
    Args:
        fig (_type_): f
        ax (_type_): ax
        colors (list, optional): colors for x/y/z axes. Defaults to ['r', 'g', 'b'].
        chosen_setting (int, optional): eci - 0, gf - 1. Defaults to 0.
        all_settings (list, optional): _description_. Defaults to ['eci', 'gf', 'rsw'].
        axis_or (str, optional): Axis label for text in plot. Defaults to ''.
        rot_gf (dcm, optional): rotation matrix to be plotted for setting == 1. Defaults to None.
        length (int, optional): length of axes. Default leads to some long axis. Defaults to None.
        use_axis_labels (bool, optional): bool to add axis label text Defaults to 1.
        origin (_type_, optional): _description_. Defaults to None.

    Returns:
        f, ax: _description_
    """    
    if type(origin) == type(None):
        origin = [0,0,0]
    if all_settings[chosen_setting] == 'eci':
        
        axis_or = 'ECI'        
        if type(length)== type(None):
            length = 6378e3*1.3

        xaxis = [1, 0, 0]
        yaxis = [0, 1, 0]
        zaxis = [0, 0, 1]        
    elif all_settings[chosen_setting] == 'gf': # global-frame
        if len(axis_or) ==0 :
            axis_or = 'GF'
        if type(length)== type(None):
            length = 15e5
        xaxis = rot_gf[0,:]
        yaxis = rot_gf[1,:]
        zaxis = rot_gf[2,:]
    elif all_settings[chosen_setting] == 'rsw': # global-frame
        if len(axis_or) ==0 :
            axis_or = 'RSW'
        if type(length)== type(None):
            length = 15e5
        xaxis = rot_gf[0,:]
        yaxis = rot_gf[1,:]
        zaxis = rot_gf[2,:]
    ax.quiver(origin[0], origin[1], origin[2], xaxis[0], xaxis[1], xaxis[2], length = length, color = colors[0], alpha = 0.5)
    ax.quiver(origin[0], origin[1], origin[2], yaxis[0], yaxis[1], yaxis[2], length = length, color = colors[1], alpha = 0.5)
    ax.quiver(origin[0], origin[1], origin[2], zaxis[0], zaxis[1], zaxis[2], length = length, color = colors[2], alpha = 0.5)
    if use_axis_labels:
        axis_labels = [f'{axis_or}_X', f'{axis_or}_Y', f'{axis_or}_Z']
        ax.text(origin[0]+ xaxis[0]*length, origin[1] + xaxis[1]*length, origin[2] + xaxis[2]*length, axis_labels[0], color = colors[0], fontweight = 'bold')
        ax.text(origin[0]+ yaxis[0]*length, origin[1] + yaxis[1]*length, origin[2] + yaxis[2]*length, axis_labels[1], color = colors[0], fontweight = 'bold')
        ax.text(origin[0]+ zaxis[0]*length, origin[1] + zaxis[1]*length, origin[2] + zaxis[2]*length, axis_labels[2], color = colors[0])   
    return fig, ax

def add_glossary_basic(fig, ax, title, legend_loc = [0.9,0.2], x_title = 0.5, y_title = 0.98, axlim = None):
    fig.suptitle(title, fontsize = 'large', fontweight = 'bold', x = x_title, y = y_title, backgroundcolor = 'white')
    fig.legend(bbox_to_anchor = legend_loc)
    set_axes_equal(ax, axlim)
    return fig, ax
def set_axes_equal(ax, axlim = None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      axlim - define axis limits, int, will set to +/-
    '''
    if type(axlim) == type(None):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    else:
        x_limits = [-axlim, axlim]
        y_limits = [-axlim, axlim]
        z_limits = [-axlim, axlim]
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
if __name__ == '__main__':
    # import splines.quaternion
    make_3d_plot_ae_test = 1
    if make_3d_plot_ae_test:
        import pathlib
        import os
        import numpy as np
        import sys
        parent_dir = pathlib.Path(__file__).parent.parent.resolve()
        os.chdir(parent_dir)
        sys.path.insert(1, os.getcwd())
        import attitude_tools.conversions as conv

        print(f'3d FRAME plotting tests')
        axlim_used = 10
        axis_length = 5
        los_length = 5

        c_eci = ['r', 'y', 'c']    
        c_gf = ['g', 'r', 'b']        


        s_host = np.array([[-100, 100, 0, 50, 10, 0]])
        s_target = np.array([[0, 100, 100, 0, 0, 0]])
        rpy_given = [-90, 90, 90]
        los_given = s_host[0,:3]
        AE_calc = [10, 10]
        rot_90yaw90roll = conv.convert_ea2dcm(rpy_given)
        title = f'''RPY : {rpy_given}
        LOS : {los_given}
        az, el : {AE_calc}
        '''
        f, ax = make_3dplot()
        f, ax = add_single_los(f, ax, s_host, s_target, draw_at_origin=1, normalize = 1, color = 'm', len_normalized=los_length)
        f, ax = add_ref_frame(f, ax, length = axis_length, use_axis_labels = 1, colors = c_eci)
        f, ax = add_ref_frame(f, ax, chosen_setting = 1, length = axis_length, use_axis_labels = 1, rot_gf=rot_90yaw90roll, colors = c_gf)
        f, ax = add_glossary_basic(f, ax, title = title, axlim = axlim_used)
        plt.show()
        
