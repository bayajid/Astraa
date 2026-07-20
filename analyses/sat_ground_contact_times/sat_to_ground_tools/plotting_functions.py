import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
def orbit_projection(gs_coordinates,
                     gt_coord,
                     swath_bounds,
                     vis_area,
                     title_f='',
               R=6.371e6*1e3,
               labels=['Earth']):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    # Function used to plot a 3d orbit of a spacecraft around the body p
    # gs coord : [long, lat] [deg]
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # Draw Earth sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v) * R
    y = np.sin(u)*np.sin(v) * R
    z = np.cos(v) * R
    # ax.plot_surface(x, y, z)  # plot the planet
    ax.plot_wireframe(x, y, z)  # plot the planet
    # set plot limits
    r_lim = np.linalg.norm((R, R, R))
    limits = [-r_lim, r_lim]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    
    # Plot ground station
    R_gs = R * 1.01 # scale up for visibility
    gs_colors = ['r', 'b','y','g', 'm', 'k', 'c']
    for ii, gs_coord in enumerate(gs_coordinates):
        long_gs = gs_coord[0]
        lat_gs = gs_coord[1]
        x_gs = R_gs * np.cos(np.deg2rad(long_gs)) * np.cos(np.deg2rad(lat_gs)) # r*cos(long)*cos(lat) [km]
        y_gs = R_gs * np.sin(np.deg2rad(long_gs)) * np.cos(np.deg2rad(lat_gs)) # r*sin(long)*cos(lat)
        z_gs = R_gs * np.sin(np.deg2rad(lat_gs)) # r*sin(lat)
        if 0: # print GS cartesian coordinates
            print(f'GS x = {x_gs:.1e}, y = {y_gs:.1e}, z = {z_gs:.1e}')
        ax.scatter3D(x_gs, y_gs, z_gs, c=gs_colors[ii], s = 100,label = f'Ground station [{long_gs};{lat_gs}]');
    # Plot ground track
    R_gt = R * 1.01
    long_gt_all = gt_coord[0]
    lat_gt_all = gt_coord[1]
    x_gt, y_gt, z_gt = [], [], []
    for long_gt, lat_gt in zip(long_gt_all, lat_gt_all):
        x_gt.append(R_gt * np.cos(np.deg2rad(long_gt)) * np.cos(np.deg2rad(lat_gt))) # r*cos(long)*cos(lat) [km]
        y_gt.append(R_gt * np.sin(np.deg2rad(long_gt)) * np.cos(np.deg2rad(lat_gt))) # r*sin(long)*cos(lat)
        z_gt.append(R_gt * np.sin(np.deg2rad(lat_gt))) # r*sin(lat)
    ax.scatter3D(x_gt, y_gt, z_gt, c='g', s = 50,label = 'Satellite ground track');
    
    # plot visible area bounds
    x_bound, y_bound, z_bound = [], [], []
    for bound in vis_area:
        for long_bound, lat_bound in zip(bound[:,0], bound[:,1]):
            x_bound.append(R_gt * np.cos(np.deg2rad(long_bound)) * np.cos(np.deg2rad(lat_bound))) # r*cos(long)*cos(lat) [km]
            y_bound.append(R_gt * np.sin(np.deg2rad(long_bound)) * np.cos(np.deg2rad(lat_bound))) # r*sin(long)*cos(lat)
            z_bound.append(R_gt * np.sin(np.deg2rad(lat_bound))) # r*sin(lat)        
    ax.scatter3D(x_bound, y_bound, z_bound, c='y', s = 5,label = 'Entire ISS access area ');
    
    # plot swath bounds
    x_bound, y_bound, z_bound = [], [], []
    for bound in swath_bounds:
        for long_bound, lat_bound in zip(bound[:,0], bound[:,1]):
            x_bound.append(R_gt * np.cos(np.deg2rad(long_bound)) * np.cos(np.deg2rad(lat_bound))) # r*cos(long)*cos(lat) [km]
            y_bound.append(R_gt * np.sin(np.deg2rad(long_bound)) * np.cos(np.deg2rad(lat_bound))) # r*sin(long)*cos(lat)
            z_bound.append(R_gt * np.sin(np.deg2rad(lat_bound))) # r*sin(lat)        
    ax.scatter3D(x_bound, y_bound, z_bound, c='g', s = 5,label = 'LCT FOV bounds');
    plt.title(f'LCT FOV plot {title_f}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z [m]', fontsize=14, fontweight='bold')
    ax.legend()
    plt.grid()
    plt.legend()
    return fig, ax
def process_and_plot_pass_data(chosen_data, label, eps_min):
    # Function to plot the GS-elevation time data
    # and histograms for different visibility conditions
    # input pass data, label for the plot title and minimum GS elevation threshold [deg]
    data_vis = chosen_data[chosen_data['is_visible'] == True]
    data_above_min = chosen_data[chosen_data['is_above_el'] == True]
    data_fov = chosen_data[chosen_data['is_in_fov'] == True]
    data_obs = chosen_data[chosen_data['is_observable'] == True]
    # get elevation for each condition
    el_vis = data_vis['gs_elevation'].values.astype(float)
    el_above_min = data_above_min['gs_elevation'].values.astype(float)
    el_fov = data_fov['gs_elevation'].values.astype(float)
    el_obs = data_obs['gs_elevation'].values.astype(float)
    # get time vectors
    t_vis = data_vis['t'].values.astype(float)
    t_0 = t_vis[0]
    t_vis = data_vis['t'].values.astype(float) - t_0
    t_above_min = data_above_min['t'].values.astype(float) - t_0
    t_fov = data_fov['t'].values.astype(float) - t_0
    t_obs = data_obs['t'].values.astype(float) - t_0
    # get histogram data
    x_data = []
    y_data = []
    d_eps = 5 # elevation jumps for the histogram
    bin_range = np.arange(0, 90+5, d_eps)
    el_used = el_fov # used elevation data
    for ii, bin_0 in enumerate(bin_range[:-1]):
        el_l = bin_0
        el_h = bin_0 + d_eps
        el_bin = (el_used >= el_l) & (el_used < el_h)
        n_el_bin = sum(el_bin)
        rel_n_el_bin = int(np.round(n_el_bin / len(el_used)*100,0))
        # print(f'[{el_l}:{el_h}) - {n_el_bin} and {rel_n_el_bin}%')
        x_data.append(el_l)
        y_data.append(rel_n_el_bin)


    # make figure
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
    fig.suptitle(label, fontsize = 16)

    # Plot el-time
    ax1.plot(t_vis, el_vis, label = f'ISS in GS view, {int(t_vis[-1]-t_vis[0])} s', c = 'r', linewidth = 2)
    ax1.plot(t_above_min, el_above_min, label = f'Above el_min, in GS view, {int(t_above_min[-1]-t_above_min[0])} s', c='m', linewidth = 2 )
    ax1.plot(t_fov, el_fov, label = f'GS in FOV of terminal, {int(t_fov[-1]-t_fov[0])} s', c = 'y', linewidth = 2)
    ax1.plot(t_obs, el_obs, label = f'GS accessible by terminal, {int(t_obs[-1]-t_obs[0])} s', c = 'g', linewidth = 1)
    ax1.plot([t_vis[0], t_vis[-1]], [eps_min, eps_min], label = 'Elevation threshold')
    ax1.legend(loc = 'upper right')
    ax1.grid()
    ax1.set_xlabel('Time since start of pass [s]', fontsize = 14)
    ax1.set_ylabel('Elevation from GS w.r.t. horizon [deg]', fontsize = 14)
    ax1.set_title(f'GS Elevation time-series.', fontsize = 16)
    fig.show()

    ax2.bar(x_data, y_data, width = 5, align = 'edge')    
    ax2.grid()
    ax2.set_xlabel('GS elevation [deg]', fontsize = 14)
    ax2.set_xticks(np.arange(0, 90+10, 10))
    ax2.set_ylabel('Occurences [%]', fontsize = 14)
    ax2.set_title(f'Elevation Histogram in Terminals FOV', fontsize = 16)
    return fig, [x_data, y_data]