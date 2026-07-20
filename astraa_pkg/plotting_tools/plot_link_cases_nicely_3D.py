#%% 
## Script used to process intermediate outputs (links, Az/El/Slant range angles, etc) into
# the final outputs (link statistics) in the forms of tables, histograms, etc
# desired outputs- histograms of link windows, max angles, max angular rates
# per different host sat, link cases, terminal placements, terminal type
# and imposed terminal limitations (at minimum, considering visiblity)
# at maximum, adding AE limitations, slant range limitations
# DIRECTLY CARRIED OVER FROM THESIS WORK
import pathlib
import json
import csv
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
# import loading_functions.data_loading as load
# import data_processing.rotations as rot
# import data_processing.data_processing_utilities as dputil
# import plotting_functions.modular_plotting as modplot
# import matplotlib.pyplot as plt
# import plotting_functions.plotting_basic as bplt
# import plotting_functions.modular_plotting as modplot
# import basic_tools.operations as basic
# import basic_tools.link_cases as link_case
import matplotlib.backends.backend_pdf
from matplotlib.offsetbox import AnchoredText
# importlib.reload(rot)
# importlib.reload(load)
# importlib.reload(dputil)
# importlib.reload(bplt)
# ## base paths
# pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
# link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
# output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
# output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")

# sat_hosts = [pos[:-23] for pos in os.listdir(f'{output_path_link}\\mk3') if 'meo_lct1_linkrates' in pos]
def get_label_names(case):
    # function to convert satellite names to cleaner labels
    # input lsit of sat names
    names_nice = [name.replace('sat_leo_incl_', 'LEO_I_') for name in case]
    if case == names_nice:
        names_nice = [name.replace('sat_leo_polar_', 'LEO_P_') for name in case]
    if case == names_nice:
        names_nice = [name.replace('sat_meo_0', 'MEO') for name in case]
    
    names_nice = [name.replace('link', 'Link ') for name in names_nice]
    names_nice = [name.replace('leo_incl_', 'LEO_I_') for name in names_nice]
    names_nice = [name.replace('leo_polar_', 'LEO_P_') for name in names_nice]
    names_nice = [name.replace('_meo_', ' MEO_') for name in names_nice]
    names_nice = [name.replace('_LEO', ' LEO') for name in names_nice]
    return names_nice

hp_folder = r'simulation_output\intersatellite_links\high_precision'
mp_folder = r'simulation_output\intersatellite_links\medium_precision'
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#%% 3D plot
# importlib.reload(modplot)
# cases_all = os.listdir(hp_folder)
range_meo = [13e6, 25e6]
range_leo = [50e3, 7000e3]
# for case in cases_all[:1]:
for ii, case in enumerate(cases_all):    
    if ii > -1:
        ## GEt path and sim parameters
        dir_case = f'{hp_folder}/{case}'
        if 'meo' in case:
            range_link = range_meo        
        else:
            range_link = range_leo        
        path_parameters_full = os.path.normpath(f'{dir_case}\simulation_parameters.json')
        path_states_full = os.path.normpath(f'{dir_case}\state_history.dat')
        # longer interval of orbital data
        path_states_ref = os.path.normpath(f'{mp_folder}\{case}\state_history.dat')
        
        with open(path_parameters_full, 'r') as j:
            sim_parameters = json.load(j)
        states_loaded = load.open_dat(path_states_full)
        states_full = load.open_dat(path_states_ref)
        ind_r_host = [1,2,3]
        ind_r_target = [7,8,9]
        r_host = states_loaded[:,ind_r_host]
        r_target = states_loaded[:,ind_r_target]
        
        name_host = get_label_names([sim_parameters['sat_names'][0]])[0]
        name_target = get_label_names([sim_parameters['sat_names'][1]])[0]
        if 'LEO' in name_host:
            ii_max_host = int(6600 /15)
        elif 'MEO' in name_host:
            ii_max_host = -1
        if 'LEO' in name_target:
            ii_max_target = int(6600 /15)
        elif 'MEO' in name_target:
            ii_max_target = -1
        case_title = get_label_names([case])[0]
        case_title = case_title[6:]
        case_title = f'Link Case {ii+1}{case_title}'
        label_host = f'Host {name_host}'
        label_target = f'Target {name_target}'


        set_axes_equal(ax)
        if ii in [1, 3]:
            ax.view_init(30, 60)
        if ii in [4]:
            ax.view_init(15, 120)
        bplt.savefig(fig, case, 'plots_results/link_cases')
plt.show()