#%% IMPORTS
# Here, 3d plots will be made
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\terran_near_polar_split\NearPolar12x244.00h'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
save_folder = r'outputs\tables\terran_const'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as combplt
import plotting_tools.modular_plotting as modplot
import attitude_tools.terminal_rotations as lct_rot
import pointing_calculations.ae_calculation as ae_calc
import attitude_tools.conversions as conv
import basic_tools.time_conversion as t_conv
import tudat_tools.tudat_converter as tudatconv
from tudat_tools.data_processing.data_saving_utilities import dict2txt
import tudat_tools.data_processing.data_processing_utilities as dputil

limit_rows = 0 # cut down datapoitns used for faster runs
# output aer;aer_dot for host
opposite_plane_ind = 11
host_ind_chosen = '0_0'

if limit_rows:
    rows_used = 1e3
else:
    rows_used = None
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = rows_used)

sat_names = simulation_parameters['sat_names']
opposite_plane_ind = 11
adjacent_plane_ind = 1
targets_whole_plane_seam = [sat for sat in sat_names if f'_{opposite_plane_ind}_' in sat]
targets_whole_plane_adjacent = [sat for sat in sat_names if f'_{adjacent_plane_ind}_' in sat]

host_pack = [[ii, name] for ii, name in enumerate(sat_names) if host_ind_chosen in name][0]
host_name = host_pack[1]
host_state_ind = [1+ ii + host_pack[0]*6 for ii in range(6)]

t_j2000 = data_raw[:,0]
s_host = data_raw[:,host_state_ind]
r_host = s_host[:,[0,1,2]]
v_host = s_host[:,[3,4,5]]
nrows = s_host.shape[0]
#%%
los_dict = {}
los_dict[host_name] = r_host
for ii, sat_target in enumerate(targets_whole_plane_seam):
    target_ind = simulation_parameters['state_ind'][sat_names.index(sat_target)]
    s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
    r_target = s_target[:,[0,1,2]]
    los_dict[sat_target] = r_target
for ii, sat_target in enumerate(targets_whole_plane_adjacent):
    target_ind = simulation_parameters['state_ind'][sat_names.index(sat_target)]
    s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
    r_target = s_target[:,[0,1,2]]
    los_dict[sat_target] = r_target

ind_0 = 1
ind_1 = 2

targets_chosen = ['sat_leo_t_11_7', 'sat_leo_t_0_0', 'sat_leo_t_11_6']
labels = ['S1', 'N1', 'S2']
# make 3d plots
f, ax = modplot.make_3dplot()
f, ax = modplot.add_earth(f, ax)
# Host sat
# f, ax = modplot.add_orbit_basic(f, ax, c = 'b', state_i = r_host, label_f='', s = 3, alpha = 0.2)
# f, ax = modplot.add_orbit_basic(f, ax, states = r_host, c = 'b', label = '', linewidth = 3)

rot_rsw = lct_rot.calc_rotrsweci(s_host[0,:3], s_host[0,3:], option = 'swr')
rot2 = conv.convert_ea2dcm([0,0,180])
rot_rsw_back = rot2@rot_rsw
f, ax = modplot.add_scatters_simple(f, ax, c = 'b', data_used = los_dict[targets_chosen[ind_0]], label_f = labels[ind_0], s = 50)
f, ax = modplot.add_orbit_basic(f, ax, states = los_dict[targets_chosen[1]], c = 'b', label = '', linewidth = 3)
f, ax = modplot.add_scatters_simple(f, ax, c = 'y', data_used = los_dict[targets_chosen[0]], label_f = labels[0], s = 50)
f, ax = modplot.add_orbit_basic(f, ax, states = los_dict[targets_chosen[0]], c = 'y', label = '', linewidth = 3)
# f, ax = modplot.add_scatters_simple(f, ax, c = 'orange', data_used = los_dict[targets_chosen[3]], label_f = labels[3], s = 50)
f, ax = modplot.add_orbit_basic(f, ax, states = los_dict[targets_chosen[2]], c = 'r', label = '', linewidth = 3)
f, ax = modplot.add_scatters_simple(f, ax, c = 'r', data_used = los_dict[targets_chosen[ind_1]], label_f = labels[ind_1], s = 50)
f, ax = modplot.add_ref_frame(f, ax, rot_gf = rot_rsw_back, origin = s_host[0,:3], chosen_setting=1)
f, ax = modplot.add_glossary_basic(f, ax, title = 'Verification - Terran Constellation at t=0')

# ax.view_init(30,70)
# ax.view_init(10,-50)
ax.view_init(0,-50)
# ax.view_init(45,-50)
bplt.autosave(f, subfolder='terran', timetag = 0)
f.show()
plt.show()