## Templates for loading satellite data
# generating attitude
# and whatnot. 

#%% IMPORTS
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
csv_output_path = r'orbital_simulations/leo_sat/j2_leo_gt'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
dep_var = 'dependent_variables.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out

## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.data_processing.data_loading as dload
import analyses.sat_ground_contact_times.sat_to_ground_tools.pass_process as pass_calc
import analyses.sat_ground_contact_times.sat_to_ground_tools.heading_angle_calc as heading_calc
from pyproj import Transformer
# nrows = int(1e4)
nrows = 86400*30*10
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = nrows)

host_chosen = 'leo_sat'
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
#%%
# dep_var, s_par = dload.open_dat(folder = csv_output_path, fname_states = dep_var, nrows = nrows)
# Get ECEF pos/vel
import tudat_tools.tudat_converter as tud_conv

tud_rot = tud_conv.tudat_predictor()
# tud_rot.set_time(t_j2000[0])
states_ecef = np.zeros((r_host.shape[0],6))
longlat = np.zeros((r_host.shape[0],2))
lon = np.zeros(r_host.shape[0])
lat = np.zeros(r_host.shape[0])
alt = np.zeros(r_host.shape[0])

for ii, t_ii in enumerate(t_j2000):
    X_ecef = tud_rot.rotate_eci2ecef(np.hstack((r_host[ii,[0,1,2]], v_host[ii,[0,1,2]])), t_ii)
    states_ecef[ii,:] = X_ecef
    # get long, lat
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon[ii], lat[ii], alt[ii] = transformer.transform(X_ecef[0],X_ecef[1],X_ecef[2])
    # longlat[ii,:] = np.rad2deg([np.arctan2(X_ecef[1], X_ecef[0]) ,np.arcsin(X_ecef[2] / np.linalg.norm(X_ecef[:3]))])


# heading_angles, v_enu_all = heading_calc.get_heading_angle(states_ecef[:,3:], longlat[:,0], longlat[:,1])
heading_angles, v_enu_all = heading_calc.get_heading_angle(states_ecef[:,3:], lon, lat)
# Get long/lat/heading
#%%
# long, lat, heading = longlat[:,0], longlat[:,1], heading_angles # deg
long, lat, heading = lon, lat, heading_angles # deg
data_gt = {}
data_gt['jd'] = t_j2000/86400
data_gt['long'] = long
data_gt['lat'] = lat
data_gt['heading'] = heading
data_gt_df = pd.DataFrame.from_dict(data_gt)
# da
for gs_ind in [0,1,2,3,4,5]:
    ground_track_df = pass_calc.process_gt_to_passes(data_gt_df, gs_ind, save_clean = 1)

#%% Get stats
#%% Get plots
import analyses.sat_ground_contact_times.sat_to_ground_tools.plotting_functions as plt_fct
import analyses.sat_ground_contact_times.sat_to_ground_tools.GS_coordinates as GS_coord

dat_all = os.listdir(r'outputs/tables/contact_times')
dat_raw_all = [dat for dat in dat_all if '_all_' in dat]
dat_vis = [dat for dat in dat_all if '_visible_' in dat]

ind_needed = 2

loc = GS_coord.gs_coordinates[ind_needed]

dat_name_raw = [d for d in dat_raw_all if str(loc) in d][0]
dat_name_vis = [d for d in dat_vis if str(loc) in d][0]
dat_raw_used = fr'outputs/tables/contact_times/{dat_name_raw}'
dat_vis_used = fr'outputs/tables/contact_times/{dat_name_vis}'
chosen_data = pd.read_csv(dat_vis_used)


f, ax = plt_fct.process_and_plot_pass_data(chosen_data, label = GS_coord.gs_names[ind_needed],
                                           eps_min = GS_coord.eps_min)
# %%
