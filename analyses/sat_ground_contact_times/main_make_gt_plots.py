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
csv_output_path = r'orbital_simulations\leo_sat\j2_leo_gt'
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
import basic_tools.parsing as parse

## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.data_processing.data_loading as dload
# import analyses.sat_ground_contact_times.sat_to_ground_tools.draft_tle_to_ground_track as gt_calc
import analyses.sat_ground_contact_times.sat_to_ground_tools.plotting_functions as plt_fct
import analyses.sat_ground_contact_times.sat_to_ground_tools.GS_coordinates as GS_coord

dat_all = os.listdir(r'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_clean\astropynaric_repo\astropynaric\outputs\tables\contact_times')
dat_raw_all = [dat for dat in dat_all if '_all_' in dat]
dat_vis = [dat for dat in dat_all if '_visible_' in dat]
t_max = 3e4 # [s]

for ind_needed in [0,1,2,3,4,5]:
    # ind_needed = 1
    loc = GS_coord.gs_coordinates[ind_needed]

    dat_name_raw = [d for d in dat_raw_all if str(loc) in d][0]
    dat_name_vis = [d for d in dat_vis if str(loc) in d][0]
    dat_raw_used = fr'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_clean\astropynaric_repo\astropynaric\outputs\tables\contact_times\{dat_name_raw}'
    dat_vis_used = fr'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_clean\astropynaric_repo\astropynaric\outputs\tables\contact_times\{dat_name_vis}'
    chosen_data_vis = pd.read_csv(dat_vis_used)
    chosen_data_raw = pd.read_csv(dat_raw_used)
    
    dat_df = chosen_data_vis[chosen_data_vis['t'] < t_max]
    dat_df_draw = chosen_data_raw[chosen_data_raw['t'] < t_max]
    longlat_vis = dat_df['coord_gt'].values
    longlat_raw = dat_df_draw['coord_gt'].values
    long_lat = np.array([parse.parse_col(ii)  for ii in longlat_vis])
    long_lat_raw = np.array([parse.parse_col(ii)  for ii in longlat_raw])

    f, ax = plt.subplots()


    ax.scatter(long_lat_raw[:,0], long_lat_raw[:,1], c = 'b')
    ax.scatter(long_lat[:,0], long_lat[:,1], c = 'g')
