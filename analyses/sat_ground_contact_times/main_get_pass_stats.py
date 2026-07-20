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

out_path = r'outputs/tables/contact_times' #r'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_clean\astropynaric_repo\astropynaric\outputs\tables\contact_times'
dat_all = os.listdir(r'outputs/tables/contact_times')#r'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_clean\astropynaric_repo\astropynaric\outputs\tables\contact_times')
dat_raw_all = [dat for dat in dat_all if '_all_' in dat]
dat_vis = [dat for dat in dat_all if '_visible_' in dat]
dat_ov = [dat for dat in dat_all if 'overview' in dat]
t_max = 86e6 # [s]

names = []
coord = []
pass_stats = np.zeros((6, 6), dtype = int)
for ind_needed in [0,1,2,3,4,5]:
    # ind_needed = 1
    loc = GS_coord.gs_coordinates[ind_needed]
    vis = GS_coord.mean_annual_vis[ind_needed]
    dat_name_raw = [d for d in dat_raw_all if str(loc) in d][0]
    dat_name_vis = [d for d in dat_vis if str(loc) in d][0]
    dat_name_ov = [d for d in dat_ov if str(loc) in d][0]
    dat_raw_used = fr'outputs/tables/contact_times/{dat_name_raw}'
    dat_vis_used = fr'outputs/tables/contact_times/{dat_name_vis}'
    dat_overview_used = fr'outputs/tables/contact_times/{dat_name_ov}'
    chosen_data_vis = pd.read_csv(dat_vis_used)
    chosen_data_raw = pd.read_csv(dat_raw_used)
    chosen_data_ov = pd.read_csv(dat_overview_used)
    names.append(GS_coord.gs_names[ind_needed])
    t_obs = chosen_data_ov['length_observable'].values
    t_tot = np.sum(t_obs)
    t_min = np.min(t_obs)
    t_max = np.max(t_obs)
    t_med = np.median(t_obs)
    nr_passes = np.max(chosen_data_ov['pass_nr'].values)
    coord.append(loc)
    pass_stats[ind_needed,:] = [int(nr_passes), int(t_max), int(t_min), int(t_med), int(t_tot/60), np.round(t_tot/60*vis,1)]
#%%
dict_names = {'OGS':names,
              '[Long., Lat.]' : coord,
              'Passes [-]' : pass_stats[:,0],
              't axn. [s]' : pass_stats[:,1],
              't min. [s]' : pass_stats[:,2],
              't med. [s]' : pass_stats[:,3],
              't tot. [min]' : pass_stats[:,4],
              't tot. adj. [min]' : pass_stats[:,5],

              }
output_df = pd.DataFrame.from_dict(dict_names)
output_df.to_csv(f'{out_path}/stats_passes.csv', index = 0)

#%% HISTOGRAM
# for ind_needed in [0,1,2,3,4,5]:
for ind_needed in [1,2]:    
    # ind_needed = 1
    name = GS_coord.gs_names[ind_needed]
    loc = GS_coord.gs_coordinates[ind_needed]
    vis = GS_coord.mean_annual_vis[ind_needed]
    dat_name_raw = [d for d in dat_raw_all if str(loc) in d][0]
    dat_name_vis = [d for d in dat_vis if str(loc) in d][0]
    dat_name_ov = [d for d in dat_ov if str(loc) in d][0]
    dat_raw_used = fr'outputs/tables/contact_times/{dat_name_raw}'
    dat_vis_used = fr'outputs/tables/contact_times/{dat_name_vis}'
    dat_overview_used = fr'outputs/tables/contact_times/{dat_name_ov}'
    chosen_data_vis = pd.read_csv(dat_vis_used)
    chosen_data_raw = pd.read_csv(dat_raw_used)
    chosen_data_ov = pd.read_csv(dat_overview_used)
    chosen_data_ov['day_index'] = chosen_data_ov['start_t_s'].values/86400
    chosen_data_ov['day_index'] = chosen_data_ov['day_index'].astype(int)

    nr_day = []
    nr_ppd = []
    max_contact = []
    min_contact = []
    t_per_day = []
    for ii in chosen_data_ov['day_index'].unique():
        day_vals = chosen_data_ov[chosen_data_ov['day_index'] == ii]
        nr_passes_per_day = max(day_vals['pass_nr']) - min(day_vals['pass_nr'])+1
        cummulative_pass_time = sum(day_vals['length_observable'])/60
        max_contact.append(max(day_vals['length_observable'])/60)
        min_contact.append(min(day_vals['length_observable'])/60)

        nr_day.append(ii+1)
        nr_ppd.append(nr_passes_per_day)
        t_per_day.append(cummulative_pass_time)


    f, ax = plt.subplots()

    ax.bar(nr_day, t_per_day, align = 'center')    
    ax.set_xlim([0.5, 31])
    ax.set_ylabel('Cummulative visibility time [min]', fontweight = 'bold')
    ax.set_xlabel('t [day]', fontweight = 'bold')
    ax.plot(nr_day, max_contact, c = 'y', linewidth = 4, label = 'Max. pass time')
    ax.plot(nr_day, min_contact, c = 'r', linewidth = 4, label = 'Min. pass time')
    ax.legend()

    f.suptitle(f'Contact time for OGS in {name}', fontweight = 'bold')
    # bplt.autosave(f, subfolder='contact_times')

    t_lim = 86400
    chosen_data_vis = chosen_data_vis[chosen_data_vis['t']<t_lim]
    chosen_data_raw = chosen_data_raw[chosen_data_raw['t']<t_lim]
    long_lat = np.array([parse.parse_col(ii)  for ii in chosen_data_vis['coord_gt'].values])
    long_lat_raw = np.array([parse.parse_col(ii)  for ii in chosen_data_raw['coord_gt'].values])
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd
    from geopandas import GeoDataFrame

    # df = .read_csv("Long_Lats.csv", delimiter=',', skiprows=0, low_memory=False)
    ll = long_lat_raw[:,:]
    df_nopass = pd.DataFrame.from_dict({
        'Longitude' : ll[:,0],
        'Latitude' : ll[:,1]
    })
    ll = long_lat[:1000,:]
    df_pass = pd.DataFrame.from_dict({
        'Longitude' : ll[:,0],
        'Latitude' : ll[:,1]
    })

    geometry = [Point(xy) for xy in zip(df_nopass['Longitude'], df_nopass['Latitude'])]

    gdf = GeoDataFrame(df_nopass, geometry=geometry)   

    #this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = gdf.plot(ax=world.plot(figsize=(10, 6), color="lightgrey"), marker='o', color='red', markersize=0, zorder = 0.3)
    ax.scatter(df_nopass['Longitude'], df_nopass['Latitude'], c ='r', s = 4, marker = 'x', label = '')
    # ax.plot(df_nopass['Longitude'], df_nopass['Latitude'], c ='r', label = '', zorder = 0.5)
    ax.scatter(df_pass['Longitude'], df_pass['Latitude'], c ='g', s = 8, marker = 'o', label = 'Satellite Visible')
    ax.scatter(loc[0], loc[1], s = 200, c = 'black', marker = 'x', label = name)

    gap_size = 60
    # ax.set_ylim([loc[1]-gap_size,80])
    ax.set_ylim([-85, 85])
    ax.set_xlim([-170, 170])
    # ax.set_xlim([loc[0]-gap_size,loc[0]+gap_size])

    ax.set_title(f'Ground Track over OGS in {name} over 1 day', fontweight = 'bold')
    ax.set_xlabel('Longitude [deg]', fontweight = 'bold')
    ax.set_ylabel('Latitude [deg]', fontweight = 'bold')

    ax.legend()
    plt.show()


# %%
