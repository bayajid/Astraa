#%% Generate histograms and time-serie plots
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
path_cwd = os.getcwd()
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.astro_targets as where_sun
import basic_tools.time_conversion as t_conv
import link_processing_tools.visibility_checks as vis_check
import tudat_tools.data_processing.data_processing_utilities as dputil
import plotting_tools.basic_plotting as bplt
import plotting_tools.modular_plotting as modplot
# path jazz
# length_chosen = 1 # days
# length_chosen = 7  # days
length_chosen = 62 # days
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

output_folder_overview = r'outputs\tables\moon_conops_conditions\final_overview'

files_all = [f for f in os.listdir(output_folder_overview) if '62' in f]
# files_all = files_all[:1]

sat = []
nr_passes = []
min_obs_time = []
max_obs_time = []
median_obs_time = []
max_nopass = []
for file in files_all:
    sat_host = file[:file.index('_62')]
    if 'sat_leo_eq' in sat_host:
        sat_name = 'LEO 1000 km altitude, Equatorial.'
    elif 'sat_leo_incl' in sat_host:
        sat_name = 'LEO 1000 km altitude, Inclined 53 deg.'
    elif 'sat_leo_polar' in sat_host:
        sat_name = 'LEO 1000 km altitude, Near-Polar 89 deg.'
    elif 'sat_meo' in sat_host:
        sat_name = 'MEO 13880 km altitude, Equatorial.'
    df = pd.read_csv(fr'{output_folder_overview}\{file}')
    sat.append(sat_name)
    nr_passes.append(int(np.max(df['pass_nr'])))
    min_obs_time.append(int(min(df['t_pass_s'])/60))
    max_obs_time.append(int(max(df['t_pass_s'])/60))
    median_obs_time.append(int(np.median(df['t_pass_s'])/60))
    max_nopass.append(np.round(np.max(df['t_nopass_s'])/86400,2))
    plt.subplots()
    plt.hist(df['t_pass_s']/60)
    plt.title(sat_name)
    plt.ylabel('Nr. Passes [-]')
    plt.xlabel('Pass length [min]')
overview_df = pd.DataFrame.from_dict({
    'Satellite' : sat,
    'Total Passes' : nr_passes,
    'Min. Pass t [min]' : min_obs_time,
    'Max. Pass t [min]' : max_obs_time,
    'Median Pass t [min]' : median_obs_time,
    'Max wait interval t [days]' : max_nopass,

})
overview_df.to_csv(fr'{output_folder_overview}/pass_time_stats_all.csv',index = 0)
print(f'Saved pass_time_stats_all to {output_folder_overview}')