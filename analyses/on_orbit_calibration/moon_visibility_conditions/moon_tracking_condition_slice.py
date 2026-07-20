#%%
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

# path jazz
downsample = 1
t_req = 60
save_csv = 1
# length_chosen = 1 # days
# length_chosen = 7  # days
length_chosen = 62 # days
csv_output_path = fr'orbital_simulations\moontrackers\leomeo_mixincl{length_chosen:.0f}d'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
output_folder = r'outputs\tables\moon_vis'
## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

# host_chosen = 'sat_leo_polar'
# host_chosen = 'sat_leo_incl'
host_chosen = 'sat_leo_eq'
# host_chosen = 'sat_meo'

host_name_full = f'{host_chosen}_0_0'
t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
nrows = len(t_j2000)
sat_index = simulation_parameters['sat_names'].index(host_name_full)
r_index = [1+sat_index*6, 1 + sat_index*6+1, 1 + sat_index*6+2]
v_index = [1+sat_index*6+3, 1 + sat_index*6+4, 1 + sat_index*6+5]

# Get host orbit
r_host = data_raw[:,r_index]
v_host = data_raw[:,v_index]
if downsample and length_chosen > 10:
    t_step_curr = t_gps[1] - t_gps[0]
    d_i = np.round(t_req / t_step_curr,0)
    print(f'Downsamplingdata from {t_step_curr} s to {t_req}')
    nrows = int(nrows / d_i)
    ii_sliced = [ii*6 for ii in range(nrows)]
    t_gps = t_gps[ii_sliced]
    r_host = r_host[ii_sliced,:]
    v_host = v_host[ii_sliced,:]
#%%
importlib.reload(where_sun)
# get moon orbit
r_moon = np.zeros((nrows,3))
moon_illumination = np.zeros((nrows,1))
for ii, t_gps_ii in enumerate(t_gps):
    r_moon_ii, illumination = where_sun.compute_moon_vector_eci(t_gps_ii, what_brightness=1)
    r_moon[ii,:] = r_moon_ii
    moon_illumination[ii] = illumination

#%%
# calculate passes and check occultation
ii_vis = vis_check.check_occultation(r_host, r_moon, R_atm = 100e3, limit_nr_links=0)
bool_vis = [True if ii in ii_vis else False for ii in range(nrows)]
# store
#%%
data = np.hstack((t_gps.reshape((nrows, 1)), r_host, r_moon, moon_illumination, np.array(bool_vis).reshape((nrows, 1))))
df_stored = pd.DataFrame(data = data, columns = ['t_gps', 'x_h', 'y_h', 'z_h', 'x_m', 'y_m', 'z_m', 'illumination','is_visible'])
save_title = f'{host_chosen}_moonpos_{length_chosen:.0f}d.csv'
if save_csv:
    df_stored.to_csv(f'{output_folder}/{save_title}', index = 0)
    print(f'Saved {save_title}')