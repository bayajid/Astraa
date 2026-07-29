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
csv_output_path = r'orbital_simulations/leo_meo_srpcheck_high_precision/leo_meo_srpcheck'#'orbital_simulations/leo_polar_meo_eq_medium_precision/leo_polar_meo_eq'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as cmbplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out

## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data and simulation parameters
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
sat_names = simulation_parameters['sat_names']
print(f'Satellites found: {sat_names}')
host_chosen = 'leo_polar'
target_chosen = sat_names[1]
print(f'Chosen satellites: {host_chosen} to {target_chosen}')
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
# get host states
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]

# get target states
r_target = data_raw[:,simulation_parameters['r_index'][target_chosen]]
v_target = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
import tudat_tools.tudat_converter as tudatconv
import attitude_tools.conversions as att_conv
## Getting ECI to RSW rotation matrix as host attitude
tudconv = tudatconv.tudat_predictor()
ii = 0
ROT_RSWfromECI = np.array([tudconv.calc_rotrsweci(r_h = r_host[ii,:], v_h = v_host[ii,:]) for ii, r in enumerate(r_host)])
# convert to quaternions
quat_eci2rsw = np.array([att_conv.convert_dcm2quat(dcm_ii) for dcm_ii in ROT_RSWfromECI])

# Get azimuth elevation slant-range from host to target
import pointing_calculations.ae_calculation as ae_calc
aer = ae_calc.calc_ae_full(r_host, r_target, attitude_eci2bf=quat_eci2rsw,
                           check_occultation = 0) # rad rad m

# Plot
f, ax = cmbplt.plot_aer(t_j2000, aer, 
                        title='Example AER plot, host pointing to Nadir', 
                        unit='rad',
                        setting='normal')
bplt.savefig(f, 'example_aer_plot')
plt.show()
# f, ax = bplt
# %%
