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
csv_output_path = r'orbital_simulations/leo_meo_srpcheck_high_precision/leo_meo_srpcheck'#'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

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
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

# host_chosen = 'leo_host_polar'
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'
host_chosen = "leo_polar"

t_j2000 = data_raw[:,0]
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
import tudat_tools.tudat_converter as tudatconv

## Getting ECI to RSW rotation matrix
tudconv = tudatconv.tudat_predictor()
ii = 0
ROT_RSWfromECI = tudconv.calc_rotrsweci(r_h = r_host[ii,:], v_h = v_host[ii,:])
r_host_rsw = ROT_RSWfromECI @ r_host[ii,:]
