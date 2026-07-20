## Script to evaluate max timestamp delay of PVTR data 
# by calculating j2 propagation errors with expected initial errors of target position
# and a max propagation time of 100s
# first doing worst case analysis using analytical approximations of dr -> PE, later 
# analyzing it for a full link case
# Date 8/8/2024
# Author Kipras Paliusis

#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import scipy as sp
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop
## Loading satellite orbital data
import prediction_methods.error_generation as err_gen
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = 'leo_host_polar'
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
t_gps_from0 = t_gps - t_gps[0]
r_target_true = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_target_true = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]

ii_0 = 100

r_initial = r_target_true[ii_0, :]
v_initial = v_target_true[ii_0, :]

s_true = np.hstack((r_target_true, v_target_true))
s_initial = np.hstack((r_initial, v_initial))
# Initial Error https://mynaric.atlassian.net/wiki/spaces/EN/pages/714571952/Host+and+Target+Position; PEB communicated to customers indicates 100m target error
dr_initial = 100 # [m]
dv_initial = 0.3
dt_prop = 100 # [s], fixed propagation time

dt_stamp_delays = [60] # [s]

r_err_host = err_gen.pos_err_gen(0, dr_initial, 1, ncols = 3, seed_used = 0)
v_err_host = err_gen.pos_err_gen(0, dv_initial, 1, ncols = 3, seed_used = 1)
# timestamp of data
t0 = t_gps_from0[ii_0]
t_end = t_gps_from0[-1]
t_step = t_gps_from0[1] - t_gps_from0[0]

# Function. Input: Dt delay, prop_time, initial conditions, truth orbit
for dt_stamp_delay in dt_stamp_delays:
    t_now = t0 + dt_stamp_delay
    t_final = t_now + dt_prop    
    # Get initial position
    s_0 = s_initial + np.hstack((r_err_host, v_err_host))
    s_0 = s_0.flatten()
    # Get propagated position
    prop_output = j2prop.propagate_orbit(s_0, t0, t_final, t_step)
    s_prop = prop_output[:,1:]
    t_vec = prop_output[:,0]
    # get respective true positions
    interpolant = sp.interpolate.interp1d(t_gps_from0,s_true, axis =0)
    s_interp = interpolant(t_vec)
    # get final error
    prop_err_final = s_interp - s_prop
    # convert to PE contribution analytically
    
ds_0 = s_0 - s_initial    
ds_f = s_interp[-1,:] - s_prop[-1,:]
dr = s_interp[:,:3] - s_prop[:,:3]

dr_0 = np.linalg.norm(ds_0[:3])
dv_0 = np.linalg.norm(ds_0[3:])

dr_f = np.linalg.norm(ds_f[:3])
dv_f = np.linalg.norm(ds_f[3:])
print(f'Prop err start : dr = {dr_0:.0f} m; dv = {dv_0:.1f} m/s')
print(f'Prop err at end : dr = {dr_f:.0f} m; dv = {dv_f:.1f} m/s')
    # OPTIONAL recompute same PE with the min distance link case
# Compute PAA true vs PAA approx with predicted target positions
