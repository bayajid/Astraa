## Checking if the moon position is interpolated correctly without velocity inputs
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import basic_tools.vector_operations as vec_op
import astronomy_tools.astro_targets as where_sun
import prediction_methods.interpolators as interp
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out

## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
t_j2000 = data_raw[:,0]
t_0 = t_j2000[0]
#%% Get true moon, dt = 1 s

body_source = where_sun.body_fromsp(t_0)
t_vec_full = np.arange(t_0, t_0+3.6e3, 1)

moon_pos_prec = np.zeros((t_vec_full.shape[0],3))
for ii, t in enumerate(t_vec_full):
    moon_pos_prec[ii,:] = body_source.get_sun(t - t_0, 'moon')

#%%

moon_errors_lin_all = np.zeros((t_vec_full.shape[0],3))
moon_errors_quad_all = np.zeros((t_vec_full.shape[0],3))


ind_sliced = [ii*10 for ii in range((int(t_vec_full.shape[0]/10)))]
pos_interpolator = interp.we_interpolating()
pos_interpolator_lin = interp.we_interpolating()

for ii_0 in ind_sliced[:-1]:
    ii_end = ii_0 + 10
    
    t_interpolated = t_vec_full[ii_0:ii_end]
    moon_pos_true = moon_pos_prec[ii_0:ii_end,:]
    ii_2 = [ii_0, ii_end]
    pos_interpolator.get_quad_interpolant(t_vec_full[ii_2].flatten(), moon_pos_prec[ii_2,:], np.zeros((2,3)))
    moon_interpolated_quad = pos_interpolator.interpolate(t_interpolated)
    pos_interpolator_lin.get_lin_interpolant(t_vec_full[ii_2].flatten(), moon_pos_prec[ii_2,:])
    moon_interpolated_lin = pos_interpolator_lin.interpolate(t_interpolated)
    
    moon_err_quad = moon_interpolated_quad - moon_pos_true
    moon_err_lin = moon_interpolated_lin - moon_pos_true

    moon_errors_quad_all[ii_0:ii_end,:] = moon_err_quad
    moon_errors_lin_all[ii_0:ii_end,:] = moon_err_lin

plt.plot(np.linalg.norm(moon_errors_quad_all, axis = 1), label = 'quad, v=0')
plt.plot(np.linalg.norm(moon_errors_lin_all, axis = 1), label = 'lin')
plt.legend()
plt.ylabel('dr [m]')
# PE plot
pe_quad = [vec_op.calc_dot_angle(v1, v2) for v1, v2 in zip(moon_interpolated_quad, moon_pos_true)]
pe_lin = [vec_op.calc_dot_angle(v1, v2) for v1, v2 in zip(moon_interpolated_lin, moon_pos_true)]
f, ax = plt.subplots()

ax.plot(pe_quad, label = 'quad, v=0')
ax.plot(pe_lin, label = 'lin')
ax.legend()
ax.set_ylabel('PE [rad]')