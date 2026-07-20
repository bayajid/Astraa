## Host pos extrapolation errors using quadrat extrapolator
# IMPORTS
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
import prediction_methods.error_generation as errgen
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
t_j2000 = data_raw[:,0]
t_0 = t_j2000[0]

states_host = data_raw[:,1:7]
t_analyzed = 20 # [s]
states_host = states_host[:t_analyzed,:]
t_vec_full = t_j2000[:t_analyzed]
#%% input errors; 3-sig
dr_3sig = 10 # m
dv_3sig = 0.06 # m/s

dr_dv = np.array([[dr_3sig, dv_3sig],
                  [dr_3sig*2, dv_3sig*2],
                  [dr_3sig*3, dv_3sig*3],
                  [dr_3sig*5, dv_3sig*5],
                  [dr_3sig*8, dv_3sig*8],
                  [dr_3sig*10, dv_3sig*10]])
output_dict_dr = {}
output_dict_pe = {}
link_dist = 250e3 # m

states_full_true = states_host


## Get error directions
ii_0 = 0
s_0_known = states_full_true[ii_0,:]
s_1_known = states_full_true[ii_0+1,:]
r_0_known = s_0_known[:3]
r_0_known_norm = r_0_known / np.linalg.norm(r_0_known)
v_0_known = s_0_known[3:]
v_0_known_norm = v_0_known / np.linalg.norm(v_0_known)
r_1_known = s_1_known[:3]
r_1_known_norm = r_1_known / np.linalg.norm(r_1_known)
v_1_known = s_1_known[3:]
v_1_known_norm = v_1_known / np.linalg.norm(v_1_known)

for jj, [dr, dv] in enumerate(dr_dv):

    states_full_actual = np.copy(states_full_true)
    pos_errors_quad_all = np.zeros((t_vec_full.shape[0],3))
    pos_interpolator = interp.we_interpolating()

    # Error directions: Position in pos vector direction
    # velocity error opposite of vector
    dr_0 = r_0_known_norm * dr_3sig
    dr_1 = r_1_known_norm * dr_3sig
    dv_0 = -v_0_known_norm * dv_3sig
    dv_1 = -v_1_known_norm * dv_3sig
    r_0_known = r_0_known + dr_0
    r_1_known = r_1_known + dr_1
    v_0_known = v_0_known + dv_0
    v_1_known = v_1_known + dv_1

    # total errors
    extrap_errors_pos = np.zeros((states_host.shape[0],3))
    extrap_errors_pointing = np.zeros((states_host.shape[0],1))
    # added errors
    extrap_errors_pos_added  = np.zeros((states_host.shape[0],3))
    extrap_errors_pointing_added  = np.zeros((states_host.shape[0],1))

    r_known = np.vstack((r_0_known,
        r_1_known))
    v_known = np.vstack((v_0_known,
        v_1_known))

    t_0 = t_vec_full[ii_0] # first t
    t_1 = t_vec_full[ii_0+1] # next t
    t_2 = float(t_vec_full[ii_0+2]) # for error calc; 1 s away

    pos_interpolator.get_quad_interpolant([t_0, t_1], r_known, v_known)
    for ii, t_ii in enumerate(t_vec_full):
        r_interp = pos_interpolator.interpolate(float(t_ii))    

        r_2_true = states_full_true[ii,:3]
        dr = r_interp - r_2_true
        dr_tot = np.linalg.norm(dr)

        extrap_errors_pos[ii,:] = dr
        extrap_errors_pointing[ii] = np.linalg.norm(dr)/link_dist*1e6 # urad
    
    extrap_errors_pos_from_0 = np.linalg.norm(extrap_errors_pos,axis=1)
    extrap_errors_pos_from_0 = extrap_errors_pos_from_0 - extrap_errors_pos_from_0[1]
    extrap_errors_pointing_from_0 = extrap_errors_pointing - extrap_errors_pointing[1]
    
    output_dict_pe[jj] = extrap_errors_pointing_from_0.flatten()
    output_dict_dr[jj] = extrap_errors_pos_from_0
    
df_dr = pd.DataFrame.from_dict(output_dict_dr)
df_pe = pd.DataFrame.from_dict(output_dict_pe)

f, axs = plt.subplots(nrows = 2)
f.suptitle('Quadratic Extrapolation Results; Added Error by Predictions')
for jj, dr in enumerate(dr_dv):
    label = f'dr={dr[0]:.0f}m, dv={dr[1]*100:.0f}cm/s'
    axs[0].plot(t_vec_full-t_vec_full[1],df_dr.iloc[:,jj], label = label)
    axs[1].plot(t_vec_full-t_vec_full[1],df_pe.iloc[:,jj], label = label)
axs[0].set_ylabel('3D Pos error [m]')
axs[1].set_ylabel(f'PE [urad] at R = {link_dist/1e3:.0f} km')
axs[1].set_xlabel('t extrap [s]')
axs[0].legend()
axs[1].legend()
axs[0].grid('on')
axs[1].grid('on')
axs[1].set_ylim([0,100])
axs[0].set_xlim([0,10])
axs[1].set_xlim([0,10])
# f, axs = plt.subplots(nrows = 2)
# f.suptitle('Quadratic Extrapolation Results; Added Error')
# axs[0].plot(extrap_errors_pos_added[:-3])
# axs[0].set_ylabel('3D Pos error added [m]')
# axs[1].plot(extrap_errors_pointing[:-3])
# axs[1].set_ylabel(f'PE added [urad] at R = {link_dist/1e3:.0f} km')
# axs[1].set_xlabel('t [s]')
# plt.legend()