## Script to evaluate what the expected PE are for
# 

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
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import attitude_tools.attitude_simulation as att_sim
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import pointing_calculations.ae_calculation as ae_calc
from scipy.interpolate import CubicSpline
import prediction_methods.interpolators as interp
import prediction_methods.j2propagator as j2prop
## Load input data
input_path = r'/home/kpaliusis/git/astropynaric/outputs/tables/prop_testing/Colimator_[-89.7, 0.46]_2025-01-28'
fname = r'gs2moon_data.csv'
## Interpolate to 5 ms
path_dat = f'{input_path}/{fname}'
data_df = pd.read_csv(path_dat)  
plot_pe = 1
plot_ae = 0
propagators_enabled = 1
dt_req = 50e-3 # s, 50ms

update_rates = [1, 0.1, 0.1] # Hz. r_h, r_t, q_h

dt_prop = 10

data_sliced = data_df.values # TODO rename to better varaible name when setup fully
q_bf_used = data_sliced[:,16:24]   
t_sliced = data_sliced[:,0] 
n_digits = 3
t_req = np.arange(t_sliced[0], t_sliced[-1]+dt_req, dt_req)
t_req = np.round(t_req, n_digits)
t_gps_interp = CubicSpline(t_sliced, data_sliced[:,0], axis = 0)
s_host_interp = CubicSpline(t_sliced, data_sliced[:,4:10], axis = 0)
s_target_interp = CubicSpline(t_sliced, data_sliced[:,10:16], axis = 0)
q_host_interp = CubicSpline(t_sliced, q_bf_used, axis = 0)

## Compute AzEl at 5 ms
t_gps_interp_5ms = t_gps_interp(t_req)
s_host_true_5ms = s_host_interp(t_req)
s_target_true_5ms = s_target_interp(t_req)
q_host_true_5ms = q_host_interp(t_req)
ae_5ms = ae_calc.calc_ae_full(s_host_true_5ms, s_target_true_5ms, q_host_true_5ms)

## Create data with host/target/att propagated/predicted at chosen rates
t_gps_pred_5ms = np.zeros(t_gps_interp_5ms.shape)
s_host_pred_5ms = np.zeros(s_host_true_5ms.shape)
s_target_pred_5ms = np.zeros(s_target_true_5ms.shape)
q_host_pred_5ms = np.zeros(q_host_true_5ms.shape)

## Compute AzEl at 5 ms with prop/pred data
update_freq_pos_h = update_rates[0]
update_freq_pos_t = update_rates[1]
update_freq_att_h = update_rates[2]
dt_gap_pos_h = np.round(1/update_freq_pos_h,3)
dt_gap_att_h = np.round(1/update_freq_att_h,3)
dt_gap_pos_t = np.round(1/update_freq_pos_t,3)
t_interp_pos_h = np.round(np.arange(t_sliced[0], t_sliced[-1]+dt_gap_pos_h, dt_gap_pos_h),3)
t_interp_att_h = np.round(np.arange(t_sliced[0], t_sliced[-1]+dt_gap_att_h, dt_gap_att_h),3)
t_interp_pos_t = np.round(np.arange(t_sliced[0], t_sliced[-1]+dt_gap_pos_t, dt_gap_pos_t),3)

data_full_pos_h = s_host_interp(t_interp_pos_h)
data_full_pos_t = s_target_interp(t_interp_pos_t)
data_full_att_h = q_host_interp(t_interp_att_h)

ii_next_pos_h = 0
ii_next_pos_t = 0
ii_next_att_h = 0
pos_h_interp = interp.we_interpolating()
pos_t_interp = interp.we_interpolating()
quat_interp = interp.we_interpolating()
for ii, t_ii in enumerate(t_gps_interp_5ms):
    if t_ii >= t_interp_pos_h[ii_next_pos_h]:
        data_pos_h = data_full_pos_h[ii_next_pos_h]
        data_pos_h_held = data_pos_h # Propagator-off case
        if ii_next_pos_h >= 1 and ii_next_pos_h < len(t_interp_pos_h):
            pos_h_interp.get_quad_interpolant(
                t_both=t_interp_pos_h[ii_next_pos_h-1:ii_next_pos_h+1],
                r_both=data_full_pos_h[ii_next_pos_h-1:ii_next_pos_h+1,:3],
                v_both=data_full_pos_h[ii_next_pos_h-1:ii_next_pos_h+1,3:],                
            )
        ii_next_pos_h += 1
        
    else:
        data_pos_h = None
    if t_ii >= t_interp_pos_t[ii_next_pos_t]:
        data_pos_t = data_full_pos_t[ii_next_pos_t]
        data_pos_t_held = data_pos_t # Propagator-off case
        t_start = t_interp_pos_t[ii_next_pos_t]
        data_pos_t_prop = j2prop.propagate_orbit(data_pos_t, t_start, t_start+dt_prop, dt_prop)
        pos_t_interp.get_quad_interpolant(
                t_both=data_pos_t_prop[:,0],
                r_both=data_pos_t_prop[:,1:4],
                v_both=data_pos_t_prop[:,4:7],                
            )
        ii_next_pos_t += 1
    else:
        data_pos_t = None
    if t_ii >= t_interp_att_h[ii_next_att_h] and ii_next_att_h < len(t_interp_att_h):
        data_att_h = data_full_att_h[ii_next_att_h]
        data_att_h_held = data_att_h # Propagator-off case
        # UPDATE INTERPOLANT
        if ii_next_att_h >= 1:
            quat_interp.get_quad_interpolant(
                t_both=t_interp_att_h[ii_next_att_h-1:ii_next_att_h+1],
                r_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,:4],
                v_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,4:],                
            )
            a=1
        ii_next_att_h += 1
    else:
        data_att_h = None
    if ii_next_pos_h >= 2:
        if propagators_enabled:
            data_pos_h_interp = pos_h_interp.interpolate_flexible(t_ii)
        else:
            data_pos_h_interp = data_pos_h_held[:3]
    else:
        data_pos_h_interp = [0,0,0]
    if ii_next_att_h >=2:
        if propagators_enabled:
            data_att_interp = quat_interp.interpolate_flexible(t_ii)
        else:
            data_att_interp = data_att_h_held[:4]
        quat_diff = data_att_interp - q_host_true_5ms[ii,:4]
        a = 1
    else:
        data_att_interp = [0,0,0,0]
    if ii_next_pos_t >=1:
        if propagators_enabled:
            data_pos_t_interp = pos_t_interp.interpolate_flexible(t_ii)
        else:
            data_pos_t_interp = data_pos_t_held[:3]
        pos_diff = data_pos_t_interp - s_target_true_5ms[ii,:3]
        a = 1
    else:
        data_pos_t_interp = [0,0,0]
    # compute ae
    t_gps_pred_5ms[ii] = t_ii
    s_host_pred_5ms[ii,:3] = data_pos_h_interp
    s_target_pred_5ms[ii,:3] = data_pos_t_interp
    q_host_pred_5ms[ii,:4] = data_att_interp

# get AE pred
ae_pred_5ms = ae_calc.calc_ae_full(s_host_pred_5ms, s_target_pred_5ms, q_host_pred_5ms)
# ae_pred_5ms = ae_calc.calc_ae_full(s_host_pred_5ms, s_target_true_5ms, q_host_pred_5ms)
# ae_pred_5ms = ae_calc.calc_ae_full(s_host_pred_5ms, s_target_true_5ms, q_host_true_5ms)
# ae_pred_5ms = ae_calc.calc_ae_full(s_host_pred_5ms, s_target_pred_5ms, q_host_pred_5ms)

ii_sliced = np.where(s_host_pred_5ms[:,0] != 0)[0][0]
t_gps_sliced = t_gps_pred_5ms[ii_sliced:]
los_pred_5ms = [vec_calc.convert_polar_to_cartesian(ae) for ae in ae_pred_5ms[ii_sliced:,:]]
los_true_5ms = [vec_calc.convert_polar_to_cartesian(ae) for ae in ae_5ms[ii_sliced:,:]]
# get PE
pe_over_time = [vec_calc.calc_dot_angle(los_true, los_pred)*1e6 for los_true, los_pred in zip(los_true_5ms, los_pred_5ms)]

# plot
if plot_pe:
    f, ax = plt.subplots()
    ax.plot(t_gps_sliced-t_gps_sliced[0], pe_over_time)
    ax.hlines(3500, 0, t_gps_sliced[-1]-t_gps_sliced[0], 'r', '--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('PE [urad]')
    ax.grid('on')
    f.suptitle(f'Propagators: {bool(propagators_enabled)}')
elif plot_ae:
    f, axs = plt.subplots(nrows=2)
    ax = axs[0]
    ax.plot(t_gps_sliced-t_gps_sliced[0], 1e6*ae_pred_5ms[ii_sliced:,0], label = 'Az pred')
    ax.plot(t_gps_sliced-t_gps_sliced[0], 1e6*ae_5ms[ii_sliced:,0], label = 'Az true')
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [urad]')
    ax.grid('on')
    ax = axs[1]
    ax.plot(t_gps_sliced-t_gps_sliced[0], 1e6*ae_pred_5ms[ii_sliced:,1], label = 'El pred')
    ax.plot(t_gps_sliced-t_gps_sliced[0], 1e6*ae_5ms[ii_sliced:,1], label = 'El true')
    
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [urad]')
    ax.grid('on')
    f.suptitle(f'Propagators: {bool(propagators_enabled)}')    
plt.show()
