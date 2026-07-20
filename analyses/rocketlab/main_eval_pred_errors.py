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
import analyses.attitude_predictions.attitude_prediction_utlities as att_pred
## Load input data
quat_path = r'outputs/tables/rocketlab_quatpred/true_quatrocketlab_march.csv'
save_folder = r'outputs/tables/rocketlab_quatpred'
## Interpolate to 5 ms
data_df = pd.read_csv(quat_path)  
propagators_enabled = 1
dt_req = 25e-3 # s, 50ms

update_rates = [1, 2, 5, 10] # Hz
latencies = [0, 1, 2, 3, 4]
#update_rates = [1]
latencies = latencies[:3]

update_intervals = [np.round(1/ii,1) for ii in update_rates]
data_sliced = data_df.values
q_bf_used = data_sliced[:,1:]
t_sliced = data_sliced[:,0] 
# t_for_interp = t_sliced[(t_sliced >= 10) * (t_sliced <= t_sliced[-1]-10)]
t_for_interp = t_sliced # Time for prediction window
n_digits = 3
t_req = np.arange(t_sliced[0], t_sliced[-1]+dt_req, dt_req)
t_req = np.round(t_req, n_digits)
t_gps_interp = CubicSpline(t_sliced, data_sliced[:,0], axis = 0)
q_host_interp = CubicSpline(t_sliced, q_bf_used, axis = 0)

# placeholder lists for indexing/labelling outputs
pe_calc = []
quat_pred = []
quat_true = []
latency_used = []
update_rate_used = []
for nn, update_freq in enumerate(update_rates):
    for mm, latency_selected in enumerate(latencies):
        update_freq_att_h = update_freq
        dt_gap_att_h = np.round(1/update_freq_att_h,3)
        dt_latency = dt_gap_att_h  * latency_selected
        t_update_arrival = np.round(np.arange(t_for_interp[0], t_for_interp[-1]+dt_gap_att_h, dt_gap_att_h),3)
        ## Adding LATENCY- receive attitude data after delay
        t_gps_interp_5ms = t_gps_interp(t_req)
        
        # get true data
        q_host_true_5ms = q_host_interp(t_req)
        t_gps_pred_5ms = np.zeros(t_gps_interp_5ms.shape)
        q_host_pred_5ms = np.zeros(q_host_true_5ms.shape)
        t_stamps_updates = t_update_arrival - dt_latency
        data_full_att_h = q_host_interp(t_stamps_updates)

        ii_next_att_h = 0
        quat_interp = interp.we_interpolating()
        for ii, t_ii in enumerate(t_gps_interp_5ms):
            if t_ii >= t_update_arrival[ii_next_att_h] and t_ii < t_update_arrival[-1]:
                data_att_h = data_full_att_h[ii_next_att_h]
                data_att_h_held = data_att_h # Propagator-off case
                # UPDATE INTERPOLANT
                if ii_next_att_h >= 1:
                    quat_interp.get_quad_interpolant(
                        t_both=t_stamps_updates[ii_next_att_h-1:ii_next_att_h+1],
                        r_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,:4],
                        v_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,4:],                
                    )
                    a=1
                ii_next_att_h += 1
            else:
                data_att_h = None
            if ii_next_att_h >=2:
                if propagators_enabled:
                    data_att_interp = quat_interp.interpolate_flexible(t_ii)
                else:
                    data_att_interp = data_att_h_held[:4]
                quat_diff = data_att_interp - q_host_true_5ms[ii,:4]
                a = 1
            else:
                data_att_interp = [0,0,0,0]

            # store
            q_host_pred_5ms[ii,:4] = data_att_interp
            t_gps_pred_5ms[ii] = t_ii

        pe_remaining = [att_pred.eval_pred_error(q_pred_ii, q_true_ii)[1] 
                        for q_pred_ii, q_true_ii in zip(q_host_true_5ms[:,:4], q_host_pred_5ms[:,:4])]
        pe_calc.append(pe_remaining)
        quat_pred.append(q_host_pred_5ms)
        latency_used.append(latency_selected)
        update_rate_used.append(update_freq)
        title_save = f'quatpred_l{latency_selected}_u{update_freq}hz.csv'
        save_path = f'{save_folder}/{title_save}'
        result_df = pd.DataFrame.from_dict({
            't_s' : t_gps_interp_5ms,
            'pe_urad' : pe_remaining,
            'q_pred_c' : q_host_pred_5ms[:,0],
            'q_pred_1' : q_host_pred_5ms[:,1],
            'q_pred_2' : q_host_pred_5ms[:,2],
            'q_pred_3' : q_host_pred_5ms[:,3],
        })
        result_df.to_csv(save_path)
        print(f'Saved {title_save}')
# Verify
# print(pe_remaining)
# Plot

plot_verif_quat = 0
if plot_verif_quat:
    f, ax = plt.subplots()
    t_from_0 = t_req - t_req[0]
    ii_q = 0
    ax.plot(t_stamps_updates, data_full_att_h[:,ii_q], marker='o', label = 'Updates', markersize = 10, linestyle = 'None')
    ax.plot(t_from_0, q_host_true_5ms[:,ii_q], label = 'True', markersize = 6)
    for ii, pe_ii in enumerate(pe_calc):
        label = f'Lat{latency_used[ii]};Upd: {1/update_rate_used[ii]:.1f} s'
        ax.plot(t_from_0, quat_pred[ii][:,ii_q], linestyle = '-', label = label, marker = 'o', markersize = 2)
    ax.legend()
    ax.grid('on')
    ax.set_ylabel('QC [-]')
    ax.set_xlabel('t [s]')
    plt.show()    
# Visualize

# Get max per case
    
if 0:
    # Evaluate Pointing Errors
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
