## Script to visualize the PE

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
load_folder = r'outputs/tables/rocketlab_quatpred'
data_files_all = os.listdir(load_folder)
data_files_pred = [f for f in data_files_all if "quatpred" in f]
pe_max_all = []
rate_all = []
lat_all = []
make_figs = 1
for f_chosen in data_files_pred:
# f_chosen = data_files_pred[0]
    df = pd.read_csv(f'{load_folder}/{f_chosen}')
    lat_ii = int(f_chosen.split('_l')[1].split('_u')[0])
    upd_r_ii = int(f_chosen.split('_u')[1].split('hz')[0])
    label_ii = f'Lat={lat_ii}; Upd={upd_r_ii}Hz'
    pe_ii = df['pe_urad']
    t_from_0 = df['t_s']
    pe_max = np.max(pe_ii)
    if make_figs:
        f, ax = plt.subplots()
        ax.plot(t_from_0, pe_ii)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('PE [urad]')
        ax.set_title(label_ii)
        ax.grid('on')
        plot_verif_quat = 0
        
        bplt.savefig(f, label_ii, save_folder=f'{load_folder}/plots', open_folder=0)
    pe_max_all.append(pe_max)
    rate_all.append(upd_r_ii)
    lat_all.append(lat_ii)
# Visualize
pe_max_all = np.array(pe_max_all)
rate_all = np.array(rate_all)
lat_all = np.array(lat_all)
lat_unique = list(set(lat_all))

data_all = np.zeros((4, len(lat_unique)+1) )
dict_full = {}
col_labels = []
for ii, lat_ii in enumerate(lat_unique):
    ii_for_lat = np.atleast_1d(lat_all== lat_ii).nonzero()
    pe_ii = pe_max_all[ii_for_lat]
    upd_ii = rate_all[ii_for_lat]
    # if ii == 0:
    df = pd.DataFrame.from_dict({
        'Up. rate [Hz]' : upd_ii,
        f'Lat. {lat_ii}' : pe_ii,
    })
    df = df.sort_values(['Up. rate [Hz]'])
    if ii == 0:
        col_labels.append('Up. rate [Hz]')
        data_all[:,0] = df['Up. rate [Hz]'].values
        # df = df.set_index('Up. rate [Hz]')
    data_all[:,ii+1] = df[f'Lat. {lat_ii}'].values
    col_labels.append(f'Lat. {lat_ii}')
    # else:
    #     df2 = pd.DataFrame.from_dict({
    #         'Up. rate [Hz]' : upd_ii,
    #         f'Lat. {lat_ii}' : pe_ii,
    #     })
    #     # df2 = df2.set_index('Up. rate [Hz]')
    #     df2.sort_values(['Up. rate [Hz]'])
    #     df2 = df2.drop(['Up. rate [Hz]'], axis = 1)
    #     print(df2)
    #     df = pd.concat([df, df2], ignore_index=1)
df_overview = pd.DataFrame(data = data_all, columns = col_labels)
df_overview.to_csv(f'{load_folder}/all_pe.csv', index = 0, float_format='%.0f')
print(f'Saved df_overview to {load_folder}')
print(df_overview)
# Get max per case`
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
