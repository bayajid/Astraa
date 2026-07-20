#%% Plotting customer provided attitude data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import scipy.io
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import attitude_tools.attitude_predictions.attitutde_plot_functions as attplt
import plotting_tools.basic_plotting as bplt

data_folder = r'attitude_tools\attitude_predictions\cust_data'
fname_att = 'BusAttEst'
fname_atterr = 'BusRateEst'
fname_quat = 'q_SVtoECI'

plot_zoom = 0
plot_whole = 0
data_att_lst = []
data_rates_lst = []
data_quat_lst = []

update_rates = [1, 5, 10]

for update_rate in update_rates:
    fname_att_upd = f'{fname_att}_{update_rate}Hz.mat'
    fname_atterr_upd = f'{fname_atterr}_{update_rate}Hz.mat'
    fname_quat_upd = f'{fname_quat}_{update_rate}Hz.mat'

    data_att = scipy.io.loadmat(f'{data_folder}\{fname_att_upd}')[fname_att_upd[:-4]][0][0][0]
    data_atterr = scipy.io.loadmat(f'{data_folder}\{fname_atterr_upd}')[fname_atterr_upd[:-4]][0][0][0]
    data_quat = scipy.io.loadmat(f'{data_folder}\{fname_quat_upd}')[fname_quat_upd[:-4]][0][0][0]

    data_att_lst.append(data_att)
    data_rates_lst.append(data_atterr)
    data_quat_lst.append(data_quat)


data_comb = [
    data_att_lst,
    data_rates_lst,
    data_quat_lst
]
data_labels = ['Customer Attitude', 'Customer Attitude Rate', 'Customer Quaternion']

markers =['.','+', 'o']
alpha = [1, 1, 1]
size = [3, 20, 50]
colors = ['r', 'm', 'g']
markers.reverse()
alpha.reverse()
size.reverse()
colors.reverse()
nrows = [3, 3, 4]
ylabels = [
    ['Roll [mdeg]',
     'Pitch [mdeg]',
     'Yaw [mdeg]'
     ],
     ['Roll rate [deg/s]',
     'Pitch rate [deg/s]',
     'Yaw rate [deg/s]'
     ],
     ['q_1',
     'q_2',
     'q_3',
     'q_4'
     ]

]
ylims = [5, 0.005, 1.1]
if plot_zoom:
    for kk in range(3): # Data index
        data_used = data_comb[kk]
        f, axs = plt.subplots(nrows = nrows[kk], figsize = (7,8))
        for jj, upd_rate in enumerate(update_rates): # jj - update freq index
            data_rate = data_used[jj]
            for ii, ax in enumerate(axs): # ii - row/data index
                if jj == 2:
                    ax.plot(data_rate[:,0], data_rate[:,ii+1], 
                        color = colors[jj],
                        alpha = 0.2,
                        label = f'{upd_rate} Hz')
                else:
                    ax.scatter(data_rate[:,0], data_rate[:,ii+1], 
                            marker = markers[jj], 
                            s = size[jj], alpha = alpha[jj],
                            color = colors[jj],
                            label = f'{upd_rate} Hz')
                ax.set_xlim([0,10])
                ax.grid()
                ax.set_ylabel(ylabels[kk][ii])
                ax.set_ylim(-ylims[kk], ylims[kk])
        ax.set_xlabel('t [s]')
        ax.legend()
        f.set_tight_layout('tight')
        bplt.savefig(f, f'{data_labels[kk]}_allHz_zoomed')
if plot_whole:
    for kk in range(3): # Data index
        data_used = data_comb[kk]
        f, axs = plt.subplots(nrows = nrows[kk], figsize = (7,8))
        for jj, upd_rate in enumerate(update_rates[:1]): # jj - update freq index
            data_rate = data_used[jj]
            for ii, ax in enumerate(axs): # ii - row/data index
                if jj == 2:
                    ax.plot(data_rate[:,0], data_rate[:,ii+1], 
                        color = colors[jj],
                        alpha = 0.2,
                        label = f'{upd_rate} Hz')
                else:
                    ax.scatter(data_rate[:,0], data_rate[:,ii+1], 
                            marker = markers[jj], 
                            s = 1, alpha = alpha[jj],
                            color = 'b',
                            label = f'{upd_rate} Hz')
                # ax.set_xlim([0,10])
                ax.grid()
                ax.set_ylabel(ylabels[kk][ii])
                # ax.set_ylim(-ylims[kk], ylims[kk])
        ax.set_xlabel('t [s]')
        # ax.legend()
        f.set_tight_layout('tight')
        bplt.savefig(f, f'{data_labels[kk]}_allHz_zoomed')