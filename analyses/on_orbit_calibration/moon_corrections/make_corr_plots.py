#%% eRROR PLOTS of moon vector for approx and precise functions VS NASA ephemeris 
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
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt

path_main =r'outputs\tables\moon_corrections'
files_av = os.listdir(path_main)
#%%
# files_pe = [file for file in  os.listdir(path_main) if 'pe' in file or 'Err' in file]
files_pe = os.listdir(path_main)
labels_pe = [
    'Const., 5-min',
    'Const., 30-min',
    'Const., 60-min',
    'Lin., 5-min',
    'Lin., 30-min',
    'Lin., 60-min',
    'No Correction'
]

rates = [
        5,
    30,
    60,
    5,
    30,
    60,
    0
]
options = [
    'Constant',
    'Constant',
    'Constant',
    'Linear',
    'Linear',
    'Linear',
    'No Correction'
]
zorders = [1, 0.9, 0.5, 1, 0.9, 0.1]
# colors = ['b', 'r', 'y', '']
plot_x = 'hour'
# plot_x = 'day'
n_days_plotted = 31
n_hours_plotted = 6



# t_fromstart_d = err_placeholder_1hr[:,0]
# t_fromstart_d = (err_placeholder_1hr[:,0] - err_placeholder_1hr[0,0])/86400
# t_fromstart_hr = (t_fromstart_d - t_fromstart_d[0])*24

if plot_x == 'hour':
    xlim = n_hours_plotted
    # t_plotted = t_fromstart_hr
    unit = 'hr'
elif plot_x == 'day':
    xlim = n_days_plotted
    # t_plotted = t_fromstart_d
    unit = 'day'


pe_max = []
pe_3sig = []
upd_rate = []
option = []

for ii in range(7):
    pe  = pd.read_csv(f'{path_main}/{files_pe[ii]}').values[:,1]
    if ii == 6:
        pe = pe * 1e3

    pe_max.append(np.round(np.max(pe),2))
    pe_3sig.append(np.round(np.std(pe)*3,2))
    upd_rate.append(rates[ii])
    option.append(options[ii])

overview_df = pd.DataFrame.from_dict(
    {
        'Update rate [min]' : upd_rate,
        'Correction Option' : options,
        'PE max [mrad]' : pe_max,
        'PE 3-sig. [mrad]' : pe_3sig,
    })
overview_df.to_csv('moon_PE_correction_overview.csv')



if 0:
    f, ax = plt.subplots(1)
    nrows = 500
    for ii in [0,2, 5,6]:
    # ax.plot(t_plotted, err_placeholder_1hr[:,1])

        data  = pd.read_csv(f'{path_main}/{files_pe[ii]}').values[:nrows,:]
        t_gps = data[:,0] - data[0,0]
        pe = data[:,1]
        if ii == 6:
            pe = pe * 1e3
        if plot_x == 'hour':
            t_plotted = t_gps / 3600 
        else:
            t_plotted = t_gps / 3600 / 24
        ax.plot(t_plotted, pe, label = labels_pe[ii])

    ax.set_ylabel(f'PE [mrad]')
    ax.grid('on')
    ax.set_xlim([0, xlim])
    # ax.set_ylim([0,1])
    ax.set_xlabel(f't since start [{unit}]')
    ax.legend()
    ax.grid('on')
    f.set_tight_layout('tight')
    f.suptitle(f'Moon Vector Error with Corrections')
    bplt.autosave(f, subfolder = 'MoonFixing')
else:
    f, axs = plt.subplots(3)
    nrows = 500
    for jj in [1,5]:
        data = pd.read_csv(f'{path_main}/{files_pe[jj]}').values[:nrows,:]
        t_gps = data[:,0] - data[0,0]
        # pe = data[:,1]
        if plot_x == 'hour':
            t_plotted = t_gps / 3600 
        else:
            t_plotted = t_gps / 3600 / 24
        for ii, ax in enumerate(axs):
            ax.plot(t_plotted, data[:,ii+5], label = labels_pe[jj])
                # ax.set_ylabel(f'PE [mrad]')
            ax.grid('on')
            ax.set_xlim([0, xlim])
            # ax.set_ylim([0,1])
            ax.grid('on')
            ax.set_ylabel(['dx', 'dy', 'dz'][ii] + ' corr. [m]')
    ax.set_xlabel(f't since start [{unit}]')
    ax.legend()
    f.set_tight_layout('tight')
    f.suptitle('Visualized Correction Terms')
    bplt.autosave(f, subfolder = 'MoonFixing')