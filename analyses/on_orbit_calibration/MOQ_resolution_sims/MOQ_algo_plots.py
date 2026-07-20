#%% March 25, 2024, verifying QUEST implementation with simulations
# repurposed from Phase B sims 
import scipy as sp
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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io
import plotting_tools.basic_plotting as bplt
import plotting_tools.plotting_utilities as plt_util

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil

importlib.reload(moon_scan)
importlib.reload(att_res)



name_filter_forall = str(50000)
name_filter_spec = 'svd'
name_filter_spec = 'triad'
storage_folder = fr'outputs\tables\ooc_mocalc_algo'

sig_chosen = 'pe_3sig'

x_data = []
y_data = []
labels = []
data_all = os.listdir(storage_folder)

dat_filt_all = [ii for ii in data_all if name_filter_forall in ii]

filt_chosen = 'svd'
dat_filt_ii = [ii for ii in dat_filt_all if filt_chosen in ii.lower()]
df_loaded = pd.read_csv(fr'{storage_folder}/{dat_filt_ii[0]}')
x_ii = df_loaded['ang_sep'].values
y_ii = df_loaded[sig_chosen].values
labels.append(filt_chosen)
x_data.append(x_ii)
y_data.append(y_ii)

filt_chosen = 'triad'
dat_filt_ii = [ii for ii in dat_filt_all if filt_chosen in ii.lower()]
df_loaded = pd.read_csv(fr'{storage_folder}/{dat_filt_ii[0]}')
x_ii = df_loaded['ang_sep'].values
y_ii = df_loaded[sig_chosen].values
labels.append(filt_chosen)
x_data.append(x_ii)
y_data.append(y_ii)


importlib.reload(plt_util)
f, ax = plt.subplots()

for ii, label in enumerate(labels):
    plot_all = 1
    plot_sep = 0
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    ax.plot(x_data[ii], y_data[ii], label = labels[ii])
    ax.set_xlabel('Angle between scanned LOS [deg]', fontweight = 'bold')
    ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
    f.suptitle(f'OOC-A : {sig_chosen}')
    ax.grid('on')
    ax.set_xlim([10, 170])
    ax.set_ylim([-0.1,10])
    ax.legend()
    # Reformat to get angle from 0:90 for non-colinearity

plt.show()