## Script to analyze the seam boundary conditions,
# link availability times
# angular rates when switching target satellites
# using link parameters computed in the near_polar_leo_states2los.py file
# Date August 9, 2023

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
import json
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\terran_near_polar_split\NearPolar12x244.00h'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
save_folder = r'outputs\tables\terran_const'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import astronomy_tools.analytical_tools as astro_calc
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as combplt
import plotting_tools.modular_plotting as modplot
import attitude_tools.terminal_rotations as lct_rot
import pointing_calculations.ae_calculation as ae_calc
import basic_tools.time_conversion as t_conv
import tudat_tools.tudat_converter as tudatconv
from tudat_tools.data_processing.data_saving_utilities import dict2txt
import tudat_tools.data_processing.data_processing_utilities as dputil

folder_links = r'outputs\tables\terran_const'
csv_output_path = r'orbital_simulations\terran_near_polar_split\NearPolar12x244.00h'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

files_all = os.listdir(folder_links)
host_index = 0
host = f'_0_{host_index}'
print(f'HOST {host}')
target_plane = 11
make_txt = 0
make_plot = 1
make_t_overlap = 0
# Load AER
r_lims = [500e3, 3e6]
fnames = [f for f in files_all if host in f and f'_{target_plane}' in f]
path_aer = fr'{folder_links}/{fnames[0]}'
path_ind = fr'{folder_links}/{fnames[1]}'

aer = np.loadtxt(path_aer, dtype = float, delimiter = ',')
t_vec = aer[:,0]
dt = t_vec[1] - t_vec[0]
with open(path_ind, 'r') as j:
    ind_sats = json.load(j)
sat_names = list(ind_sats.keys())
# Choose Target(s)
targets = sat_names
targeind_chosens_plotted = [7, 6, 5, 4]
target_labels = ['T1', 'T2', 'T3', 'T4']
if host_index == 2:
    targeind_chosens_plotted = targeind_chosens_plotted[1:]
    target_labels = target_labels[1:]
targets = []
for ind in targeind_chosens_plotted:
    for sat in sat_names:
        if f'{target_plane}_{ind}' in sat:
            targets.append(sat)
# Choose time
T_orb = astro_calc.calc_period_circular(h = 1050e3)
t_col0 = 1/36*T_orb
ind_col0 = np.where(t_vec > t_col0)[0][0]

t_col1 = 2/36*T_orb
ind_col1 = np.where(t_vec > t_col1)[0][0]

t_col2 = 22.5/360*T_orb
ind_col2 = np.where(t_vec > t_col2)[0][0]

t_col3 = 25/360*T_orb
ind_col3 = np.where(t_vec > t_col3)[0][0]

#%%
importlib.reload(combplt)
if make_txt:
    for ind in [ind_col0,
    ind_col1,
    ind_col2,
    ind_col3]:
        print('\n')
        # Check parameters
        for ii, targ in enumerate(targets):
            ind_targ = ind_sats[targ]
            aer_col0 = aer[ind, ind_targ]
            print(f't = {t_vec[ind]:.0f} s Target {target_labels[ii]} -> AER = {aer_col0[0]:.1f} deg; {aer_col0[1]:.1f} deg; {aer_col0[2]/1e3:.1f} km')
        # LAter - AER <- link windows
if make_plot:
    add_t_lines = 1
    add_rlims = 1
    t_lim = 10 # mins
    targets = []

    for ind in targeind_chosens_plotted:
        for sat in sat_names:
            if f'{target_plane}_{ind}' in sat:
                targets.append(sat)
    if not make_t_overlap:
        f, axs = None, None
        rows_used = np.where(t_vec > t_lim*60)[0][0]
        rlim = None
        for ii, target in enumerate(targets):
            ind_targ = ind_sats[target]
            aer_ii = aer[:rows_used,ind_targ[:3]]
            if target == targets[-1]:
                rlim = 1
            f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'standard', autolimscale=1, line_type='-', r_lim = rlim)

        t_cols = [t_col0, t_col1, t_col2, t_col3]
        ## Add t limits
        if add_t_lines:
            for ii, ax in enumerate(axs):
                for jj, t_col in enumerate(t_cols):
                    ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = ['t0', 't1', 't2', 't3'][jj], linestyle = 'dashdot')
            ax = axs[0]
            ax.legend()
        
        ax = axs[1]
        target_legend = [f'S{host_index+1} to {Targ}' for Targ in target_labels]
        ax.legend(target_legend)
        fig_title = f'Northbound S{host_index+1} links to {target_labels[-1]}; {target_labels[-2]}; {target_labels[-3]}'
        f.suptitle(fig_title)
        bplt.autosave(f, subfolder = f'S{host_index+1}terran_links', timetag=0)
        if 1:
            importlib.reload(combplt)
            f, axs = None, None
            rows_used = np.where(t_vec > t_lim*60)[0][0]
            rlim = None
            for ii, target in enumerate(targets):
                ind_targ = ind_sats[target]
                aer_ii = aer[:rows_used,ind_targ[3:]]
                if target == targets[-1]:
                    rlim = 1
                f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'rate', autolimscale=1.2, force_0 = 0, line_type='-', r_lim = rlim)

            t_cols = [t_col0, t_col1, t_col2, t_col3]
            ## Add t limits
            if add_t_lines:
                for ii, ax in enumerate(axs):
                    for jj, t_col in enumerate(t_cols):
                        ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = ['t0', 't1', 't2', 't3'][jj], linestyle = 'dashdot')
                ax = axs[0]
                ax.legend()
            
            ax = axs[1]
            target_legend = [f'S{host_index+1} to {Targ}' for Targ in target_labels]
            ax.legend(target_legend)
            fig_title = f'Northbound S{host_index+1} rates'
            f.suptitle(fig_title)
            bplt.autosave(f, subfolder = f'S{host_index+1}terran_links', timetag=0)
    else:
        rows_used = np.where(t_vec > t_lim*60)[0][0]
        rlim = None
        f, ax = plt.subplots(figsize = (6,4))
        y_vals = []
        for ii, target in enumerate(targets):
            ind_targ = ind_sats[target]
            aer_ii = aer[:,ind_targ[:3]]
            ii_r_overmin = np.where(aer_ii[:,2] > r_lims[0])[0]
            ii_r_belowmax = np.where(aer_ii[:,2] < r_lims[1])[0]
            ii_both = [ii for ii in ii_r_overmin if ii in ii_r_belowmax]
            t_in = t_vec[ii_both]/60
            y_val = (1 + ii * 0.1)
            ones = np.ones((t_in.shape[0],1)) * y_val
            y_vals.append(y_val)
            ax.scatter(t_in, ones, s = 6, c = 'g')
        ax.set_ylim([0.9,1.5-0.1])
        if add_t_lines and 0:
            t_cols = [t_col0, t_col1, t_col2, t_col3]
            for jj, t_col in enumerate(t_cols):
                ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = ['t0', 't1', 't2', 't3'][jj], linestyle = 'dashdot')                        
                ax.legend()
        else:
            ax.legend(['Target visible'])
        ax.set_yticks(y_vals, target_labels)
        ax.set_xticks(np.arange(0,11,1))
        ax.set_xlim([0,10])
        ax.grid(axis = 'x')
        ax.set_xlabel('t [min]', fontweight = 'bold')
        ax.set_ylabel('Target Satellite', fontweight = 'bold')
        title = f'Northbound Host S{host_index+1} - Southbound Target satellite visibility.'
        f.suptitle(title)
        bplt.autosave(f, )

        




