## Script to analyze the seam boundary conditions,
# link availability times
# angular rates when switching target satellites
# CALCULATE true anomaly
# using link parameters computed in the near_polar_leo_states2los.py file
# Date August 11, 2023

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
import astronomy_tools.element_conversion as el_conv
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
files_all = os.listdir(folder_links)
host_index = 0
host = f'_0_{host_index}'
print(f'HOST {host}')
host_label = f'N{host_index+1}'
target_plane = 11
make_txt = 1
make_plot = 0
make_t_overlap = 0
# Load AER
r_lims = [500e3, 4e6]
fnames = [f for f in files_all if host in f and f'_{target_plane}' in f]
path_aer = fr'{folder_links}/{fnames[0]}'
path_ind = fr'{folder_links}/{fnames[1]}'
# Load positions
csv_output_path = r'orbital_simulations\terran_near_polar_split\NearPolar12x244.00h'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
rows_used = None
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = rows_used)
# load states

aer = np.loadtxt(path_aer, dtype = float, delimiter = ',')[:rows_used,:]
t_vec = aer[:,0]
dt = t_vec[1] - t_vec[0]
with open(path_ind, 'r') as j:
    ind_sats = json.load(j)
sat_names = list(ind_sats.keys())
# Choose Target(s)
targets = sat_names
targeind_chosens_plotted = [7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]
# target_labels = ['S1', 'S2', 'S3', 'S4','S5']
target_labels = [f'S{ii+1}' for ii, ind in enumerate(targeind_chosens_plotted)]
if host_index == 2:
    targeind_chosens_plotted = targeind_chosens_plotted[1:]
    target_labels = target_labels[1:]
targets = []
for ind in targeind_chosens_plotted:
    for sat in sat_names:
        if f'{target_plane}_{ind}' in sat:
            targets.append(sat)

sat_names = simulation_parameters['sat_names']
target_names_plane4 = [ii for ii in sat_names if '_4_' in ii]
target_names_plane5 = [ii for ii in sat_names if '_5_' in ii]
target_names_plane6 = [ii for ii in sat_names if '_6_' in ii]
target_names_plane7 = [ii for ii in sat_names if '_7_' in ii]
target_names_plane11 = [ii for ii in sat_names if '_11_' in ii]
host_ind_chosen = f'0_{host_index}'

host_pack = [[ii, name] for ii, name in enumerate(sat_names) if host_ind_chosen in name][0]
host_name = host_pack[1]
host_state_ind = [1+ ii + host_pack[0]*6 for ii in range(6)]

t_j2000 = data_raw[:,0]
s_host = data_raw[:,host_state_ind]

tud_converter = tudatconv.tudat_predictor()
#%%
importlib.reload(el_conv)
kep_states_host, arg_lat_host = el_conv.get_kepler_param(tud_converter, s_host)
arg_lat_host = np.rad2deg(arg_lat_host)%360
## GET TRUE ANOMALY separation from host to target satellites
delta_arg_lat_targets = []
delta_arg_lat_dt_targets = [] # gradients
arg_lat_targets = []
arg_lat_targets_reverse = []
for target_name in targets:
    target_ind = simulation_parameters['state_ind'][sat_names.index(target_name)]
    s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
    kep_states_target, arg_lat_target = el_conv.get_kepler_param(tud_converter, s_target, reverse = 1)
    arg_lat_target = np.rad2deg(arg_lat_target)
    # arg_lat_target = np.rad2deg(arg_lat_target) % 360
    # arg_lat_target_reverse = 180 - arg_lat_target
    # arg_lat_target_reverse = [ii + 360 if ii < 0 else ii for ii in arg_lat_target_reverse]
    delta_arg_lat_target = np.zeros(arg_lat_target.shape)
    # for ii, lat in enumerate(arg_lat_target): # Aug 22: can remove
    #     # lat = 180 + lat
    #     # lat = lat % 360
    #     # if lat>360:
    #     #     while lat>360:
    #     #         lat = lat - 360    
    #     arg_lat_target[ii] = lat

    #     # if lat>360:
    #     #     while lat>360:
    #     #         lat = lat - 360    
    #     # arg_lat_target[ii] = lat
    #     # lat = 180 + lat
    #     # if lat>360:
    #     #     lat = lat - 360
    #     # arg_lat_target[ii] = lat
    # # delta_arg_lat_target = np.abs(arg_lat_host-delta_arg_lat_target)
    d_arg_lat = np.abs(arg_lat_host-arg_lat_target)
    for ii, d_arg in enumerate(d_arg_lat):
        if d_arg > 180:        
            d_arg = np.abs(d_arg - 360)
            d_arg_lat[ii] = d_arg
    d_arg_lat_gradient = np.gradient(d_arg_lat, t_vec) # deg/s
    delta_arg_lat_target = d_arg_lat
    delta_arg_lat_dt_targets.append(d_arg_lat_gradient)
    # delta_arg_lat_target = [d_lat - 360 if d_lat>360 else d_lat for d_lat in delta_arg_lat_target]
    delta_arg_lat_targets.append(delta_arg_lat_target)
    arg_lat_targets.append(arg_lat_target)
    arg_lat_targets_reverse.append(arg_lat_target)

calc_tlines = 1
shift_t_to_t0 = 0
if calc_tlines:
    # Get T lines
    dta_t0, dta_t1, dta_t2, dta_t3 = 10, 20, 22.5, 25
    t_col0 = np.interp(dta_t0, delta_arg_lat_targets[0], t_vec)
    calc_ta_sep = 0
    if calc_ta_sep: # not sure what this does anymore AUG 22
        ta0 = np.interp(t_col0, t_vec, arg_lat_host) # what this?
        ta1, ta2, ta3 = ta0 + 10, ta0+12.5, ta0+15
    t_cols123 = np.interp([ta1, ta2, ta3], arg_lat_host, t_vec)
    
    t_cols = [t_col0, t_cols123[0], t_cols123[1], t_cols123[2]]
else:
    t_cols = [22.090889684933366, 198.78295168610475, 242.95614280070802, 287.1294264945833]
if shift_t_to_t0:
    t0 = 20
    t_cols = [t - t0 for t in t_cols]
    ii_0 = np.where(t_vec > t0)[0][0]
    ii_0 = ii_0 - 1
else:
    ii_0 = 0
    t0 = 0
if make_txt:
    print(f'HOST {host_label}')
    for ii, targ in enumerate(targets):
        print('\n')
        ind_targ = ind_sats[targ]
        ind_targ = ind_sats[targ]
        aer_col0 = aer[:, ind_targ[0]]
        aer_vals = np.interp(t_cols, t_vec, aer_col0)
        for jj, t in enumerate(t_cols):
            print(f't{jj} = {t:.0f} s Target {target_labels[ii]} -> Az = {aer_vals[jj]:.1f} deg')    
#%% PLOTS
shift_t_to_t0 = 0# CONDITIONAL. shift time axis
make_arglat_plot = 1
make_arglat_r_verif_plot = 0
make_ang_sep_plot = 0
make_comb_az_plot = 0
make_comb_ta_r_plot = 0
make_aer_plot = 0
make_single_r_plot = 0
make_single_az_plot = 0
make_r_plot_allorbit = 0
add_t_lines = 1
lim_x = 1

t_lim = 7
y_lim = 40

## TIME LABELS
label_used = ['t0', 't1', 't2', 't3']
label_used = [f'{label} = {label_used[ii-1]} + {t_cols[ii] - t_cols[ii-1]:.0f} s' if ii > 0 else label for ii, label in enumerate(label_used)]

    
t_plotted = (t_vec - t0)/60 
t_plotted =t_plotted[ii_0:]
arg_lat_host = arg_lat_host[ii_0:]
if make_arglat_plot:
    f, ax = plt.subplots()
    ax.plot(t_plotted, arg_lat_host, label= f'{host_label}')
    for ii, target_name in enumerate(targets):
        ax.plot(t_plotted, arg_lat_targets[ii], label= target_labels[ii])

    ax.legend()
    ax.grid('on')

    if lim_x:
        ax.set_xlim([0, 9])
        ax.set_ylim([0, 360])

    ax.set_ylabel('Arg of latitude [deg]')
    ax.set_xlabel('t [min]')

    ## TA Phase plot
elif make_arglat_r_verif_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    ax = axs[0]
    ax.plot(t_plotted, arg_lat_host, label= 'N1')
    # for ii, target_name in enumerate(targets):
    #     ax.plot(t_plotted, delta_arg_lat_targets[ii][ii_0:], label= target_labels[ii], linestyle = '--')
    for ii, target_name in enumerate(targets):
        ax.plot(t_plotted, arg_lat_targets_reverse[ii], label= target_labels[ii])
    # ax.legend()
    ax.grid('on')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('Arg Lat [deg]', fontweight = 'bold')
    ax = axs[1]
    for ii, target in enumerate(targets):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0::,2]/1e3, label= target_labels[ii], linestyle = '--')
    # ax.legend()
    ax.grid('on')
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    ax.legend(loc = 'upper right')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([1500, 5000])
    ax.set_xlabel('t [min]', fontweight = 'bold')
    ax.set_ylabel('Slant Range [km]', fontweight = 'bold')
    # f.suptitle(f'{host_label} - target angular separation')
elif make_ang_sep_plot:

    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    for ii, target_name in enumerate(targets):
        ax.plot(t_plotted, delta_arg_lat_targets[ii][ii_0:], label= target_labels[ii], linestyle = '--')

    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
        ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([0, y_lim])

    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')
    ax.set_xlabel('t [min]', fontweight = 'bold')
    f.suptitle(f'{host_label} - target angular separation')
elif make_comb_ta_r_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    ax = axs[0]
    for ii, target_name in enumerate(targets):
        ax.plot(t_plotted, delta_arg_lat_targets[ii][ii_0:], label= target_labels[ii], linestyle = '--')

    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
        # ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')
    ax = axs[1]
    for ii, target in enumerate(targets):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0::,2]/1e3, label= target_labels[ii], linestyle = '--')
    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    ax.legend(loc = 'upper right')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([1500, 5000])
    ax.set_xlabel('t [min]', fontweight = 'bold')
    ax.set_ylabel('Slant Range [km]', fontweight = 'bold')
    # f.suptitle(f'{host_label} - target angular separation')
elif make_comb_az_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    ax = axs[0]
    for ii, target_name in enumerate(targets):
        ax.plot(t_plotted, delta_arg_lat_targets[ii][ii_0:], label= target_labels[ii], linestyle = '--')

    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
        # ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')

    ax = axs[1]
    for ii, target in enumerate(targets):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,0], label= target_labels[ii], linestyle = '--')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([35, 160])
    ax.legend(loc = 'lower right')
    ax.grid('on')
    ax.set_ylabel('Azimuth [deg]', fontweight = 'bold')
    ax.set_xlabel('t [min]', fontweight = 'bold')
elif make_single_r_plot:
    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    
    for ii, target in enumerate(targets):
        ind_targ = ind_sats[target]
        aer_ii = aer[ii_0:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,2]/1e3, label= target_labels[ii], linestyle = '--')
    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
        ax.legend(loc = 'upper left')
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([1500, 5000])

    ax.set_ylabel('Slant Range [km]')
    ax.set_xlabel('t [min]')
    f.suptitle(f'{host_label} Link Distances to Seam Targets')

elif make_single_az_plot:
    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    
    for ii, target in enumerate(targets):
        ind_targ = ind_sats[target]
        aer_ii = aer[ii_0:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,0], label= target_labels[ii], linestyle = '--')
    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_cols):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj])
        # ax = axs[0]
        ax.legend(loc = 'lower right')
    # ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    if lim_x:
        ax.set_xlim([0, t_lim])
        ax.set_ylim([45, 180])

    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlabel('t [min]')
    f.suptitle(f'{host_label} Azimuth to Seam Targets')    
elif make_aer_plot:
        
    if not make_t_overlap:
        f, axs = None, None
        rows_used = np.where(t_vec > t_lim*60)[0][0]
        rlim = None
        for ii, target in enumerate(targets):
            ind_targ = ind_sats[target]
            aer_ii = aer[:rows_used,ind_targ[:3]]
            if target == targets[-1]:
                rlim = 1
            f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'standard', autolimscale=1, line_type='-', r_lim = rlim, x0 = 0)
        
        ## Add t limits
        if add_t_lines:
            for ii, ax in enumerate(axs):
                for jj, t_col in enumerate(t_cols):
                    ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')                        

            ax = axs[0]
            ax.legend()
        
        ax = axs[1]
        target_legend = [f'N{host_index+1} to {Targ}' for Targ in target_labels]
        ax.legend(target_legend)
        fig_title = f'Northbound N{host_index+1} links to {target_labels[-1]}; {target_labels[-2]}; {target_labels[-3]}'
        f.suptitle(fig_title)
        bplt.autosave(f, subfolder = f'N{host_index+1}terran_links', timetag=0)
        if 0: ## RATES
            f, axs = None, None
            rows_used = np.where(t_vec > t_lim*60)[0][0]
            rlim = None
            for ii, target in enumerate(targets):
                ind_targ = ind_sats[target]
                aer_ii = aer[:rows_used,ind_targ[3:]]
                if target == targets[-1]:
                    rlim = 1
                f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'rate', autolimscale=1, line_type='-', r_lim = rlim)

            ## Add t limits
            if add_t_lines:
                for ii, ax in enumerate(axs):
                    for jj, t_col in enumerate(t_cols):
                        ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
                ax = axs[0]
                ax.legend()
            
            ax = axs[1]
            target_legend = [f'N{host_index+1} to {Targ}' for Targ in target_labels]
            ax.legend(target_legend)
            fig_title = f'Northbound N{host_index+1} rates'
            f.suptitle(fig_title)
            bplt.autosave(f, subfolder = f'N{host_index+1}terran_links', timetag=0)
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
            for jj, t_col in enumerate(t_cols):
                ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')                        
                ax.legend()
        else:
            ax.legend(['Target visible'])
        ax.set_yticks(y_vals, target_labels)
        ax.set_xticks(np.arange(0,11,1))
        ax.set_xlim([0,10])
        ax.grid(axis = 'x')
        ax.set_xlabel('t [min]', fontweight = 'bold')
        ax.set_ylabel('Target Satellite', fontweight = 'bold')
        title = f'Northbound Host N{host_index+1} - Southbound Target satellite visibility.'
        f.suptitle(title)
        bplt.autosave(f, )
elif make_r_plot_allorbit:
    # targeind_chosens_plotted = [7, 6, 5, 4, 3, 2, 1, 0]
    # target_labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    # targets_labels = [f'T{7-ii}' for ii, index in enumerate(targeind_chosens_plotted)]
    f, ax = plt.subplots()
    for ii, target_name in enumerate(target_names_plane11):
        ind_targ = ind_sats[target_name]
        aer_ii = aer[:,ind_targ[:3]]
        # if min(aer_ii[:,2]) > 500e3:
        ax.plot(t_plotted, aer_ii[:,2]/1e3)
    ax.set_ylabel('Slant Range [km]')
    ax.set_xlabel('t [min]')
    ax.set_ylim([500, 4000])
    ax.set_xlim([20, 30])
    fig_title = f'Northbound N{host_index+1}, crossing pole at t=26.5 min'
    f.suptitle(fig_title)