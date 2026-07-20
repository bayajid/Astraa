# Date August 22 AUg
# reproduce Alternating/sequential switching from Terran
# check pointing params/any oversights
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

group_used = 'A'
# group_used = 'B'

# switching = 'seq'
switching = 'alt'

if switching == 'seq':
    # TA trigger
    dta = 20 # deg
    n_cycles = 48
    # how many sats skipped when reconnecting
    nr_sats_skipped = 0
    nr_tagets_labelled = 10
elif switching == 'alt':
    # TA trigger
    dta = 35 # deg
    n_cycles = 24
    nr_sats_skipped = 1
    nr_tagets_labelled = 24

host_index = 0
host = f'_0_{host_index}'
print(f'HOST {host}')

host_label = f'N{host_index+1}'
target_plane = 11
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

target_labels = [f'S{ii+1}' for ii, ind in enumerate(targeind_chosens_plotted)]
target_labels = [label if ii<nr_tagets_labelled else '' for ii, label in enumerate(target_labels)]

if host_index == 2:
    targeind_chosens_plotted = targeind_chosens_plotted[1:]
    target_labels = target_labels[1:]
target_name_base = f'sat_leo_t_{target_plane}'
targets = [f'{target_name_base}_{ind}' for ind in targeind_chosens_plotted]
# for sat in sat_names:
#     for ind in targeind_chosens_plotted:
#         if f'{target_plane}_{ind}' in sat:
#             if sat not in targets:
#                 targets.append(sat)

sat_names = simulation_parameters['sat_names']
host_ind_chosen = f'0_{host_index}'

host_pack = [[ii, name] for ii, name in enumerate(sat_names) if host_ind_chosen in name][0]
host_name = host_pack[1]
host_state_ind = [1+ ii + host_pack[0]*6 for ii in range(6)]

t_j2000 = data_raw[:,0]
s_host = data_raw[:,host_state_ind]

tud_converter = tudatconv.tudat_predictor()
#%%GET TRUE ANOMALY separation from host to target satellites
importlib.reload(el_conv)
kep_states_host, arg_lat_host = el_conv.get_kepler_param(tud_converter, s_host)
arg_lat_host = np.rad2deg(arg_lat_host)%360
delta_arg_lat_targets = []
delta_arg_lat_dt_targets = [] # gradients
arg_lat_targets = []
arg_lat_targets_reverse = []
for target_name in targets:
    target_ind = simulation_parameters['state_ind'][sat_names.index(target_name)]
    s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
    kep_states_target, arg_lat_target = el_conv.get_kepler_param(tud_converter, s_target, reverse = 1)
    arg_lat_target = np.rad2deg(arg_lat_target)
    delta_arg_lat_target = np.zeros(arg_lat_target.shape)
    d_arg_lat = np.abs(arg_lat_host-arg_lat_target)
    for ii, d_arg in enumerate(d_arg_lat):
        if d_arg > 180:        
            d_arg = np.abs(d_arg - 360)
            d_arg_lat[ii] = d_arg
    d_arg_lat_gradient = np.gradient(d_arg_lat, t_vec) # deg/s
    delta_arg_lat_target = d_arg_lat
    delta_arg_lat_dt_targets.append(d_arg_lat_gradient)
    delta_arg_lat_targets.append(delta_arg_lat_target)
    arg_lat_targets.append(arg_lat_target)
    arg_lat_targets_reverse.append(arg_lat_target)

## t LINES STUFF
calc_tlines = 1
shift_t_to_t0 = 0
if calc_tlines:
    # Get T lines
    ii_targets_linked = list(range(host_index, len(targets), 1 + nr_sats_skipped))
    target_labels_plotted = np.array(target_labels)[ii_targets_linked]
    # to slice out time-vector and delta arg for np.interpolate interface
    delta_arg_lat_targets_increasing = []
    t_vecs_delta_arg = []
    # filter to only get growing dTA
    for ii, darglat in enumerate(delta_arg_lat_targets):
        ii_darglat_from0 = np.where(darglat < dta)[0][0]
        ii_darglat_fin = ii_darglat_from0+100
        delta_arg_lat_targets_increasing.append(darglat[ii_darglat_from0:ii_darglat_fin])
        t_vecs_delta_arg.append(t_vec[ii_darglat_from0:ii_darglat_fin])
        # ii_increasing = [jj for jj, dargdt in enumerate(delta_arg_lat_dt_targets[ii]) if dargdt>0]
        # delta_arg_lat_targets_increasing_ii = [darglat[jj] if jj in ii_increasing else 0 for jj, dlat in enumerate(darglat)]
        # delta_arg_lat_targets_increasing.append(delta_arg_lat_targets_increasing_ii)
    t_skips = [np.interp(dta, delta_arg_lat_targets_increasing[ii], t_vecs_delta_arg[ii]) for ii in ii_targets_linked]
    print(f'targets linked : {np.array(target_labels)[ii_targets_linked]}')
    # t_col0 = np.interp(dta, delta_arg_lat_targets[0], t_vec)
    # t_col3 = np.interp(dta, delta_arg_lat_targets[1 + nr_sats_skipped], t_vec)
    delta_arg_lat_targets_linked = np.array(delta_arg_lat_targets)[ii_targets_linked]
    verify_dta_at_eq = 0  # to verify dTA at equator = 10 [not possible with this phasing]
    if verify_dta_at_eq:
        # get U of host at t0:
        u_host_t0 = np.interp(t_col0, t_vec, arg_lat_host)
        print(f'Arg lat at {t_col0} s : {u_host_t0} deg')
        f, ax = plt.subplots()
        ax.plot(t_vec, arg_lat_host, label = 'N1 - TA')
        ax.plot(t_vec, delta_arg_lat_targets[0], label = 'N1 - S1 dTA')
        ax.plot([t_vec[0], t_vec[-1]], [10, 10], c = 'r')
        ax.plot([t_col0, t_col0], [-30,30], c = 'r')
        ax.legend()
        ax.set_xlim([t_lim_0, t_col0+5])
        ax.set_xlabel('sim ref time [s]')
        ax.set_ylabel('angle [deg]')
        ax.set_ylim([-20, 30])
        ax.grid()
        sys.exit()
    # if calc_ta_sep: # not sure what this does anymore AUG 22
    #     ta0 = np.interp(t_col0, t_vec, arg_lat_host) # what this?
    #     ta1, ta2, ta3 = ta0 + 10, ta0+12.5, ta0+15
    # t_cols123 = np.interp([ta1, ta2, ta3], arg_lat_host, t_vec)
    
#     t_cols = [t_col0, t_cols123[0], t_cols123[1], t_cols123[2]]
# else:
#     t_cols = [22.090889684933366, 198.78295168610475, 242.95614280070802, 287.1294264945833]
# if shift_t_to_t0:
#     t0 = 20
#     t_cols = [t - t0 for t in t_cols]
#     ii_0 = np.where(t_vec > t0)[0][0]
#     ii_0 = ii_0 - 1
# else:
#     ii_0 = 0
#     t0 = 0
# if make_txt:
#     print(f'HOST {host_label}')
#     for ii, targ in enumerate(targets):
#         print('\n')
#         ind_targ = ind_sats[targ]
#         ind_targ = ind_sats[targ]
#         aer_col0 = aer[:, ind_targ[0]]
#         aer_vals = np.interp(t_cols, t_vec, aer_col0)
#         for jj, t in enumerate(t_cols):
#             print(f't{jj} = {t:.0f} s Target {target_labels_plotted[ii]} -> Az = {aer_vals[jj]:.1f} deg')    
# if add_t_lines:
#     ## TIME LABELS
#     label_used = ['t0', 't1', 't2', 't3']
#     label_used = [f'{label} = {label_used[ii-1]} + {t_cols[ii] - t_cols[ii-1]:.0f} s' if ii > 0 else label for ii, label in enumerate(label_used)]
#%% PLOTS
shift_t_to_t0 = 0# CONDITIONAL. shift time axis
make_arglat_plot = 0
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
t_lim_0 = 0
t_lim = t_lim_0 + 30
y_lim = 40
# az_lims = [45, 180]
az_lims = [0, 90]

nr_tagets_plotted = 24
targets_potted = np.array(targets)[ii_targets_linked]
targets_potted = targets_potted[:nr_tagets_plotted]
# Shift plot to t0 [Aug22- outdated, doesnt work, not needed]
if not add_t_lines:
    ii_0 = 0
    t0 = 0
else:
    t0 = 0
    ii_0 = 0
t_plotted = (t_vec - t0)/60 
t_plotted =t_plotted[ii_0:]
arg_lat_host = arg_lat_host[ii_0:]
if make_arglat_plot:
    f, ax = plt.subplots()
    ax.plot(t_plotted, arg_lat_host, label= f'{host_label}')
    for ii, target_name in enumerate(targets_potted):
        ax.plot(t_plotted, arg_lat_targets[ii], label= target_labels_plotted[ii])

    ax.legend()
    ax.grid('on')

    if lim_x:
        ax.set_xlim([t_lim_0, 9])
        ax.set_ylim([0, 360])

    ax.set_ylabel('Arg of latitude [deg]')
    ax.set_xlabel('t [min]')

    ## TA Phase plot
elif make_arglat_r_verif_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    ax = axs[0]
    ax.plot(t_plotted, arg_lat_host, label= 'S1')
    # for ii, target_name in enumerate(targets_potted):
    #     ax.plot(t_plotted, delta_arg_lat_targets_connected[ii][ii_0:], label= target_labels_plotted[ii], linestyle = '--')
    for ii, target_name in enumerate(targets_potted):
        ax.plot(t_plotted, arg_lat_targets_reverse[ii], label= target_labels_plotted[ii])
    # ax.legend()
    ax.grid('on')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('Arg Lat [deg]', fontweight = 'bold')
    ax = axs[1]
    for ii, target in enumerate(targets_potted):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0::,2]/1e3, label= target_labels_plotted[ii], linestyle = '--')
    # ax.legend()
    ax.grid('on')
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    ax.legend(loc = 'upper right')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([1500, 5000])
    ax.set_xlabel('t [min]', fontweight = 'bold')
    ax.set_ylabel('Slant Range [km]', fontweight = 'bold')
    # f.suptitle(f'{host_label} - target angular separation')
elif make_ang_sep_plot:

    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    for ii, target_name in enumerate(targets_potted):
        ax.plot(t_plotted, delta_arg_lat_targets_linked[ii][ii_0:], label= target_labels_plotted[ii], linestyle = '--')

    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
            # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
        ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([0, y_lim])

    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')
    ax.set_xlabel('t [min]', fontweight = 'bold')
    f.suptitle(f'{host_label} - target angular separation')
elif make_comb_ta_r_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    ax = axs[0]
    for ii, target_name in enumerate(targets_potted):
        ax.plot(t_plotted, delta_arg_lat_targets_linked[ii][ii_0:], label= target_labels_plotted[ii], linestyle = '--')
        
        
    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
        # ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')
    ax.legend(loc = 'upper right')

    ax = axs[1]
    for ii, target in enumerate(targets_potted):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0::,2]/1e3, label= target_labels_plotted[ii], linestyle = '--')
    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    ax.plot([t_plotted[0], t_plotted[-1]], [0.5e3, 0.5e3], 'r--', label = 'R max')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([250, 5000])
    ax.set_xlabel('t [min]', fontweight = 'bold')
    ax.set_ylabel('Slant Range [km]', fontweight = 'bold')
    f.suptitle(f'{host_label} - Switching Mode: {switching}')
elif make_comb_az_plot:
    f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    ax = axs[0]
    for ii, target_name in enumerate(targets_potted):
        ax.plot(t_plotted, delta_arg_lat_targets_linked[ii][ii_0:], label= target_labels_plotted[ii], linestyle = '--')

    # ax.legend()
    ax.grid('on')
    if add_t_lines:
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
        # ax.legend(loc = 'upper left')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([0, y_lim])
    ax.set_ylabel('TA separation [deg]', fontweight = 'bold')

    ax = axs[1]
    for ii, target in enumerate(targets_potted):
        ind_targ = ind_sats[target]
        aer_ii = aer[:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,0], label= target_labels_plotted[ii], linestyle = '--')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([35, 160])
    ax.legend(loc = 'lower right')
    ax.grid('on')
    ax.set_ylabel('Azimuth [deg]', fontweight = 'bold')
    ax.set_xlabel('t [min]', fontweight = 'bold')
elif make_single_r_plot:
    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    
    for ii, target in enumerate(targets_potted):
        ind_targ = ind_sats[target]
        aer_ii = aer[ii_0:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,2]/1e3, label= target_labels_plotted[ii], linestyle = '--')
    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
        ax.legend(loc = 'upper left')
    ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim([1500, 5000])

    ax.set_ylabel('Slant Range [km]')
    ax.set_xlabel('t [min]')
    f.suptitle(f'{host_label} Link Distances to Seam Targets')

elif make_single_az_plot:
    f, ax = plt.subplots()
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
    
    for ii, target in enumerate(targets_potted):
        ind_targ = ind_sats[target]
        aer_ii = aer[ii_0:rows_used,ind_targ[:3]]
            
        ax.plot(t_plotted, aer_ii[ii_0:,0], label= target_labels_plotted[ii], linestyle = '--')
    ax.legend()
    ax.grid('on')
    if add_t_lines:
        # for ii, ax in enumerate(axs):
        for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
            ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
        # ax = axs[0]
        ax.legend(loc = 'lower right')
    # ax.plot([t_plotted[0], t_plotted[-1]], [4e3, 4e3], 'r--', label = 'R max')
    if lim_x:
        ax.set_xlim([t_lim_0, t_lim])
        ax.set_ylim(az_lims)

    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlabel('t [min]')
    f.suptitle(f'{host_label} Azimuth to Seam Targets, Switching - {switching}')    
elif make_aer_plot:
        
    if not make_t_overlap:
        f, axs = None, None
        rows_used = np.where(t_vec > t_lim*60)[0][0]
        rlim = None
        for ii, link_nr in enumerate(ii_targets_linked):
            target = targets[link_nr]
            ind_targ = ind_sats[target]
            aer_ii = aer[:rows_used,ind_targ[:3]]
            if target == targets[-1]:
                rlim = 1
            f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'standard', autolimscale=1, line_type='-', r_lim = rlim, x0 = 0)
        
        ## Add t limits
        if add_t_lines:
            for ii, ax in enumerate(axs):
                for jj, t_col in enumerate(t_skips):
                # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
                    ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')

            ax = axs[0]
            ax.legend()
        
        ax = axs[1]
        target_legend = [f'N{host_index+1} to {Targ}' for Targ in target_labels_plotted]
        ax.legend(target_legend)
        fig_title = f'Northbound N{host_index+1} links to {target_labels[-1]}; {target_labels[-2]}; {target_labels[-3]}'
        f.suptitle(fig_title)
        bplt.autosave(f, subfolder = f'N{host_index+1}terran_links', timetag=0)
        if 0: ## RATES
            f, axs = None, None
            rows_used = np.where(t_vec > t_lim*60)[0][0]
            rlim = None
            for ii, target in enumerate(targets_potted):
                ind_targ = ind_sats[target]
                aer_ii = aer[:rows_used,ind_targ[3:]]
                if target == targets[-1]:
                    rlim = 1
                f, axs = combplt.plot_aer(t_vec[:rows_used], aer_ii, f = f, axs = axs, setting = 'rate', autolimscale=1, line_type='-', r_lim = rlim)

            ## Add t limits
            if add_t_lines:
                for ii, ax in enumerate(axs):
                    for jj, t_col in enumerate(t_skips):
        # ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), label = label_used[jj], linestyle = 'dashdot')
                        ax.plot([t_col/60, t_col/60], list(ax.get_ylim()), 'y-')
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
        for ii, link_nr in enumerate(ii_targets_linked):
            target = targets[link_nr]
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
        ax.set_xlim([t_lim_0,10])
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