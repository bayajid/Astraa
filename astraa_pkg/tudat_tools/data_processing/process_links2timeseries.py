#%% 
## Script used to process intermediate outputs (links, Az/El/Slant range angles, etc) into
# the final outputs (link statistics) in the forms of tables, histograms, etc
# desired outputs- histograms of link windows, max angles, max angular rates
# per different host sat, link cases, terminal placements, terminal type
# and imposed terminal limitations (at minimum, considering visiblity)
# at maximum, adding AE limitations, slant range limitations
import pathlib
import json
import csv
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import loading_functions.data_loading as load
import data_processing.rotations as rot
import data_processing.data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
import basic_tools.operations as basic
import basic_tools.link_cases as cases
import matplotlib.backends.backend_pdf
from matplotlib.offsetbox import AnchoredText
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)
## base paths
pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")

sat_hosts = [pos[:-23] for pos in os.listdir(f'{output_path_link}\\mk3') if 'meo_lct1_linkrates' in pos]

## Inputs


run_verification_test = 0
# def plot_timeseries(
#     sat_host,
#     case_analyzed, 
#     lct_chosen,
#     link_type = 'general',
#     output_path_link = None,
#     t_length = 2.5,
#     t_start = 0,
#     cols_used = ['az_h', 'daz_h', 'el_h', 'del_h'],
#     link_category_chosen = 'leadfoll',
#     data_type_chosen = 'linkrates',
#     ii = 0,
#     sat_target_limits = '_4_6',
#     labels_used = ['Azimuth [deg]', 'Az. rate [deg/s]', 'Elevation [deg]', 'El. rate [deg/s]'],
#     legend = 1,
#     title = None,
#     ):
#     if type(output_path_link) == type(None):
#         output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
#     data_folder = f'{output_path_link}\{link_type}'
#     data_all = os.listdir(data_folder)
#     data_available_host = [file for file in data_all if sat_host in file]
#     data_available_lct = [file for file in data_available_host if lct_chosen in file]
#     data_chosen = [file for file in data_available_lct if link_category_chosen in file.replace(sat_host,'')]
#     # print(data_chosen)
#     data_name = [file for file in data_chosen if data_type_chosen in file][0]
#     data_csv = pd.read_csv(f'{data_folder}/{data_name}')

#     nrows, ncols = 2,2
#     t_end = t_start*3600 + t_length* 3600
#     data_cut = data_csv[data_csv['t'] < t_end]
#     # plot glossary
#     title = f'Coplanar LEO I {lct_chosen} to leader' if type(title) == type(None) else title

#     fig, axs = plt.subplots(nrows, ncols, figsize = (nrows * 5, ncols * 3))
#     for sat_target in case_analyzed:
#         if type(sat_target_limits) != type(None):   
#             if sat_target_limits in sat_target:
#                 df_used = data_cut[data_cut['sat_target'] == sat_target]
#                 if not df_used.empty:
#                     t_vec = df_used['t']/60
#                     for ii, col_needed in enumerate(cols_used):
#                         data_to_plot = df_used[col_needed]
#                         jj = ii%2 # column index
#                         if ii>1:
#                             nr = 1 # row index
#                         else:
#                             nr = 0
#                         ax = axs[nr, jj]
#                         ax.grid()
#                         ax.scatter(t_vec, data_to_plot, s = 2, label = sat_target)
#                         ax.set_ylabel(labels_used[ii], weight = 'bold')
#                         if nr==1:
#                             ax.set_xlabel('t [min]', weight = 'bold')
#     for ii in range(4):
#         jj = ii%2 # column index
#         nr = 1 if ii>1 else 0
#     if legend:
#         axs[jj,nr].legend()
#     plt.suptitle(title, size = 14, weight = 'bold')
#     plt.tight_layout()
#     plt.show()
#     return fig, axs
def get_label_names(case):
    # function to convert satellite names to cleaner labels
    # input lsit of sat names
    names_nice = [name.replace('sat_leo_incl_', 'LEO_I_') for name in case]
    if case == names_nice:
        names_nice = [name.replace('sat_leo_polar_', 'LEO_P_') for name in case]
    if case == names_nice:
        names_nice = [name.replace('sat_meo_0', 'MEO') for name in case]
    return names_nice

def loadplot_timeseries(
    sat_host,
    case_analyzed, 
    lct_chosen,
    link_type = 'general',
    output_path_link = None,
    t_length = 8,
    t_start = 0,
    cols_used = ['r_h', 'dr_h', 'az_h', 'daz_h', 'el_h', 'del_h'],
    link_category_chosen = 'leadfoll',
    data_type_chosen = 'linkrates',
    ii = 0,
    sat_target_limits = '_4_5',
    slant_limits = [0, 8000],
    sat_target_exclusion = None,
    labels_used = ['Slant range [km]', 'Slant rate [km/s]', 'Azimuth [deg]', 'Az. rate [deg/s]', 'Elevation [deg]', 'El. rate [deg/s]'],
    legend = 1,
    title = None,
    grid = 0,
    ):
    # Function to load a data file based on the provided folders
    # (output path link, link_type)
    # then choose the data based on sat_host and link-category_chosen
    # Proceed to plot the specified csv columns as a timeseries
    # with t given in [hr]
    if type(output_path_link) == type(None):
        output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
    data_folder = f'{output_path_link}\{link_type}'
    data_all = os.listdir(data_folder)
    data_available_host = [file for file in data_all if sat_host in file]
    data_available_lct = [file for file in data_available_host if lct_chosen in file]
    data_chosen = [file for file in data_available_lct if link_category_chosen in file.replace(sat_host,'')]
    # print(data_chosen)
    data_name = [file for file in data_chosen if data_type_chosen in file][0]
    print(f'Data file chosen: {data_name}')
    data_csv = pd.read_csv(f'{data_folder}/{data_name}')

    nrows, ncols = 3,2
    t_end = t_start*3600 + t_length* 3600
    data_cut = data_csv[data_csv['t'] < t_end]
    # plot glossary
    title = f'Coplanar LEO I {lct_chosen} to leader' if type(title) == type(None) else title
    fig, axs = plt.subplots(nrows, ncols, figsize = (nrows * 5, ncols * 3))
    label_sats = get_label_names(case_analyzed) # labels to be used for plots
    for mm, sat_target in enumerate(case_analyzed):
        if type(sat_target_limits) != type(None):   
            if sat_target_limits in sat_target:
                if type(sat_target_exclusion) != type(None):
                    if sat_target_exclusion in sat_target:
                        continue
                df_used = data_cut[data_cut['sat_target'] == sat_target]
                if not df_used.empty:
                    t_vec = df_used['t']/60
                    for ii, col_needed in enumerate(cols_used):
                        data_to_plot = df_used[col_needed]
                        # slant [m] convert to [km]
                        if 'km' in labels_used[ii]: 
                            data_to_plot = data_to_plot / 1e3 
                        # track indices
                        jj = ii%ncols # column index
                        if ii>1:
                            nr = 1 # row index
                        else:
                            nr = 0
                        if ii>3:
                            nr = 2 
                        ax = axs[nr, jj]
                        if grid:
                            ax.grid()
                        ax.scatter(t_vec, data_to_plot, s = 2, marker = '_', label = label_sats[mm])
                        ax.set_ylabel(labels_used[ii], weight = 'bold')
                        ax.set_xlim([0, t_end/60 + 20])
                        if nr==2:
                            ax.set_xlabel('t [min]', weight = 'bold')
                        if nr == 0:
                            if jj == 0:
                                ax.set_yticks(np.arange(slant_limits[0],slant_limits[1] + 2000,2000))
                        if nr == 1:
                            if jj == 0:
                                ax.set_yticks(np.arange(-180,180 +45, 45))
    for ii in range(6):
        jj = ii%2 # column index
        nr = 2 if ii>1 else 0
    if legend:
        # axs[nr, jj-1].legend(markerscale=5)
        axs[nr, jj].legend(markerscale=5)
    plt.suptitle(title, size = 14, weight = 'bold')
    plt.tight_layout()
    plt.show()
    return fig, axs
def get_cases(sat_host):
    # function to return the cases (lists of target satellites) and names (referring to csv names)    
    if sat_host == 'sat_leo_incl_4_4':
        cases_chosen = cases.cases_leo_i
        names_chosen = cases.names_leo_i
    elif sat_host == 'sat_leo_polar_4_4':
        cases_chosen = cases.cases_leo_p
        names_chosen = cases.names_leo_p
    elif sat_host ==  'sat_meo_0_0':
        cases_chosen = cases.cases_meoleo
        names_chosen = cases.names_meoleo
    return cases_chosen, names_chosen
save_folder = f'plots_results//time_series'
#%% Make the plots LEO I LCT2 -leader
sat_host = sat_hosts[0]

ind_case = 0
cases_chosen, names_chosen = get_cases(sat_host)
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
terminal_option = 'general'
title = 'LEO I - coplanar links'
fig_li, ax_li = loadplot_timeseries(sat_host, 
case_analyzed, 
lct_chosen = 'lct2',
sat_target_limits='',
legend = 1,
t_length = 2.5,
title = title)
for ax in ax_li:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
title = title.replace(' ', '_')
title = title.replace('-', '_')
bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)        

#%% LEO P - LCT2 - Leader
sat_host = sat_hosts[1]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 0
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
terminal_option = 'general'
title = 'LEO P - coplanar links'
fig_li, ax_li = loadplot_timeseries(sat_host, 
case_analyzed, 
title = title,
t_length =2.5,
sat_target_limits='',
lct_chosen = 'lct2',
legend = 1)
for ax in ax_li:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
title = title.replace(' ', '_')
title = title.replace('-', '_')
bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% MEO - LCT2 - Leader
sat_host = sat_hosts[2]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = -1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
terminal_option = 'general'
title = 'MEO - coplanar links'
fig_li, ax_li = loadplot_timeseries(sat_host, 
case_analyzed,  
title = title,
slant_limits=[12000, 28000],
lct_chosen = 'lct2',
sat_target_limits = '',
link_category_chosen = name_analyzed,
legend = 1)
for ax in ax_li:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
title = title.replace(' ', '_')
title = title.replace('-', '_')
bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% LEO I - crossplane all
cross_planes_plotted = [5, 8, 7, 11]
sat_host = sat_hosts[0]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
for plane_index in cross_planes_plotted:
    if plane_index < 6:
        theta_step = 1
    elif plane_index < 11:
        theta_step = 1
    else:
        theta_step = 1  
    case_analyzed = [f'{sat_host[:-4]}_{plane_index}_{ii}' for ii in range(0,14,theta_step)]
    title = f'LEO I non-coplanar plane 4- LEO I plane {plane_index}'
    fig_li, ax_li = loadplot_timeseries(sat_host, 
        case_analyzed, 
        title = title,
        lct_chosen = lct_chosen,
        sat_target_limits = '',
        t_length= 2.5,
        link_category_chosen = name_analyzed,
        legend = 1)
    for ax in ax_li:
        for a in ax:
            a.grid(visible=True, which='major', color='gray', linestyle='-')
    title = title.replace(' ', '_')
    title = title.replace('-', '_')
    bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% LEO P - crossplane  all
sat_host = sat_hosts[1]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'LEO P - non-coplanar to LEO P'
for plane_index in cross_planes_plotted:
    if plane_index < 6:
        theta_step = 1
    else:
        theta_step = 2    
    case_analyzed = [f'{sat_host[:-4]}_{plane_index}_{ii}' for ii in range(0,14,theta_step)]
    title = f'LEO P non-coplanar plane 4- LEO P plane {plane_index}'
    fig_li, ax_li = loadplot_timeseries(sat_host, 
        case_analyzed, 
        title = title,
        lct_chosen = lct_chosen,
        sat_target_limits = '',
        t_length= 2.5,
        link_category_chosen = name_analyzed,
        legend = 1)
    for ax in ax_li:
        for a in ax:
            a.grid(visible=True, which='major', color='gray', linestyle='-')
    title = title.replace(' ', '_')
    title = title.replace('-', '_')
    bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
# fig_li, ax_li = loadplot_timeseries(sat_host, 
# case_analyzed, # TODO change
# title = title,
# lct_chosen = lct_chosen,
# sat_target_limits = '_5_',
# link_category_chosen = name_analyzed,
# legend = 1)
# for ax in ax_li:
#     for a in ax:
#         a.grid(visible=True, which='major', color='gray', linestyle='-')
# bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% LEO I - LEO P
sat_host = sat_hosts[0]
cross_planes_plotted = [4, 5, 8, 11]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case+1]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'LEO I - non-coplanar to LEO P'
for plane_index in cross_planes_plotted:
    if plane_index < 6:
        theta_step = 1
    else:
        theta_step = 2
    case_analyzed = [f'sat_leo_polar_{plane_index}_{ii}' for ii in range(0,14,theta_step)]
    title = f'LEO I non-coplanar plane 4- LEO P plane {plane_index}'
    fig_li, ax_li = loadplot_timeseries(sat_host, 
        case_analyzed, 
        title = title,
        lct_chosen = lct_chosen,
        sat_target_limits = '',
        t_length= 2.5,
        link_category_chosen = name_analyzed,
        legend = 1)
    for ax in ax_li:
        for a in ax:
            a.grid(visible=True, which='major', color='gray', linestyle='-')
    title = title.replace(' ', '_')
    title = title.replace('-', '_')
    bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% LEO P - LEO I
sat_host = sat_hosts[1]
cross_planes_plotted = [4, 5, 8, 11]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'LEO P - non-coplanar to LEO I'
for plane_index in cross_planes_plotted:
    if plane_index < 6:
        theta_step = 1
    else:
        theta_step = 2
    case_analyzed = [f'sat_leo_incl_{plane_index}_{ii}' for ii in range(0,14,theta_step)]
    title = f'LEO P non-coplanar plane 4- LEO I plane {plane_index}'
    fig_li, ax_li = loadplot_timeseries(sat_host, 
        case_analyzed, 
        title = title,
        lct_chosen = lct_chosen,
        sat_target_limits = '',
        t_length= 2.5,
        link_category_chosen = name_analyzed,
        legend = 1)
    for ax in ax_li:
        for a in ax:
            a.grid(visible=True, which='major', color='gray', linestyle='-')
    title = title.replace(' ', '_')
    title = title.replace('-', '_')
    bplt.savefig(fig_li, f'tseries_{title}', save_folder = save_folder)
#%% MEO - LEO P
slant_ranges_meo = [12000, 24000]
sat_host = sat_hosts[-1]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'MEO - LEO P'
case_input = [f'sat_leo_polar_{ii}_4' for ii in range(0,14,2)]
fig, ax = loadplot_timeseries(sat_host, 
case_input, 
title = title,
lct_chosen = lct_chosen,
sat_target_limits = '',
slant_limits= slant_ranges_meo,
link_category_chosen = name_analyzed,
legend = 1)
for ax in ax:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
title = title.replace(' ', '_')
title = title.replace('-', '_')
bplt.savefig(fig, f'tseries_{title}', save_folder = save_folder)
#%% MEO - LEO I
sat_host = sat_hosts[-1]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = 0
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'MEO - LEO I'

case_input = [f'sat_leo_incl_{ii}_4' for ii in range(0,14,2)]
fig, ax = loadplot_timeseries(sat_host, 
case_input, 
title = title,
t_length= 8,
slant_limits= slant_ranges_meo,
lct_chosen = lct_chosen,
sat_target_limits = '',
link_category_chosen = name_analyzed,
legend = 1)
for ax in ax:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
bplt.savefig(fig, f'tseries_{title}', save_folder = save_folder)
#%% LEO I - MEO
slant_ranges_meo = [12000, 24000]
sat_host = sat_hosts[0]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = -1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'LEO I - MEO'

fig, ax = loadplot_timeseries(sat_host, 
case_analyzed, 
title = title,
lct_chosen = lct_chosen,
sat_target_limits = '',
slant_limits= slant_ranges_meo,
link_category_chosen = name_analyzed,
legend = 1)
for ax in ax:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
bplt.savefig(fig, f'tseries_{title}', save_folder = save_folder)
#%% LEO P - MEO
sat_host = sat_hosts[1]
cases_chosen, names_chosen = get_cases(sat_host)
ind_case = -1
case_analyzed = cases_chosen[ind_case]
name_analyzed = names_chosen[ind_case]
lct_chosen = 'lct2'
terminal_option = 'general'
title = 'LEO P - MEO'

fig, ax = loadplot_timeseries(sat_host, 
case_analyzed, 
title = title,
lct_chosen = lct_chosen,
sat_target_limits = '',
slant_limits= slant_ranges_meo,
link_category_chosen = name_analyzed,
legend = 1)
for ax in ax:
    for a in ax:
        a.grid(visible=True, which='major', color='gray', linestyle='-')
bplt.savefig(fig, f'tseries_{title}', save_folder = save_folder)
#%%
if 0:
    cols_chosen = ['t_window']
    name_analyzed = names_chosen[ii]
    # automatic plot-name
    case_plot_name =name_analyzed[:5].replace('_',' ')
    if 'leo' in sat_host:
        name_plot = f'{sat_host[4:7].upper()} {sat_host[8].upper()}-{case_plot_name.upper()}'
    elif 'meo' in sat_host:
        name_plot = f'{sat_host[4:7].upper()}-{case_plot_name.upper()}'
    data_chosen = [file for file in data_available_lct if name_analyzed in file]

#%%
if run_verification_test:
    # link cases
    # terminal type
    terminal_options = ['mk2', 'mk3', 'general'][:1]
    terminal_type = terminal_options
    # term      inal placements
    # time
    t_start = 0
    t_analysis = 60 * 60 * 2 # 2 hours
    t_end = t_start + t_analysis
    # outputs desired
    outputs = 'ae'
    output = 'time_window'
    csv_name = 'linkrates'

    target_chosen= case[5]

    nrows = 1
    ncols = 3
    data_freedom = os.path.normpath(f'{link_lct_path}\\aer_{lct_chosen}_{sat_host}.csv')
    data_aer_csv = pd.read_csv(data_freedom)
    data_aer = data_aer_csv[data_aer_csv['t'] < t_end]


    sat_target = 'sat_leo_polar_6_3'
    fig, ax = plt.subplots(nrows, ncols, figsize = (ncols*5,nrows*3))
    for terminal_type in terminal_options:
        targets = []
        data_folder = os.path.normpath(f'{output_path_link}\{terminal_type}\\')
        for lct_chosen in lct_analyzed:

            az_ranges, daz_ranges, el_ranges, del_ranges, case_lst, sat_host_lst, r_lst = [], [],[],[],[],[], []
            cols_chosen = ['az_h', 'el_h', 'daz_h', 'del_h', 'r_h']

            data_name_list = [f'{sat_host}_{name}_{lct_chosen}_{csv_name}.csv']
            data_name = data_name_list[0]# loop when analyzing multiple cases
            data_csv = pd.read_csv(f'{data_folder}/{data_name}')
            data_sliced = data_csv[data_csv['t'] < t_end]

            for ii in range(nrows):
            
                targets.append(sat_target)
                t_free = data_aer['t'].values/60
                az_free = data_aer[f'{sat_target}_az'].values
                el_free = data_aer[f'{sat_target}_el'].values
                r_free = data_aer[f'{sat_target}_slant'].values/1e3
                ae_free = np.zeros((az_free.shape[0],3))
                ae_free[:,0] = az_free
                ae_free[:,1] = el_free
                ae_free[:,2] = r_free
                data_sat = data_sliced[data_sliced['sat_target'] == sat_target]
                t_vec = data_sat['t'].values/60
                az = np.array(data_sat['az_h'].values)
                el = np.array(data_sat['el_h'].values)
                r = np.array(data_sat['r_h'].values)/1e3
                ae = np.zeros((az.shape[0],3))
                ae[:,0] = az
                ae[:,1] = el
                ae[:,2] = r
                label = ['az [deg]', 'el [deg]', 'r  [km]']
                if terminal_type == 'mk2':
                    size = 30
                elif terminal_type == 'mk3':
                    size = 10
                for jj in range(ncols):
                    ax_ii = ax[jj]
                # slice data to chosen t
                    if terminal_type == terminal_options[0]:
                        ax_ii.plot(t_free, ae_free[:,jj], label = f'{label[jj][:-6]} - NoLimits')
                    if terminal_type == 'general':
                        ax_ii.plot(t_vec, ae[:,jj], label = f'{label[jj][:-6]} - {terminal_type}')
                    else:
                        ax_ii.scatter(t_vec, ae[:,jj], label = f'{label[jj][:-6]} - {terminal_type}', s = size)
                    ax_ii.set_xlabel('t [min]')
                    ax_ii.set_ylabel(label[jj])
    for ax_ii in ax:
        ax_ii.grid('on')
        ax_ii.plot([47, 47], ax_ii.get_ylim())
        ax_ii.legend()
    ax[2].plot([0, 120], [5000, 5000], label = 'rlim mk2', color = 'r')
    ax[1].plot([0, 120], [25, 25], label = 'elim mk2', color = 'r')
    ax[1].plot([0, 120], [-5, -5], label = 'elim mk2', color = 'r')
    ax[0].plot([0, 120], [-175, -175], label = 'azlim mk2', color = 'r')
    ax[0].plot([0, 120], [175, 175], label = 'azlim mk2', color = 'r')
    fig.suptitle(f'{sat_host} links with {targets}, for terminals')
    figname = f'{sat_host}_{name}_{sat_target}_limitation_checks'
    bplt.savefig(fig, figname, 'plots_verification')
    plt.show()
                        ## Save data - Done here, 
# %%
