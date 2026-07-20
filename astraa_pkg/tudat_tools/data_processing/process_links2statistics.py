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
import rotations as rot
import data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
import basic_tools.operations as basic
import basic_tools.constants as const
import basic_tools.link_cases as cases
import matplotlib.backends.backend_pdf
from matplotlib.offsetbox import AnchoredText
importlib.reload(const)
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)
## base paths
pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")
output_path_tables = os.path.normpath(r"simulation_output\\tables\\")
make_link_histograms = 1
make_output_tables = 0
make_link_nr_histograms = 0
#%% Getting Azimuth/Elevation/AzElgrad limits for every case
if 0:
    sat_hosts = [pos[8:-4] for pos in os.listdir(pos_path) if 'los' in pos]
    print(f'Available hosts: {*sat_hosts,}')

    sat_host = sat_hosts[2]
    if sat_host == 'sat_leo_incl_4_4':
        cases_chosen = cases.cases_leo_i
        names_chosen = cases.names_leo_i
    elif sat_host == 'sat_leo_polar_4_4':
        cases_chosen = cases.cases_leo_p
        names_chosen = cases.names_leo_p
    elif sat_host ==  'sat_meo_0_0':
        cases_chosen = cases.cases_meoleo
        names_chosen = cases.names_meoleo
    # link cases
    # terminal type
    terminal_options = ['mk2', 'mk3', 'general']
    terminal_type = terminal_options[-1]
    # terminal placements
    # time
    t_start = 0
    t_analysis = 60 * 60 * 2 # 2 hours
    t_end = t_start + t_analysis
    # outputs desired
    outputs = 'ae'
    output = 'time_window'
    csv_name = 'linkrates'

    data_folder = os.path.normpath(f'{output_path_link}\{terminal_type}\\')

    lct_analyzed = ['lct1', 'lct2']
    for lct_chosen in lct_analyzed:

        az_ranges, daz_ranges, el_ranges, del_ranges, case_lst, sat_host_lst, r_lst = [], [],[],[],[],[], []
        cols_chosen = ['az_h', 'el_h', 'daz_h', 'del_h', 'r_h']
        for ii, case_analyzed in enumerate(cases_chosen):    
            name_analyzed = names_chosen[ii]
            data_name_list = [f'{sat_host}_{name_analyzed}_{lct_chosen}_{csv_name}.csv']
            data_name = data_name_list[0]# loop when analyzing multiple cases
            data_csv = pd.read_csv(f'{data_folder}/{data_name}')
            # slice data to chosen t
            data_sliced = data_csv[data_csv['t'] < t_end]
            out_array = np.zeros((data_sliced.shape[0], len(cols_chosen)))
            for jj, col in enumerate(cols_chosen):
                # Get absolute values of data column
                data_col = np.abs(data_sliced.loc[:,col].values)
                out_array[:,jj] = data_col
            ## Save data
            az_ranges.append(np.round([min(out_array[:,0]), max(out_array[:,0])], 1))
            el_ranges.append(np.round([min(out_array[:,1]), max(out_array[:,1])], 1))
            daz_ranges.append(np.round([min(out_array[:,2]), max(out_array[:,2])], 3))
            del_ranges.append(np.round([min(out_array[:,3]), max(out_array[:,3])], 3))
            r_lst.append(np.round([min(out_array[:,4])/1e3, max(out_array[:,4])/1e3], 0))
            case_lst.append(name_analyzed)
            sat_host_lst.append(sat_host)
        # format to look nice
        az_ranges = [str(f'[{a[0]}-{a[1]}]') for a in az_ranges]
        el_ranges = [str(f'[{a[0]}-{a[1]}]') for a in el_ranges]
        daz_ranges = [str(f'[{a[0]}-{a[1]}]') for a in daz_ranges]
        del_ranges = [str(f'[{a[0]}-{a[1]}]') for a in del_ranges]
        r_lst = [str(f'[{a[0]}-{a[1]}]') for a in r_lst]
        
        results_dict = {
            'Host' : sat_host_lst,
            'Link Case': case_lst,
            'Az [deg]' :az_ranges,
            'El [deg]': el_ranges,
            'Slant [km]' : r_lst,
            r'$$\dot{Az}$$ [deg/s]' :  daz_ranges,
            r'$$\dot{El}$$ [deg/s]' : del_ranges,
        }
        result_df = pd.DataFrame.from_dict(results_dict)
        save_name = f'{sat_host}_{name_analyzed}_{lct_chosen}_general_max.csv'
        save_full = f'{output_path_stats}/{terminal_type}/{save_name}'
        result_df.to_csv(save_full, index = False)
        print(f'saved {save_full}')

if make_link_nr_histograms:
    importlib.reload(modplot)
    sat_hosts = [pos[8:-4] for pos in os.listdir(pos_path) if 'los' in pos]
    print(f'Available hosts: {*sat_hosts,}')
    

    for sat_host in sat_hosts:
        cases_chosen, names_chosen = dputil.get_cases(sat_host)        
        # link cases
        # terminal type
        terminal_options = ['mk2', 'mk3']

        # terminal placements
        # time

        if 'leo' in sat_host: 
            cases_chosen = cases_chosen[1:]
            names_chosen = names_chosen[1:]
            t_analysis = 2
            ylims = [0, 50, 5]
            nrows = 2
        else:
            nrows = 1
            terminal_options = ['mk3']
            t_analysis = 8
            ylims = [0, 80, 10]
        t_start = 0
        t_analysis = 60 * 60 * t_analysis
        t_end = t_start + t_analysis
        # outputs desired
        output = 'link_nr'
        csv_name = 'linkoverview'
        xlabels = []
        ncols = 2            
        fig, ax = plt.subplots(nrows, ncols, figsize = (ncols*5,nrows*3))
        for ll, terminal_type in enumerate(terminal_options):
            data_folder = os.path.normpath(f'{output_path_link}\{terminal_type}\\')

            nr_link_lst = []
            lct_analyzed = ['lct2']

            ## Complete code
            # Used for plotting
            xlabel = 'Constellation Shell/Plane [-]'
            ylabel = 'Nr. Links [-]'

            data_all = os.listdir(data_folder)
            data_available_host = [file for file in data_all if sat_host in file]
            for lct_chosen in lct_analyzed:
                data_available_lct = [file for file in data_available_host if lct_chosen in file]
                for ii in range(ncols):    
                    # only same-shell in 1st row, other shell and MEO in 2nd
                    if ii < 1: 
                        ind_0 = 0
                        ind_f = 1
                    else:
                        ind_0 = 1
                        ind_f = 3
                    name_analyzed = names_chosen[ind_0 : ind_f]
                    case_analyzed = cases_chosen[ind_0 : ind_f]
                    names_used = [name[:5].upper().replace('_', ' ') if 'leo' in name else name.upper() 
                        for name in name_analyzed]

                    # print(name_plot)
                    data_all = []
                    y_data = []
                    x_data = []
                    for name in name_analyzed:
                        data_chosen = [file for file in data_available_lct if name in file]
                        data_chosen = [file for file in data_chosen if csv_name in file]
                        data_all.extend(data_chosen)
                    data_chosen = data_all
                    for data_name in data_chosen:
                        data_csv = pd.read_csv(f'{data_folder}/{data_name}')       
                        # filter data to only include the t_end analysis
                        data_csv_t = data_csv[data_csv['t_start'] < t_end/60]
                        for cases in case_analyzed:
                            # split to separate LEO planes and single MEO plane
                            if 'leo' in cases[0]:
                                sat_target_cut = cases[0][:-3]
                                sat_labels = [f'{sat_target_cut}{ii}_' for ii in range(13)]
                            else:
                                sat_labels = [case[:-1] for case in cases[:1]]
                            for sat_label in sat_labels:
                                # make label for histogram column
                                name_xdat = dputil.get_planelabel_fromcase(sat_label)
                                name_xdat = name_xdat[:-1] if name_xdat[-1] == ' ' else name_xdat
                                # get nr of links per constellation plane
                                link_data_plane = data_csv_t[data_csv_t['sat_target'].str.contains(sat_label)]
                                sats_linked_unique = sorted(set(link_data_plane['sat_target'].values))
                                # sum links per target satellite in current plane
                                nr_links = 0
                                for sat_target in sats_linked_unique:
                                    nr_links+=max(link_data_plane[link_data_plane['sat_target'] == sat_target][output])
                                y_data.append(nr_links)
                                x_data.append(name_xdat)
                        case_plot_name = sat_host[:5].replace('_',' ') # TODO change
                        if 'leo' in sat_host:
                            # name_plot = f'{sat_host[4:7].upper()} {sat_host[8].upper()}-{case_plot_name.upper()}'
                            host_label = f'{sat_host[4:9].upper()}-'
                            name_host = f'{host_label[:-1]}_4'
                        elif 'meo' in sat_host:
                            host_label = f'{sat_host[4:7].upper()}-'
                            name_host = host_label
                        name_plot = host_label.replace('_', ' ')
                        for name in names_used:
                            name_plot = f'{name_plot} {name},'
                        name_plot = name_plot[:-1] if name_plot[-1] == ',' else name_plot
                        name_plot = f'{name_plot} using {terminal_type.capitalize()}'
                        # name_plot = name_plot + f'. No. links: {len(t_windows_filt)}'
                        # plot
                        if nrows == 2:
                            ax_chosen = ax[ll, ii]
                        else:
                            ax_chosen = ax[ii]
                        if ii>0:
                            c = 'khaki'
                        else:
                            c = 'cornflowerblue'
                        ax_chosen = modplot.plot_hist_nrlinks(ax_chosen,
                        x_data,                        
                        y_data,
                        c,
                        ylim= ylims,
                        )
                        if 'leo' in sat_host:
                            if ll == 1:
                                ax_chosen.set_xlabel(xlabel, fontsize = 12, fontweight = 'bold')
                        else:
                            ax_chosen.set_xlabel(xlabel, fontsize = 12, fontweight = 'bold')
                        if ii == 0:
                            ax_chosen.set_ylabel(ylabel, fontsize = 12, fontweight = 'bold')

                        ax_chosen.set_xticks(ticks = x_data, rotation = 90)
                        ax_chosen.set_xticklabels(labels = x_data, rotation = 90)
                        ax_chosen.set_yticks(np.arange(ylims[0], ylims[1] + ylims[2], ylims[2]))
                        
                        ax_chosen.set_title(name_plot, fontsize = 12)    
                        # plt.show()
                        # Add ylabels
                        # if ii == 0:
                        #     ax_chosen.set_ylabel('Occurence [%]', size = 12)
                        #     # ax_chosen.annotate(f'{lct_chosen.upper()}', xy = (0,20))
                        #     anchored_text = AnchoredText(f'{lct_chosen.upper()}',loc=loc)
                        #     ax_chosen.add_artist(anchored_text)

                        # if ii == ncols-1:
                        #     ax_twin.set_ylabel('Cumulative Prob. [-]', size = 12)
                        # # add xlabels
                        # if ll == 1:
                        #     ax_chosen.set_xlabel('Communication time [min]', size = 12)

                    # else:
                    #     print(f'{name_analyzed} for {sat_host} not found')
                    #     continue
            # host_name = sat_host[4:7].upper().strip('-0')
            # fig.suptitle(f'{host_name}-{sat_host[8].upper()} Non-coplanar {int(t_end/3600)} hr Link Window Distribution using {terminal_type.capitalize()}', size = 16)
            # fig.tight_layout(h_pad=1, w_pad=1)   
            # save_name = f'link_histogram_{sat_host}_{int(t_end/3600)}hr_{terminal_type}'
            # bplt.savefig(fig, save_name, save_folder = r'plots_results/link_histograms') 
            # print(f'Saved {save_name}')
        title = f'Nr. of links by {name_host} in {int(t_analysis/3600)} hr with LCT position {lct_chosen.upper()}.'
        fig.suptitle(title, fontsize = 14, fontweight = 'bold')
        fig.set_tight_layout('tight') 
        bplt.savefig(fig, f'nrlink_histogram_{sat_host}_{int(t_analysis/3600)}hr', r'plots_results/link_histograms')
    plt.show()            
#%% histograms of communication windows
if make_link_histograms:
    importlib.reload(modplot)
    sat_hosts = [pos[8:-4] for pos in os.listdir(pos_path) if 'los' in pos]
    print(f'Available hosts: {*sat_hosts,}')
    

    for sat_host in sat_hosts:
        if sat_host == 'sat_leo_incl_4_4':
            cases_chosen = cases.cases_leo_i
            names_chosen = cases.names_leo_i
        elif sat_host == 'sat_leo_polar_4_4':
            cases_chosen = cases.cases_leo_p
            names_chosen = cases.names_leo_p
        elif sat_host ==  'sat_meo_0_0':
            cases_chosen = cases.cases_meoleo
            names_chosen = cases.names_meoleo
        # link cases
        # terminal type
        terminal_options = ['mk2', 'mk3']

        # terminal placements
        # time

        
        if 'leo' in sat_host: 
            cases_chosen = cases_chosen[1:]
            names_chosen = names_chosen[1:]
            t_analysis = 2
            ylims = [0, 50, 5]
            nrows = 2
        else:
            nrows = 1
            terminal_options = ['mk3']
            t_analysis = 8
            ylims = [0, 90, 10]
        loc = 2
        t_start = 0
        t_analysis = 60 * 60 * t_analysis
        t_end = t_start + t_analysis
        t_cutoff = 60 # orbital period of 105 minutes for 1000km LEO altitude
        # outputs desired
        outputs = 'ae'
        output = 'time_window'
        csv_name = 'linkoverview'


        for terminal_type in terminal_options:
            data_folder = os.path.normpath(f'{output_path_link}\{terminal_type}\\')

            t_window_lst = []
            lct_analyzed = ['lct1', 'lct2']

            ## Complete code
            # Used for plotting
            t_step = 5 # 5 minute steps
            bin_range = np.arange(t_start, t_cutoff+t_step*2, t_step)
            nrows = 2
            ncols = len(names_chosen)
            # if 'meo' in sat_host:# remove leader-follower from histograms
            #     print('reducing ncols for MEO, HARDCODED')
            #     ncols = 2 # known that MEO
            fig, ax = plt.subplots(nrows, ncols, figsize = (ncols*5,nrows*3))
            xlabel = 'Available link time [min]',
            ylabel = 'Occurences [%]',

            data_all = os.listdir(data_folder)
            data_available_host = [file for file in data_all if sat_host in file]
            for ll, lct_chosen in enumerate(lct_analyzed):
                data_available_lct = [file for file in data_available_host if lct_chosen in file]
                cols_chosen = ['t_window']
                for ii, case_analyzed in enumerate(cases_chosen):    
                    name_analyzed = names_chosen[ii]
                    # automatic plot-name
                    case_plot_name =name_analyzed[:5].replace('_',' ')
                    if 'leo' in sat_host:
                        name_plot = f'{sat_host[4:7].upper()} {sat_host[8].upper()}-{case_plot_name.upper()}'
                    elif 'meo' in sat_host:
                        name_plot = f'{sat_host[4:7].upper()}-{case_plot_name.upper()}'
                    data_chosen = [file for file in data_available_lct if name_analyzed in file[10:]]
                    data_chosen = [file for file in data_chosen if csv_name in file]
                    if len(data_chosen) == 1:
                        data_name = data_chosen[0]
                        data_csv = pd.read_csv(f'{data_folder}/{data_name}')## finished here            
                        # filter data to only include the t_end analysis
                        data_csv_t = data_csv[data_csv['t_start'] < t_end/60]
                        t_windows = data_csv_t['t_window'].values
                        t_windows_filt = [t_cutoff if t == np.inf else t for t in t_windows]
                        t_windows_filt = [t if t < t_cutoff  else t_cutoff for t in t_windows_filt]
                        name_plot = name_plot + f'. No. links: {len(t_windows_filt)}'
                        # plot
                        ax_chosen, ax_twin = modplot.plot_cum_hist_twindow(ax[ll, ii], 
                        t_windows_filt,
                        t_step,
                        name_plot,
                        cut_y = 0,
                        t_cutoff = t_cutoff,
                        reverse_cumul = 0,
                        ylim = ylims,
                        invert_x = 0)
                        # Add ylabels
                        if ii == 0:
                            ax_chosen.set_ylabel('Occurence [%]', size = 12, fontweight = 'bold')
                            # ax_chosen.annotate(f'{lct_chosen.upper()}', xy = (0,20))
                            anchored_text = AnchoredText(f'{lct_chosen.upper()}',loc=loc)
                            ax_chosen.add_artist(anchored_text)

                        if ii == ncols-1:
                            ax_twin.set_ylabel('Cumulative Prob. [-]', size = 12, fontweight = 'bold')
                        # add xlabels
                        if ll == 1:
                            ax_chosen.set_xlabel('Communication time [min]', size = 12, fontweight = 'bold')

                    else:
                        print(f'{name_analyzed} for {sat_host} not found')
                        continue
            host_name = sat_host[4:7].upper().strip('-0')
            fig.suptitle(f'{host_name}-{sat_host[8].upper()} {int(t_end/3600)} hr Link Window Distribution using {terminal_type.capitalize()}', size = 14, fontweight = 'bold' )
            fig.tight_layout(h_pad=1, w_pad=1)   
            save_name = f'link_histogram_{sat_host}_{int(t_end/3600)}hr_{terminal_type}'
            bplt.savefig(fig, save_name, save_folder = r'plots_results/link_histograms') 
            print(f'Saved {save_name}')    
#%% Create tabular data      
def calc_tabdata_below_thresholds(sat_host, 
    terminal_type,
    names_chosen,
    lct_chosen = 'lct2',
    csv_name = 'linkoverview',
    unit_chosen = 'min',
    output_col = 't_window',
    t_col = 't_start',
    upper_lim_vals = [10, 15, 30, 45, 60],
    t_start = 0,
    window_length = 480,
    output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
    ):
    t_end = t_start + window_length
    
    data_folder = f'{output_path_link}/{terminal_type}'
    # placeholders for outputs
    output_vals = np.zeros((len(names_chosen), len(upper_lim_vals)))
    row_label_case = []
    output_col_labels = [f'>{lim} {unit_chosen}' for lim in upper_lim_vals]
    ind_del = 0
    for ii, name_analyzed in enumerate(names_chosen):
        ## Load data
        # filter out names which are not needed
        data_all = os.listdir(data_folder)
        data_available_host = [file for file in data_all if sat_host in file]
        data_available_lct = [file for file in data_available_host if lct_chosen in file]
        data_filt = [file for file in data_available_lct if csv_name in file]
        data_name = [file for file in data_filt if name_analyzed in file.removeprefix(sat_host)]
        if len(data_name) != 0:
            ind_del +=1
            data_name = data_name[0]  
            full_data_path = os.path.normpath(f'{data_folder}/{data_name}')
            # make labels names
            case_plot_name =name_analyzed[:5].replace('_',' ')
            if 'leo' in sat_host:
                name_label = f'{sat_host[4:7].upper()} {sat_host[8].upper()}-{case_plot_name.upper()}'
            elif 'meo' in sat_host:
                name_label = f'{sat_host[4:7].upper()}-{case_plot_name.upper()}'
            data_csv = pd.read_csv(full_data_path)
            # slice data to chosen t
            data_sliced = data_csv[data_csv[t_col] < t_end]
            ## Get data requested
            data_col = data_sliced[output_col]
            total_links = len(data_col)
            for jj, lim in enumerate(upper_lim_vals):
                data_below_lim = data_col[data_col < lim]
                output_vals[ii,jj] = int(np.round(len(data_below_lim) / total_links*100,0))
            row_label_case.append(name_label)
        else:
            print(f'{sat_host}-{name_analyzed} is empty for {terminal_type}')
            output_vals = np.delete(output_vals, ind_del, 0)
            ind_del -=1
            continue
    if len(row_label_case) ==0:
        return None
    else:
        return row_label_case, output_col_labels, output_vals
if make_output_tables:
    # ## base paths
    pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
    link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
    output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
    output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")          
    sat_hosts = [pos[8:-4] for pos in os.listdir(pos_path) if 'los' in pos]
    sat_host = sat_hosts[0]
    # link cases
    # terminal type
    terminal_options = ['mk2', 'mk3', 'general']

    # terminal placements
    # time
    t_start = 0
    lct_analyzed = ['lct1', 'lct2']
    lct_chosen = lct_analyzed[1]
    # Inputs:

    for terminal_type in terminal_options[:2]:
        links_found = 0
        for ii, sat_host in enumerate(sat_hosts):
            if sat_host == 'sat_leo_incl_4_4':
                cases_chosen = cases.cases_leo_i[1:]
                names_chosen = cases.names_leo_i[1:]
                t_analysis = 2
            elif sat_host == 'sat_leo_polar_4_4':
                cases_chosen = cases.cases_leo_p[1:]
                names_chosen = cases.names_leo_p[1:]
                t_analysis = 2
            elif sat_host ==  'sat_meo_0_0':
                cases_chosen = cases.cases_meoleo[:-1]
                names_chosen = cases.names_meoleo[:-1]
                t_analysis = 8
            link_tabular = calc_tabdata_below_thresholds(sat_host, 
            terminal_type, names_chosen,
            lct_chosen = lct_chosen,
            window_length = t_analysis*60
            )
            if type(link_tabular) != type(None):
                links_found = 1
                cols_df = link_tabular[1]
                if ii == 0:
                    tab_data = link_tabular[-1]
                    link_col = link_tabular[0]
                else:
                    tab_data = np.vstack((tab_data,link_tabular[-1]))
                    link_col = link_col + link_tabular[0]
        if links_found:
            tab_data = tab_data.astype(int)
            # output
            tab_dict = {
                'Link case' : link_col
            }
            for ii, col in enumerate(cols_df):
                tab_dict[col] = tab_data[:,ii]
            tab_df = pd.DataFrame.from_dict(tab_dict)
            tab_name = f'linktimes_cross_{terminal_type}_{lct_chosen}.csv'
            save_path = f'{output_path_tables}\\link_durations\{tab_name}'
            tab_df.to_csv(save_path, index = False)
            print(f'Saved {tab_name}')
        else:
            print(f'{terminal_type} - {sat_host}, no links found')