#%% # initially this was the complete script to process raw states into link statistics
# since 20 Jul, it has become repurposed to process raw states to intermediate outputs
# raw states -> rel positions -> az/el/slant for each LCT position
# the az/el/slant and other outputs are then processed into links
# and subsequently into link statistics
import pathlib
import json
import csv
import os
import numpy as np
import pandas as pd
import sys
import importlib
parent_dir = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(parent_dir)
print(f'\nCWD : {os.getcwd()}\n')
sys.path.insert(1, os.getcwd())
import loading_functions.data_loading as load
import data_processing.rotations as rot
import data_processing.data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
import basic_tools.operations as basic
import matplotlib.backends.backend_pdf
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)

lct_chosen = 'lct1'
long_sim = 0 # setting to process 30 days of orbital data
#output parameters
n_digits = 3
save_cs_intermediate_los = 1
save_cs_intermediate_angles = 1
csv_output_path = r'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\thesis_work\simulation_output\tables\link_parameters'

# simulation parameters
slant_limits_mk2 = [10e3, 5000e3] # slant range limits [m]
slant_limits_mk3 = [None, None]
R_atm = 200e3 # altitude above which links are established
dt = 60
t_lim = None # option to limit nr of loaded rows
if long_sim:
    print('Processing LONG simulation')
    chosen_constellation = 'Leo_globalMeo_equator720.00h'
else:
    chosen_constellation = 'Leo_globalMeo_equator24.00h'
simulation_folder = dputil.get_sim_path(parent_dir)
n_data_rows_loaded = int(t_lim/dt) if t_lim != None else None # number of data point to load

## generate lists of target names to analyze in initial link analysis
subfolder = 'initial_constellation'
target_sat_polar0 = 'sat_leo_polar'
target_sat_incl0 = 'sat_leo_incl'
target_sat_meo = 'sat_meo_0'
# LEO P - LEO P
polarleo_basic = [f'{target_sat_polar0}_4_{ii}' for ii in [2,3,5,6]] # leader/follower links
polarleo_cross_close = [f'{target_sat_polar0}_5_{ii}' for ii in range(1,8)] + [f'{target_sat_polar0}_3_{ii}' for ii in range(1,8)]
polarleo_cross_far = [f'{target_sat_polar0}_6_{ii}' for ii in range(1,8)] + [f'{target_sat_polar0}_2_{ii}' for ii in range(1,8)] # far cross-plane links
# LEO P - LEO I -> incl_leo sats
# LEO I - LEO I
incl_leo_closest = [f'{target_sat_incl0}_4_{ii}' for ii in range(2,7)] # 
incl_leo_further = [f'{target_sat_incl0}_3_{ii}' for ii in range(2,7)] + [f'{target_sat_incl0}_5_{ii}' for ii in range(2,7)] 
incl_leo_furthest = [f'{target_sat_incl0}_2_{ii}' for ii in range(2,7)] + [f'{target_sat_incl0}_6_{ii}' for ii in range(2,7)] 
# LEO I - LEO P
incl_leo_polar_closest = [f'{target_sat_polar0}_4_{ii}' for ii in [2,3,5,6]] # leader/follower links
incl_leo_polar_further = [f'{target_sat_polar0}_5_{ii}' for ii in range(1,8)] + [f'{target_sat_polar0}_3_{ii}' for ii in range(1,8)]
incl_leo_polar_furthest = [f'{target_sat_polar0}_6_{ii}' for ii in range(1,8)] + [f'{target_sat_polar0}_2_{ii}' for ii in range(1,8)] # far cross-plane links
target_meo_all = [f'{target_sat_meo}_{ii}' for ii in range(5)]
planes_meoleo = [0, 4, 8, 12]
theta_meoleo = [0, 4, 8, 12]
incl_meoleo_0 = [f'{target_sat_incl0}_{planes_meoleo[0]}_{ii}' for ii in theta_meoleo] # 
incl_meoleo_4 = [f'{target_sat_incl0}_{planes_meoleo[1]}_{ii}' for ii in theta_meoleo]
incl_meoleo_8 = [f'{target_sat_incl0}_{planes_meoleo[2]}_{ii}' for ii in theta_meoleo]
incl_meoleo_12 = [f'{target_sat_incl0}_{planes_meoleo[3]}_{ii}' for ii in theta_meoleo]
polar_meoleo_0 = [f'{target_sat_polar0}_{planes_meoleo[0]}_{ii}' for ii in theta_meoleo] # 
polar_meoleo_4 = [f'{target_sat_polar0}_{planes_meoleo[1]}_{ii}' for ii in theta_meoleo]
polar_meoleo_8 = [f'{target_sat_polar0}_{planes_meoleo[2]}_{ii}' for ii in theta_meoleo]
polar_meoleo_12 = [f'{target_sat_polar0}_{planes_meoleo[3]}_{ii}' for ii in theta_meoleo]
# MEO host - LEO polar/incl
names_meoleo_cases =['incl_meoleo_0','incl_meoleo_4','incl_meoleo_8','incl_meoleo_12','polar_meoleo_0','polar_meoleo_4','polar_meoleo_8','polar_meoleo_12']
sats_meoleo_cases = [incl_meoleo_0,incl_meoleo_4,incl_meoleo_8,incl_meoleo_12,polar_meoleo_0,polar_meoleo_4,polar_meoleo_8,polar_meoleo_12]
# Polar LEO host - LEO case
names_polar_links = ['polarleo_basic','polarleo_cross_close','polarleo_cross_far','incl_leo_closest','incl_leo_further','incl_leo_furthest', 'target_meo_all']
sats_polar_cases = [polarleo_basic,polarleo_cross_close,polarleo_cross_far,incl_leo_closest,incl_leo_further,incl_leo_furthest, target_meo_all]
# Inclined LEO host - LEO case
sats_incl_cases  = [incl_leo_closest, incl_leo_further, incl_leo_furthest, polarleo_basic,polarleo_cross_close,polarleo_cross_far, target_meo_all]
names_incl_links = ['incl_leo_closest','incl_leo_further','incl_leo_furthest', 'polarleo_basic','polarleo_cross_close','polarleo_cross_far', 'target_meo_all']
## load raw data
data_raw, simulation_parameters = dputil.load_constellation_data(chosen_constellation, 
simulation_folder, nrows = n_data_rows_loaded)
sat_names = simulation_parameters['sat_names']

all_cases = ['incl', 'polar', 'meo']
# iterate through all cases
print(f'Analyzing {all_cases} in a loop.')

if long_sim:
    sats_incl_cases = sats_incl_cases[:1]
    names_incl_links  = names_incl_links[:1]
    all_cases = ['incl']
for host_analyzed in all_cases:
    # LEO-LEO links
    # host_analyzed = 'incl'
    # host_analyzed = 'polar'
    # MEO-LEO
    # host_analyzed = 'meo'
    # LEO-MEO
    # host_analyzed = 'LEO_POLAR_MEO'
    # host_analyzed = 'LEO_ INCL_MEO'
    print(f'Analyzed ROOT case: {host_analyzed}')
    if host_analyzed == 'polar':
        cross_orbital = False
        cases_analyzed = sats_polar_cases
        names_analyzed = names_polar_links
        sat_host = 'sat_leo_polar_4_4'
        pdf_title = f'LEO_{host_analyzed}'
    elif host_analyzed == 'incl':
        cross_orbital = False
        cases_analyzed = sats_incl_cases
        names_analyzed = names_incl_links
        sat_host = 'sat_leo_incl_4_4'
        pdf_title = f'LEO_{host_analyzed}'
    elif host_analyzed == 'meo':
        cross_orbital = True
        sat_host = 'sat_meo_0_0'
        names_analyzed = names_meoleo_cases
        cases_analyzed = sats_meoleo_cases
        pdf_title = f'{host_analyzed}_LEO'
    if long_sim:
        sats_incl_cases = sats_incl_cases[:1]
        names_incl_links  = names_incl_links[:1]
        sat_host = 'sat_leo_incl_0_4'
        all_cases = ['incl']
        sats_incl_cases = [sat_name.replace('_0_', '_4_') for sat_name in sats_incl_cases]
    data_used, t_vec, indices_dict, los_dict = dputil.calculate_link_parameters(data_raw, simulation_parameters, sat_host, 
    t_lim = None)
    if save_cs_intermediate_los:
        print(f'Saving LOS in J2000') # TODO Add ref frame to sim_parameters and read it here
        path_rel_pos = r'simulation_output\processed_outputs\2rel_position'
        pandas_dict = {}
        pandas_dict['t'] = t_vec
        pos_h = los_dict[sat_host]['pos_h']
        pandas_dict['posh_x'] = pos_h[:,0]
        pandas_dict['posh_y'] = pos_h[:,1]
        pandas_dict['posh_z'] = pos_h[:,2]
        for key in los_dict[sat_host].keys():
            if key != 'pos_h':
                rho = los_dict[sat_host][key]['rel_pos']
                pandas_dict[f'{key}_los_x'] = rho[:,0]
                pandas_dict[f'{key}_los_y'] = rho[:,1]
                pandas_dict[f'{key}_los_z'] = rho[:,2]
        rel_pos_df = pd.DataFrame.from_dict(pandas_dict)
        output_path_relpos = f'{path_rel_pos}\los_eci_{sat_host}.csv'
        rel_pos_df.to_csv(output_path_relpos, index=False)
        print(f'Saved rel_pos for {sat_host} in {output_path_relpos}')
    if save_cs_intermediate_los:
        print(f'Saving All POS in J2000')
        path_rel_pos = r'simulation_output\processed_outputs\2rel_position'
        pandas_dict = {}
        pandas_dict['t'] = t_vec
        pos_h = los_dict[sat_host]['pos_h']
        vel_h = data_used[:,indices_dict[sat_host]['ind_vel']]
        pandas_dict['posh_x'] = pos_h[:,0]
        pandas_dict['posh_y'] = pos_h[:,1]
        pandas_dict['posh_z'] = pos_h[:,2]
        pandas_dict['velh_x'] = vel_h[:,0]
        pandas_dict['velh_y'] = vel_h[:,1]
        pandas_dict['velh_z'] = vel_h[:,2]
        for key in los_dict[sat_host].keys():
            if key != 'pos_h':
                pos_t = los_dict[sat_host][key]['pos_t']
                vel_t = data_used[:,indices_dict[key]['ind_vel']]
                pandas_dict[f'{key}_post_x'] = pos_t[:,0]
                pandas_dict[f'{key}_post_y'] = pos_t[:,1]
                pandas_dict[f'{key}_post_z'] = pos_t[:,2]
                pandas_dict[f'{key}_velt_x'] = vel_t[:,0]
                pandas_dict[f'{key}_velt_y'] = vel_t[:,1]
                pandas_dict[f'{key}_velt_z'] = vel_t[:,2]
        pos_df = pd.DataFrame.from_dict(pandas_dict)
        output_path_pos = f'{path_rel_pos}\pos_eci_{sat_host}.csv'
        pos_df.to_csv(output_path_pos, index=False)
        print(f'Saved host and target pos for {sat_host} in {output_path_pos}')        
    if save_cs_intermediate_angles:
        print(f'Saving Az, El, R')
        path_angles = r'simulation_output\processed_outputs\3azelslant'
        pandas_dict = {}
        pandas_dict['t'] = t_vec
        pos_h = los_dict[sat_host]['pos_h']
        pandas_dict['posh_x'] = pos_h[:,0]
        pandas_dict['posh_y'] = pos_h[:,1]
        pandas_dict['posh_z'] = pos_h[:,2]
        for ii in range(4):
            lct_chosen = f'lct{ii+1}'
            angle_dict = pandas_dict.copy()
            for key in los_dict[sat_host].keys():
                if key != 'pos_h':                    
                    lct_data = los_dict[sat_host][key][lct_chosen]
                    ae = np.rad2deg(lct_data['ae'])
                    slant = basic.calc_dmat(lct_data['los'])
                    angle_dict[f'{key}_az'] = ae[:,0]
                    angle_dict[f'{key}_el'] = ae[:,1]
                    angle_dict[f'{key}_slant'] = slant
            lct_aer_df = pd.DataFrame.from_dict(angle_dict)
            output_path_aer = f'{path_angles}\\aer_{lct_chosen}_{sat_host}.csv'
            lct_aer_df.to_csv(output_path_aer, index=False)
            print(f'Saved AER_{lct_chosen} for {sat_host} in {output_path_aer}')
    #%% Separate processing into 
    for case_chosen, cc in enumerate(cases_analyzed[:0]):
        daz_maxlst, del_maxlst, daz_meanlst, del_meanlst, hostslst, targetslst, t_contactlst, links_nrlst = [],[], [],[],[],[],[],[]
        name_case, sats_case = names_analyzed[case_chosen], cases_analyzed[case_chosen]
        print(f'---Analyzing {name_case}---\n')
        try:
            sats_case.remove(sat_host) # make sure host isnt in analyzed targets
        except:
            sats_case = sats_case
        terminal_type = 'mk2'
        link_dict, sats_linked, sats_not_linked, sats_not_checked = dputil.process_los_dict(data_used, t_vec, sat_host, los_dict, indices_dict,
            sat_target_chosen=sats_case, lct_chosen = lct_chosen, n_point_shown= data_used.shape[0], R_atm = R_atm)
    
        print('Combining multiple target sats into a single plot')
        # pdf = matplotlib.backends.backend_pdf.PdfPages(f'plots_results/link_cases/{pdf_title}_{name_case}.pdf')
        overview_tab = {}
        sats_case_clean = [sat for sat in sats_case if sat not in sats_not_linked]
        for ii, sat_target in enumerate(sats_case_clean): # combined plots
            link_dict_target = link_dict[sat_target]
            az_h = []
            el_h = []
            r_h = []
            # Gradients
            daz_h = []
            del_h = []
            t_h = []
            for mm, link_nr in enumerate(link_dict_target.keys()):
                link_ii = link_dict_target[link_nr]
                ae = link_ii['ae_host']
                t_range = link_ii['t_vec']
                t_len = link_ii['t_window']
                slant = link_ii['slant_range']/1e3
                az = ae[:,0]
                el = ae[:,1]
                az_range = np.rad2deg(np.unwrap(az))
                el_range = np.rad2deg(np.unwrap(el))
                if az_range.shape[0]>1:
                    daz_range = np.gradient(az_range, t_range)
                    del_range = np.gradient(el_range, t_range)
                    t_h = np.hstack((t_h,t_range))
                    r_h = np.hstack((r_h, slant))
                    az_h = np.hstack((az_h,az_range))
                    el_h = np.hstack((el_h,el_range))
                    daz_h = np.hstack((daz_h,daz_range)) # gradients
                    del_h = np.hstack((del_h,del_range)) # gradients
                                            # track overview table data
                    az_hmax = np.max(np.abs(daz_h))
                    el_hmax = np.max(np.abs(del_h))
                    az_h_mean = np.mean(np.abs(daz_h))
                    el_h_mean = np.mean(np.abs(del_h))
                    # store
                    daz_maxlst.append(np.round(az_hmax,n_digits))
                    del_maxlst.append(np.round(el_hmax,n_digits))
                    daz_meanlst.append(np.round(az_h_mean,n_digits))
                    del_meanlst.append(np.round(el_h_mean,n_digits))
                    hostslst.append(sat_host)
                    targetslst.append(sat_target)
                    t_len = 'inf' if t_len == np.inf else int(t_len/60)
                    t_contactlst.append(t_len) 
                    links_nrlst.append(link_nr)
            y_data = [az_h, daz_h, el_h, del_h, r_h]
            if ii == 0:
                fig_title = f'{sat_host}-{sat_target[:-2]}-{lct_chosen}-AER'
                f, a = bplt.plot_aer_plusgrad(y_data, t_h, title = f'{fig_title}', label = sat_target, x_limits = [0,24*60])
            else:
                t_h = t_h/60
                for jj, ax in enumerate(a):
                    ax.scatter(t_h, y_data[jj], s = 2, label = sat_target)
            for ax in a:
                ax.legend()
            overview_table = {'host':hostslst,
            'target' : targetslst,
            't_len' : t_contactlst,
            'link' : links_nrlst,
            '||daz||_max' : daz_maxlst,
            '||del||_max' : del_maxlst,
            '||daz||_mean' : daz_meanlst,
            '||del||_mean' : del_meanlst,
            }
            overview_df = pd.DataFrame.from_dict(overview_table)
            overview_df.to_csv(f'{csv_output_path}/{fig_title}.csv', index = False)
            bplt.savefig(f, name = fig_title, save_folder = 'plots_results/link_cases')
                    
            print(f'Not linked: {sats_not_linked}')
print('All done')

# %%
