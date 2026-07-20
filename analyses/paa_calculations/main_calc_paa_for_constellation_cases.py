## 12-04-2023 Computing PAA for multiple link cases in
# a LEO Inclined/Polar + MEO satellite constellation
# TODO update description - plots, calculations, tables, unit tests
#%% Imports
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
parent_dir = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(parent_dir)
print(f'\nCWD : {os.getcwd()}\n')
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.data_processing.data_loading as load
import paa_tools.paa_calculation as paa_calc
import matplotlib.pyplot as plt
import plotting_tools.combined_plots as plt_tool
import plotting_tools.modular_plotting as modplot
import attitude_tools.terminal_rotations as lct_rot
import link_processing_tools.visibility_checks as vis_check
import link_processing_tools.leomeo_link_cases as link_cases
import attitude_tools.conversions as conv
import basic_tools.string_conversion as s_conv
importlib.reload(s_conv)
importlib.reload(plt_tool)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(modplot)
importlib.reload(paa_calc)
importlib.reload(vis_check)
importlib.reload(link_cases)
### Paths
sim_folder ='orbital_simulations'
subfolder = 'constellation_leomeo'
folder_outputs = f'{sim_folder}/{subfolder}'
paa_table_name = 'PAA_overview_leomeoconstellation'
save_folder = f'outputs'
subfolder_tables = f'{save_folder}//tables'
subfolder_plots = f'{save_folder}//plots'
make_aer_plots = 1
make_3d_figs = 0
make_overview_df = 0
use_full_data = 1
# conditionals for outputs
save_figs = 1
save_tables = 0
do_test_cases = 0 # 1 run Pointing angle and PAA unit tests
print(f'''Settings: 
    save figs : {bool(save_figs)}
    save tables : {bool(save_tables)}
    run unit tests : {bool(do_test_cases)}
''')
if do_test_cases:
    def parse_col(col):
        col_strip = col.strip('[').strip(']')
        col_split = col_strip.split(' ')
        lst = []
        for val in col_split:
            if val != '':
                lst.append(float(val))
        return lst

    test_cases = pd.read_csv(r'unit_tests\PAA_tests_inout.csv')

    for ii in range(test_cases.shape[1]):
        # parse list inputs
        if ii >1:
            test_cases.iloc[:,ii] = test_cases.iloc[:,ii].apply(lambda x : parse_col(x))
    ## test cases for PAA
    for ii, case_list in enumerate(test_cases['case']):
        
        states_host = test_cases.iloc[ii, 2]
        states_host = np.reshape(states_host, (1, len(states_host)))
        states_target = test_cases.iloc[ii, 3]
        states_target = np.reshape(states_target, (1, len(states_target)))
        attitude_host = test_cases.iloc[ii, 4]
        attitude_host = np.reshape(attitude_host, (1, len(attitude_host)))
        t_vec = np.array([[0]])
        ae_exp = test_cases.iloc[ii,6]
        paa_exp = test_cases.iloc[ii,5]
        pt_angles = paa_calc.compute_azel_paa(states_host, states_target, attitude_host, t_vec, official_convention = 1)[0]
        # pt_angles = paa_calc.compute_azel_paa(states_host, states_target, attitude_host, t_vec, official_convention = 1)[0]
        # checks
        fail_ae = 0
        fail_paa = 0
        if not np.abs(ae_exp[0] - pt_angles[1]) < 1e-6:
            fail_ae = 1
        elif not np.abs(ae_exp[1] - pt_angles[2]) < 1e-6:            
            fail_ae = 1
        if not np.abs(paa_exp[0] - pt_angles[3]) < 1e-7:
            fail_paa = 1
        elif not np.abs(paa_exp[1] - pt_angles[4]) < 1e-7:
            fail_paa = 1
        if fail_paa:
            print(f'Case {case_list} FAIL : PAA expected {paa_exp}, computed {pt_angles[3:5]} ')
        else:
            print(f'Case {case_list} Pass : PAA expected {paa_exp}, computed {pt_angles[3:5]} ')
        if fail_ae:
            print(f'Case {case_list} FAIL : AE expected {ae_exp}, computed {pt_angles[1:3]} ')
        else:
            print(f'Case {case_list} Pass : AE expected {ae_exp}, computed {pt_angles[1:3]} ')
        (f'T')
            
if use_full_data:
    # Load states
    data_raw, sim_parameters = load.open_dat(folder_outputs, base_path = r'C:\\Users\\KPaliusis\\Documents\\Github_Repos\\astropynaric')
    t_vec, data_used, indices_dict = dputil.get_sat_ind(data_raw, sim_parameters)
    # Select link cases + host/target states
    case_titles = link_cases.case_titles
    hosts = link_cases.hosts
    targets = link_cases.targets
    
    #placeholders
    r_min, case_nr, v_rel_max, paa_az_max, paa_el_max, paa_1d_max = [], [], [], [], [], []

    for ii, host_name in enumerate(hosts):
        # if ii > 5 and ii < 6:
        # if ii in [3]:
        if 1:
        # if ii == 1:
            mm = 100
            case_title = f'Case {ii+1} {case_titles[ii]}'
            target_name = targets[ii]
            ind_host = np.hstack((indices_dict[host_name]['ind_pos'], (indices_dict[host_name]['ind_vel']) ))
            ind_target = np.hstack((indices_dict[target_name]['ind_pos'], (indices_dict[target_name]['ind_vel']) ))
            states_host = data_used[:,ind_host]
            states_target = data_used[:,ind_target]

            r_h, v_h = states_host[:,:3],states_host[:,3:]

            v_rel = np.linalg.norm(states_target[:,3:] - states_host[:,3:], axis = 1)
            print(f'{case_title}. {host_name}-{target_name}')
            
            # generate host attitude
            attitude_host, rotq_eci2lct = lct_rot.calc_quat_eci2lct(r_h=r_h, v_h=v_h)
            lct_pointing = paa_calc.compute_azel_paa(states_host, states_target, attitude_host, official_convention = 1)
            los_lct, aer_lct, paa_full = lct_pointing[:,:3], lct_pointing[:,[3,4,5 ]], lct_pointing[:,6:] 
            paa_analytical = paa_calc.calc_paa_analytical(states_host=states_host, states_target=states_target)
            if 0:
                # 23-04-2023 Added for Pointing unit test cases. 
                np.set_printoptions(2)
                jj = 10
                q_used = attitude_host[[jj],:]
                sh_used = states_host[[jj],:]
                st_used = states_target[[jj],:]
                pt_test = paa_calc.compute_azel_paa(sh_used, st_used, q_used)
                los_lct, aer_lct, paa_full = pt_test[:,:3], pt_test[:,[3,4,5 ]], pt_test[:,6:] 
                print(f'Direct AE output : {np.rad2deg(aer_lct[0,:2])} deg')
                dcm = conv.convert_quat2dcm(q_used[0])
                q2dcm_recalc = conv.convert_dcm2quat(dcm)

                rpy = conv.convert_dcm_2_ea(dcm)
                dcm_recalc = conv.convert_ea_2_dcm(rpy)
                if 0: # conversion tests 24-4-2023
                    rpy_45yaw = [0, 0, 95]
                    # rpy_45yaw = [0, -40, 0]
                    # rpy_45yaw = [95, 0, 0]
                    dcm_45 = conv.convert_ea_2_dcm(rpy_45yaw)
                    rpy_recalc = conv.convert_dcm_2_ea(dcm_45)
                    print(f'{rpy_45yaw} == {rpy_recalc}?')
                    # rpy_45yaw = [90, 0, 45]
                    # dcm_45 = conv.convert_ea_2_dcm(rpy_45yaw)
                    # rpy_recalc = conv.convert_dcm_2_ea(dcm_45)
                    # print(f'{rpy_45yaw} == {rpy_recalc}?')

                q_recalc = conv.convert_ea2quat(rpy)
                rpy_recalc = conv.convert_dcm_2_ea(dcm_recalc)
                # q_recalc = q2dcm_recalc
                pt_test_redo = paa_calc.compute_azel_paa(sh_used, st_used, q_recalc.reshape((1,4)))
                los_lct2, aer_lct2, paa_full2 = pt_test_redo[:,:3], pt_test_redo[:,[3,4,5 ]], pt_test_redo[:,6:] 
                print(f'Redone AE output : {np.rad2deg(aer_lct2[0,:2])} deg')

            # slice out data where link is not available
            ii_vis = vis_check.check_occultation(states_host, states_target)
            t_proc = t_vec[ii_vis]
            los_lct, aer_lct, paa_full = los_lct[ii_vis,:], aer_lct[ii_vis,:], paa_full[ii_vis,:]*1e6
            paa_analytical = (paa_analytical[0][ii_vis], paa_analytical[1][ii_vis])
            if "meo" in host_name.lower() or "meo" in target_name.lower():
                r_lim = 24e6
            else:
                r_lim = 7e6
            
            # print(f'PAA max Az: {np.max(np.abs(paa_full[ii_vis,0])):.1f}, El: {np.max(np.abs(paa_full[ii_lim,1])):.1f}. Tot : {np.max(np.abs(paa_analytical[0][ii_lim])):.1f}')
            
            title_given = case_title
            ## Save overview table data
            case_nr.append(f'{ii+1} {case_titles[ii]}')
            r_min.append(np.round(np.min(aer_lct[:,2])/1e3,0))
            v_rel_max.append(np.round(np.max(v_rel[ii_vis]),0))
            paa_az_max.append(np.round(np.max(np.abs(paa_full[:,0])),0))
            paa_el_max.append(np.round(np.max(np.abs(paa_full[:,1])),0))
            paa_1d_max.append(np.round(np.max(paa_analytical[0]),0))
            
            if make_aer_plots:
                if 0: # occultation considered
                    f, ax_aer = plt_tool.plot_aer(t_proc, aer_lct, title = title_given, setting = '', ii_vis = ii_vis)
                    # f_dot, ax_aerdot = plt_tool.plot_aer(t_proc, aer_dot, title = title_given, ii_vis = ii_vis)
                    f_paa, ax_paa = plt_tool.plot_paa(t_proc, paa_full, paa_analytical, ii_vis = ii_vis)
                else:
                    aer_deg = np.copy(aer_lct)
                    # aer_dot_deg = np.copy(aer_dot)
                    # aer_dot_deg[:,[0,1]] = np.rad2deg(aer_dot_deg[:,[0,1]]) # convert to deg/s
                    # f_dot, ax_aerdot = plt_tool.plot_aer(t_proc, aer_dot_deg, title = title_given, save_figure=save_figs)
                    aer_deg[:,[0,1]] = np.rad2deg(aer_lct[:,[0,1]]) # convert to deg
                    f, ax_aer = plt_tool.plot_aer(t_proc, aer_deg, title = title_given, setting = '', save_figure=save_figs)
                    f_paa, ax_paa = plt_tool.plot_paa(t_proc, paa_full, paa_analytical, fname = title_given, save_figure=save_figs)
            ## make 3d link plot
            if make_3d_figs:
                rot_ver = 15
                # rot_ver = 90

                rot_hor = [45,
                        30,
                        45,
                        45,
                        45,
                        195, # good
                        175,
                        0,
                        60,
                        125][ii] #125 
                print(f'{ii}, {case_title}, {rot_hor} deg')
                fig, ax= modplot.make_3dplot(unit = 'Mm')
                fig, ax = modplot.add_earth(fig, ax)
                label_host = ''
                label_target = ''
                r_host = states_host[:,:3]
                r_target = states_target[:,:3]
                ii_max_host = 100
                ii_max_target = 100
                if 'meo' in target_name:
                    range_link = [13e6, 23e6]
                else:
                    range_link = [100e3, 7000e3]
                fig, ax = modplot.add_orbit_basic(fig, ax, r_host[ii_vis,:], label = label_host, c = 'b', linewidth = 6)
                fig, ax = modplot.add_orbit_basic(fig, ax, r_target[ii_vis,:], label = label_target, c = 'r', linewidth = 6)
                fig, ax = modplot.add_orbit_basic(fig, ax, data_used[:,ind_host[:3]], label = f'Host {host_name}', c = 'b', linewidth = 1)
                fig, ax = modplot.add_orbit_basic(fig, ax, data_used[:,ind_target[:3]], label = f'Target {target_name}', c = 'r', linewidth = 1)
                fig, ax = modplot.add_los(fig, ax, r_host[ii_vis,:], r_target[ii_vis,:], link_range = range_link, split = 10)
                fig, ax = modplot.add_glossary_basic(fig, ax, f'Link case {ii+1} : {case_title}')
                ax.view_init(rot_ver, rot_hor)
                if save_figs:                    
                    fig.savefig(f'{subfolder_plots}//{case_title}_3D.png', bbox_inches='tight')
    if make_overview_df:
        # get link case titles for the overview table
        host_labels = s_conv.get_label_names(hosts)
        target_labels = s_conv.get_label_names(targets)
        case_labels = [f'{ii+1} {case}' for ii, case in enumerate(link_cases.case_titles_brief)]
        overview_df = pd.DataFrame.from_dict({
            'Link' :case_labels,
            'Host' : host_labels,
            'Target' : target_labels,
            'R min [km]':r_min,
            'V_r max [m/s]':v_rel_max,
            'dAz max [urad]':paa_az_max,
            'dEl max [urad]':paa_el_max,
            'PAA 1d max [urad]':paa_1d_max                
            })
        # print(overview_df)
        if save_tables:
            overview_df.to_csv(f'{subfolder_tables}/{paa_table_name}.csv', index = False)
            print(f'Saved {paa_table_name}')
plt.show()            