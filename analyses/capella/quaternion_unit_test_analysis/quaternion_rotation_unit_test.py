## February 27, 2024
# attempting to determine quaternions
# which perform the same rotation
# when used as q_conj V q and q V q_conj
import matplotlib.pyplot as plt
# import splines.quaternion
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.conversions as conv
import paa_tools.paa_calculation as paa_calc
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import basic_tools.in_out as point_io
import pointing_calculations.conversion_pointing as ae
importlib.reload(modplot)
importlib.reload(ae)
## Paths
np.set_printoptions(2)
subfolders_used = r'unit_tests'
# PLOTS FOR VERIFICATION/ANALYSIS
make_plots = 0
make_3d_plots = 0
save_outputs = 0
print_outputs = 0

## Placeholders
output_dir = {}
output_dir['case'] = []
output_dir['t_stamp_gps_s'] = []
output_dir['t_now_gps_s'] = []
output_dir['state_host_m'] = []
output_dir['state_target_m'] = []
output_dir['attitude_host_quat'] = []
output_dir['output_paa_azel_rad'] = []
output_dir['output_cpa_azel_rad'] = []

test_nr = []
in_t_gps_s = []
in_t_now_s = []
in_states_host = []
in_states_target = []
in_attitude_host = []
in_mounting_offset = []
out_ae_expected = []
paa_out_expected = []
test_case_names = [
    '0rot',
    '90az_0el',
    '-90az_0el',
]
for ii, case_name in enumerate(test_case_names):
    if 1:
        print(f'ii = {ii}, case {case_name}')
        attitude_host = [0,0,0,0,0,0,0,0]
        test_title = test_case_names[ii]
        test_case = ii + 1
        mounting_offset = np.array([1, 0, 0, 0])

        if test_case == 1: # unity quaternion, no rotation, 45 el
            states_host = [1, 1, 1, 1, 1, 1]
            states_target = [3e6, 1, 3e6, 0, 0, 0]
            RPY = [0,0,0]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 2: # 90, 0
            states_host = [1, 1, 1, 1, 1, 1]
            states_target = [2, 1, 3e6, 0, 0, 0]
            RPY = [-90,0,90]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 3: # -90, 0
            states_host = [1, 1, 1, 1, 1, 1]
            states_target = [3e6, 3e6, 1, 0, 0, 0]
            RPY = [0,0,135]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        states_host, states_target, attitude_host = np.array(states_host), np.array(states_target), np.array(attitude_host)
        # get outputs
        outputs = paa_calc.compute_azel_paa(states_host.reshape((1,6)), states_target.reshape((1,6)), 
                                            attitude_host.reshape((1,8)), mounting_offset = mounting_offset, 
                                            official_convention= 1)
        kk = 0
        for jj in range(4):
            if jj in [0,2]:
                kk = kk + 1
            
            t_gps = 1325030348.816 + ii*3600 + 0.25*(kk-2)
            t_now = 1325030348.816 + ii*3600 + 0.005*(jj-2)

            AE_out = outputs[:,[3,4]][0]
            AE_out_deg = np.rad2deg(AE_out)
            paa_out = outputs[:,[-2,-1]]
            test_nr.append(ii+1)
            in_t_gps_s.append(t_gps)
            in_t_now_s.append(t_now)
            in_states_host.append(states_host)
            in_states_target.append(states_target)
            in_attitude_host.append(attitude_host)
            in_mounting_offset.append(mounting_offset)
            out_ae_expected.append(np.round(AE_out,5))
            paa_out_expected.append(np.round(paa_out,6))
            # print(f'RPY input : {RPY}. ')
            if print_outputs:
                print(f''' TEST {test_case}
                RPY = {RPY}
                host : {states_host}
                target : {states_target}
                Quaternion : {attitude_host[:4]}
                
                Outputs : 
                AE : {AE_out} rad / {AE_out_deg} deg
                dAz : {outputs[:,[6]]} dEl {outputs[:,[7]]}
                ''')
                
        rot_eci2bf = conv.convert_ea2dcm(RPY)
        rot_bf2gf = conv.convert_quat2dcm(mounting_offset)
        RPY_mounting = conv.convert_dcm2ea(rot_bf2gf)
        los_given = states_target[:3] - states_host[:3]

        los_rotated = outputs[0,:3]
        los_length = np.linalg.norm(los_given)
        axlim_used = los_length*1.5
        axis_length = los_length
        if make_plots:
            f_2d, axs = plt.subplots(nrows = 2, ncols = 2)
            for ii, ax_row in enumerate(axs):
                for jj, ax_col in enumerate(ax_row):
                    if jj == 0: # COLUMNS 
                        # ECI
                        lot_plotted = los_given
                        title = 'ECI'
                        # plot eci
                    elif jj == 1:
                        # plot gf
                        title = 'GLOBAL '
                        lot_plotted = los_rotated
                        
                    if ii == 0: # ROWS 
                        # plot xy
                        xlabel = 'X'
                        ylabel = 'Y'
                        x_plot = [0, lot_plotted[0]]
                        y_plot = [0, lot_plotted[1]]
                    else:
                        # plot yz
                        xlabel = 'X/Y'
                        ylabel = 'Z'
                        x_plot = [0, np.linalg.norm(lot_plotted[0:2])]
                        y_plot = [0, lot_plotted[2]]
                    if 0:
                        print(f'{ii}-{jj} : {x_plot}')
                    title = f'{title}: {xlabel} - {ylabel} plane.'
                    max = np.max([np.abs(x_plot), np.abs(y_plot)])
                    axs[ii,jj].set_ylim(-max, max)
                    axs[ii,jj].set_xlim(-max, max)
                    axs[ii,jj].plot(x_plot, y_plot, label = 'LOS')
                    if jj == 1:
                        if ii == 0:
                            label_used = 'Azimuth = 0'
                            # axs[ii,jj].plot([axs[ii,jj].get_xlim()[0], 0], [0,0], c = 'm', linewidth = 1, label = 'Azimuth = 180')
                        elif ii == 1:
                            label_used = 'Elevation = 0'
                        axs[ii,jj].plot([0,axs[ii,jj].get_xlim()[1]], [0,0], c = 'r', linewidth = 1, label = label_used)
                        axs[ii,jj].legend()
                    axs[ii,jj].set_ylabel(ylabel)
                    axs[ii,jj].set_xlabel(xlabel)
                    axs[ii,jj].set_title(title)
                    axs[ii,jj].grid('on')
            f_2d.set_tight_layout('tight')
            f_2d.suptitle(f'Test case {test_case} - {test_title}\nRPY : {*np.round(RPY,2),} deg. Mt Offset : {*np.round(RPY_mounting,2),}\nAz El : {*np.round(AE_out_deg,2),} deg')

            if save_outputs or 0:
                bplt.savefig(f_2d, f'{test_case}_2d_{test_title}', subfolder = subfolders_used)        
        if make_3d_plots:
            print(f'3d FRAME plotting tests')

            c_eci = ['r', 'r', 'r']    
            c_gf = ['g', 'g', 'g']        
            

            rot_complete = rot_bf2gf @ rot_eci2bf
            title = f'''Test case {test_case} - {test_title}
            Input Attitude RPY : {*np.round(RPY,2),} deg
            Input Mounting Offset RPY : {*np.round(RPY_mounting,2),} deg
            Output Global Frame az, el : {*np.round(AE_out_deg,2),} deg
            '''
            f, ax = modplot.make_3dplot()
            f, ax = modplot.add_single_los(f, ax, states_host, states_target, draw_at_origin=1, normalize = 1, color = 'm', len_normalized=los_length, label_used = 'Host - target LOS')
            f, ax = modplot.add_ref_frame(f, ax, length = axis_length/1.3, use_axis_labels = 1, colors = c_eci)
            f, ax = modplot.add_ref_frame(f, ax, chosen_setting = 1, length = axis_length, use_axis_labels = 1, rot_gf=rot_complete, colors = c_gf)
            f, ax = modplot.add_glossary_basic(f, ax, y_title = 0.8,legend_loc = [0.2, 0.8], title = title, axlim = axlim_used)
            if save_outputs:
                view_angle = [RPY[1], RPY[2]-90]
                ax.view_init(view_angle[0], view_angle[1])
                bplt.savefig(f, f'{test_case}_3dViewside_{test_title}', subfolder = subfolders_used)
                view_angle = [5, 15]
                ax.view_init(view_angle[0], view_angle[1])
                bplt.savefig(f, f'{test_case}_3dTopview_{test_title}', subfolder = subfolders_used)
