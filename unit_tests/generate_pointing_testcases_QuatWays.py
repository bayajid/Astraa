## April 20, 2023
# script to generate truth vectors for Pointing Computations
# provide inputs for host/target states and host attitude
# and expected outputs of the rotated LOS; Az El (global frame) and PAA dAz dEl 
#%%
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
save_outputs = 1
print_outputs = 1

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
    '145az_-30el',
    '-90az_-90el',
    '-135az_-45el',
    'coplanar_leo_link',
    'crossplane_leo_link',
    'coplanar_leo_mtsideways',
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

        elif test_case == 4: # 90, 0
            states_host = [0, 0, 0, 1, 1, 1]
            states_target = [0.612, 0.612, -0.5, 0, 0, 0]
            RPY = [0,179.9,90]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 5: # -90, -90
            states_host = [1e6, 1e6, 1e6, 1, 1, 1]
            states_target = [-1e6, 1e6, -1e8, 2, 2, 2]
            RPY = [0,0,270]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 6: # -90, -90
            states_host = [0, 0, 0, 1, 1, 1]
            states_target = [-1e3, 1.414e3, 1e3, 2, 2, 2]
            RPY = [270, 90, 180]
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 7: # coplanar sat pointing 
            states_host = np.array([-2.34786955e+06,  1.21948903e+05,  6.98634834e+06, -6.97157247e+03,
        -4.08058220e+01, -2.33974544e+03])
            states_target = np.array([-5.14722258e+06,  9.20964927e+04,  5.27612558e+06, -5.26523471e+03,
                -8.95750927e+01, -5.13331212e+03])
            quat_raw = np.array([ 0.00140588,  0.16136584, -0.0086119 , -0.98685608])
            dcm_raw = np.array([[-9.47918181e-01, -5.55413286e-03, -3.18465500e-01],        [-4.53263848e-06, -9.99847717e-01,  1.74511325e-02],        [-3.18513929e-01,  1.65436892e-02,  9.47773804e-01]])
            view_angle = [90, 5]

            RPY = conv.convert_dcm2ea(dcm_raw)
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 8: # cross-plane LEO Inclined sat pointing 
            states_host = np.array([-6.93018337e+06,  1.51949847e+06,  2.01105039e+06, -2.51791513e+03,
        -4.15493048e+03, -5.52209166e+03])
            states_target = np.array([-6.75541786e+06, -2.83587490e+06,  8.29322062e+05,  1.11419255e+03,
        -4.36114448e+03, -5.81801600e+03])
            quat_raw = np.array([ 0.18164497,  0.54415496, -0.25988549, -0.77676252])
            view_angle = [90, 5]
            dcm_converted = conv.convert_quat2dcm(quat_raw)
            RPY = conv.convert_dcm2ea(dcm_converted)
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat

        elif test_case == 9: # coplanar LEO with mounting offset 45 deg yaw
            states_host = np.array([-2.34786955e+06,  1.21948903e+05,  6.98634834e+06, -6.97157247e+03,
        -4.08058220e+01, -2.33974544e+03])
            states_target = np.array([-5.14722258e+06,  9.20964927e+04,  5.27612558e+06, -5.26523471e+03,
                -8.95750927e+01, -5.13331212e+03])
            quat_raw = np.array([ 0.00140588,  0.16136584, -0.0086119 , -0.98685608])
            dcm_raw = np.array([[-9.47918181e-01, -5.55413286e-03, -3.18465500e-01],        [-4.53263848e-06, -9.99847717e-01,  1.74511325e-02],        [-3.18513929e-01,  1.65436892e-02,  9.47773804e-01]])
            view_angle = [90, 5]

            RPY = conv.convert_dcm2ea(dcm_raw)
            quat = conv.convert_ea2quat(RPY)
            attitude_host[:4] = quat        # convert to arrays
            mounting_offset = np.array([
                np.cos(np.deg2rad(22.5)), 
                0, 
                0,
                np.sin(np.deg2rad(22.5))
            ])
        states_host, states_target, attitude_host = np.array(states_host), np.array(states_target), np.array(attitude_host)
        # get outputs
        outputs = paa_calc.compute_azel_modular(states_host.reshape((1,6)), states_target.reshape((1,6)), 
                                            attitude_host.reshape((1,8)), mounting_offset = mounting_offset, 
                                            rotation_function=2,
                                            official_convention= 1)
        kk = 0
        for jj in range(4):
            if jj in [0,2]:
                kk = kk + 1
            
            t_gps = 1325030348.816 + ii*3600 + 0.25*(kk-2)
            t_now = 1325030348.816 + ii*3600 + 0.005*(jj-2)

            AE_out = outputs[[3,4]]
            AE_out_deg = np.rad2deg(AE_out)
            paa_out = outputs[[-2,-1]]
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
            if print_outputs and jj == 3:
                print(f''' TEST {test_case}  AE : {AE_out} rad / {AE_out_deg} deg''')