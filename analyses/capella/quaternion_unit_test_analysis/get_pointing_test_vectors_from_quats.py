## Mar 1, 2024
# Process grid-search results into pointing test-cases
import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
os.chdir(os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.conversions as conv
import attitude_tools.rotations as att_rot
import basic_tools.vector_operations as vec_op
import paa_tools.paa_calculation as paa_calc
import basic_tools.in_out as point_io

# To check if other quaternions lead to different rotations, 
# set to append_verification_cases=1
append_verification_cases = 0
rotation_option = 1


input_path = r'analyses\capella\quaternion_unit_test_analysis\outputs'
input_name = r'quat_output_filt.csv'
df_q = pd.read_csv(fr'{input_path}/{input_name}')

inputs_1 = df_q[df_q['los_eci_y'] != 0]
inputs_2 = df_q[df_q['los_eci_x'] != 0]


set_1 = inputs_1.values[[150, -1],:]
set_2 = inputs_2.values[[150, 0],:]
set_3 = df_q.values[6666, :]
# print(f'{set_1}')
# print(f'{set_2}')

v_q_chosen = np.array([set_3, set_1[0], set_1[1], set_2[1]])

los_eci = v_q_chosen[:,[1,2,3]]*10
quat_eci2bf = v_q_chosen[:,[4,5,6,7]]
print(f'Quat Given : {quat_eci2bf}')
# get host positions
r_host = np.array([[6e3, 5e3, -1e3],
                  [-1e3, -6e3, -6e3],
                  [-1e3, -6e3, -6e3],
                  [2e3, 0, 6e3]])
r_target = los_eci + r_host

quat_mo = np.array([1,0,0,0])
test_nr = []
in_t_gps_s = []
in_t_now_s = []
in_states_host = []
in_states_target = []
in_attitude_host = []
in_mounting_offset = []
out_ae_expected = []
paa_out_expected = []

for ii, q_ii in enumerate(quat_eci2bf):
    
    r_host_full = np.hstack([r_host[ii,:].flatten(), [0, 0, 0]]).reshape(1,6)
    r_target_full = np.hstack([r_target[ii,:].flatten(), [0, 0, 0]]).reshape(1,6)
    q_ii_input = np.hstack([q_ii, [0, 0, 0, 0]]).reshape(1,8)
    outputs_1 = paa_calc.compute_azel_modular(r_host_full, r_target_full, 
                                              q_ii_input, mounting_offset=quat_mo,
                                            rotation_function=rotation_option) 
    
    AE_out_1 = outputs_1[[3,4]]
    AE_out_deg_1 = np.rad2deg(AE_out_1)
    print(f'AE deg : {AE_out_deg_1}')
    paa_out = outputs_1[[-2,-1]]
    
    t_gps = 1325030348.816 + ii*3600 
    t_now = 1325030348.816 + ii*3600 + 0.25
    test_nr.append(ii+1)
    in_t_gps_s.append(t_gps)
    in_t_now_s.append(t_now)
    in_states_host.append(r_host_full.flatten())
    in_states_target.append(r_target_full.flatten())
    in_attitude_host.append(q_ii_input.flatten())
    in_mounting_offset.append(quat_mo)
    out_ae_expected.append(np.round(AE_out_1,5))
    paa_out_expected.append(np.round(paa_out,6))

append_name = ''    
if append_verification_cases:
    append_name = 'PlusVerifCheck'
    for ii, q_ii in enumerate(quat_eci2bf):
    
        r_host_full = np.hstack([r_host[ii,:].flatten(), [0, 0, 0]]).reshape(1,6)
        r_target_full = np.hstack([r_target[ii,:].flatten(), [0, 0, 0]]).reshape(1,6)
        q_ii_input = np.hstack([np.flip(q_ii), [0, 0, 0, 0]]).reshape(1,8)
        outputs_1 = paa_calc.compute_azel_modular(r_host_full, r_target_full, 
                                                q_ii_input, mounting_offset=quat_mo,
                                                rotation_function=rotation_option) 
        
        AE_out_1 = outputs_1[[3,4]]
        AE_out_deg_1 = np.rad2deg(AE_out_1)
        print(f'AE deg : {AE_out_deg_1}')
        paa_out = outputs_1[[-2,-1]]
        
        t_gps = 1325030348.816 + ii*3600 
        t_now = 1325030348.816 + ii*3600 + 0.25
        test_nr.append(ii+1)
        in_t_gps_s.append(t_gps)
        in_t_now_s.append(t_now)
        in_states_host.append(r_host_full.flatten())
        in_states_target.append(r_target_full.flatten())
        in_attitude_host.append(q_ii_input.flatten())
        in_mounting_offset.append(quat_mo)
        out_ae_expected.append(np.round(AE_out_1,5))
        paa_out_expected.append(np.round(paa_out,6))
        
    
in_t_gps_s = np.stack(in_t_gps_s)
in_states_host = np.stack(in_states_host)
in_states_target = np.stack(in_states_target)
in_attitude_host = np.stack(in_attitude_host)
in_mounting_offset = np.stack(in_mounting_offset)
out_ae_expected = np.stack(out_ae_expected)
importlib.reload(point_io)
point_io.save_azel(
    in_t_gps_s,
    in_states_host,
    in_states_target,
    in_attitude_host,
    in_mounting_offset,
    out_ae_expected,
    paa=paa_out_expected,
    full_folder = input_path,
            fname = f'quat_pointing_testvectors_rot{rotation_option}{append_name}'    
)