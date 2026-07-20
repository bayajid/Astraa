## Mar 1, 2024
# grid-search for quaternions which
# lead to simialr rotations in multiple conventions
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
import attitude_tools.rotations as att_rot
import basic_tools.vector_operations as vec_op
# PLOTS FOR VERIFICATION/ANALYSIS
make_plots = 1
save_outputs = 1
## Paths
np.set_printoptions(2)
subfolders_used = r'unit_tests'

## Generate coarse grid to search
vectors_to_rotate = np.array([[1, 0, 1], [1, -1, -1], [0, -1, -1]])*1e3
q_range = [1, -1]
q_steps = 0.1

q_1_range = np.arange(q_range[-1], q_range[0]+q_steps, q_steps)
q_2_range = np.copy(q_1_range)
q_3_range = np.copy(q_1_range)
q_4_range = np.copy(q_1_range)

# Placeholders
nr_iterations = int(vectors_to_rotate.shape[0] * len(q_1_range)**4)
rot_vec_array = np.zeros((nr_iterations, 3))
quat_array = np.zeros((nr_iterations, 4))
angle_array = np.zeros((nr_iterations, 1))
## Choose rotation functions
rotation_op_1 = att_rot.rotate_with_quat_mat
rotation_op_2 = att_rot.rotate_with_quat_mat_swaperoo
## loop over and store
# loss-function- angular difference between two vectors
# make loops
ii_loop = 0
for ii_v, v_ii in enumerate(vectors_to_rotate):
    for ii_1, q_1 in enumerate(q_1_range):
        for ii_2, q_2 in enumerate(q_2_range):
            for ii_3, q_3 in enumerate(q_3_range):
                for ii_4, q_4 in enumerate(q_4_range):
                    q_ii = np.array([q_1, q_2, q_3, q_4])
                    q_ii = q_ii/np.linalg.norm(q_ii)
                    
                    vec_1 = rotation_op_1(v_ii, q_ii, norm = 0)
                    vec_2 = rotation_op_2(v_ii, q_ii, norm = 0)
                    
                    loss_ii = vec_op.calc_dot_angle(vec_1, vec_2)
                    
                    # store
                    rot_vec_array[ii_loop,:] = v_ii
                    quat_array[ii_loop,:] = q_ii
                    angle_array[ii_loop,:] = loss_ii
                    
                    ii_loop+=1
                    
#%% Save Outputs
urad_threshold = 1
output_df_whole = pd.DataFrame(data = np.hstack((angle_array*1e6, rot_vec_array, quat_array)),
                               columns = ['err_urad', 'los_eci_x', 'los_eci_y', 'los_eci_z', 'q_1', 'q_2', 'q_3', 'q_4']
                               )

output_df_filtered = output_df_whole[output_df_whole['err_urad'] < urad_threshold]
print(f'Outputs total : {output_df_whole.shape[0]}; filtered to {urad_threshold} urad -> {output_df_filtered.shape[0]}')
                    

if save_outputs:
    output_path = r'analyses\quaternion_unit_test_analysis\outputs'
    output_df_whole.to_csv(fr'{output_path}/quat_output_full.csv', index = 0)
    output_df_filtered.to_csv(fr'{output_path}/quat_output_filt.csv', index = 0)
    print('Outputs Saved')
                    
#%%                    
if make_plots:
    f, axs = plt.subplots(nrows = 5)
    f.suptitle('Used Quaternion and error')
    for ii, ax in enumerate(axs[:-1]):
        ax.plot(quat_array[:,ii], 'bo', alpha = 0.6, markersize = 0.1)
        ax.set_ylabel(f'q_{ii+1}')
    
    ax = axs[-1]
    ax.plot(angle_array*1e6, 'bo', alpha = 0.6, markersize = 0.1)
    ax.set_ylabel('Error [urad]')
    
    # add vertical lines
    for ii, ax in enumerate(axs):
        ax.set_ylim(ax.get_ylim())
        ax.set_xlim(0, nr_iterations)
        ax.grid('on')
        for jj in range(3):
            ax.plot(np.array([nr_iterations/3, nr_iterations/3])*(jj+1), np.array(ax.get_ylim())*1.2, 'r')
    f.set_in_layout('tight')
    plt.show()
    
    

if save_outputs:
    pass