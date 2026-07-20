## Templates for loading satellite data
# generating attitude
# and whatnot. 

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
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()


## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import attitude_tools.attitude_resolution as att_res
if 0:
    # Feb 15, 21:30:12 raster-verification-scan
    AE_t = np.array([[17.34532322, 44.45992062], 
                    [ 6.3180629 , 36.92409322]])
    AE_c = np.array([[17.77380973, 42.21518552],
                    [ 6.74558644, 34.50332529]])                       
                    
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c, 
                                                            AE_found=AE_t,
                                                            deg = 1,
        check_non_colin=1)
    print(f'Angle: {np.rad2deg(angle_between_los)} deg')
    # Feb 15, 22:00:12 raster-verification-scan
    AE_t = np.array([[-0.48357638, 31.90945201],[-0.57972372, 29.44630738]])
    AE_c = np.array([[-3.01627902, 29.7551307 ],[-3.16289932, 27.24645249]])

    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c, 
                                                            AE_found=AE_t,
                                                            deg = 1,
    check_non_colin=1)
    print(f'Angle: {np.rad2deg(angle_between_los)} deg')
    #Pre-calibration Quat:
    q_precal = np.array([0.6756,0.0000,0.0000,0.7373])
    #Post-calibration:
    q_postcal = np.array([0.6781,-0.0190,0.0112,0.7346])


    print(f'Approximating MO errors. 21:30')

    AE_1 = np.array([[4.1, 35.2,], [4.25, 35.45]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_1,
                                                            deg = 1,
        check_non_colin=1)
    print(f'Solved difference: {angle_between_los*1e3:.1f} mrad')
    AE_initial = np.array([[4.75, 35.6,], [5.1, 33.2]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial,
                                                            deg = 1,
        check_non_colin=1)
    print(f'Initial difference: {angle_between_los*1e3:.1f} mrad')


    print(f'Approximating MO errors. 22:00')

    AE_2 = np.array([[-4.1, 28.9], [-4.65, 28.25]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'Solved difference: {angle_between_los*1e3:.1f} mrad')
    AE_initial_2 = np.array([[-3.85, 29.13], [-3.9, 25.6]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'Initial difference: {angle_between_los*1e3:.1f} mrad')


    AE_initial_2 = np.array([[-113, 67.6], [-111.5, 67.5]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'CHECK MID FEB 20 SOLUTION: {angle_between_los*1e3:.1f} mrad')

    AE_initial_2 = np.array([[-110, 67.5], [-106.5, 67.25]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'CHECK MID FEB 20: {angle_between_los*1e3:.1f} mrad')

    AE_initial_2 = np.array([[-113.12, 67.6], [-111.76, 67.4]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'CHECK NOWPO FEB 20: {angle_between_los*1e3:.1f} mrad')


    AE_initial_2 = np.array([[-111, 67.25], [-106.25, 67]])
    quat_mo, angle_between_los = att_res.get_mo_quat_fromscan(AE_exp=AE_c*0, 
                                                            AE_found=AE_initial_2,
                                                            deg = 1,
        check_non_colin=1)
    print(f'CHECK NOWPRE FEB 20: {angle_between_los*1e3:.1f} mrad')
else:
    ae_sets_start = np.array([[152, 60], [180,67], [-126, 61], [-95.4, 21.7], [-107, 9]])
    ae_sets_end = np.array([[67, 9], [74, 20], [102, 48], [153, 50.5], [-165, 35]])
    
    for ii, ae in enumerate(ae_sets_start):
        importlib.reload(att_res)
        angle = att_res.what_angle_between_ae(ae_sets_start[ii,:], ae_sets_end[ii,:], deg = 1, round = 1)
        print(f'{ae_sets_start[ii,:], ae_sets_end[ii,:]} -> {angle}')
    # angle_between_los = att_res.what_angle_between_ae(AE_initial_2[0,:], AE_initial_2[1,:], deg = 1)

