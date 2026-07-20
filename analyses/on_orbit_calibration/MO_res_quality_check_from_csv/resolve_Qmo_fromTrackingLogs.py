## Necessary tools and usage example for the Mounting Offset Quaternion Resolution
# Using 2 sets of Expected (Commanded) Azimuth/Elevation angles in the Terminal Frame
# and 2 sets of Measured (Detected) Azimuth/Elevation angles in the Terminal Frame
# in addition to the Mounting Offset Quaternion that was commanded during the 
# the current step of the calibration procedure.
# Date November 2, 2023
# Author: Kipras Paliusis

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
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import attitude_tools.conversions as conv
import attitude_tools.attitude_resolution as att_res
import analyses.on_orbit_calibration.MO_res_for_cust.resolve_Qmo as Q_mo_fix
importlib.reload(conv)

# read log
def get_cmd_from_log(logname, rownr = None, arr = None):
    
    if '15-02' in logname:
        ii_opmode = -2
        ii_amcemc = [5, 6]
        # ii_opmode = -4
        # ii_amcemc = [-2, -1]
        mode_desired = 7.0
    else:
        ii_opmode = -4
        ii_amcemc = [-2, -1]
        mode_desired = 7.0
    if arr is None:
        try:
            arr = np.genfromtxt(logname, delimiter = ';', skip_header=4)
        except:
            print(f'Cant load {logname}')
            return 0
        if len(arr) == 0:
            # print(f'No log rows found')
            return
        arr = arr[1:,:]
        comm_ind = np.where(arr[:,ii_opmode] == mode_desired)[0]
        if len(comm_ind) == 0:
            print(f'No communication rows found')
            return
        
        arr = arr[comm_ind,:]
    ae_cmd_all =  np.rad2deg(arr[:,[3, 4]])
    ae_cmd_all = ae_cmd_all
    
    threshold = 360
    
    ae_tracked =  np.rad2deg(1e-6*arr[:,ii_amcemc])
    ae_tracked = ae_tracked
    ii_where_cmd_above_threshold = np.where(np.abs(ae_cmd_all[:,0] > threshold))[0]
    if len(ii_where_cmd_above_threshold) != 0:
        ae_cmd_all[ii_where_cmd_above_threshold,:] = ae_cmd_all[ii_where_cmd_above_threshold,:]/1e6
    ii_where_track_above_threshold = np.where(np.abs(ae_tracked[:,0] > threshold))[0]
    if len(ii_where_cmd_above_threshold) != 0:
        ae_tracked[ii_where_track_above_threshold,:] = ae_tracked[ii_where_track_above_threshold,:]/1e6        
        
    if rownr is None:
        row_chosen = int(arr.shape[0]/1.3)
    else:
        row_chosen = rownr
        
    n_checks = 5
    fac = int(ae_tracked.shape[0]/n_checks)
    for ii in range(n_checks):
        ii = ii * fac 
        AE_initial_2 = np.array([ae_cmd_all[ii,:], ae_tracked[ii,:]])
        quat_mo, angular_offset = att_res.get_mo_quat_fromscan(AE_exp=np.zeros((2,2)), 
                                                                AE_found=AE_initial_2,
                                                                deg = 1,
                                                                check_non_colin=1)
        angular_offset = angular_offset*1e3 # mrad
        if angular_offset<1e3:
            
            print(f'AE: {AE_initial_2[0,:]};AE: {AE_initial_2[1,:]} -> {angular_offset:.1f} mrad')
    
    

if __name__ == '__main__':
    
    path = r'analyses\on_orbit_calibration\MO_res_quality_check_from_csv\csv_logs'
    fname_tracked_log = 'Moon_20-02-2024_213047_PM.csv'
    logs_all = os.listdir(path)
    # logs_all = [ii for ii in logs_all if '15-02' in ii]
    logs_all = [ii for ii in logs_all if '20-02' in ii]
    # logs_all = ['Moon_15-02-2024_220510_PM.csv']
    for log in logs_all:
        print(f'log: {log}')
        fname_tracked_log = log
        full_path = f'{path}/{fname_tracked_log}'
        
            
        get_cmd_from_log(full_path)