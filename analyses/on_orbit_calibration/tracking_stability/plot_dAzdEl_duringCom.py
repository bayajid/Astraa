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
import cactus_tools.ooc.quat_mo_calculation as mo_calc
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

importlib.reload(mo_calc)
setting = 'pmg'
setting = 'moon'


try:
    folder_logs = r'analyses\on_orbit_calibration\tracking_stability\pmg_tracking_logs'
    logs_all = os.listdir(folder_logs)
except:
    folder_logs = r'pmg_tracking_logs'    

folder_logs= folder_logs.replace('pmg',setting)
logs_all = os.listdir(folder_logs)


logs_used = logs_all[:1]
logs_used = logs_all

d_angles_all = []
t_vecs_all = []
f_pe, ax_pe = plt.subplots()

f_dae, ax_dae = plt.subplots()
for ii, log in enumerate(logs_used):
    # read
    if '15-02' in log:
        col_track = -2
    else:
        col_track = -4
    
    log_data = mo_calc.get_cmd_from_log(f'{folder_logs}/{log}', rownr = 'all', give_tvec = 1, col_track = col_track)
    if log_data[0] is not None:
        # get az;el0
        ae_tracked = log_data[0]
        ae_0 = ae_tracked[0,:] # deg
        t_vec = log_data[2][1:]
        dae = ae_0 - ae_tracked[1:,:] 
        dae = np.deg2rad(dae)*1e3
        dangle = [1e3*vec_calc.calc_dot_angle(ae_0, ae_ii, 'polar', deg = 1) for ae_ii in ae_tracked[1:,:]]
        d_angles_all.append(dangle)
        t_vecs_all.append(t_vec)
        ax_pe.plot(t_vec, dangle, label = log)
ax_dae.plot(t_vec, dae[:,0], c = 'orange', label = 'dAz')
ax_dae.plot(t_vec, dae[:,1], c = 'green', label = 'dEl')
ax_dae.legend()
ax_dae.grid('on')
ax_dae.set_ylabel('Delta Az/El [mrad]')
ax_dae.set_xlabel('t [s]')
f_pe.suptitle(f'{setting.upper()} Delta Az/El since Comm start')

ax_pe.legend()
ax_pe.grid('on')
ax_pe.set_ylabel('Angular difference [mad]')
ax_pe.set_xlabel('t [s]')
ax_pe.set_ylim([0,1.8])
if setting == 'moon':
    ax_pe.set_xlim([0,5])
    
f_pe.suptitle(f'{setting.upper()} Angular changes of tracked Az/El since Comm starts')
