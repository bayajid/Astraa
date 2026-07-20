## CODE TO ANALYZE how the output mounting offset quaternion is sensitive
# to chosen row indices of the tracking logs

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
import analyses.on_orbit_calibration.MO_res_for_cust.resolve_Qmo_fromTrackingLogs as Q_mo_fix
importlib.reload(conv)
importlib.reload(Q_mo_fix)
#%%
path_base = r'analyses/on_orbit_calibration/MO_res_for_cust'
lognames = [
    'Moon_19-01-2024_200027_PM.csv', 
    'Moon_19-01-2024_203825_PM.csv'
           ]

arrs = [
    np.genfromtxt(f'{path_base}/{lognames[0]}', delimiter = ';', skip_header=3),
    np.genfromtxt(f'{path_base}/{lognames[1]}', delimiter = ';', skip_header=3)
]

q_mo_initial = conv.convert_daz_to_qmo(az_correction_manual=-(-60+0.4))

q_mo_ref = Q_mo_fix.get_q_mo_corrected(lognames, 
                                       arrs = arrs,
                                        q_mountingoffset_precal = q_mo_initial,
                                        row_nr_2 = None,
                                        row_nr_1 = None,
                                        path_base = path_base                                                    
                                           )
nrows_tried = 7000
stepsize = 10
outputs = np.zeros((int((nrows_tried/stepsize)**2), 6))


rowtracker = 0

for ii, row in enumerate(np.arange(1, nrows_tried, stepsize)):
    for jj, row in enumerate(np.arange(1, nrows_tried, stepsize)):
        q_mo_resolved = Q_mo_fix.get_q_mo_corrected(lognames, 
                                                    arrs = arrs,
                                                    q_mountingoffset_precal = q_mo_initial,
                                                    row_nr_2 = -ii*stepsize,
                                                    row_nr_1 = -jj*stepsize,
                                                    path_base = path_base                                                    
                                           )
        
        outputs[rowtracker, [0,1]] = [-ii, -jj]
        outputs[rowtracker, 2:] = q_mo_resolved
        rowtracker += 1

delta_mo_df = pd.DataFrame(data = outputs, columns = ['i1', 'i2', 'q0','q1','q2','q3'])
#%%        
        
df_chosen = delta_mo_df[delta_mo_df.values[:,1] != 0]

delta_mo = df_chosen.values[:,2:] - q_mo_ref
rel_diff = np.abs(delta_mo) / q_mo_ref*100
f, ax = plt.subplots()

for ii in range(4):
    ax.plot(rel_diff[:,ii], label = f'q{ii}')
ax.set_xlabel('Row combination nr [-]')
ax.set_ylabel('Rel difference from ref Q: delta Q / Q [%]')
ax.grid()
ax.legend()
# ax.set_ylim([-15,15])
# ax.set_xlim([0, 100])
