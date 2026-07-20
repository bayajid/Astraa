#%% March 25, 2024, verifying QUEST implementation with simulations
# repurposed from Phase B sims 
import scipy as sp
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
os.chdir(sys.path[1])
import plotting_tools.basic_plotting as bplt
results_folder = r'/outputs/tables/ooc_ground_test'
all_files = os.listdir(results_folder)

f, ax = plt.subplots()
use_time = 0
# use_time = 1
storage_folder = 'ooc_'
error_text_box = f'''PEB Components Used:
r_h : 0 m
r_t : 0 m
att_h : 0 mrad
rss : 0.9 mrad
mean : 0.9 mrad'''
markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


for ii, file in enumerate(all_files):
    # load data
    data_csv = pd.read_csv(f'{results_folder}/{file}')
    
    # get label for trolley tilt
    trolley_angle = float(file[file.index('0_trolley')+9:-4])
    
    y_data = data_csv['pe_3sig'].values
    if use_time:
        x_data = data_csv['t_gap_s'].values/3600
        xlabel = 'Time waited [h]'
        x_lims = [0, 6]
        title_app = '_time'
    else:
        x_data = data_csv['ang_sep'].values
        xlabel = 'Angle between scanned LOS [deg]'
        x_lims = [5, 170]
        title_app = '_angsep'
    
    ax.plot(x_data, y_data, label = trolley_angle, marker = markers[ii], markevery=15)
ax.plot(ax.get_xlim(), [3,3], label = '3 mrad', c = 'y')
ax.set_xlabel(xlabel, fontweight = 'bold')
ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
ax.grid('on')
ax.set_xlim(x_lims)
ax.legend(bbox_to_anchor=(1, 0.6))
# ax.text(140.85, 0.2, 
#             error_text_box,
#             fontsize = 12,            
#             bbox = {'boxstyle': 'square',
#                     'facecolor':'peachpuff',                    
#                     }
#             )
    # ax.set_xscale('log')
    # ax.set_yscale('log')
ax.set_ylim([-0.1,20])
title = f'''Phase A GROUND Simualtions.
Expected MOQ vs Trolley/Ground tilt angle.
'''
f.suptitle(f'{title}')
# Reformat to get angle from 0:90 for non-colinearity
# ax.text(0.14, 0.2, 
#         error_text_box,
#         fontsize = 12,            
#         bbox = {'boxstyle': 'square',
#                 'facecolor':'peachpuff',                    
#                 }
#         )
figname = f'ooc_ground_sim_all_angles_{title_app}'
bplt.savefig(f, figname, subfolder =storage_folder, y_coord_tag = -5)
plt.show()