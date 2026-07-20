#%% analyze customer provided data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import scipy.io
import importlib
import os, sys
# path jazz
sys.path.insert(0, os.getcwd()[:os.getcwd().index('astropynaric')+13])
os.chdir(sys.path[0])

import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import plotting_tools.basic_plotting as bplt
import prediction_methods.interpolators as interp
import prediction_methods.attitude_prediction_methods as att_pred
import basic_tools.vector_operations as vec_op

import pynaric.simple_calibrate_cpa as pyn_cal
importlib.reload(pyn_cal)
pmg_nr = 1

path_raw_logs = r'analyses\pmg_fm_investigations'
# terminal_used = 'sn10003'
terminal_used = 'sn10006'
if pmg_nr!= 1:
    log_path = fr'{path_raw_logs}/{terminal_used}/cactus_logs_{pmg_nr}'
else:
    log_path = fr'{path_raw_logs}/{terminal_used}/cactus_logs'
out_path = fr'{path_raw_logs}/outputs/{terminal_used}'
storage_files = [out_path]
all_logs = os.listdir(log_path)



