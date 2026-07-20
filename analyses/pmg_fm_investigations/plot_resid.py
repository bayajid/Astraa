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
from mat4py import loadmat

# path jazz
sys.path.insert(0, os.getcwd()[:os.getcwd().index('astropynaric')+13])
os.chdir(sys.path[0])

import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import plotting_tools.basic_plotting as bplt
import prediction_methods.interpolators as interp
import prediction_methods.attitude_prediction_methods as att_pred
import basic_tools.vector_operations as vec_op
# terminal_nr = 10003
terminal_nr = 10006
path_raw_logs = fr'analyses\pmg_fm_investigations\sn{terminal_nr}\ws_logs'
# path_outputs = r'analyses\attitude_predictions\outputs'
all_logs = os.listdir(path_raw_logs)
used_logs = all_logs[:1]

# for log in used_logs:
log = used_logs[0]
full_log_path = f'{path_raw_logs}/{log}'

# Load

# data = loadmat(full_log_path)
# data_mat = 0
data_mat = scipy.io.loadmat(full_log_path)['data']
columns = data_mat.dtype.descr

# columns = data_mat[case_used].dtype.descr
# loaded_mat = data_mat[case_used][0][0]

columns_needed =['AMC_Measurement',
'AMC_PowerStatus',
'AMC_Status',
'CTC_FsmPointingNom',
'CTC_FsmPointingRed',
'CTC_PointingNom',
'CTC_PointingRed',
'CTC_Status',
'EMC_Measurement',
'EMC_Status',
'FTC_Measurement1',
'FTC_Measurement2',
'FTC_PositionNom',
'FTC_PositionRed',
'OC_SystemControl',
'OC_Time',
'TSP_OpticalPower1',
'TSP_OpticalPower2',
'TSP_RawQuad1',
'TSP_RawQuad2']
columns_needed = ['OC_Time']
data_dict = {}
for ii, col in enumerate(columns_needed):
    data_dict[col] = data_mat[0][0][col][0][0][0][0][3].flatten()
data_df = pd.DataFrame.from_dict(data_dict)

a = 1