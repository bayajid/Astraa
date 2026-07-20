#%% Functi ont ogenerate some crazy spacecraft attitude
# with angular velocities and rates. Several possible settings

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
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import attitude_tools.attitude_predictions.attitutde_plot_functions as attplt
import plotting_tools.basic_plotting as bplt
importlib.reload(attplt)
update_rate = 10

ea_to_rate = 0
rate_to_ea = 1
ea_to_quat = 1
quat_to_ea = 1

data_folder = r'attitude_tools\attitude_predictions\cust_data'
fname_att = 'BusAttEst'
fname_atterr = 'BusRateEst'
fname_quat = 'q_SVtoECI'
fname_att_upd = f'{fname_att}_{update_rate}Hz.mat'

fname_atterr_upd = f'{fname_atterr}_{update_rate}Hz.mat'
fname_quat_upd = f'{fname_quat}_{update_rate}Hz.mat'

data_att = scipy.io.loadmat(f'{data_folder}\{fname_att_upd}')[fname_att_upd[:-4]][0][0][0][1:,:]
data_atterr = scipy.io.loadmat(f'{data_folder}\{fname_atterr_upd}')[fname_atterr_upd[:-4]][0][0][0][1:,:]
data_quat = scipy.io.loadmat(f'{data_folder}\{fname_quat_upd}')[fname_quat_upd[:-4]][0][0][0][1:,:]

if ea_to_rate:
    f, ax = attplt.plot_ea_gradient(data_att[:,0], data_att[:,[1,2,3]])
    bplt.savefig(f, 'RPYrate_from_ea_cust')
if rate_to_ea:
    f, ax = 0,0
if ea_to_quat:
    data_quat_converted = np.zeros(data_quat.shape)
    for ii, row in enumerate(data_att):
        rpy_deg = row[1:] / 1e3
        quat_converted = conv.convert_ea2quat(rpy_deg)
        data_quat_converted[ii,0] = row[0]
        data_quat_converted[ii,1:] =  quat_converted
        # pass
    f, ax = 0,0