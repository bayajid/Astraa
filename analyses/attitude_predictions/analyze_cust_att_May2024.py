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
path_cust_data = r'analyses\attitude_predictions\cust_data'
path_outputs = r'analyses\attitude_predictions\outputs'
plots_saved_path = 'customer_attitude_may2024'
debug_mode = 0
save_outputs = 1 # WONT SAVE ANTHING
use_deg = 1 # conditional if deg are used
scalar_flip = 0
plot_simulated_att = 1 
plot_simulated_vsnoise = 1

plot_inputs = 0
plot_simstuff = 0 
plot_single = 0
plot_diff = 0
plot_omegas = 0
plot_ea_numrates = 0
plot_quat = 1
plot_ea_fromquat = 1

# Load customer data - euler angle (noise) + estimated euler rates
# data_used = 'may_start'
data_used = 'may_end'
if data_used == 'may_start':
    fname = 'AttEstDataforMynaric_2024-05-03.mat'
    scalar_flip = 1
    comp_own_qrate = 1
elif data_used == 'may_end':
    fname = 'AttEstDataforMynaric_2024-05-20.mat'
    scalar_flip = 0
    comp_own_qrate = 0
full_rate_path = fr'{path_cust_data}/{fname}'
data_mat = scipy.io.loadmat(full_rate_path)


for case_used in ['worstCaseX', 'worstCaseY', 'worstCaseZ']:
    columns = data_mat[case_used].dtype.descr
    loaded_mat = data_mat[case_used][0][0]
    for scalar_flip in [scalar_flip]:

        data_dict = {}
        for ii, col in enumerate(columns):
            data_dict[col[0]] = loaded_mat[ii].flatten()
        data_df = pd.DataFrame.from_dict(data_dict)
        data_array = data_df.values
        # loaded data
        quat_est = data_array[:,[8, 5,6,7]] # given as scalar last, flip to scalar-first
        quat_true = data_array[:,[4, 1,2,3]] # given as scalar last, flip to scalar-first

        if scalar_flip:
            quat_est[:,0] = - quat_est[:,0]
        # quat_true[:,-1] = -quat_true[:,-1]
        w_true = data_array[:,[9,10,11]]

        t_vec_loaded = data_array[:,0]
        if comp_own_qrate:
            # compute data
            quat_rate_est = np.array([conv.calc_qdot(rpy = None, q = q_ii, w = w_true[ii,:], deg = 1)[1].flatten() for ii, q_ii in enumerate(quat_est)])
        else:
            quat_rate_est = data_array[:,[15, 12, 13, 14]]
        if debug_mode:
            importlib.reload(bplt)
            savefig = 1
            quat_rate_est_num = np.gradient(quat_true, t_vec_loaded, axis = 0)
            colors = 'bryg'
            
            f, ax = plt.subplots()
            
            for ii in range(4):
                ax.plot(t_vec_loaded, quat_rate_est_num[:,ii], f'{colors[ii]}--', label = f'num_{ii}', markevery = 50)
                ax.plot(t_vec_loaded, quat_rate_est[:,ii], f'{colors[ii]}o-', label = f'q_c_{ii}', markevery = 50)
            ax.legend()
            ax.grid()
            # ax.set_xlim([0,50])
            ax.set_title('Quat rate calculation')

            bplt.savefig(f, 'quat_rate_calc', subfolder = plots_saved_path,y_coord_tag=-0.5, save = savefig)
            
            
            f, ax = plt.subplots()
            for ii in range(4):
                ax.plot(t_vec_loaded, quat_est[:,ii], f'{colors[ii]}--', label = f'q_est{ii}', markevery = 50)
                ax.plot(t_vec_loaded, quat_true[:,ii], f'{colors[ii]}o-', label = f'q_true{ii}', markevery = 50)
            ax.legend()
            ax.grid()
            # ax.set_xlim([0,50])
            ax.set_title('Quaternions- true and estimated')
            bplt.savefig(f, 'quat_true_est', subfolder = plots_saved_path,y_coord_tag=-0.5, save = savefig)
            quat_rate_est = np.array([conv.calc_qdot(rpy = None, q = q_ii, w = w_true[ii,:], deg = 1)[1].flatten() for ii, q_ii in enumerate(quat_true)])
            pe_stored, q_pre, t_vec_loaded = att_pred.get_quad_pred_error(quat_true, quat_rate_est, quat_true, t_vec_loaded)
            # plot
            f, ax = plt.subplots()
            ax_title = f'Data: {case_used}, TRUE DATA USED'
            ax.plot(t_vec_loaded, pe_stored)
            ax.set_ylabel('Pointing Error [urad]')
            ax.set_xlabel('t since start [s]')
            ax.set_title(ax_title)
            ax.grid()
            ax.set_xlim([0, max(t_vec_loaded)])
            bplt.savefig(f, name = f'{case_used}_qpred_flip{scalar_flip}VERIF', subfolder = plots_saved_path)
        else:
            pe_stored, q_pre, t_vec_loaded = att_pred.get_quad_pred_error(quat_est, quat_rate_est, quat_true, t_vec_loaded)
            # plot
            f, ax = plt.subplots()
            ax_title = f'Data: {case_used}, scalar term flipped : {bool(scalar_flip)}; 3-sigma PE : {np.percentile(pe_stored, 99.7):.0f} urad'
            ax.plot(t_vec_loaded, pe_stored)
            ax.set_ylabel('Pointing Error [urad]')
            ax.set_xlabel('t since start [s]')
            ax.set_title(ax_title)
            ax.grid()
            ax.set_xlim([0, max(t_vec_loaded)])
            bplt.savefig(f, name = f'{case_used}_qpred_flip{scalar_flip}', subfolder = plots_saved_path)