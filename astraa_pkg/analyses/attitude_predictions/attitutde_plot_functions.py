# Small attitude plotting scripts
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotting_tools.basic_plotting as bplt

def plot_ea_earates(t_vec, ea_input, ea_rates_est, ii_lim = -1, fname = 'EA and Rates', EA_unit = 'mdeg', savefig = 0):
    # Plot input euler angles and euler angle rates in 2 columns    
    f, axs = plt.subplots(nrows = 1, ncols = 2)
    ax = axs[0]
    for ii in range(3):
        ax.plot(t_vec[:ii_lim], ea_rates_est[:ii_lim,ii], label = 'RPY'[ii])
    ax.set_ylabel('Angular rates [deg/s]')
    ax.set_xlabel('t [s]')
    ax.grid('on')
    ax.legend()
    f.suptitle(fname)    

    ax = axs[1]
    for ii in range(3):
        ax.plot(t_vec[:ii_lim], ea_input[:ii_lim,ii], label = 'RPY'[ii])
    ax.set_ylabel('Euler Angles ' + f'[{EA_unit}]')
    ax.set_xlabel('t [s]')
    ax.grid('on')
    ax.legend()
    f.suptitle(fname)
    f.set_tight_layout('tight')
    if savefig:
        bplt.savefig(f, fname[:-4])
    return f, axs

def plot_ea_gradient(t_vec, ea_input):
    # get gradients and plot them of euler angle inputs
    r_input_diff = np.gradient(ea_input[:,0]/1e3, t_vec) 
    p_input_diff = np.gradient(ea_input[:,1]/1e3, t_vec) 
    y_input_diff = np.gradient(ea_input[:,2]/1e3, t_vec) 

    f, ax = plt.subplots()
    plotted_data = [r_input_diff, p_input_diff, y_input_diff]
    zorders = [0.8, 0.1, 1]
    alphas = [1, 0.3, 0.5]
    for ii in range(3):
        ax.scatter(t_vec, plotted_data[ii], label = ['Roll','Pitch', 'Yaw'][ii], s = 3, alpha = alphas[ii], zorder = zorders[ii])
    ax.set_ylabel('Differentitated RPY rates [deg/s]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid()
    ax.set_xlim([0, t_vec[-1]])
    ax.set_ylim([-2e-2, 2e-2])
    # f.suptitle('Numerically Differentiated Est Att, 10 Hz')
    return f, ax 
def plot_sim_ea_noisysea(t_vec, ea_clean, ea_noisy, ii_lim = -1, fname = 'EA simulated', EA_unit = 'deg', savefig = 0):
    # make plots of simulated euler angles
    # with the noisy euler angles overlaid with transparent scatters
    f, axs = plt.subplots(nrows = 3)
    
    for ii, ax in enumerate(axs):
        ax.plot(t_vec[:ii_lim], ea_clean[:ii_lim,ii], label = 'ea')
        ax.scatter(t_vec[:ii_lim], ea_noisy[:ii_lim,ii], label = 'ea + err', alpha = 0.3, s = 30)
        ax.grid('on')
        ax.legend()
        ax.set_ylabel(f'RPY'[ii] + f' [{EA_unit}]')

    ax.set_xlabel('t [s]')
    f.suptitle(fname)
    f.set_tight_layout('tight')
    if savefig:
        bplt.savefig(f, fname)
    return f, ax
def plot_ea(t_vec, ea_plotted, ii_0 = 0, ii_lim = -1, fname = 'EA simulated', EA_unit = 'deg', savefig = 0):
    # make plots of simulated euler angles
    # with the noisy euler angles overlaid with transparent scatters
    f, ax = plt.subplots()
    
    for ii in [0,1,2]:
        ax.plot(t_vec[ii_0:ii_lim], ea_plotted[ii_0:ii_lim,ii])
        ax.grid('on')
        ax.legend()
    ax.set_ylabel(f'EA' + f' [{EA_unit}]')

    ax.set_xlabel('t [s]')
    f.suptitle(fname)
    f.set_tight_layout('tight')
    if savefig:
        bplt.savefig(f, fname)
    return f, ax

def add_lines_and_annotation(ax, x_data, y_data, x_chosen, 
                             c, text_annotation, 
                             xy_text = None,
                             alpha = 0.5):    
    # get 2 indices before and after req time
    # TODO add docstring and move to modular plotting
    ii_higher = [ii for ii, xval in enumerate(x_data) if xval > x_chosen][0]
    ii_lower = [ii for ii, xval in enumerate(x_data) if xval < x_chosen][-1]

    # interpoalte value
    interpolant = sp.interpolate.interp1d(x_data[[ii_lower, ii_higher]], y_data[[ii_lower, ii_higher]])
    yval_h = interpolant(x_chosen)

    ymin = ax.get_ylim()[0]
    # horizontal line
    ax.plot([0, x_chosen], [yval_h, yval_h], f'{c}--', alpha = alpha)
    # vertical line
    ax.plot([x_chosen, x_chosen], [ymin, yval_h], f'{c}--', alpha = alpha)

    ## Add annotation
    if type(xy_text) == type(None):
        xy_text = (-0.65, yval_h*.6)
    ax.annotate(text_annotation, xy = [0.02, yval_h], xytext = xy_text, c = c, 
                arrowprops=dict(facecolor=c, arrowstyle='-'),
                annotation_clip = False, 
                bbox=dict(boxstyle="round", fc="w"))
    return ax
def plot_dq_pe(t_vec_prediction, quat_true_sliced, q_predicted, pe_pred, attitude_setting, update_rate
                                    ):
    f, axs = plt.subplots(ncols = 2)

    ax = axs[0] # dq plot
    for ii in range(4):
        ax.plot(t_vec_prediction, quat_true_sliced[:,ii] - q_predicted[:,ii+1], label = f'dq{ii}')
    ax.legend()
    ax.grid('on')
    ax.set_ylabel('Pred Quaternion Error')
    ax.set_ylim([-1e-4, 1e-4])
    ax = axs[1] # PE
    
    ax.plot(t_vec_prediction, pe_pred)
    ax.set_ylabel('PE [urad]')
    ax.set_xlabel('t [s]')
    ax.set_xlim([t_vec_prediction[0], t_vec_prediction[0]+5])
    ax.grid('on')
    ax.set_yscale('log')
    ax.set_ylim([1e1, 10e4])
    f.suptitle(f'{attitude_setting}, {update_rate} Hz - Quaternion error and PE')
    f.set_tight_layout('tight')
    return f, axs
def plot_pe(t_vec, pe_pred, t_pred_start, attitude_setting):
    f, ax = plt.subplots()
    ax.plot(t_vec, pe_pred)
    ax.set_ylabel('PE [urad]')
    ax.set_xlabel('t [s]')
    ax.set_xlim([-2 + t_pred_start, t_pred_start+2])
    ax.grid('on')
    ax.set_yscale('log')
    ax.set_ylim([1e1, 10e4])
    ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')

    f.suptitle(f'{attitude_setting}, PE')
    f.set_tight_layout('tight')
    return f, ax
def plot_dq(t_vec, t_pred_start, quat_true, quat_est, attitude_setting):
    f, ax = plt.subplots()

    for ii in range(4):
        ax.plot(t_vec, quat_true[:,ii] - quat_est[:,ii], label = f'dq{ii}')
    
    
    ax.grid('on')
    ax.set_ylabel('Pred Quaternion Error')
    ax.set_ylim([-1e-4, 1e-4])
    ax.set_xlim([t_pred_start-2, t_pred_start+2])
    ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
    f.suptitle(f'{attitude_setting}, - True vs Est. Quaternion')
    ax.legend()    
    f.set_tight_layout('tight')
    return f, ax

def plot_q_msg_pred(t_vec, quat_true, quat_predicted, 
                    q_message, attitude_setting):
    f, axs = plt.subplots(nrows = 4)
    f.suptitle(f'Quaternions [Scalar-first] - {attitude_setting}')
    t_pred_start = q_message[-1,0]
    t_vec_prediction = quat_predicted[:,0]
    for ii, ax in enumerate(axs):
        ax.set_ylabel(['q1', 'q2', 'q3', 'q_0'][ii] + ' [-]', fontweight = 'bold')
        ax.plot(t_vec, quat_true[:,ii], label = 'True', zorder = 0.6)
        ax.scatter(q_message[:,0], q_message[:,ii+1], label = 'MSG', c = 'y', s = 10)
        ax.plot(t_vec_prediction, quat_predicted[:,ii+1], label = 'Predicted', marker = 'x', markevery = 5, zorder = 0.5)
        ax.grid()
        ax.set_ylim([(q_message[-1,1+ii])-0.1, q_message[-1,1+ii]+0.1])
        ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
        if ii == 3:
            ax.legend()
        ax.set_xlim([t_pred_start-5, t_pred_start+10])
    ax.set_xlabel('t [s]', fontweight = 'bold')
    f.set_tight_layout('tight')
    return f, axs

def read_and_add_pe(f, ax, 
                    path_outputs, 
                    attitude_case, 
                    setting, 
                    pe_filter = '',
                    rows_used = 100,
                    normalize_x = 1,
                    plot_label = '',
                    colors = ['c', 'b', 'g', 'r', 'm'],
                    marker = 'o',
                    zord = 0.1, 
                    upd_rates_plotted = [10, 5, 1],
                    add_hz_to_label = 1,
                    add_annotation = 0):
    """function to detect PE csv files
    for a chosen attitude case and prediction setting
    first finds .csv's fitting the criteria, reads them,
    stores into a single dataframe and then plots all of them
    if their update rates match the upd_rates_plotted

    Args:
        f (figure object): original figure to have the PE plotted on
        ax (ax): 
        path_outputs (path string): where the PE outputs are located
        attitude_case (string): attitude case description
        setting (string): prediction type description (eg 'quadratic')
        pe_filter (str, optional): used to exclude additionally detected results. 
        eg nojerk to only get jerk. Defaults to ''.
        rows_used (int, optional): Nr rows loaded. Defaults to 100.
        normalize_x (bool, optional): Whether x axis should be [Nr Updates] or [s]. Defaults to 1.
        plot_label (str, optional): First part of label used. Defaults to ''.
        colors (list, optional): List of colors. Defaults to ['c', 'b', 'g', 'r', 'm'].
        marker (str, optional): Marker for type. Defaults to 'o'.
        zord (float, optional): Order of plotted lines. Defaults to 0.1.
        upd_rates_plotted (list, optional): Filter to choose which Hz to plot. Defaults to [10, 5, 1].
        add_hz_to_label (bool, optional): whether to mention update rate. Defaults to 1.
        add_annotation (bool, optional): if annotations are added. NOT SUPPORTED YET. Defaults to 0.

    Returns:
        f, ax, PE_df: PE_df - overview of PE found as a function of time
    """    
    # Get list of files
    PE_all = os.listdir(path_outputs)
    PE_scenario = [PE for PE in PE_all if attitude_case in PE]
    PE_chosen = [PE for PE in PE_scenario if setting in PE]
    if len(pe_filter) != 0:
        PE_chosen = [PE for PE in PE_chosen if pe_filter not in PE]
    # Check if anything was found
    if len(PE_chosen) == 0:
        print(f'''PE PLOTTING : No inputs were found for:
        attitude: {attitude_case}
        prediction setting: {setting}
        removing entries with : {pe_filter} (filter not used if empty)
        results searched for in :{path_outputs}        
        ''')
        return f, ax, 0
    
    # Create overview dataframe
    pe_overview = np.zeros((rows_used,len(PE_chosen)))
    upd_rates_found = []
    for ii, PE_file in enumerate(PE_chosen):
        pe_df = pd.read_csv(fr'{path_outputs}/{PE_file}')
        # determine update interval from filename
        for upd_rate in upd_rates_plotted:
            if str(upd_rate) in PE_file:
                upd_rate_detected = upd_rate
                break
        upd_rates_found.append(upd_rate_detected)
        pe_overview[:,ii] = pe_df['pe_urad'][:rows_used]
    pe_overview_df = pd.DataFrame(columns = upd_rates_found, data = pe_overview)
    upd_rates_found = [float(ii) for ii in upd_rates_found]
    upd_rates_found = list(sorted(upd_rates_found))

    # get time vector
    t_vec = pe_df['t_s'][:rows_used]
    t_vec_plot = t_vec - t_vec[0]
    
    ## PLOT
    for ii, upd_rate in enumerate(upd_rates_found):  
        if upd_rate in upd_rates_plotted:            
            if normalize_x:
                x_data = t_vec_plot * upd_rate
                marker_int = int((1/upd_rate)/0.1)
            else:
                x_data = t_vec_plot
                marker_int = int(1/0.1)

            # PLOT
            if add_hz_to_label:
                label = f'{plot_label} {upd_rate:.0f} Hz'
            else:
                label = f'{plot_label}'

            ax.plot(x_data, pe_overview_df[upd_rate],
                    label = label,
                    markevery=marker_int , marker = marker, 
                    c = colors[ii], zorder = zord)
            text_annotation = f'PE, {int(1/upd_rate):.0f} Hz \n{plot_label[:-1]}'
            if add_annotation:
                ax = add_lines_and_annotation(ax, t_vec_plot, pe_overview_df[upd_rate].values,
                                            upd_rate, c = colors[ii], text_annotation=text_annotation )
            ii += 1
    return f, ax, pe_overview_df