#%% imports/paths make combined plots for a single extrap method
# in no/max jerk conditions (no - constant accelerations. Max - acceleration swap at t=0 (prediction time-scale))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import os
import importlib
import sys
import scipy as sp
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import plotting_tools.basic_plotting as bplt

path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/attitude_tools/outputs'
# attitude_used_nojerk = 'rotate_all_axes'
# attitude_used_jerk = 'rotate_swap' # default - all 3 axes rotating like mad. Switch acceleration sign to negative at t=40s
attitude_used_nojerk = 'rotate_all_pred084'
attitude_used_jerk = 'rotate_all_pred084_swap'
# attitude_used = 'rotate_all_delayed'
# attitude_used = 'real_stable_sat'

# def add_horizontal_vertical_line(ax, ydata, xdata, x_value, c):
#     # function to add a horizontal and vertical line
#     # to show Y value of a graph at a certain X value

if 'real' in attitude_used_jerk:
    rows_used = 800
    marker_int = 1
else:
    rows_used = 1000
    marker_int = 50
PE_all = os.listdir(path_outputs)
PE_all_jerk = [file for file in PE_all if attitude_used_jerk in file]
PE_all_nojerk = [file for file in PE_all if attitude_used_nojerk in file]
PE_all_nojerk = [file for file in PE_all_nojerk if 'swap' not in file]
zoom = 0
savefig = 0
log = 1
plot_interp_smart = 1
upt_intervals_plotted = [0.25, 1] 
plots_shown_title = f'AttitudeExtrapError'
colors = ['b', 'g', 'y', 'r', 'm']
plot_jerk = 1
plot_nojerk = 1
plot_nothing = 1


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
#%%
f, ax = plt.subplots()
# setting_plotted = 'quadratic_interp_basic'

if plot_jerk:
    zord = 0.7
    plot_label = 'max jerk,'
    setting_plotted = 'interp_2pts_smarter'
    marker = '^'
    PE_all = PE_all_jerk
    PE_chosen = [PE for PE in PE_all if setting_plotted in PE]
    # make overview for plotting
    pe_overview = np.zeros((rows_used,len(PE_chosen)))
    upd_intervals_overview = []
    for ii, PE_file in enumerate(PE_chosen):
        # upd_int = PE_al
        pe_df = pd.read_csv(fr'{path_outputs}/{PE_file}')
        # determine update interval from filename
        for upd_interval in upt_intervals_plotted:
            if str(upd_interval) in PE_file:
                interval_file = upd_interval
                break
        upd_intervals_overview.append(interval_file)
        pe_overview[:,ii] = pe_df['pe_urad'][:rows_used]
    pe_overview_df = pd.DataFrame(columns = upd_intervals_overview, data = pe_overview)
    upd_intervals_overview = [float(ii) for ii in upd_intervals_overview]
    upd_intervals_overview = list(sorted(upd_intervals_overview))
    t_vec = pe_df['t_s'][:rows_used]
    t_vec_plot = t_vec - t_vec[0]
    ## Add plots
    ii = 0 
    for upd_interval in upd_intervals_overview:  
        if upd_interval in upt_intervals_plotted:
            upd_freq = 1/ upd_interval
            ax.plot(t_vec_plot, pe_overview_df[upd_interval],markevery=marker_int , marker = marker, label = f'{plot_label} {upd_freq:.0f} Hz', c = colors[ii], zorder = zord)
            text_annotation = f'PE, {int(1/upd_interval):.0f} Hz \n{plot_label[:-1]}'
            ax = add_lines_and_annotation(ax, t_vec_plot, pe_overview_df[upd_interval].values,
                                          upd_interval, c = colors[ii], text_annotation=text_annotation )
            ii += 1

    if plot_nothing and 0: #
        ## Add error if nothing is done
        PE_file_nothing = [PE for PE in PE_all if 'do_nothing' in PE][0]
        pe_nothing_df = pd.read_csv(fr'{path_outputs}/{PE_file_nothing}')
        pe_nothing_df = pe_nothing_df.fillna(0)
        pe_nothing = pe_nothing_df['pe_urad'][:rows_used]
        t_vec_nothing = pe_nothing_df['t_s'][:rows_used] - pe_nothing_df['t_s'][0]
        ax.plot(t_vec_nothing, pe_nothing, label = f'{plot_label} no prediction', linewidth = 2, c = 'm', zorder = zord)
        
        
        
if plot_nojerk:
    zord = 0.7
    plot_label = 'no jerk,'
    setting_plotted = 'interp_2pts_smarter'
    marker = 'o'
    PE_all = PE_all_nojerk
    PE_chosen = [PE for PE in PE_all if setting_plotted in PE]
    # make overview for plotting
    pe_overview = np.zeros((rows_used,len(PE_chosen)))
    upd_intervals_overview = []
    for ii, PE_file in enumerate(PE_chosen):
        # upd_int = PE_al
        pe_df = pd.read_csv(fr'{path_outputs}/{PE_file}')
        # determine update interval from filename
        for upd_interval in upt_intervals_plotted:
            if str(upd_interval) in PE_file:
                interval_file = upd_interval
                break
        upd_intervals_overview.append(interval_file)
        pe_overview[:,ii] = pe_df['pe_urad'][:rows_used]
    pe_overview_df = pd.DataFrame(columns = upd_intervals_overview, data = pe_overview)
    upd_intervals_overview = [float(ii) for ii in upd_intervals_overview]
    upd_intervals_overview = list(sorted(upd_intervals_overview))
    t_vec = pe_df['t_s'][:rows_used]
    t_vec_plot = t_vec - t_vec[0]
    
    ## Add plots
    ii = 0 
    for upd_interval in upd_intervals_overview:  
        if upd_interval in upt_intervals_plotted:
            upd_freq = 1/ upd_interval
            ax.plot(t_vec_plot, pe_overview_df[upd_interval],markevery=marker_int , marker = marker, label = f'{plot_label} {upd_freq:.0f} Hz', c = colors[ii], zorder = zord)
            text_annotation = f'PE, {int(1/upd_interval):.0f} Hz \n{plot_label[:-1]}'
            xy_text = [(-0.65, 0.2), None][ii]
            ax = add_lines_and_annotation(ax, t_vec_plot, pe_overview_df[upd_interval].values,
                                          upd_interval, c = colors[ii], text_annotation=text_annotation, xy_text = xy_text
                                         )
            ii += 1

    if plot_nothing:
        ## Add error if nothing is done
        PE_file_nothing = [PE for PE in PE_all if 'nothing' in PE][0]
        pe_nothing_df = pd.read_csv(fr'{path_outputs}/{PE_file_nothing}')
        pe_nothing_df = pe_nothing_df.fillna(0)
        pe_nothing = pe_nothing_df['pe_urad'][:rows_used]
        t_vec_nothing = pe_nothing_df['t_s'][:rows_used] - pe_nothing_df['t_s'][0]
        ax.plot(t_vec_nothing, pe_nothing, label = f'no prediction', linewidth = 2, c = 'm', zorder = zord)

if 1:
    ax.plot([0, 3], [100, 100], 'r--', label = r'PE = 100 $\mu$rad')
ax.legend()
ax.set_ylabel('Estimation\nError [$\\mu$rad]', rotation = 0)
ax.yaxis.set_label_coords(-0.15, 0.94)
ax.set_xlabel('Prediction time [s]')
# ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75])
ax.set_xticks([0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3])
if zoom:
    ax.set_xlim([0,1])
    ax.set_ylim([0,100])
else:
    # ax.set_xlim([0,10])
    ax.set_xlim([0,3])
ax.grid()
f.set_tight_layout('tight')
if log and not zoom:
    ax.set_yscale('log')
    ax.set_ylim([1e-2, 1e5])
else:
    if not zoom:
        ax.set_ylim([0, 500])
# f.suptitle(plots_shown_title, fontweight = 'bold')
if savefig:
    if zoom:
        name_start = f'PE_zoom_'
    else:
        name_start = f'PE_'
    bplt.savefig(f, f'{name_start}{plots_shown_title}', timetag = 1)
