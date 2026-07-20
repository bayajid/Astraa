# ADDING PLOTS WITH NEW CUSTOMER DATA
#  X AXIS "normalized" - unit is t/upd interval -> nr. updates available
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
# if os.getcwd()[:os.getcwd().index('astropynaric_repo')+30] != os.getcwd():
#     os.chdir(os.getcwd()[:os.getcwd().index('astropynaric_repo')+30])

import plotting_tools.basic_plotting as bplt
import attitutde_plot_functions as attplt

import importlib
importlib.reload(attplt)

path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/analyses/attitude_predictions/outputs'
attitude_used_nojerk = 'attitude_nojerk_noisy'
attitude_used_jerk = 'attitude_jerk_noisy'
attitude_used_customer = 'customer_may11'

savefig = 1
# def add_horizontal_vertical_line(ax, ydata, xdata, x_value, c):
#     # function to add a horizontal and vertical line
#     # to show Y value of a graph at a certain X value
plot_no_prediction_customer = 1
plot_no_prediction_nojerk = 1
plot_cust_noise = 1
plot_jerk_pred_noisy = 1
plot_nojerk_pred_noisy = 1
plot_cust_pred_noisy = 1
xlim = 6

x_norm = 1

rows_used = 100
f, ax = plt.subplots()
if plot_jerk_pred_noisy:
    setting = 'quadratic'
    attitude_case = 'jerk_noisy'
    pe_filter = 'nojerk'
    plot_label = 'jerk'
    zorder = 0.5
    marker = 'o'
    f, ax, jerk_noise_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm, zord = zorder)

if plot_nojerk_pred_noisy:
    setting = 'quadratic'
    attitude_case = 'nojerk_noisy'
    pe_filter = ''
    plot_label = 'no jerk'
    marker = 'x'
    f, ax, nojerk_noise_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm)

if plot_cust_pred_noisy:
    setting = 'quadratic'
    attitude_case = 'customer_may11'
    zorder = 0.3
    pe_filter = ''
    plot_label = 'cust. propagated'
    marker = '>'
    f, ax, cust_pred_noisy_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm, zord = zorder)

if plot_no_prediction_customer:
    setting = 'nothing_5Hz'
    attitude_case = 'customer_may11'
    colors = ['m']
    zorder = 0.1
    pe_filter = ''
    plot_label = 'cust. no prediction'
    marker = 0
    f, ax, no_prediction_customer_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm, zord = zorder, colors= colors)

if plot_no_prediction_nojerk:
    setting = 'nothing_5Hz'
    attitude_case = 'nojerk_noisy'
    colors = ['m']
    zorder = 0.3
    pe_filter = ''
    plot_label = 'no jerk, no prediction'
    marker = 10
    f, ax, no_prediction_nojerk_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm, zord = zorder, colors= colors)
if plot_cust_noise:
    setting = 'ideal_5Hz'
    attitude_case = 'customer_may11'
    colors = ['r']
    zorder = 0.7
    pe_filter = ''
    plot_label = 'APE+AKE'
    marker = 10
    add_hz_to_label = 0
    f, ax, cust_noise_pe = attplt.read_and_add_pe(f, ax, path_outputs, attitude_case, plot_label = plot_label, setting = setting, pe_filter = pe_filter, marker = marker, rows_used = rows_used, normalize_x= x_norm, zord = zorder, colors= colors, add_hz_to_label = add_hz_to_label)

if 1:
    ax.plot([0, xlim], [100, 100], 'r--', label = r'PE = 100 $\mu$rad', linewidth = 2)
    # ax.legend(loc='right')
    ax.legend(loc='upper right')
    ax.set_ylabel('Estimation\nError [$\\mu$rad]')
    if x_norm:
        ax.set_xlabel('t [available updates]')
    else:
        ax.set_xlabel('t [s]')

    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75])
    # ax.set_xticks([0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

    ax.set_xlim([0,xlim])
    ax.set_ylim([1e1, 1e5])
    ax.set_yscale('log')

    ax.grid()
    f.set_tight_layout('tight')
    if savefig:
        name_start = 'PE_customer'
        bplt.savefig(f, name_start, timetag = 1)
