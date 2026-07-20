import matplotlib.pyplot as plt
import numpy as np
from plotting_tools.basic_plotting import savefig
from matplotlib.ticker import MaxNLocator
# Functions to allow for plotting data in a loop
def autoscale_yaxis(ax, autolimscale, set_bins = 0, n_bins = 5, force_0 = 1):
    axlims = ax.get_ybound()
    ylims = [0,0]
    for ll in range(2):
        if axlims[ll] < 0:
            ylims[ll] = axlims[ll] * (autolimscale**(-1*(ll*2-1)))
        else:
            ylims[ll] = axlims[ll] / (autolimscale**(-1*(ll*2-1)))
    if force_0:
        # show 0 if Y data does not intersect 0
        if ylims[0] > 0:
            ylims[0] = 0
        if ylims[1] < 0:
            ylims[1] = 0
    ax.set_ylim(ylims)
    if set_bins: # set max number of Y axis ticks (generally adds more ticks)        
        ax.yaxis.set_major_locator(MaxNLocator(n_bins)) 
    return ax
def plot_aer(t, aer,
             f = None, 
             axs = None,
               title = '', setting = 'rate', 
             axlim = 'auto', unit = 'deg', 
             line_type = 'o-',
             r_scale = 3,
             autolimscale = 1.5, 
             set_xlims = 1,
             x0 = 1,
             force_0 = 1,
             r_lim = None,
             ii_vis = None, 
             save_figure = 0):
    """Function to plat aer as a fct of time

    Args:
        t (_type_): time
        aer (_type_): az, el, r or rates [deg, m]
        f (_type_, optional): _description_. Defaults to None.
        axs (_type_, optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to ''.
        setting (str, optional): Val or rate. Defaults to 'rate'.
        axlim (str, optional): set xlimits automatically scaled. Defaults to 'auto'.
        unit (str, optional): angle unit. Defaults to 'deg'.
        line_type (str, optional): for plots. Defaults to '--'.
        r_scale (int, optional): _description_. Defaults to 3.
        autolimscale (float, optional): _description_. Defaults to 1.5.
        set_xlims (int, optional): _description_. Defaults to 1.
        x0 (int, optional): set t0 to 0. Otherwise auto from t_vec. Defaults to 1.
        force_0 (int, optional): force y axis to cross 0. Defaults to 1.
        r_lim (_type_, optional): _description_. Defaults to None.
        ii_vis (_type_, optional): _description_. Defaults to None.
        save_figure (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """    # function to plot az, el, slant-range
    # time [s]    
    
    if type(f) == type(None):
        make_newfig = 1
    else:
        make_newfig = 0
    if make_newfig:
        f,axs = plt.subplots(nrows = 3, sharex=1)
        f.suptitle(title, fontweight = 'bold')
    if r_scale == 1:
        r_unit = 'm'
    elif r_scale == 3:
        r_unit = 'km'
    for ii, ax in enumerate(axs):
        if type(ii_vis) != type(None):
            y_plotted = aer[ii_vis,ii]
            x_plotted = t[ii_vis]/60
        else:
            x_plotted = t/60
            y_plotted = aer[:,ii]
        
        if ii == 2:
            if r_scale == 3:
                y_plotted = y_plotted / 10**r_scale

        ax.plot(x_plotted,y_plotted, line_type)
        
        if setting != 'rate':
            ax.set_ylabel([f'Az [{unit}]', f'El [{unit}]', f'Slant-range [{r_unit}]'][ii], fontweight = 'bold')
        else:
            ax.set_ylabel([f'Az rate [{unit}/s]', f'El rate [{unit}/s]', f'Slant-rate [{r_unit}/s]'][ii], fontweight = 'bold')
        ax.grid('on')
        if set_xlims:
            if x0 == 1:
                ax.set_xlim([min(x_plotted), max(x_plotted)])
            else:
                ax.set_xlim([x0, max(x_plotted)])
        if axlim == 'auto': # Auto-scale y axes
            ax = autoscale_yaxis(ax, autolimscale, set_bins = 1, force_0 = force_0)
    if type(r_lim) != type(None):
        ax.set_ylim([50, 5e3])
        ax.plot([x_plotted[0], x_plotted[-1]], [4e3, 4e3], 'ro-', label = 'R max')
        ax.plot([x_plotted[0], x_plotted[-1]], [0.5e3, 0.5e3], 'go-', label = 'R min')
        ax.legend()
    ax.set_xlabel('t [min]', fontweight = 'bold')
    if make_newfig:
        f.set_tight_layout('tight')

    if save_figure:
        savefig(f, f'{title}_AER{setting}', save_as_matfig=1)
    return f, axs
def plot_paa(t_vec, paa, paa_1d, fname = '', 
             save_figure = 0, axlim = 'auto', autolimscale = 1.5, 
             ii_vis = None):


    paa_1dim, paa_dot_relmotion = paa_1d
    
    #% plot PAA
    # f, axs = plt.subplots(nrows = 3, figsize = (12, 8))
    f, axs = plt.subplots(nrows = 3)
    f.suptitle(fname, fontweight = 'bold')
    for ii, ax in enumerate(axs[:2]):
        if type(ii_vis) != type(None):
            y_plotted = paa[ii_vis,ii]
            x_plotted = t_vec[ii_vis]/60
        else:
            x_plotted = t_vec/60
            y_plotted = paa[:,ii]
        ax.plot(x_plotted, y_plotted, label = 'robust ' + ['dAz', 'dEl'][ii] + ', tangential vel.', marker = 'o', markevery = 30)
        # ax.plot(t_vec, paa_analytical[:,ii+1], label = 'analytical ' + ['dAz', 'dEl'][ii] + ', v_rel')
        ax.set_ylabel('2-way d' + ['Az','El'][ii] + ' [urad]', fontweight = 'bold')
        # ax.set_xlabel('t [s]', fontweight = 'bold')
        # ax.legend()
        ax.grid()
    ax = axs[2]
    if type(ii_vis) != type(None):
        paa_1dim_plotted = paa_1dim[ii_vis]
        paa_dot_relmotion_plotted = paa_dot_relmotion[ii_vis]
    else:    
        paa_1dim_plotted = paa_1dim
        paa_dot_relmotion_plotted = paa_dot_relmotion
    
    ax.plot(x_plotted, paa_1dim_plotted, label = 'PAA analytical - 2 v_rel_tangential / c ', marker = 'o', markevery = 30)
    ax.plot(x_plotted, paa_dot_relmotion_plotted, label = 'PAA - dot product LOT_Tx, LOS_Rx')
    
    if axlim == 'auto': # Auto-scale y axes
        for ax in axs:
            ax = autoscale_yaxis(ax, autolimscale, set_bins = 1)
    ax.set_ylabel('2-way PAA [urad]', fontweight = 'bold')
    ax.set_xlabel('t [min]', fontweight = 'bold')
    ax.legend()
    ax.grid()
    f.set_tight_layout('tight')
    if save_figure:
        savefig(f, f'{fname}_PAA', save_as_matfig = 1)
    return f, axs


def plot_ae(t, ae, title = '', setting = 'notrate', 
            axis_label_appends = '',
             axlim = 'auto', unit = 'deg', 
             s = 5,
             autolimscale = 1.5, ii_vis = None, save_figure = 0, label = '',
             alpha = 1):
    # function to plot az, el, slant-range
    # time [s]    
    f,axs = plt.subplots(nrows = 2)
    f.suptitle(title, fontweight = 'bold')
    for ii, ax in enumerate(axs):
        if type(ii_vis) != type(None):
            y_plotted = ae[ii_vis,ii]
            x_plotted = t[ii_vis]/60
        else:
            x_plotted = t/60
            y_plotted = ae[:,ii]
        
            
        ax.plot(x_plotted,y_plotted, '--', label = label, alpha = alpha)
        
        if setting != 'rate':
            ax.set_ylabel([f'{axis_label_appends}Az [{unit}]', f'{axis_label_appends}El [{unit}]', 'Slant-range [m]'][ii], fontweight = 'bold')
        else:
            ax.set_ylabel([f'{axis_label_appends}Az rate [{unit}/s]', f'{axis_label_appends}El rate [{unit}/s]', 'Slant-rate [m/s]'][ii], fontweight = 'bold')
        ax.grid('on')
        if axlim == 'auto': # Auto-scale y axes
            ax = autoscale_yaxis(ax, autolimscale, set_bins = 1)
    if axlim == 'equal':
        ax_lims = []
        ax_gap_az = np.max(ae[:,0]) - np.min(ae[:,0])
        ax_gap_el = np.max(ae[:,1]) - np.min(ae[:,1])
        if ax_gap_az > ax_gap_el:
            axlims = [np.max(ae[:,0]), np.min(ae[:,0])]
        else:
            axlims = [np.max(ae[:,1]), np.min(ae[:,1])]
        for ii, ax in enumerate(axs):
            ax.set_ylim(axlims[0], axlims[1])
    ax.set_xlabel('t [min]', fontweight = 'bold')
    f.set_tight_layout('tight')

    if save_figure:
        savefig(f, f'{title}_AER{setting}')
    return f, axs

def plot_states(states_in, t_in = None,title=''):
    if t_in is None:
        t_in = np.arange(0, states_in.shape[0], 1)
    f, axs = plt.subplots(nrows = 2, sharex=1)
    for ii, ax in enumerate(axs):
        ax.plot(t_in, states_in[:,ii*3+0], label = 'x')
        ax.plot(t_in, states_in[:,ii*3+1], label = 'y')
        ax.plot(t_in, states_in[:,ii*3+2], label = 'z')
        ax.set_ylabel(['r ', 'v '][ii] + ['m', 'm/s'][ii])
    ax.legend()
    f.suptitle(title)
    plt.tight_layout()
    return f,axs
def plot_quats(quats_in, t_in = None,title=''):
    if t_in is None:
        t_in = np.arange(0, quats_in.shape[0], 1)
    f, axs = plt.subplots(nrows = 2, sharex=1)
    for ii, ax in enumerate(axs):
        ax.plot(t_in, quats_in[:,ii*4+0], label = 'c')
        ax.plot(t_in, quats_in[:,ii*4+1], label = '1')
        ax.plot(t_in, quats_in[:,ii*4+2], label = '2')
        ax.plot(t_in, quats_in[:,ii*4+3], label = '3')
        ax.set_ylabel(['q', 'q_dot'][ii])
    ax.legend()
    f.suptitle(title)
    plt.tight_layout()
    
    return f,axs


def add_ae(f, axs, t, ae, label, alpha = 1):
    for ii, ax in enumerate(axs):
        x_plotted = t/60
        y_plotted = ae[:,ii]
        ax.plot(x_plotted,y_plotted, '--', label = label, alpha = alpha)
        ax.legend()
    return f, axs