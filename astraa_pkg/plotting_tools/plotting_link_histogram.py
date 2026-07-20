## DRAFT - histogram plots, carried over from thesis work. 
def loadplot_timeseries(
    sat_host,
    case_analyzed, 
    lct_chosen,
    link_type = 'general',
    output_path_link = None,
    t_length = 8,
    t_start = 0,
    cols_used = ['r_h', 'dr_h', 'az_h', 'daz_h', 'el_h', 'del_h'],
    link_category_chosen = 'leadfoll',
    data_type_chosen = 'linkrates',
    ii = 0,
    sat_target_limits = '_4_5',
    sat_target_exclusion = None,
    labels_used = ['Slant range [km]', 'Slant rate [km/s]', 'Azimuth [deg]', 'Az. rate [deg/s]', 'Elevation [deg]', 'El. rate [deg/s]'],
    legend = 1,
    title = None,
    grid = 0,
    ):
    # Function to load a data file based on the provided folders
    # (output path link, link_type)
    # then choose the data based on sat_host and link-category_chosen
    # Proceed to plot the specified csv columns as a timeseries
    # with t given in [hr]
    # carried over from MSc thesis work
    if type(output_path_link) == type(None):
        output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
    data_folder = f'{output_path_link}\{link_type}'
    data_all = os.listdir(data_folder)
    data_available_host = [file for file in data_all if sat_host in file]
    data_available_lct = [file for file in data_available_host if lct_chosen in file]
    data_chosen = [file for file in data_available_lct if link_category_chosen in file.replace(sat_host,'')]
    # print(data_chosen)
    data_name = [file for file in data_chosen if data_type_chosen in file][0]
    data_csv = pd.read_csv(f'{data_folder}/{data_name}')

    nrows, ncols = 3,2
    t_end = t_start*3600 + t_length* 3600
    data_cut = data_csv[data_csv['t'] < t_end]
    # plot glossary
    title = f'Coplanar LEO I {lct_chosen} to leader' if type(title) == type(None) else title
    fig, axs = plt.subplots(nrows, ncols, figsize = (nrows * 5, ncols * 3))
    label_sats = dputil.get_label_names(case_analyzed) # labels to be used for plots
    for mm, sat_target in enumerate(case_analyzed):
        if type(sat_target_limits) != type(None):   
            if sat_target_limits in sat_target:
                if type(sat_target_exclusion) != type(None):
                    if sat_target_exclusion in sat_target:
                        continue
                df_used = data_cut[data_cut['sat_target'] == sat_target]
                if not df_used.empty:
                    t_vec = df_used['t']/60
                    for ii, col_needed in enumerate(cols_used):
                        data_to_plot = df_used[col_needed]
                        # slant [m] convert to [km]
                        if 'km' in labels_used[ii]: 
                            data_to_plot = data_to_plot / 1e3 
                        # track indices
                        jj = ii%ncols # column index
                        if ii>1:
                            nr = 1 # row index
                        else:
                            nr = 0
                        if ii>3:
                            nr = 2 
                        ax = axs[nr, jj]
                        if grid:
                            ax.grid()
                        ax.scatter(t_vec, data_to_plot, s = 2, marker = '_', label = label_sats[mm])
                        ax.set_ylabel(labels_used[ii], weight = 'bold')
                        ax.set_xlim([0, t_end/60 + 20])
                        if nr==2:
                            ax.set_xlabel('t [min]', weight = 'bold')
    for ii in range(6):
        jj = ii%2 # column index
        nr = 2 if ii>1 else 0
    if legend:
        axs[nr, jj].legend(markerscale=5)
    plt.suptitle(title, size = 14, weight = 'bold')
    plt.tight_layout()
    plt.show()
    return fig, axs

def plot_cum_hist_twindow(ax_chosen,
        t_used,
        t_step,
        title_used,
        t_cutoff,
        cut_y= 1,
        ylim = None,
        reverse_cumul = 1,
        cumul_color = 'red',
        invert_x = 1,
        c = 'cornflowerblue'
        ):
    # Function to plot a histogram on a single ax
    # inputting t_step and t_used (range of communication times)
    # carried over from MSc thesis work
    bin_range = np.arange(0, max(t_used)+t_step*2, t_step)
    x_data, y_data, cumul_data = [],[], []
    cumul = 0
    for ii, bin_0 in enumerate(bin_range[:-1]):
        t_l = bin_0
        t_h = bin_0 + t_step
        t_bin = (t_used >= t_l) & (t_used < t_h)
        n_t_bin = sum(t_bin)
        rt_n_t_bin = np.round(n_t_bin / len(t_used)*100,2)
        # print(f'[{t_l}:{t_h}) - {n_t_bin} and {rt_n_t_bin}%')
        cumul +=(rt_n_t_bin)/100
        cumul_data.append(cumul)
        x_data.append(t_l)
        y_data.append(rt_n_t_bin)    
    if reverse_cumul:
        cumul = 1
        cumul_data = []
        for y in y_data:
            cumul -= y/100
            cumul_data.append(cumul)

    ax_chosen.bar(x_data, y_data, width = t_step, align = 'edge', color = c)    
    ax_chosen.xaxis.grid('on')
    ax_bis = ax_chosen.twinx()
    # ax_chosen.set_xlabel(xlabel, fontsize = 14)
    ax_chosen.set_xticks(np.arange(0, t_cutoff+t_step*2, t_step))
    ax_chosen.set_xlim([0, t_cutoff+t_step*2])
    # ax_chosen.set_ylabel(ylabel, fontsize = 14)
    ax_chosen.set_title(title_used, fontsize = 16)
    ax_bis.spines['right'].set_color(cumul_color)
    ax_bis.tick_params(axis='y', colors=cumul_color)
    ax_bis.yaxis.label.set_color('red')
    if cut_y:
        ii_lim = [cc for cc,cumul in enumerate(cumul_data) if np.round(cumul,1) >= 1][-1]
        if x_data[ii_lim] == t_cutoff:
            ax_chosen.set_xlim([0, x_data[ii_lim] + t_step])
        else:
            ax_chosen.set_xlim([0, x_data[ii_lim]])
    else:
        ax_chosen.set_xlim([0, t_cutoff+5])
    if type(ylim) != type(None):
        ax_chosen.set_ylim([min(ylim), max(ylim)])
    x_datac = x_data.copy()
    x_datac = [x + t_step for x in x_datac]
    if reverse_cumul:
        cumul_data.insert(0,1)
        cumul_data.append(0)
        x_datac.append(t_cutoff + t_step)
    else:
        cumul_data.insert(0,0)
        cumul_data.append(1)
        x_datac.append(t_cutoff + t_step)
    x_datac.insert(0,0)
    ax_bis.plot(x_datac, cumul_data, c = cumul_color)
    # ax_bis.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax_bis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_bis.grid()
    ax_bis.set_ylim([0, 1.1])
    if invert_x:
        # ax_bis.invert_xaxis()
        ax_chosen.invert_xaxis()
    return ax_chosen, ax_bis

def plot_hist_nrlinks(ax_chosen,
        x_data,
        y_data,
        c = 'blue',
        title_used = None,
        xlabel = 'Constellation Shell/Plane [-]',
        ylabel = 'Nr. Links [-]',
        ticks = 'vertical',
        ylim = [0, 50]
        ):
    # Function to plot a histogram on a single ax
    # inputting t_step and t_used (range of communication times)
    # carried over from MSc thesis work
    ax_chosen.bar(x_data, y_data, align = 'center', color = c)    
    ax_chosen.set_ylim([min(ylim), max(ylim)])
    # ax_chosen.grid('on')
    # ax_bis = ax_chosen.twinx()

    return ax_chosen