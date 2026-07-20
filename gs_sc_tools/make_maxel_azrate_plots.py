#%% IMPORTS
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

path_overviews = r'sat_GS_contacts'


make_azrate_plot = 0
make_stats_gsel = 0
make_high_pass_tseries_plots = 0

all_data = os.listdir(path_overviews)
pass_data = [ii for ii in all_data if 'pass_overview' in ii]
# pass_data = [ii for ii in pass_data if 'T2d' in ii]
h_desired = [500, 1000] # altitudes for filtering
# h_desired = [1000] # altitudes for filtering
el_max = 85
azrate_max = 10
length_min = 60

if make_azrate_plot:
    f, ax = plt.subplots()
    # for h_a in h_desired[:1]: 
    for h_a in h_desired:    
        pass_with_h = [ii for ii in pass_data if str(h_a) in ii]
        path_full = fr'{path_overviews}/{pass_with_h[0]}'
        passes_df = pd.read_csv(path_full)
        if 1:
            print(f'{h_a} -> {passes_df.head()}')
        x_data = passes_df['peak_gs_el']
        y_data = passes_df['peak_azrate_gs']

        ax.scatter(x_data, y_data, label = f'h = {h_a:.0f} km', s = 2)

    ax.legend()
    ax.set_ylabel('Max. Azimuth Rate for GS telescope [deg/s]')
    ax.set_xlabel('Peak Elevation from GS [deg]')
    # ax.set_ylim([0, 15])
    ax.grid()


if make_stats_gsel:
    for h_a in h_desired:    
        pass_with_h = [ii for ii in pass_data if str(h_a) in ii]
        path_full = fr'{path_overviews}/{pass_with_h[0]}'
        passes_df = pd.read_csv(path_full)

        passes_total = passes_df[passes_df['length_observable']>length_min].shape[0]
        passes_above85 = passes_df[passes_df['peak_gs_el']>el_max]
        passes_aboveazrate = passes_df[passes_df['peak_azrate_gs']>azrate_max]

        print(f'''
            h = {h_a} km. Sim time: 180 days.
            total passes : {passes_total}
            above {el_max} deg el : {passes_above85.shape[0]}
            above {azrate_max} deg/s azrate : {passes_aboveazrate.shape[0]}

              ''')

if make_high_pass_tseries_plots:
    for h_a in h_desired:    
        f, ax = plt.subplots()
        pass_with_h = [ii for ii in pass_data if str(h_a) in ii]
        path_full = fr'{path_overviews}/{pass_with_h[0]}'
        passes_df = pd.read_csv(path_full)
        passes_total = max(passes_df['pass_nr'])
        passes_above85 = passes_df[passes_df['peak_gs_el']>el_max]
        passes_aboveazrate = passes_df[passes_df['peak_azrate_gs']>azrate_max]
        # get Timeseries data
        pass_ts_data = [ii for ii in os.listdir(path_overviews) if f'ble_h{h_a}' in ii][0]
        path_full_ts = fr'{path_overviews}/{pass_ts_data}'
        passes_df_ts = pd.read_csv(path_full_ts)
        for pass_nr in passes_aboveazrate['pass_nr'][1:]:
            # get time for pass_nr
            pass_dat = passes_aboveazrate[passes_aboveazrate['pass_nr'] == pass_nr]
            if pass_dat['peak_azrate_gs'].values[0] > 30 or 1:
                t_start = pass_dat['start_t_s'].values[0]
                t_end = t_start + pass_dat['length_observable'].values[0]
                pass_over = passes_df_ts[passes_df_ts['t'] >= t_start]
                pass_bothend = pass_over[pass_over['t'] <= t_end]        
                # plot  
                ax.plot(pass_bothend['t'] - pass_bothend['t'].values[0], np.abs(pass_bothend['gs_azimuth_rate']), 'o-')
                # ax.plot(pass_bothend['t'] - pass_bothend['t'].values[0], np.abs(pass_bothend['gs_azimuth']), 'o-')            
        ax.set_ylabel('Max. Azimuth Rate for GS telescope [deg/s]')
        ax.set_xlabel('t [s]')
        ax.grid()
        f.suptitle(f'Azimuth rates for passes over Wessling with Sat h = {h_a} km')
    for h_a in h_desired:    
        f, ax = plt.subplots()
        pass_with_h = [ii for ii in pass_data if str(h_a) in ii]
        path_full = fr'{path_overviews}/{pass_with_h[0]}'
        passes_df = pd.read_csv(path_full)
        passes_total = max(passes_df['pass_nr'])
        passes_above85 = passes_df[passes_df['peak_gs_el']>el_max]
        passes_aboveazrate = passes_df[passes_df['peak_azrate_gs']>azrate_max]
        # get Timeseries data
        pass_ts_data = [ii for ii in os.listdir(path_overviews) if f'ble_h{h_a}' in ii][0]
        path_full_ts = fr'{path_overviews}/{pass_ts_data}'
        passes_df_ts = pd.read_csv(path_full_ts)
        for pass_nr in passes_aboveazrate['pass_nr'][1:]:
            # get time for pass_nr
            pass_dat = passes_above85[passes_above85['pass_nr'] == pass_nr]
            if pass_dat['peak_gs_el'].values[0] > 30 or 1:
                t_start = pass_dat['start_t_s'].values[0]
                t_end = t_start + pass_dat['length_observable'].values[0]
                pass_over = passes_df_ts[passes_df_ts['t'] >= t_start]
                pass_bothend = pass_over[pass_over['t'] <= t_end]        
                # plot  
                ax.plot(pass_bothend['t'] - pass_bothend['t'].values[0], np.abs(pass_bothend['gs_elevation']), 'o-')
                # ax.plot(pass_bothend['t'] - pass_bothend['t'].values[0], np.abs(pass_bothend['gs_azimuth']), 'o-')            
        ax.set_ylabel('Max. Elevation [deg]')
        ax.set_xlabel('t [s]')
        ax.grid()
        f.suptitle(f'High GS Elevation passes over Wessling with Sat h = {h_a} km')
