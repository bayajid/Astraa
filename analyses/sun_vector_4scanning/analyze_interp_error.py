import datetime as dt
import pandas as pd
import numpy as np
from time import process_time
from scipy.interpolate import interp1d
### KP:MAY 30 UPDATES FROM HERE
    
# cb = mk3can()
# cb.start_receive()
# cb.init_fts()

# Load csv
path_to_ae = "ae_gs2sun.csv"
ae_df = pd.read_csv(path_to_ae)
ae_vals = ae_df.iloc[:,2:].values
# KP May 30 : New file to track ae_gs2sun.csv reference time!
path_to_tinfo = 'ref_time.csv'
t_info_df = pd.read_csv(path_to_tinfo)
# Auto-parse start time from ref_time.csv
month_ref = t_info_df['month_used'].values[0]
day_ref =  t_info_df['day_used'].values[0] 
hour_ref = t_info_df['h_start'].values[0]
minute_ref = 0 # KP: OK as long as ref time is also from minute=0
t_ref = dt.datetime(2023, 5, day_ref, hour_ref, minute_ref, 0)
# current time
t_now = dt.datetime.now() 
# tiem difference
t_difference = (t_now - t_ref)
t_res_sun_angles = t_info_df['t_res'].values[0]
print(f'''Sun-pointing angle data:
month start {month_ref}
day start {day_ref}
hour start {hour_ref}
time res [s]: {t_res_sun_angles}
        ''')
seconds_since_start = int(t_difference.seconds) + t_difference.microseconds/1e6 # seconds
# round
n_digits_used = len(str(t_res_sun_angles))-2 # 3 digits for 5 ms
seconds_since_start = np.round(seconds_since_start, n_digits_used)
ii_start = int(seconds_since_start/t_res_sun_angles)

print(f'''Runtime:{t_now} SLICING INSPECTION
    Reference data time {t_ref}
    Time difference between now and ref start time: {seconds_since_start} s
    for dt = {t_res_sun_angles} s, start row : {ii_start}''')

# get start row
print(f'Automatically chosen row : {ae_df.iloc[ii_start,:]}')
ae_sun_used = ae_vals[ii_start:, :]
t_since_start = ae_df.iloc[:,0] - ae_df.iloc[0,0] # Time vector since start [s]
t_since_start = t_since_start.values
az_offsets  = ae_sun_used[:,0] - ae_sun_used[0,0] # azimuth vector since start, from 0
el_used = ae_sun_used[:,1] # elevation vector since start, actual value
az_used  = ae_sun_used[:,0]
## Jun 1st - Making sun az/el interpolator 
spline_length = 30 # minutes
ii_end = np.where(t_since_start < spline_length*60)[0][-1]

az_true = az_used[:ii_end]
el_true = az_used[:ii_end]
t_true = t_since_start[:ii_end]


az_interpolant = interp1d(t_true, az_true)
el_interpolant = interp1d(t_true, el_true)
use_interpolant = 1

az_interp = az_interpolant(t_true)
el_interp = el_interpolant(t_true)

az_error = az_interp - az_true
el_error = az_interp - el_true
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot(t_true, az_error, label = 'd Az')
ax.plot(t_true, el_error, label = 'd El')
ax.legend()
ax.set_ylabel('Interp error [urad]')