# Originally used in June to get starting row of sun-vector csv's. 
# best off just using an interpolated sun-vector, avoid slicing
import datetime as dt
import pandas as pd
import numpy as np
# Hardcoded refernce time
day_ref =  9 
hour_ref = 16
minute_ref = 0
t_ref = dt.datetime(2023, 5, day_ref, hour_ref, minute_ref, 0)
# current time
t_now = dt.datetime.now() 
# tiem difference
t_difference = (t_now - t_ref)
t_resolution = 1 # s
seconds_since_start = int(t_difference.seconds) # seconds
ii_start = int(seconds_since_start/t_resolution)

print(f'''Runtime:{t_now} SLICING INSPECTION
    Reference data time {t_ref}
    Time difference between now and ref start time: {seconds_since_start} s
    for dt = {t_resolution} s, start row : {ii_start}''')

# read file
# UPDATE PATH!
path_to_ae = r"C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\astropynaric_repo\astropynaric\outputs\tables\sun_vector\2023-05-09\ae_gs2sun.csv"
ae_df = pd.read_csv(path_to_ae)
ae_vals = ae_df.iloc[:,2:].values
# get start row
print(f'Automatically chosen row : {ae_df.iloc[ii_start,:]}')
ae_sun_used = ae_vals[ii_start:, :]
t_since_start = ae_df.iloc[:,0] - ae_df.iloc[0,0] # Time vector since start [s]
t_since_start = t_since_start.values
az_offsets  = ae_sun_used[:,0] - ae_sun_used[0,0] # azimuth vector since start, from 0
el_used = ae_sun_used[:,1] # elevation vector since start, actual value