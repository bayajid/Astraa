#%% Final block of code to compute
# pass statistics/cummulative contact times
# over the entire year
import numpy as np
import pandas as pd
import gt_calc as gt_tools
import matplotlib.pyplot as plt
import os
from astropy.time import Time
import datetime as dt
folder = r'monthly_outputs\contact_times_good'
pass_overview_all = os.listdir(folder)
passes_all = [f for f in pass_overview_all if 'pass_overview' in f]
indices = []
t_stamp = []
longest_pass_time = []
total_passes = []
median_pass_time = []
total_pass_time = []
avg_pass_time = []
for pass_dat in passes_all:
    loaded_pass = pd.read_csv(f'{folder}/{pass_dat}')
    month = loaded_pass.iloc[0,2]
    t_ap = Time(month, format = 'iso')
    t_dt = Time.to_datetime(t_ap)
    yrmo = dt.datetime.strftime(t_dt, '%Y-%b')
    index = dt.datetime.strftime(t_dt, '%Y-%b')
    longest_pass = np.max(loaded_pass['length_observable'])
    total_passtime = np.sum(loaded_pass['length_observable'])
    med_pass_time = np.median(loaded_pass['length_observable'])
    avg_time = np.average(loaded_pass['length_observable'])
    nr_passes = np.max(loaded_pass['pass_nr'])
    indices.append(int(dt.datetime.strftime(t_dt, '%m')))
    t_stamp.append(yrmo)
    median_pass_time.append(med_pass_time)
    avg_pass_time.append(avg_time)
    longest_pass_time.append(longest_pass)
    total_passes.append(nr_passes)
    total_pass_time.append(total_passtime)
#%%
df = pd.DataFrame.from_dict({
    'month_ind' :indices,
    'month' : t_stamp,
    'pass_median_s' : median_pass_time,
    'cumulative_contactable_time_s' : total_pass_time,
    'nr_observable_passes' : total_passes,
    'longest_contactable_time_s' : longest_pass_time,
    'average_pass_s' : avg_pass_time
})
df = df.sort_values('month_ind')
df.to_csv('Summary_monthly_contacts.csv', index = False)