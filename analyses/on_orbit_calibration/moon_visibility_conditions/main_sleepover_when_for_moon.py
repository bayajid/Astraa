#%%
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
path_cwd = os.getcwd()
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.astro_targets as where_sun
import basic_tools.time_conversion as t_conv
import link_processing_tools.visibility_checks as vis_check
import tudat_tools.data_processing.data_processing_utilities as dputil
import plotting_tools.basic_plotting as bplt
import plotting_tools.modular_plotting as modplot
### SCRIPT to get moon illuminations and rise times to see when good moon-scanning opportunities
# arrise
path_moon_dat = r'outputs\tables\sun_vector\moon_2023-11-26_big\Long_aeILUM_gs2moon.csv'

full_df = pd.read_csv(path_moon_dat)
#%% settings - filter out unfavourable times
el_min = 0
illum_min = 0
time_morning_max = 8*3600
time_evening_min = 16*3600
# filter
full_df_illum = full_df[full_df['illum']>illum_min]
full_df_el = full_df_illum[full_df_illum['el_deg']>0]
full_df_morning = full_df_el[full_df_el['s_of_day']<time_morning_max]
full_df_evening = full_df_el[full_df_el['s_of_day']>time_evening_min]

full_df_out = pd.concat((full_df_morning, full_df_evening))
full_df_out =  full_df_out.sort_index()
save_title = fr'outputs\tables\moon_view_conditions\moon_IL{illum_min}_{full_df_out.iloc[-1,1]}.csv'
full_df_out.to_csv(save_title, index = 0)
print(f'Sleepover times saved to {save_title}')