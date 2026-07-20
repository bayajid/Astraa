## Templates for loading satellite data
# generating attitude
# and whatnot. 

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

# path jazz
path_cwd = os.getcwd()
path_to_corr = r'analyses/pmg_corr_analysis/data/Terminal_data_m2_cor/Terminal_data/Terminal_10067_WITH_INCREMENT_LOSS_CORRECTION_mirror_M2_23-12-2024_150138_.csv'
path_to_nocorr = r'analyses/pmg_corr_analysis/data/Terminal_data_m2_nocor/Terminal_data/Terminal_10067_WITHOUT_INCREMENT_LOSS_CORRECTION_mirror_M2_23-12-2024_122654_.csv'
## MVP imports

# Load Data
path_chosen = path_to_corr
data = pd.read_csv(path_chosen, delimiter = ';', header = 4)
# Get tracked CPA Az/El

az = data['cpa_az'] # urad
el = data['cpa_el']
# Get diff of CPA Az/El for both
diff_az = np.gradient(az)
diff_el = np.gradient(el)
# Visualize
f, ax = plt.subplots()

ax.plot(np.rad2deg(az/1e6), diff_az)
plt.show()