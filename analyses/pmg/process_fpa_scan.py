## Processing FPA scan to tip/tilt vs Power-plots

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

import data_loading.load_terminal_logs as dat_load
importlib.reload(dat_load)

fname = 'fpa_scan.csv'
folder = 'analyses\pmg\scan_files'

data_df = dat_load.log2df(fname,
                          folder)