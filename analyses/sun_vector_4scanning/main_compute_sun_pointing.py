#%% Integrated script to compute sun pointing directions for a given host
# state, gps time and attitude quaternion

import matplotlib.pyplot as plt
# import splines.quaternion
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import json
import attitude_tools.conversions as conv
import paa_tools.paa_calculation as paa_calc
import pointing_calculations.ae_calculation as ae_calc
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.parsing as parse
import basic_tools.vector_operations as vec_op
import basic_tools.data_loading as load
import astronomy_tools.astro_targets as where_sun
import importlib

importlib.reload(ae_calc)

inputs_testvec = r"outputs\pointing_testvectors.csv"
input_data_df = pd.read_csv(inputs_testvec)

#%% parse
ii_lst = [-1]
for ii_used in ii_lst:
    input_row = input_data_df.iloc[ii_used,:]
    states_host = np.array(parse.parse_col(input_row[2]))
    states_target = np.array(parse.parse_col(input_row[3]))
    attitude_eci2bf = np.array(parse.parse_col(input_row[4]))
    link_case = input_row[0]
    t_gps = input_row[1]

    r_e2s = where_sun.compute_sun_vector_eci_better(t_gps)
    r_sun_los = r_e2s - states_host[:3]

    sun_pointing = ae_calc.calc_ae_full(states_host=states_host.reshape([1,6]), states_target=r_e2s.reshape([1,3]), attitude_eci2bf=attitude_eci2bf.reshape([1,8]))
    print(f'''
    host pos : {states_host}
    attitude host : {attitude_eci2bf}
    time : {t_gps}
    SUN VECTOR:
    AZ {sun_pointing[0][0]:.1f} 
    EL {sun_pointing[0][1]:.1f} 
    ''')