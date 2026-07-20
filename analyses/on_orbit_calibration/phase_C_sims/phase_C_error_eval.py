#%% Feb 1, 2024 - generating inputs with expected errors
# according to the in-orbit phase C simulation plan
import scipy as sp
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
csv_output_path = r'orbital_simulations\leo_for_pmg\leo_leo_srpcheck'
fname_simparam = 'simulation_parameters.json'

# Use for full-1s time-step data:
fname_states = 'states_fine.dat'
# Use for coarse 60-s step data:
# fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io
import plotting_tools.basic_plotting as bplt
import plotting_tools.plotting_utilities as plt_util
import plotting_tools.combined_plots as cmbplt
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import prediction_methods.error_generation as err_gen
import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.ae_profiles as ae_profile
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil

pm_input_unknown = {
    "aoff": 0.00010152158522895261,
    "aan": -3.8234775853129714e-05,
    "aae": -3.3010734516465985e-05,
    "npae": -6.706053888476065e-05,
    "bnp": 5.408797683279241e-05,
    "aes": -0.00014384616855501767,
    "aec": 0.00010905073526353,
    "eoff": -4.7575431305943927e-05,
    "ean": 1.9939943503568663e-05,
    "eae": -1.558564846733813e-05,
    "eec": 9.138174606531089e-05,
    "es2a": -0.00012875879434360338,
    "ec2a": -2.015107525084422e-05,
    "es3a": -2.400339716677598e-05,
    "ec3a": 7.086059014596483e-05
}

pm_output_perfect_knowledge = {'azoff': -0.00031474109674899087, 'azan': -2.5938667904969702e-05, 'azae': -3.202604521633582e-05, 'azes': -0.00014809891921100112, 'azec': 0.00013400063415387796, 'npaze': -6.84143427429211e-05, 'bnp': 1.4013040489269008e-06, 'eloff': -4.434218158202272e-05, 'elan': 2.404812439929875e-05, 'elae': -2.0335430918183566e-05, 'elec': 8.625326931178301e-05, 'elsa2': -0.00012694319608831726, 'elca2': 0.00038938180128706204, 'elsa3': -0.0002924921940945238, 'elca3': -0.0010412855797313815}
pm_output_real_knowledge = {'azoff': -0.0003285913740915233, 'azan': -0.0005781361784979609, 'azae': 0.00023250859646814135, 'azes': -0.00016159732728982619, 'azec': 0.0001535594452619991, 'npaze': -7.194136694847826e-05, 'bnp': 2.012542518694344e-06, 'eloff': -5.6381634329073834e-05, 'elan': -0.000514976753948574, 'elae': -0.000274231179157155, 'elec': 0.00010425387636331324, 'elsa2': -0.00012350154110047284, 'elca2': 0.00038938180128706204, 'elsa3': -0.0002924921940945238, 'elca3': -0.0010473410341837748}

# Convert Bayajid's coefficient notation to paper
pm_samename_perfect_knowledge = {}
pm_samename_real_knowledge = {}
for key_0 in pm_output_perfect_knowledge.keys():
    key = key_0.replace('az','a')    
    key = key.replace('el','e')
    for ii in [1,2,3]:
        key = key.replace(f'a{ii}', f'{ii}a')
        key = key.replace(f'e{ii}', f'{ii}e')
    # print(f'{key_0} -> {key}')
    pm_samename_perfect_knowledge[key] = pm_output_perfect_knowledge[key_0]
    pm_samename_real_knowledge[key] = pm_output_real_knowledge[key_0]
    
residuals_perfect_knowledge = {}
residuals_real_knowledge = {}

for key in pm_input_unknown.keys():
    err_ii_perf = pm_input_unknown[key] - pm_samename_perfect_knowledge[key]
    err_ii_real = pm_input_unknown[key] - pm_samename_real_knowledge[key]
    residuals_perfect_knowledge[key] = err_ii_perf
    residuals_real_knowledge[key] = err_ii_real
#%%
# get errors or just plot
if 1:
    f, axs = plt.subplots(nrows = 2, figsize = (16,8))
    ax = axs[0]
    yvals_true_unknown = []
    xvals_true_unknown = []
    for ii, key in enumerate(pm_input_unknown):
        xvals_true_unknown.append(key)
        yvals_true_unknown.append(1e3*pm_input_unknown[key])
    
    xvals = []
    yvals_solved_real = []
    pm_used = pm_samename_real_knowledge
    for ii,key in enumerate(pm_used):
        xvals.append(key)
        yvals_solved_real.append(1e3*pm_used[key])
    
    
    ax.stem(xvals_true_unknown, yvals_true_unknown, 'b', label = 'True, input', )
    ax.stem(xvals, yvals_solved_real, 'g', label = 'Cmd/Log, output', )

    ax.legend()
    ax.grid()
    ax.set_ylabel('PM Components [mrad]')

    ax = axs[1]
    xvals = []
    yvals_errors_real = []
    yvals_errors_perf = []
    pm_used = pm_samename_real_knowledge
    for ii,key in enumerate(pm_used):
        xvals.append(key)
        yvals_errors_perf.append(residuals_perfect_knowledge[key]*1e3)
        yvals_errors_real.append(residuals_real_knowledge[key]*1e3)

    # ax.stem(xvals, yvals_errors_perf, 'r', label = 'True/log - perfect knowledge', )
    ax.stem(xvals, yvals_errors_real, 'g', label = 'Cmd/Log - real knowledge', )
    ax.set_ylabel('Errors - PM Component [mrad]', fontweight = 'bold')
    ax.legend()
    ax.grid()

sum_pm = 0
ls_errors_pm_real = 0
ls_errors_pm_perf = 0
for key in pm_input_unknown.keys():
    sum_pm += np.abs(pm_input_unknown[key])
    ls_errors_pm_real += np.abs(residuals_real_knowledge[key])
    ls_errors_pm_perf += np.abs(residuals_perfect_knowledge[key])
sum_pm = sum_pm*1e3
ls_errors_pm_real = ls_errors_pm_real*1e3
ls_errors_pm_perf = ls_errors_pm_perf*1e3

print(f'''Linear sum:
Input PM coefficients : {sum_pm:.0f} mrad    
errors coefficients, real knowledge: {ls_errors_pm_real:.0f} mrad
errors coefficients, perfect knowledge: {ls_errors_pm_perf:.0f} mrad
      ''')

#%% Get errors in az/el
## Errors as a function of az/el

csv_for_azel_true = r"C:\Users\KPaliusis\Documents\Github_Repos\astropynaric\outputs\tables\pmg_sim_aefixed\pmg_trackingtrue_fixedae.csv"    
ae_csv = pd.read_csv(csv_for_azel_true)
ae_true = np.deg2rad(ae_csv.values[:,[1,2]]) # rad
t_gps = ae_csv.values[:,0]
pmg_components_az_solved = {
    'aoff' : pm_samename_real_knowledge['aoff'],
    'aan' : pm_samename_real_knowledge['aan'],
    'aae' : pm_samename_real_knowledge['aae'],
    'npae' : pm_samename_real_knowledge['npae'],
    'bnp' : pm_samename_real_knowledge['bnp'],
    'aes' : pm_samename_real_knowledge['aes'],
    'aec' : pm_samename_real_knowledge['aec'],
}
pmg_components_el_solved = {
    'eoff' :pm_samename_real_knowledge['eoff'],
    'ean' :  pm_samename_real_knowledge['ean'],
    'eae' :  pm_samename_real_knowledge['eae'],
    'eec' :  pm_samename_real_knowledge['eec'],
    'es2a' : pm_samename_real_knowledge['es2a'],
    'ec2a' : pm_samename_real_knowledge['ec2a'],
    'es3a' : pm_samename_real_knowledge['es3a'],
    'ec3a' : pm_samename_real_knowledge['ec3a'],
}

pmg_components_az_input = {
    'aoff' : pm_input_unknown['aoff'],
    'aan' : pm_input_unknown['aan'],
    'aae' : pm_input_unknown['aae'],
    'npae' : pm_input_unknown['npae'],
    'bnp' : pm_input_unknown['bnp'],
    'aes' : pm_input_unknown['aes'],
    'aec' : pm_input_unknown['aec'],
}
pmg_components_el_input = {
    'eoff' :pm_input_unknown['eoff'],
    'ean' :  pm_input_unknown['ean'],
    'eae' :  pm_input_unknown['eae'],
    'eec' :  pm_input_unknown['eec'],
    'es2a' : pm_input_unknown['es2a'],
    'ec2a' : pm_input_unknown['ec2a'],
    'es3a' : pm_input_unknown['es3a'],
    'ec3a' : pm_input_unknown['ec3a'],
}

ae_with_pm_solved, ae_pm_errors_solved = err_gen.calculate_pmg_errors(ae_true, pmg_components_az_solved, pmg_components_el_solved)
ae_with_pm_input, ae_pm_errors_input = err_gen.calculate_pmg_errors(ae_true, pmg_components_az_input, pmg_components_el_input)
delta_ae = ae_with_pm_solved - ae_with_pm_input
delta_ae_neglect = ae_true - ae_with_pm_input
f, ax = cmbplt.plot_ae(t_gps - t_gps[0], 1e3*delta_ae, title = 'Az/El PM-errors with PMG', unit = 'mrad', axlim = 'equal')
f, ax = cmbplt.plot_ae(t_gps - t_gps[0], 1e3*delta_ae_neglect, title = 'Az/El PM-errors without PMG', unit = 'mrad', axlim = 'equal')