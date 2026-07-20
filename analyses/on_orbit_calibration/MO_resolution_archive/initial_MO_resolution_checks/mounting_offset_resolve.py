#%% 2023 July
## Script to use simulated sun-vector measurements with an unkown mounting offset
# and to try and reconstruct the unknown mounting offset with the sets of
# expected and determined sun angles
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
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.tudat_converter as tudatconv
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec
import plotting_tools.basic_plotting as bplt
import astronomy_tools.astro_targets as where_sun
import pointing_calculations.ae_calculation as ae_calc
import basic_tools.parsing as parser
import attitude_tools.attitude_resolution as att_solve
# save_folder = r'outputs\tables\simulated_sun_tracks'
# switch to moon
save_folder = r'outputs\tables\simulated_moon_tracks'
chosen_index = 2

scenarios_av = os.listdir(save_folder)
print(f'Scenarios found. Chosen - {chosen_index}')
for ii, scenario in enumerate(scenarios_av):
    print(f'{ii} -> {scenario}')
scen_chosen = scenarios_av[chosen_index]

importlib.reload(parser)
try:
      MO_true = parser.parse_col(scen_chosen[scen_chosen.index('MO')+2:scen_chosen.index('.csv')], delim = ',')
except:
      MO_true = parser.parse_col(scen_chosen[scen_chosen.index('MO')+2:scen_chosen.index('_nosp')], delim = ',')
DCM_true = conv.convert_ea2dcm(MO_true)
show_pe_overview = 1
#%%
# LOAD data
path_full = f'{save_folder}\{scen_chosen}'
sun_df = pd.read_csv(path_full)
# slice
ae_meas = sun_df[['a_meas', 'e_meas']].values
ae_exp = sun_df[['a_true', 'e_true']].values
r_exp = sun_df[['r']].values
t_vec = sun_df[['t_s']].values
if show_pe_overview:
     for ii, t in enumerate(t_vec):
          print(f'{ii} -> PE = {sun_df.iloc[ii,-1]:.2f} mrad')

#%% reconstruct X Y Z
importlib.reload(ae_calc)
# ii_used = [60, 180] # 20 min apart
# ii_used = [2, 33] # 5 minutes apart
ii_used = [66, 61] # 0.18 min apart
ae_meas_1 = ae_meas[ii_used[0],:] # terminal frame
ae_meas_2 = ae_meas[ii_used[1],:]
ae_exp_1 = ae_exp[ii_used[0],:] # body frame
ae_exp_2 = ae_exp[ii_used[1],:]

# xyz_meas_1 = ae_calc
aer_meas_1 = np.hstack((ae_meas_1, r_exp[ii_used[0]]))
aer_meas_2 = np.hstack((ae_meas_2, r_exp[ii_used[1]]))
aer_exp_1 = np.hstack((ae_exp_1, r_exp[ii_used[0]]))
aer_exp_2 = np.hstack((ae_exp_2, r_exp[ii_used[1]]))

los_meas_1 = ae_calc.xyz_from_aer(aer_meas_1)
los_meas_2 = ae_calc.xyz_from_aer(aer_meas_2)
los_exp_1 = ae_calc.xyz_from_aer(aer_exp_1)
los_exp_2 = ae_calc.xyz_from_aer(aer_exp_2)
#%% TRIAD
S_bf = los_exp_1 / np.linalg.norm(los_exp_1)
S_lf = los_meas_1 / np.linalg.norm(los_meas_1 )
cross_bf = np.cross(los_exp_1, los_exp_2)
cross_lf = np.cross(los_meas_1, los_meas_2)
M_bf = cross_bf / np.linalg.norm(cross_bf)
M_lf = cross_lf / np.linalg.norm(cross_lf)

mat_bf = np.hstack((S_bf.reshape([3,1]), M_bf.reshape([3,1]), np.cross(S_bf, M_bf).reshape([3,1])))
mat_lf = np.hstack((S_lf.reshape([3,1]), M_lf.reshape([3,1]), np.cross(S_lf, M_lf).reshape([3,1])))

# DCM_resolved = mat_bf @ mat_lf.transpose()
# RPY_resolved = conv.convert_dcm2ea(DCM_resolved.transpose())
DCM_resolved = mat_lf @ mat_bf.transpose()
RPY_resolved = conv.convert_dcm2ea(DCM_resolved)
RPY_true = np.deg2rad(conv.convert_dcm2ea(DCM_true))*1000
RPY_resolved = np.deg2rad(RPY_resolved)*1000
importlib.reload(att_solve)
quat_res_auto = att_solve.get_mo_quat_fromscan(ae_meas, ae_exp)
dcm_res_auto = conv.convert_quat2dcm(quat_res_auto)
ea_res_auto = np.deg2rad(conv.convert_dcm2ea(dcm_res_auto))*1000
print(f'RPY tru : {RPY_true}. Auto resolved : {ea_res_auto}')
#%%
print(f'CHOSEN : {scen_chosen}')
print(f'Unknown mount offset - Roll Pitch Yaw [deg] : {MO_true}')
print(f'''Simulated sun vector measurements:
      at t = {t_vec[ii_used[0]]} s. PE = {sun_df.iloc[ii_used[0],-1]:.2f} mrad')
      Expected AE -> {ae_exp_1} deg
      Found AE -> {ae_meas_1} deg
      at t = {t_vec[ii_used[1]]} s. PE = {sun_df.iloc[ii_used[1],-1]:.2f} mrad')
    Expected AE -> {ae_exp_2} deg
      Found AE -> {ae_meas_2} deg''')
print(f''' 
Rotation matrix calculated from TRUE unknown MOunting offset : 
{DCM_true}
q true = {conv.convert_dcm2quat(DCM_true)}
Rotation matrix resolved using TRIAD algorithm and 2 sun-vector measurements:
{DCM_resolved}
q resolved = {conv.convert_dcm2quat(DCM_resolved)}
Converted back to Euler Angles -> [{RPY_resolved[0]:.3f}, {RPY_resolved[1]:.3f}, {RPY_resolved[2]:.3f}]
Vs truth : [{RPY_true[0]:.3f}, {RPY_true[1]:.3f}, {RPY_true[2]:.3f}]
Difference : [{RPY_true[0] - RPY_resolved[0]:.3f}, {RPY_true[1] - RPY_resolved[1]:.3f}, {RPY_true[2] - RPY_resolved[2]:.3f}] mrad
      ''')