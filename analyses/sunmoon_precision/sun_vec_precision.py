#%%
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
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec_op
import basic_tools.data_loading as load
import astronomy_tools.astro_targets as where_sun

import astropy.coordinates as ap_coord
import astropy.time as ap_time
from skyfield.api import load as sfload
import skyfield.framelib as framelib
import astronomy_tools.astro_rotations as ast_rot
import datetime as dt
importlib.reload(modplot)
importlib.reload(vec_op)
importlib.reload(where_sun)
importlib.reload(t_conv)
#%% get sun _vec
# Full script to compute sun-vector Azimuth/Elevation 
# for a given LEO satellite with some provided attitude

# placeholders for test cases
r_host_in = []
t_gps_in = []
sun_vec_out = []
sun_vec_source = []
verify_vs_skyfield = 1
verify_vs_vallado = 1
verify_vs_tudat = 0
make_test_case = 1
if verify_vs_tudat:
    # LOAD
    setting = 'default_2020'
    input_folder = fr'orbital_simulations\sun_verification\high_precision_{setting}\LEO_MEO_medhigh_incl'
    state_history = load.open_dat(f'{input_folder}\state_history.dat')
    with open(f'{input_folder}/simulation_parameters.json') as j:
        sim_parameters = json.load(j)
    dep_var =  load.open_dat(f'{input_folder}\dependent_variables.dat')

    ind_r = sim_parameters['r_index']['leo_host_incl']
    r_host_true = state_history[:,ind_r]
    r_host2sun_true = dep_var[:,[1,2,3]]
    r_earth2sun_true = dep_var[:,[4,5,6]]
    r_host2moon_true = dep_var[:,[7,8,9]]
    t_vec_j2000 = state_history[:,[0]]
    t_gps = t_vec_j2000 + t_conv.dt_j2000tt2gps()
    if 0:
        r_host2sun_true = r_earth2sun_true
    if 0:
        # verification of inputs plots, check which is earth2sun and which host2sun
        t_vec = t_gps - t_gps[0]
        plt.title('TUDAT earth/host-Sun vector component comparison')
        plt.plot(t_vec, r_host2sun_true[:,0] -r_host2sun_true[0,0], label = 'x_host2sun')
        plt.plot(t_vec, r_earth2sun_true[:,0] -r_earth2sun_true[0,0], label = 'x_e2sun')
        plt.plot(t_vec, r_host2sun_true[:,1] -r_host2sun_true[0,1], label = 'y_host2sun')
        plt.plot(t_vec, r_earth2sun_true[:,1] -r_earth2sun_true[0,1], label = 'y_e2sun')
        plt.plot(t_vec, (r_earth2sun_true[:,0]-r_host_true[:,0])  -(r_earth2sun_true[0,0]-r_host_true[0,0]), label = 'x_e2sun - x_e2h', marker = 'o', markevery = 300, alpha = 0.4)
        plt.legend()
    ii_light_offset = 499
    ii_tudat_app = 2000 # data index for apparent sun-pos
    ii_tudat_true = 2000 # data index for true sun-pos
    # slice vector, get apparent Host - Sun vector by offsetting sun position with 
    # light travel time
    r_host_app = r_host_true[ii_light_offset:,:]
    r_host2sun_app = r_earth2sun_true[:-ii_light_offset,:] - r_host_app
    # [ii_light_offset:,:]
    # r_earth2sun_app = r_earth2sun_true[:-ii_light_offset:,0] # Apparent Earth-Sun (490 sec delay)
    # r_host2sun_app = np.zeros(r_earth2sun_true.shape)
    # [:-ii_light_offset,:]
    # r_host2sun_app = r_host2sun_app - r_host_app

    # SLICE single test vectors
    # APPARENT
    t_gps_tudat_app = t_gps[:-ii_light_offset] # app sunvec tiemstamp
    t_gps_testvec_app = float(t_gps_tudat_app[ii_tudat_app])
    r_host_given_app = r_host_app[ii_tudat_app] # Input of host pos for apparent testvec
    r_h2s_tudat_app = r_host2sun_app[ii_tudat_app,:] # Testvec, host to sun
    
    # TRUE VECTOR
    t_gps_host2suntrue = float(t_gps[ii_tudat_true])  # true sunvec tiemstamp
    r_host_given_true = r_host_true[ii_tudat_true,:]# Input of host pos for apparent testvec
    r_h2s_tudat_true = r_host2sun_true[ii_tudat_true,:]
    # [ii_tudat_true,:]

    r_e2s_app_math = where_sun.compute_sun_vector_eci_better(t_gps_testvec_app, norm = 0, rotate = 1, conv2tt = 1, t_ltravel=ii_light_offset) 
    r_e2s_true_math = where_sun.compute_sun_vector_eci_better(t_gps_host2suntrue, norm = 0, rotate = 1, conv2tt = 1, t_ltravel=0) 
    # 
    # Math earth2sun VS apparent Tudat !HOST!2sun
    e2s_mathvstudat_hostapp = vec_op.calc_dot_angle(r_e2s_app_math.flatten(),r_h2s_tudat_app)*1e6
    # Math host2sun VS apparent Tudat !HOST!2sun
    h2s_mathvstudat_hostapp = vec_op.calc_dot_angle(r_e2s_app_math.flatten()-r_host_given_app,r_h2s_tudat_app)*1e6
    # math host2sun vs host-sun true! Tudat HOST2sun
    h2s_mathvstudat_hosttrue = vec_op.calc_dot_angle(r_e2s_true_math.flatten()-r_host_given_true,r_h2s_tudat_true)*1e6
    # math earth2sun vs host-sun true! Tudat HOST2sun
    e2s_mathvstudat_hosttrue = vec_op.calc_dot_angle(r_e2s_true_math.flatten(),r_h2s_tudat_true)*1e6
    print(f'''PE Sun vector vs TUDAT
    Math earth2sun VS apparent Tudat !HOST!2sun : {e2s_mathvstudat_hostapp:.0f} urad
    FULL - Math host2sun VS apparent Tudat !HOST!2sun : {h2s_mathvstudat_hostapp:.0f} urad
    Math earth2sun vs true! Tudat HOST2sun : {e2s_mathvstudat_hosttrue:.0f} urad
    FULL - Math host2sun vs true! Tudat HOST2sun : {h2s_mathvstudat_hosttrue:.0f} urad
    '''
        )
    if make_test_case:
        # add apparent host-sun vec
        t_gps_in.append(t_gps_testvec_app)
        r_host_in.append(r_host_given_app)
        sun_vec_out.append(r_e2s_app_math.flatten()-r_host_given_app)
        sun_vec_source.append('TUDAT_simulation')
if verify_vs_skyfield:
    get_unit_vec = 0
    ## Own function
    # t_gps = 1277693948.816 # 646930800.0 J2000 - [2020-Jul]
    t_gps = 1325030348.816
    r_e2s_app_math = where_sun.compute_sun_vector_eci_better(t_gps, norm = get_unit_vec, rotate = 1, conv2tt = 1) 
    ## Skyfield
    ts = sfload.timescale()
    t_utc = t_conv.gws2utc(t_gps)
    # t_utc = dt.datetime(2021, 12, 31, 23, 58, 50, 816000)
    # print(f't utc to GWS {t_utc}\ngws:{t_conv.utc2gws(t_utc)}: ')
    t = ts.utc(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond/1e6)
    # t = ts.utc(t_utc)
    skyfield_eph = 'de440.bsp'
    eph = sfload(skyfield_eph)
    sun, earth = eph['sun'], eph['earth']
    e = earth.at(t)
    s_e2s_app_sf = e.observe(sun)
    s_e2s_app_sf = s_e2s_app_sf.apparent().frame_xyz(framelib.ICRS).km*1e3
    # get sample host states and attitude quaternions in ECI

    ## Compare - own vs skyfield
    e2sun_diff_mathvsskyfiel = vec_op.calc_dot_angle(r_e2s_app_math.flatten(),s_e2s_app_sf)*1e6

    print(f'''PE
    vector Math : {r_e2s_app_math}
    vector Skyfield : {s_e2s_app_sf}
    e2sun MATH vs Skyfield : {e2sun_diff_mathvsskyfiel:.0f} urad
    '''
        )
    if 1: # ADDITIONAL TEST, added May 5th
        date_cest = dt.datetime(2023, 5, 8, 9, 0, 0)
        print(f'Math vs Skyfield repeated for {date_cest.date().isoformat()} CEST')
        t_utc = date_cest + dt.timedelta(hours = -2)
        t_gps = t_conv.utc2gws(t_utc)
        t = ts.utc(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond/1e6)
        # t = ts.utc(t_utc)
        skyfield_eph = 'de440.bsp'
        eph = sfload(skyfield_eph)
        sun, earth = eph['sun'], eph['earth']
        e = earth.at(t)
        s_e2s_app_sf = e.observe(sun).apparent().frame_xyz(framelib.ICRS).km*1e3        
        # get sample host states and attitude quaternions in ECI
        r_e2s_app_math = where_sun.compute_sun_vector_eci_better(t_gps, norm = get_unit_vec, rotate = 1, conv2tt = 1) 
        ## Compare - own vs skyfield
        e2sun_diff_mathvsskyfiel = vec_op.calc_dot_angle(r_e2s_app_math.flatten(),s_e2s_app_sf)*1e6

        print(f'''PE separate NEW test
        vector Math : {r_e2s_app_math}
        vector Skyfield : {s_e2s_app_sf}
        e2sun MATH vs Skyfield : {e2sun_diff_mathvsskyfiel:.0f} urad
        '''
            )
        sys.exit()
    if make_test_case:
        # add apparent host-sun vec
        t_gps_in.append(t_gps)
        r_host_in.append(r_host_given_app*0)
        sun_vec_out.append(s_e2s_app_sf.flatten())
        sun_vec_source.append('Skyfield_python_library')
if verify_vs_vallado: # comapre to Vallado 5-1
    t_utc_input = dt.datetime(2006, 4, 2, 0, 0, 0, 0)
    # Vallado 5-1 expected outputs
    r_e2s_icrs_verif = np.array([146259922, 28595947, 12397430])*1e3
    r_e2s_mod_verif = np.array([146186212, 28788976, 12481064])*1e3
    
    t_gps_verif = t_conv.utc2gws(t_utc_input, ls = 14) 
    # own MOD
    r_e2s_mod_own = where_sun.compute_sun_vector_eci_better(t_gps_verif+ 499, norm = 0, rotate = 0, conv2tt = 1)
    # own ICRS
    r_e2s_icrs_own = where_sun.compute_sun_vector_eci_better(t_gps_verif+ 499, norm = 0, rotate = 1, conv2tt = 1)

    dtheta_mod = vec_op.calc_dot_angle(r_e2s_mod_own,r_e2s_mod_verif)*1e6
    dtheta_icrs = vec_op.calc_dot_angle(r_e2s_icrs_own,r_e2s_icrs_verif)*1e6
    print(f'''PE-TRUE ANGLE (not apparent), Vallado 5-1 example    
    MOD : {dtheta_mod:.0f} urad
    ICRS : {dtheta_icrs:.0f} urad
    '''
        )
    if make_test_case:
        # add apparent host-sun vec
        t_gps_in.append(t_gps_verif+ 499)
        r_host_in.append(r_host_given_app*0)
        sun_vec_out.append(r_e2s_icrs_verif)
        sun_vec_source.append('Vallado2013_5-1_example')
if make_test_case:
        # add apparent host-sun vec
    t_gps_old = 1277693948.816
    t_gps_in.append(t_gps_old)
    r_host_in.append(r_host_given_app*0)
    sun_vec_out.append(where_sun.compute_sun_vector_eci_better(t_gps_old, norm = 0, rotate = 1, conv2tt = 1))
    sun_vec_source.append('astropynaric')

if make_test_case:
    output_dict = {}
    output_dict['input_tgps_s'] = t_gps_in
    output_dict['input_r_earth2host_m'] = r_host_in
    output_dict['outputs_host2sun_m'] = sun_vec_out
    output_dict['data_source'] = sun_vec_source
    sun_vec_csv = pd.DataFrame.from_dict(output_dict)
    save_path_full = f'outputs/sun_vector_testvectors.csv'
    sun_vec_csv.to_csv(save_path_full, index = 0)
    print(f'Saved Sun vector test cases to {save_path_full}')