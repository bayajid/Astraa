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
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import attitude_tools.conversions as conv
import analyses.on_orbit_calibration.MO_res_for_cust.resolve_Qmo as Q_mo_fix
importlib.reload(conv)

# read log
def get_cmd_from_log(logname, rownr = None, arr = None):
    if arr is None:
        arr = np.genfromtxt(logname, delimiter = ';', skip_header=3)

    ae_cmd_all = arr[:,[3,4]]
    ae_tracked = arr[:,[5,6]]
    if rownr is None:
        row_chosen = int(arr.shape[0]/1.3)
    else:
        row_chosen = rownr
    ae_tracked_chosen = np.rad2deg(1e-6*ae_tracked[row_chosen,:])
    ae_cmd_chosen = ae_cmd_all[row_chosen,:]

    return ae_tracked_chosen, ae_cmd_chosen 


if __name__ == '__main__':
    import scipy as sp

    path = r'analyses/on_orbit_calibration/MO_res_for_cust'
    fname_log1 = 'Moon_29-01-2024_230229_PM.csv'

    arr = np.genfromtxt(f'{path}/{fname_log1}', delimiter=';', skip_header=4)

    ii_start_man = np.where(arr[:,-2] == 1)[0][0]-1
    
    t_vec_all_from0 = arr[:,1] + arr[:,2]*1e-3
    t_vec_all_from0 = t_vec_all_from0 - t_vec_all_from0[0]
    t_vec_used = t_vec_all_from0[:ii_start_man]
    for ii, t in enumerate(t_vec_used[:-1]):
        if t > t_vec_used[ii+1]:
            t_vec_used[ii+1] = t + 0.025
    ae_moon = arr[:ii_start_man,[3,4]]

    tstart_man = t_vec_all_from0[ii_start_man:]
    t_vec_man = [tstart_man[0] + ii*0.05 for ii, t in enumerate(tstart_man)]
    tstart_man = tstart_man[0]

    az_moon_exrapoator = sp.interpolate.interp1d(t_vec_used, ae_moon[:,0],  fill_value='extrapolate')
    el_moon_exrapoator = sp.interpolate.interp1d(t_vec_used, ae_moon[:,1],  fill_value='extrapolate')

    ae_at_tman = [
    float(az_moon_exrapoator(tstart_man)),
    float(el_moon_exrapoator(tstart_man))
    ]

    az_moon_cmd_from_man = np.rad2deg(az_moon_exrapoator(t_vec_man) - ae_at_tman[0])
    el_moon_cmd_from_man = np.rad2deg(el_moon_exrapoator(t_vec_man) - ae_at_tman[1])
    print(f'''

          ''')
    a = pd.DataFrame.from_dict(
        {'d_az' : az_moon_cmd_from_man,
         'd_el' : az_moon_cmd_from_man,
         't_from0' : t_vec_man
         }
    )
    
    a.to_csv('delta_ae_for_moon.csv', index = False)

