## Necessary tools and usage example for the Mounting Offset Quaternion Resolution
# Using 2 sets of Expected (Commanded) Azimuth/Elevation angles in the Terminal Frame
# and 2 sets of Measured (Detected) Azimuth/Elevation angles in the Terminal Frame
# in addition to the Mounting Offset Quaternion that was commanded during the 
# the current step of the calibration procedure.
# Date November 2, 2023
# Author: Kipras Paliusis

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

def get_q_mo_corrected(lognames, q_mountingoffset_precal, 
                       path_base, arrs = None, row_nr_1=None, row_nr_2=None,  az_correction_precal = None):
    fname_log1 = lognames[0]
    fname_log2 = lognames[1]
    if arrs is None:
        ae_t1, ae_c1 = get_cmd_from_log(fr'{path_base}/{fname_log1}', rownr = row_nr_1)
        ae_t2, ae_c2 = get_cmd_from_log(fr'{path_base}/{fname_log2}', rownr = row_nr_2)
    else:
        ae_t1, ae_c1 = get_cmd_from_log(logname = fr'{path_base}/{fname_log1}', rownr = row_nr_1, 
        arr = arrs[0])
        ae_t2, ae_c2 = get_cmd_from_log(logname = fr'{path_base}/{fname_log2}', rownr = row_nr_2, 
        arr = arrs[1])



    ae_set_t = np.deg2rad(np.vstack((ae_t1, ae_t2)))
    ae_set_c = np.deg2rad(np.vstack((ae_c1, ae_c2)))    

    if q_mountingoffset_precal is None:
        q_mountingoffset_precal = conv.convert_daz_to_qmo(az_correction_manual=-az_correction_precal)

    q_resolved = Q_mo_fix.compute_q_mountingoffset(az_el_expected = ae_set_c,
                                        az_el_current = ae_set_t,
                                        q_mountingoffset_precal = q_mountingoffset_precal)

    return q_resolved
if __name__ == '__main__':
    path = r'analyses/on_orbit_calibration/MO_res_for_cust'
    fname_log1 = 'Moon_19-01-2024_200027_PM.csv'
    fname_log2 = 'Moon_19-01-2024_203825_PM.csv'
    ae_t1, ae_c1 = get_cmd_from_log(fr'{path}/{fname_log1}')
    ae_t2, ae_c2 = get_cmd_from_log(fr'{path}/{fname_log2}')
    # ae_c1[0] = ae_c1[0] - 59.6
    ae_set_t = np.deg2rad(np.vstack((ae_t1, ae_t2)))
    ae_set_c = np.deg2rad(np.vstack((ae_c1, ae_c2)))
    print(f'''
    set t1 : {ae_t1} vs c1: = {ae_c1}
    set t2 : {ae_t2} vs c2: = {ae_c2}

        ''')
    q_mountingoffset_precal = conv.convert_daz_to_qmo(az_correction_manual=-(-60+0.4))
    q_resolved = Q_mo_fix.compute_q_mountingoffset(az_el_expected = ae_set_c,
                                            az_el_current = ae_set_t,
                                            q_mountingoffset_precal = q_mountingoffset_precal)
    ae_t1, ae_c1 = get_cmd_from_log(fr'{path}/{fname_log1}')
    ae_t2, ae_c2 = get_cmd_from_log(fr'{path}/{fname_log2}')

    # ae_c1[0] = ae_c1[0] - 59.6
    ae_set_t = np.deg2rad(np.vstack((ae_t1, ae_t2)))
    ae_set_c = np.deg2rad(np.vstack((ae_c1, ae_c2)))
    print(f'''
    set t1 : {ae_t1} vs c1: = {ae_c1}
    set t2 : {ae_t2} vs c2: = {ae_c2}

        ''')
    q_mountingoffset_precal = conv.convert_daz_to_qmo(az_correction_manual=-(-60+0.4))
    q_resolved = Q_mo_fix.compute_q_mountingoffset(az_el_expected = ae_set_c,
                                            az_el_current = ae_set_t,
                                            q_mountingoffset_precal = q_mountingoffset_precal)
    print(f'''Pre-calibration Quat:
        {q_mountingoffset_precal}
            Post-calibration:
        {q_resolved}
            ''')