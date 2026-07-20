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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import attitude_tools.rotations as att
import basic_tools.in_out as out
def get_heading_angle(v_ecef, long_data, lat_data):
    # function to rotate from ECEF to East North Up and compute
    # the heading angle using the velocity vector
    # heading_angle - heading angle wrt North [deg]
    
    v_enu_all = np.zeros(v_ecef.shape)
    heading_angles = np.zeros(v_ecef.shape[0])

    for ii, v_ecef_ii in enumerate(v_ecef):
        lat = lat_data[ii]
        long = long_data[ii]

        rot_1 = att.rot_basic(90 - lat, rot_ax = 1)
        rot_2 = att.rot_basic(90 + long, rot_ax = 3)

        # rotate that idiot velocity
        rot_comb = rot_1 @ rot_2
        v_enu_ii = rot_comb @ v_ecef_ii # get rotated, idiot

        # compute heading angle
        # heading_rad = np.arctan2(-v_enu_ii[0], v_enu_ii[1])
        heading_rad = np.arctan(-v_enu_ii[0]/v_enu_ii[1])
        heading_angle = np.rad2deg(heading_rad)
        # store
        v_enu_all[ii,:] = v_enu_ii
        heading_angles[ii] = heading_angle

    heading_angles = heading_angles + 360
    for ii, angle in enumerate(heading_angles):
        if  angle > 360:
            heading_angles[ii] = heading_angles[ii] - 360
    return heading_angles, v_enu_all