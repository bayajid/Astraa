# Collect astronomical rotations
import numpy as np
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.rotations as rot
def rot_mod2j2000_iau76(t_tt, r_mod):
    # Rotate Of Date (MOD) frame to J200/GCRS
    # according to IAU76 conmventions
    # tested with Vallado 2013, page 231
    # t_tt - Julian Centuries since J2000 in Terrestrial Time
    zeta_arcsec =  2306.2181*t_tt + 0.30188*t_tt**2  + 0.017998*t_tt**3
    teta_arcsec = 2004.3109*t_tt**1 - 0.42665*t_tt**2 - 0.041833*t_tt**3
    z_arcsec = 2306.2181*t_tt**1 + 1.09468*t_tt**2 + 0.018203*t_tt**3
    
    rot3_zeta = rot.rot_basic(zeta_arcsec/3600, rot_ax = 3)
    rot2_theta = rot.rot_basic(-teta_arcsec/3600, rot_ax = 2)
    rot3_z = rot.rot_basic(z_arcsec/3600, rot_ax = 3)
    P_rot = rot3_zeta @ rot2_theta @ rot3_z
    # r_gcrs = P_rot.transpose() @ r_mod
    r_gcrs = P_rot @ r_mod
    return r_gcrs