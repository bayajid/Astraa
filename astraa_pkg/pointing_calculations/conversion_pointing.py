import os
import sys
import pathlib
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import basic_tools.vector_operations as vec
# calc_aelct
def calc_aersw(los_rsw):
    # TODO remove/replace
    # function to calculate the Azimuth and Elevation [rad]
    # for a given LOS vector in the RSW frame
    # azimuth is right-hand positive for in the NEGATIVE R direction (pointing towards Nadir)
    # elevation is positive between the SW plane in the POSITIVE R direction (pointing towards zenith)
    az = np.arctan2(-los_rsw[2], los_rsw[1]) # rad
    el = np.arcsin(los_rsw[0]/np.linalg.norm(los_rsw, axis = 1)) # rad
    return [az, el]
def calc_ae(los, official_convention = 1, wrap =1):
    # function to calculate the Azimuth and Elevation [rad]
    # for a given LOS vector in the Global/LCT frame
    # az = np.arctan2(-los[2], los[1]) # rad
    # az = np.arctan((los[1]/ los[0])) # rad
    # el = np.arctan(los[2]/np.linalg.norm(los[:2])) # rad
    if not official_convention:
        az = np.arctan((los[1]/ los[0])) # rad
        el = np.arctan(-los[2]/np.linalg.norm(los[:2])) # rad
    else:
        # Sep 7 - Betul tells me to remove this
        # if los[0] > 0:
        #     az = np.arctan(los[1] / los[0])
        # else:
        #     az = np.pi + np.arctan(los[1] / los[0])        
        az = np.arctan2(los[1], los[0]) # rad
        el = np.arcsin(los[2] / np.linalg.norm(los))
        if wrap:
            if az > np.pi:
                az = az - 2 * np.pi
    return [az, el]
def conv_los2ae(los_all, official_convention, wrap = 1):
    """CONVERT cartesian LOs to Az/El without rotating

    Args:
        los_all (Nx3): LOS cartesian
        official_convention (bool): whether official convention is used. 1 for
        wrap (int, optional): Whether Az is wrapped. Defaults to 1.

    Returns:
        ae_all: Nx2 azimuth elevation angles [rad]
    """
    if len(los_all.shape) == 1:
        nrows = 1
        los_all = los_all.reshape((1,3))
    else:
        nrows = los_all.shape[0]
    ae_all = np.zeros((nrows, 2))
    for ii, los_ii in enumerate(los_all):
        ae_ii = calc_ae(los_ii, official_convention = official_convention, wrap = wrap)    
        ae_all[ii,:] = ae_ii
    return ae_all