## Tools to compute/evaluate link geometry conditions
import numpy as np
import os
import pathlib
import sys
ind_ap = str(pathlib.Path(__file__).parent.resolve()).index('astropynaric')
path_main = str(pathlib.Path(__file__).parent.resolve())[:ind_ap+12]
os.chdir(path_main)
sys.path.insert(1, os.getcwd())
import astronomy_tools.constants as const
def calc_eta_l(r_h, r_l):
    """ Precursor to checking for Earth Occultation during LEO or MEO links
    Calculates the angle between the LEO 
    satellite's nadir vector and the LEO-MEO LOS
    # asume circular orbits
    Args:
        r_h (array): higher satellite position array
        r_l (array): lower satellite position array
    Returns:
        vector: angle between lower sat's nadir vector and the lower-higher sat LOS
    """    
    len_low  = np.linalg.norm(r_l[0,:])
    len_high  = np.linalg.norm(r_h[0,:])
    len_prod = len_low * len_high
    eta_l = np.zeros(np.shape(r_h)[0])
    for ii, r_higher in enumerate(r_h):
        r_lower = r_l[ii,:]
        dr = r_lower - r_higher
        cos_eta = np.dot(r_lower, dr) / (len_prod)
        eta_l[ii] = np.arccos(cos_eta)
    return eta_l
def check_occultation(states_host, states_target, R_atm = None, R_E = None, limit_nr_links = True):
    """Function to check for Earth occultation conditions
    for given host and target states. Return indices of 
    states when visibility is possible
    ASSUMES CIRCULAR ORBITS
    Args:
        states_host (array): host states r, v [m, m/s]
        states_target (array): target states r, v [m, m/s]
        limit_nr_links (bool) : 1 : cut to a single link. 0 - keep discontinuous link data

    Returns:
        ii_returned: list of indices for link visibility. 
    """    
    # fetch atmoshperic height and Earth equatorial radius
    if R_atm == None:
        R_atm = const.R_mesosphere
    if R_E == None:
        R_E = const.R_E 
    r_h = states_host[:,:3]
    r_t = states_target[:,:3]

    # semi-major axis for host and target sat at first time-step
    a_h = np.linalg.norm(r_h[0,:])
    a_t = np.linalg.norm(r_t[0,:])
    # determine which satellite is higher/lower
    if a_h <= a_t:
        r_low = r_h
        r_high = r_t
    else:
        r_low = r_t
        r_high = r_h
    # LEO satellite orbital radius [m]
    r_l = np.linalg.norm(r_low[0,:]) 
    # Earth angular radius [rad] as seen from the LEO satellites. 
    rho_l = np.arcsin((R_E + R_atm) / r_l) 
    
    eta_l = calc_eta_l(r_h = r_high, r_l = r_low)

    ii_vis = [ii for ii, r in enumerate(r_h) if eta_l[ii] >= rho_l]
    if limit_nr_links:
        try: # check where index gap is more than 1 between consecutive link indices
            ii_returned = []
            for ii, ii_v in enumerate(ii_vis):
                ii_returned.append(ii_v)
                if ii_vis[ii+1] - ii_v == 1:
                    pass
                else:
                    break            
        except:
            print(f'No link cut detected')
            ii_returned = [ii for ii in ii_vis if ii - ii_vis[0] < 360]
            # ii_returned = ii_vis
    else:
        ii_returned = ii_vis
    return ii_returned