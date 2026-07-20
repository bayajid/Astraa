#%% IMPORTS
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import astronomy_tools.constants as const
def calc_period_circular(h):
    """h [m] -> orbital period [s]

    Args:
        h (altitude): alt [m]

    Returns:
        T: period [s], circular orbit
    """    # input altitude [m]
    T = 2*np.pi * np.sqrt((h+const.R_E)**3/const.mu_e)
    return T

def calc_orbital_precession(h, i):    
    """    # function to compute the ascending node precession
    # as a function of altitude [m] and inclination [rad]
    # for circular orbits due to J2

    Args:
        h (float): m, altitutde
        i (incl): rad

    Returns:
        d_raan_rate: rate of ascending node drift [deg/s]
    """    
    n = 1/(calc_period_circular(h)/2/np.pi)
    d_raan_rate = -3/2 * const.R_E**2 / (h + const.R_E)**2 * const.J2 * n * np.cos(i)
    return np.rad2deg(d_raan_rate)
def link_dist_cos(r1, r2, theta):
    """Calculate link distance for coplanar sats
    using cosine rule. Assumes coplanar sats

    Args:
        r1 (altitude): altitude [m] of sat 1
        r2 (float): altitude [m] of sat 2
        theta (float): angular sep [deg] between sats
    Reutnrs:
        R - link distance [m]
    """    
    a1 = r1 + const.R_E
    a2 = r2 + const.R_E
    link_dist = np.sqrt(a1**2 + a2**2 - 2*a1*a2*np.cos(np.deg2rad(theta)))
    return link_dist
if __name__ == '__main__':
    h_all = [1e6, 1e6, 1e6, 1050e3, 13892e3]
    i_all = [0, 53, 89, 89, 0]

    for ii, h_ii in enumerate(h_all):
        print(f'h = {h_ii/1e3:.1f} km; i = {i_all[ii]} -> period = {calc_period_circular(h_ii):.0f} s')
        # print(f'h = {h_ii/1e3:.1f} km; i = {i_all[ii]} -> period = {calc_period_circular(h_ii)/60:1f} min; \nprecession/day: {calc_orbital_precession(h_ii, np.deg2rad(i_all[ii]))*86400:.2f} deg')
        