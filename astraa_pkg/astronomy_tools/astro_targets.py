##
import importlib
import os
import sys
import pathlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.constants as const
import astronomy_tools.astro_rotations as astrot
import numpy as np
import basic_tools.time_conversion as t_conv
from skyfield.api import load as sfload
import skyfield.framelib as framelib
import basic_tools.vector_operations as vec

R = const.R_E
mu_E = const.mu_e
class body_fromsp:
    """Class to give sun or moon vectors using skyfield
    which utilized NASA's planetary ephemerides de440.bsp
    initialize with initial time in [s] since j2000
    # DE440.bsp is the ephemeris that is precise to !m-level! for any body and is compared
    # to ground and space observations.
    # it is derived by fitting numerical orbits with years and years of observations
    # https://iopscience.iop.org/article/10.3847/1538-3881/abd414 
    """    
    def __init__(self, t_input, t_type = 'j2000'):
        skyfield_eph = 'de440.bsp'
        eph = sfload(skyfield_eph)
        self.eph = eph
        if t_type == 'j2000':
            t_utc = t_conv.gws2utc(t_input + t_conv.dt_j2000tt2gps())
        elif t_type == 'gps':
            t_utc = t_conv.gws2utc(t_input)
        self.ts = sfload.timescale()
        self.t = self.ts.utc(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond/1e6)
    def get_sun(self, dt_second, body  = 'sun'):
        # update time
        t_used = self.t + dt_second/86400
        body2, earth = self.eph[body], self.eph['earth']
        e = earth.at(t_used)
        r_e2body = e.observe(body2).apparent().frame_xyz(framelib.ICRS).km*1e3
        return r_e2body
    

def compute_sun_vector_eci(t_input, gps = 1):
    # input time [s] since GPS week 0
    # return normalized sun-pointing vector in ECI
    # if gps=1, then converts to Julian Centuries
    # otherwise time is input as Julian Centuries
    if gps:
        julian_centuries_ut1 = t_conv.tgps2jc(t_input)
    else:
        julian_centuries_ut1 = t_input
        
    mean_long = mean_long_of_sun(julian_centuries_ut1)
    mean_anomaly = anomaly_of_sun(julian_centuries_ut1)
    lambd = ecliptic_longitude(mean_long, mean_anomaly)
    epsilon = ecliptic_obliquity(julian_centuries_ut1)
    s_i = np.cos(lambd)
    s_j = np.cos(epsilon)*np.sin(lambd)
    s_k = np.sin(epsilon)*np.sin(lambd)
    r_sun = np.array([s_i, s_j, s_k]) # Earth-Sun vector in ECI
    return r_sun

def compute_sun_vector_eci_better(t_input, norm = 0, gps = 1, t_ltravel = 499, rotate = 1, conv2tt = 1, ls = 18):
    # input time [s] since GPS week 0
    # return normalized sun-pointing vector in ECI
    # if gps=1, then converts to Julian Centuries
    # otherwise time is input as Julian Centuries
    if gps:
        dt_gps2tt = 0
        if conv2tt:
            t_offsets = t_conv.t_sys_conv()
            dt_gps2tt = t_offsets['dt_GPS2TT']
            dt_gps2ut1 = - t_offsets['dt_UTC2GPS']# UTC follows UT1 by +/- 0.9s

        t_ut1_sun = t_conv.tgps2jc(t_input+dt_gps2ut1-t_ltravel) # to get apparent sun vector
        t_tt = t_conv.tgps2jc(t_input+ dt_gps2tt) # for rotation
    else:
        t_ut1_sun = t_input
        
    mean_long = mean_long_of_sun(t_ut1_sun)
    mean_anomaly = anomaly_of_sun(t_ut1_sun)        
    lambd = ecliptic_longitude(mean_long, mean_anomaly)
    epsilon = ecliptic_obliquity(t_ut1_sun)

    s_i = np.cos(lambd)
    s_j = np.cos(epsilon)*np.sin(lambd)
    s_k = np.sin(epsilon)*np.sin(lambd)

    r_e2s = np.array([s_i, s_j, s_k]).reshape((3,1)) # Earth to sun in Mean Of Date (MOD) frame
    if rotate:
        r_e2s = astrot.rot_mod2j2000_iau76(t_tt, r_e2s)
    
    if not norm: # give length to unit vector
        dist_earth2sun = 1.000140612 - 0.016708617 * np.cos(mean_anomaly) 
        - 0.000139589*np.cos(2 * mean_anomaly)
        r_sun = dist_earth2sun * const.AU * r_e2s # Earth-Sun vector in ECI
    else:
        r_sun = r_e2s
    return r_sun.flatten()
def mean_long_of_sun(jc):
    # input Julian Centuries
    # compute Mean longitude of the Sun, Vallado 2013
    # return [rad]
    Lambda_sun_deg = 280.46 + 36000.771 * jc
    return np.deg2rad(Lambda_sun_deg)
def anomaly_of_sun(jc):
    # input Julian Centuries
    # compute Mean anomaly of the Sun, Vallado 2013
    # return [rad]
    M_sun_deg = 357.5291092+35999.05034*jc
    return np.deg2rad(M_sun_deg)
def ecliptic_longitude(mean_long, mean_anomaly):
    # compute longitude of the ecliptic 
    v_sun = mean_long + np.deg2rad(1.914666471 * np.sin(mean_anomaly) + 0.019994643 * np.sin(2 * mean_anomaly))
    v_sun = np.deg2rad(np.rad2deg(v_sun)%360)
    return v_sun
def ecliptic_obliquity(jc):
    # compute obliquity of the ecliptic plane
    eps = np.deg2rad(23.439291 - 0.0130042 * jc)
    return eps
## intermediate functions from Bayajid (using JD? Vallado refers to JC)
def meanLongitudeDegrees(time):
    return ((280.4606184 + 0.9856474 * time) % 360)
# mean anomaly
def meanAnomalyRadians(time):
    return (np.deg2rad((357.528 + 0.9856003 * time) % 360))
# ecliptic longitude
def eclipticLongitudeRadians(mean_long, mean_anomaly):
    return (np.deg2rad((mean_long + 1.915 * np.sin(mean_anomaly) + 0.020 * np.sin(2 * mean_anomaly)) % 360))
#ecliptic obliquitey, €
def eclipticObliquityRadians(time):
    return(np.deg2rad(23.439 - 0.0000004 * time))

## MOON FUNCTIONS 0.3 deg accurac

def moon_longitude(time):
    # Moon's longitude in [deg]
    # time - Julian centuries lambda_moon
    lambda_moon = 218.32 + 481267.883 * time
    return lambda_moon

def mean_anomaly_moon(time):
    # Return mean anomaly of moon in [deg] # M_moon
    M_moon_arcsec = 485868.249036
    M_moon_arcsec += 1717915923.2178*time 
    M_moon_arcsec += 31.8792*time**2 
    M_moon_arcsec += 0.051635*time**3 
    M_moon_arcsec -= 0.00024470*time**4
    M_moon_deg = M_moon_arcsec/3600
    return M_moon_deg

def mean_elongation_sun(time):
    # mean elongation of sun [deg] D_sun
    d_sun_arcsec = 1072260.70369 
    d_sun_arcsec += 1602961601.2090*time**1
    d_sun_arcsec -= 6.3706*time**2 
    d_sun_arcsec += 0.006593*time**3 
    d_sun_arcsec -= 0.00003169*time**4
    d_sun_deg = d_sun_arcsec/3600
    return d_sun_deg
def mean_argument_of_latitude_moon(time):
    # mean argument of latitude of the moon [deg] u_Moon
    u_moon_arcsec = 335779.526232 
    u_moon_arcsec += 1739527262.8478*time**1 
    u_moon_arcsec -= 12.7512*time**2
    u_moon_arcsec -=0.001037*time**3 
    u_moon_arcsec += 0.00000417*time**4
    u_moon_deg = u_moon_arcsec / 3600
    return u_moon_deg
def where_moon_for_real(t_input, t_ltravel = 1.27, rotate = 1, gps = 1, conv2tt = 1, ls = 18, what_brightness = 0, truncation = 10, single_precision = 0):
    # moon equation but more precise, using Jean Meeus algoritm from Ch 47 of Astronomical Algorithms
    # same inputs
    dt_gps2tt = 0
    if gps:
        if conv2tt:
            t_offsets = t_conv.t_sys_conv()
            dt_gps2tt = t_offsets['dt_GPS2TT']
            dt_gps2ut1 = - t_offsets['dt_UTC2GPS']# UTC follows UT1 by +/- 0.9s

        t_ut1_app = t_conv.tgps2jc(t_input+dt_gps2ut1-t_ltravel)+1 # For Meeus moon vector
        t_ut1_rot = t_conv.tgps2jc(t_input+ dt_gps2tt) # for rotation
    else:
        t_ut1_app = 1
    if single_precision :
        t_ut1_app = np.float32(t_ut1_app)
        t_ut1_rot = np.float32(t_ut1_rot)
    Om = 259.183275 - 1934.1420 * t_ut1_app + 0.002078 * t_ut1_app * t_ut1_app + 0.0000022 * t_ut1_app * t_ut1_app * t_ut1_app
    Om_rad = np.deg2rad(Om) # good

    # Moon mean longitude
    Lm = 270.434164 + 481267.8831 * t_ut1_app - 0.001133 * t_ut1_app * t_ut1_app + 0.0000019 * t_ut1_app * t_ut1_app * t_ut1_app
    # Lm = 218.3164477 + 481267.8831 * T_mod - 0.0015786 * T_mod * T_mod + 0.0000019 * T_mod * T_mod * T_mod
    Lm+=0.000233 * np.sin(np.deg2rad(51.2 + 20.2 * t_ut1_app))
    Lm+=0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * t_ut1_app - 0.0091731 * t_ut1_app * t_ut1_app))
    Lm+=0.001964 * np.sin(Om_rad) # good within 0.005 deg
    Lm_rad = np.deg2rad(Lm)

    # Sun mean anomaly
    M = 358.475833 + 35999.0498 * t_ut1_app - 0.000150 * t_ut1_app * t_ut1_app - 0.0000033 * t_ut1_app * t_ut1_app * t_ut1_app
    M-=0.001778 * np.sin(np.deg2rad(51.2 + 20.2 * t_ut1_app)) # good within 0.001 deg
    M_rad = np.deg2rad(M)

    # Moon mean anomaly
    Mm = 296.104608 + 477198.8491 * t_ut1_app + 0.009192 * t_ut1_app * t_ut1_app + 0.0000144 * t_ut1_app * t_ut1_app * t_ut1_app
    Mm +=0.000817 * np.sin(np.deg2rad(51.2 + 20.2 * t_ut1_app))
    Mm +=0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * t_ut1_app - 0.0091731 * t_ut1_app * t_ut1_app))
    Mm +=0.002541 * np.sin(Om_rad) # good within 0.001 deg
    Mm_rad = np.deg2rad(Mm)

    # Moon mean elongation
    D  = 350.737486 + 445267.1142 * t_ut1_app - 0.001436 * t_ut1_app * t_ut1_app + 0.0000019 * t_ut1_app * t_ut1_app * t_ut1_app
    D+= 0.002011 * np.sin(np.deg2rad(51.2 + 20.2 * t_ut1_app))
    D+= 0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * t_ut1_app - 0.0091731 * t_ut1_app * t_ut1_app))
    D+= 0.001964 * np.sin(Om_rad) # within 0.001 deg
    D_rad = np.deg2rad(D) 

    # Mean distance of Moon from its ascending node
    F = 11.250889 + 483202.0251 * t_ut1_app - 0.003211 * t_ut1_app * t_ut1_app - 0.0000003 * t_ut1_app * t_ut1_app * t_ut1_app
    F+=0.003964 * np.sin(np.deg2rad(346.560 + 132.870 * t_ut1_app - 0.0091731 * t_ut1_app * t_ut1_app))
    F-=0.024691 * np.sin(np.deg2rad(Om))
    F-=0.004328 * np.sin(np.deg2rad(Om + 275.05 - 2.30 * t_ut1_app))# 0.0001 deg
    F_rad = np.deg2rad(F)        

    e = 1 - 0.002495 * t_ut1_app - 0.00000752 * t_ut1_app * t_ut1_app # GOOD
    e2 = e * e # [deg] 

    lam = Lm + 6.288750 * np.sin(Mm_rad)
    lam+= 1.274018 * np.sin(2 * D_rad-Mm_rad)
    lam+= 0.658309 * np.sin(2 * D_rad)
    lam+= 0.213616 * np.sin(2 * Mm_rad)
    lam-= e * 0.185596 * np.sin(M_rad)
    lam-= 0.114336 * np.sin(2 * F_rad)
    lam+= 0.058793 * np.sin(2 * D_rad-2 * Mm_rad)
    lam+= e * 0.057212 * np.sin(2 * D_rad -M_rad - Mm_rad)
    lam+= 0.053320 * np.sin(2 * D_rad +Mm_rad)
    lam+= e * 0.045874 * np.sin(2 * D_rad -M_rad)
    if truncation > 10 or not truncation:
        lam+= e * 0.041024 * np.sin(Mm_rad - M_rad)
        lam-= 0.034718 * np.sin(D_rad)
        lam-= e * 0.030465 * np.sin(M_rad + Mm_rad)
        lam+= 0.015326 * np.sin(2 * D_rad - 2 * F_rad)
        lam-= 0.012528 * np.sin(2 * F_rad + Mm_rad)
        lam-= 0.010980 * np.sin(2 * F_rad - Mm_rad)
        lam+= 0.010674 * np.sin(4 * D_rad - Mm_rad)
        lam+= 0.010034 * np.sin(3 * Mm_rad)
        lam+= 0.008548 * np.sin(4 * D_rad - 2 * Mm_rad)
        lam-= e * 0.007910 * np.sin(M_rad - Mm_rad + 2 * D_rad)
        lam-= e * 0.006783 * np.sin(2 * D_rad + M_rad)
        lam+= 0.005162 * np.sin(Mm_rad - D_rad)
        lam+= e * 0.005000 * np.sin(M_rad + D_rad)
        lam+= e * 0.004049 * np.sin(Mm_rad - M_rad + 2 * D_rad)
        lam+= 0.003996 * np.sin(2 * Mm_rad + 2 * D_rad)
        lam+= 0.003862 * np.sin(4 * D_rad)
        lam+= 0.003665 * np.sin(2 * D_rad - 3 * Mm_rad)
        lam+= e * 0.002695 * np.sin(2 * Mm_rad - M_rad)
        lam+= 0.002602 * np.sin(Mm_rad - 2 * F_rad - 2 * D_rad)
        lam+= e * 0.002396 * np.sin(2 * D_rad - M_rad - 2 * Mm_rad)
        lam-= 0.002349 * np.sin(Mm_rad + D_rad)
        lam+= e2 * 0.002249 * np.sin(2 * D_rad -2 * M_rad)
        lam-= e * 0.002125 * np.sin(2 * Mm_rad + M_rad)
        lam-= e2 * 0.002079 * np.sin(2 * M_rad)
        lam+= e2 * 0.002059 * np.sin(2 * D_rad - Mm_rad - 2 * M_rad)
        lam-= 0.001773 * np.sin(Mm_rad + 2 * D_rad - 2 * F_rad)
        lam-= 0.001595 * np.sin(2 * F_rad + 2 * D_rad)
        lam+= e * 0.001220 * np.sin(4 * D_rad - M_rad - Mm_rad)
        lam-= 0.001110 * np.sin(2 * Mm_rad + 2 * F_rad)
        lam+= 0.000892 * np.sin(Mm_rad - 3 * D_rad)
        lam-= e * 0.000811 * np.sin(M_rad + Mm_rad + 2 * D_rad)
        lam+= e * 0.000761 * np.sin(4 * D_rad - M_rad - 2 * Mm_rad)
        lam+= e2 * 0.000717 * np.sin(Mm_rad - 2 * M_rad)
        lam+= e2 * 0.000704 * np.sin(Mm_rad - 2 * M_rad - 2 * D_rad)
        lam+= e * 0.000693 * np.sin(M_rad - 2 * Mm_rad + 2 * D_rad)
        lam+= e * 0.000598 * np.sin(2 * D_rad - M_rad - 2 * F_rad)
        lam+= 0.000550 * np.sin(Mm_rad + 4 * D_rad)
        lam+= 0.000538 * np.sin(4 * Mm_rad)
        lam+= e * 0.000521 * np.sin(4 * D_rad - M_rad)
        lam+= 0.000486 * np.sin(2 * Mm_rad - D_rad) # 0.0003 deg
    lam_rad = np.deg2rad(lam)

    B = + 5.128189 * np.sin(F_rad)
    B+= 0.280606 * np.sin(Mm_rad + F_rad)
    B+= 0.277693 * np.sin(Mm_rad - F_rad)
    B+= 0.173238 * np.sin(2 * D_rad - F_rad)
    B+= 0.055413 * np.sin(2 * D_rad + F_rad - Mm_rad)
    B+= 0.046272 * np.sin(2 * D_rad - F_rad -Mm_rad)
    B+= 0.032573 * np.sin(2 * D_rad + F_rad)
    B+= 0.017198 * np.sin(2 * Mm_rad + F_rad)
    B+= 0.009267 * np.sin(2 * D_rad + Mm_rad - F_rad)
    B+= 0.008823 * np.sin(2 * Mm_rad - F_rad)
    if truncation > 10 or not truncation:
        B+= e * 0.008247 * np.sin(2 * D_rad - M_rad - F_rad)
        B+= 0.004323 * np.sin(2 * D_rad - F_rad - 2 * Mm_rad)
        B+= 0.004200 * np.sin(2 * D_rad + F_rad + Mm_rad)
        B+= e * 0.003372 * np.sin(F_rad - M_rad - 2 * D_rad)
        B+= e * 0.002472 * np.sin(2 * D_rad + F_rad - M_rad - Mm_rad)
        B+= e * 0.002222 * np.sin(2 * D_rad + F_rad - M_rad)
        B+= e * 0.002072 * np.sin(2 * D_rad - F_rad - M_rad - Mm_rad)
        B+= e * 0.001877 * np.sin(F_rad - M_rad + Mm_rad)
        B+= 0.001828 * np.sin(4 * D_rad - F_rad - Mm_rad)
        B-= e * 0.001803 * np.sin(F_rad + M_rad)
        B-= 0.001750 * np.sin(3 * F_rad)
        B+= e * 0.001570 * np.sin(Mm_rad - M_rad - F_rad)
        B-= 0.001487 * np.sin(F_rad + D_rad)
        B-= e * 0.001481 * np.sin(F_rad + M_rad + Mm_rad)
        B+= e * 0.001417 * np.sin(F_rad - M_rad - Mm_rad)
        B+= e * 0.001350 * np.sin(F_rad - M_rad)
        B+= 0.001330 * np.sin(F_rad - D_rad)
        B+= 0.001106 * np.sin(F_rad + 3 * Mm_rad)
        B+= 0.001020 * np.sin(4 * D_rad - F_rad)
        B+= 0.000833 * np.sin(F_rad + 4 * D_rad - Mm_rad)
        B+= 0.000781 * np.sin(Mm_rad - 3 * F_rad)
        B+= 0.000670 * np.sin(F_rad + 4 * D_rad - 2 * Mm_rad)
        B+= 0.000606 * np.sin(2 * D_rad - 3 * F_rad)
        B+= 0.000597 * np.sin(2 * D_rad + 2 * Mm_rad - F_rad)
        B+= e * 0.000492 * np.sin(2 * D_rad + Mm_rad - M_rad - F_rad)
        B+= 0.000450 * np.sin(2 * Mm_rad - F_rad - 2 * D_rad)
        B+= 0.000439 * np.sin(3 * Mm_rad - F_rad)
        B+= 0.000423 * np.sin(F_rad + 2 * D_rad + 2 * Mm_rad)
        B+= 0.000422 * np.sin(2 * D_rad - F_rad - 3 * Mm_rad)
        B-= e * 0.000367 * np.sin(M_rad + F_rad + 2 * D_rad - Mm_rad)
        B-= e * 0.000353 * np.sin(M_rad + F_rad + 2 * D_rad)
        B+= 0.000331 * np.sin(F_rad + 4 * D_rad)
        B+= e * 0.000317 * np.sin(2 * D_rad + F_rad - M_rad + Mm_rad)
        B+= e2 * 0.000306 * np.sin(2 * D_rad - 2 * M_rad - F_rad)
        B-= 0.000283 * np.sin(Mm_rad + 3 * F_rad) # 0.001 deg off
    B_rad = np.deg2rad(B)             

    W1 = 0.0004664 * np.cos(Om_rad)# G
    W2 =  0.0000754 * np.cos(np.deg2rad(Om + 275.05 - 2.30 * t_ut1_app))

    # Geocentric latitude of Moon
    beta = B * (1 - W1 - W2) # deg 
    beta_rad = np.deg2rad(beta) # goodge

    # Equatorial horizontal parallax
    Pm = 0.950724
    Pm+= 0.051818 * np.cos(Mm_rad)
    Pm+= 0.009531 * np.cos(2 * D_rad - Mm_rad)
    Pm+= 0.007843 * np.cos(2 * D_rad)
    Pm+= 0.002824 * np.cos(2 * Mm_rad)
    Pm+= 0.000857 * np.cos(2 * D_rad + Mm_rad)
    Pm+= e * 0.000533 * np.cos(2 * D_rad - M_rad)
    Pm+= e * 0.000401 * np.cos(2 * D_rad - M_rad - Mm_rad)
    Pm+= e * 0.000320 * np.cos(Mm_rad - M_rad)
    Pm-= 0.000271 * np.cos(D_rad)
    if truncation > 10 or not truncation:
        Pm-= e * 0.000264 * np.cos(M_rad + Mm_rad)
        Pm-= 0.000198 * np.cos(2 * F_rad - Mm_rad)
        Pm+= 0.000173 * np.cos(3 * Mm_rad)
        Pm+= 0.000167 * np.cos(4 * D_rad - Mm_rad)
        Pm-= e * 0.000111 * np.cos(M_rad)
        Pm+= 0.000103 * np.cos(4 * D_rad - 2 * Mm_rad)
        Pm-= 0.000084 * np.cos(2 * Mm_rad - 2 * D_rad)
        Pm-= e * 0.000083 * np.cos(2 * D_rad + M_rad)
        Pm+= 0.000079 * np.cos(2 * D_rad + 2 * Mm_rad)
        Pm+= 0.000072 * np.cos(4 * D_rad)
        Pm+= e * 0.000064 * np.cos(2 * D_rad - M_rad + Mm_rad)
        Pm-= e * 0.000063 * np.cos(2 * D_rad + M_rad - Mm_rad)
        Pm+= e * 0.000041 * np.cos(M_rad + D_rad)
        Pm+= e * 0.000035 * np.cos(2 * Mm_rad - M_rad)
        Pm-= 0.000033 * np.cos(3 * Mm_rad - 2 * D_rad)
        Pm-= 0.000030 * np.cos(Mm_rad + D_rad)
        Pm-= 0.000029 * np.cos(2 * F_rad- 2 * D_rad)
        Pm-= e * 0.000029 * np.cos(2 * Mm_rad + M_rad)
        Pm+= e2 * 0.000026 * np.cos(2 * D_rad - 2 * M_rad)
        Pm-= 0.000023 * np.cos(2 * F_rad - 2 * D_rad + Mm_rad)
        Pm+= e * 0.000019 * np.cos(4 * D_rad - M_rad - Mm_rad) # GOODGE
    Pm_rad = np.deg2rad(Pm)
    # Distance of Moon in Earth radii = 1/(np.sin(Pm)
    r_moon = 1/np.sin(Pm_rad) * 6378.14e3

    # Nutation correction:
    L = 279.6967 + 36000.7689 * t_ut1_app + 0.000303 * t_ut1_app * t_ut1_app
    L_rad = np.deg2rad(L)

    dphi = -17.2 * np.sin(Om_rad) - 1.3 * np.sin(2 * L_rad) # arcsec

    # Apparent geocentric longitude

    lam_app = lam + dphi/3600
    lam_app = np.deg2rad(lam_app)
    ## Obliquity of ecliptic and corrections
    eps = eclipticObliquityRadians(t_ut1_app)
    d_eps = (9.2100 + 0.00091* t_ut1_app) * np.cos(Om_rad)
    d_eps+= (0.5522 - 0.00029 * t_ut1_app) * np.cos(2*L_rad)
    d_eps-= 0.0904 * np.cos(2*Om_rad)
    d_eps+= 0.0884 *np.cos(2*Lm_rad)
    d_eps+= 0.0216 *np.cos(L_rad + M_rad)
    d_eps+= 0.0183 *np.cos(2*Lm_rad - Om_rad)
    d_eps+= 0.0113 *np.cos(2*Lm_rad + Mm_rad)
    d_eps-= 0.0093 *np.cos(2*L_rad - M_rad)
    d_eps-= 0.0066 *np.cos(2*L_rad - Om_rad)
    eps += np.deg2rad(d_eps/3600)
    
    # Converting to equatorial from ecliptic
    
    # right ascension
    alpha_ra = np.arctan2(np.sin(lam_app) * np.cos(eps) - np.tan(beta_rad) * np.sin(eps), np.cos(lam_app))
    # declination
    beta_decl = np.arcsin(np.sin(beta_rad) * np.cos(eps) + np.cos(beta_rad) * np.sin(eps) * np.sin(lam_app))

    r_m2s = np.array([
        np.cos(beta_decl) * np.cos(alpha_ra) * r_moon,
        np.cos(beta_decl) * np.sin(alpha_ra) * r_moon,
        np.sin(beta_decl) * r_moon,
    ])
    if rotate:
        r_m2s = astrot.rot_mod2j2000_iau76(t_ut1_rot, r_m2s)
    # if what_brightness:
    #     return r_m2s.flatten(), illum_perc
    # else:
    return r_m2s.flatten()

def compute_moon_vector_eci(t_input, t_ltravel = 1.27, rotate = 1, gps = 1, conv2tt = 1, ls = 18, what_brightness = 0):
    """Function to mathematically comptue moon position in ECI
    according to the Brower method described in Vallado 2013, chapter 5

    Args:
        t_input (float): time, preferably in GPS seconds
        t_ltravel (float, optional): Light travel time to moon. Defaults to 1.27.
        rotate (bool, optional): Whether to rotate from MOD to ECI. Defaults to 1.
        gps (bool, optional): whether GPS time is input. Defaults to 1.
        conv2tt (int, optional): _description_. Defaults to 1.
        ls (float, optional): Leap seconds used in time conversion. Defaults to 18.
        what_brightness (bool, optional): Whether to also output moon illumination. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    dt_gps2tt = 0
    if gps:
        if conv2tt:
            t_offsets = t_conv.t_sys_conv()
            dt_gps2tt = t_offsets['dt_GPS2TT']
            dt_gps2ut1 = - t_offsets['dt_UTC2GPS']# UTC follows UT1 by +/- 0.9s

        t_ut1_app = t_conv.tgps2jc(t_input+dt_gps2ut1-t_ltravel) # to get apparent sun vector
        t_ut1_rot = t_conv.tgps2jc(t_input+ dt_gps2tt) # for rotation
    else:
        t_ut1_app = 1

    t_ut1_app = t_ut1_app
    moon_longitude_deg = moon_longitude(t_ut1_app) # lambda_moon
    moon_longitude_rad = np.deg2rad(moon_longitude_deg)
    moon_anomaly_rad = np.deg2rad(mean_anomaly_moon(t_ut1_app)) # M_moon
    sun_anomal_rad = anomaly_of_sun(t_ut1_app) # M_sun
    mean_elong_sun_rad = np.deg2rad(mean_elongation_sun(t_ut1_app)) # D
    moon_mean_arg_latitude_rad = np.deg2rad(mean_argument_of_latitude_moon(t_ut1_app))
    
    # ecliptic LONGITUDE the moon [rad] lambda_ecliptic LAMBDA_ecl
    moon_ecl_longitude_deg = moon_longitude_deg
    moon_ecl_longitude_deg += 6.29 * np.sin(moon_anomaly_rad) 
    moon_ecl_longitude_deg -= 1.27 * np.sin(moon_anomaly_rad - 2*mean_elong_sun_rad)
    moon_ecl_longitude_deg += 0.66 * np.sin(2*mean_elong_sun_rad)
    moon_ecl_longitude_deg += 0.21 * np.sin(2*moon_anomaly_rad)
    moon_ecl_longitude_deg -= 0.19 * np.sin(sun_anomal_rad)
    moon_ecl_longitude_deg -= 0.11 * np.sin(2*moon_mean_arg_latitude_rad)
    
    moon_ecl_longitude_rad = np.deg2rad(moon_ecl_longitude_deg)

    if what_brightness:
        # compute moon illumination
        t_ut1_app_sun = t_conv.tgps2jc(t_input+dt_gps2ut1-499)
        mean_long = mean_long_of_sun(t_ut1_app_sun)
        mean_anomaly = anomaly_of_sun(t_ut1_app_sun)        
        sun_ecl_long_rad = ecliptic_longitude(mean_long, mean_anomaly)
        phase_diff = sun_ecl_long_rad - moon_ecl_longitude_rad
        # moon illumination
        illum_perc = 100/2 * (1-np.cos(phase_diff))
        
    # ecliptic latitude of the moon [deg]
    phi_ecliptic_moon_deg = 5.13 * np.sin(moon_mean_arg_latitude_rad)
    phi_ecliptic_moon_deg+= 0.28 * np.sin(moon_anomaly_rad + moon_mean_arg_latitude_rad)
    phi_ecliptic_moon_deg-= 0.28 * np.sin(moon_mean_arg_latitude_rad - moon_anomaly_rad)
    phi_ecliptic_moon_deg-= 0.17 * np.sin(moon_mean_arg_latitude_rad - 2 * mean_elong_sun_rad)
    
    phi_ecliptic_moon_rad = np.deg2rad(phi_ecliptic_moon_deg)

    obliquity_ecliptic = eclipticObliquityRadians(t_ut1_app) # epsilon

    # # parallax
    eta_moon_deg = 0.9508 + 0.0518*np.cos(moon_anomaly_rad) 
    eta_moon_deg += 0.0095*np.cos(moon_anomaly_rad - 2 * mean_elong_sun_rad)
    eta_moon_deg += 0.0078*np.cos(2 * mean_elong_sun_rad) 
    eta_moon_deg += 0.0028*np.cos(2 * moon_anomaly_rad)
    
    eta_moon_rad = np.deg2rad(eta_moon_deg)
    
    r_moon_m = const.R_E / np.sin(eta_moon_rad)
    

    s_i = np.cos(phi_ecliptic_moon_rad) *  np.cos(moon_ecl_longitude_rad)
    
    s_j = np.cos(obliquity_ecliptic) * np.cos(phi_ecliptic_moon_rad) * np.sin(moon_ecl_longitude_rad)
    s_j -= np.sin(obliquity_ecliptic) * np.sin(phi_ecliptic_moon_rad)
    
    s_k = np.sin(obliquity_ecliptic) * np.cos(phi_ecliptic_moon_rad) * np.sin(moon_ecl_longitude_rad)
    s_k+= np.cos(obliquity_ecliptic) * np.sin(phi_ecliptic_moon_rad)

    r_m2s = r_moon_m * np.array([s_i, s_j, s_k]).reshape((3,1)) # Moon to sun in Mean Of Date (MOD) frame
    if rotate:
        r_m2s = astrot.rot_mod2j2000_iau76(t_ut1_rot, r_m2s)
    if what_brightness:
        return r_m2s.flatten(), illum_perc
    else:
        return r_m2s.flatten()

def check_shadow(r_host, r_sun, body = 'sun'):
    """conditionals if sat is in penumbra or umbra using conical shadow model
    from Vallado 2013 book
    Args:
        r_host (_type_): host pos
        r_sun (_type_): sun pos
        body (str, optional): body. TODO add moon. Defaults to 'sun'.

    Returns:
        um, pen: booleans regarding sat being in umbra and penumbra
    """    
    pen = False
    um = False
    if body=='sun':
        # TODO replace with floats
        alpha_umb = np.deg2rad(0.264121687)
        alpha_pen = np.deg2rad(0.269007205)
    elif body == 'moon':
        alpha_umb = np.deg2rad(0.265896)
        alpha_pen = np.deg2rad(0.267227)
    cos_zeta = np.dot(r_sun[:3], r_host[:3]) / (np.linalg.norm(r_sun[:3]) * np.linalg.norm(r_host[:3]))
    if np.dot(r_sun[:3], r_host[:3]) < 0:
        r_norm = np.linalg.norm(r_host[:3])
        sat_hor = r_norm * cos_zeta

        sat_ver = r_norm * np.sin(np.arccos(cos_zeta))

        x = const.R_E / np.sin(alpha_pen)

        pen_ver = np.tan(alpha_pen) * (x + sat_hor)

        if sat_ver <= pen_ver:
            pen = True

            y = const.R_E / np.sin(alpha_umb)

            umb_ver = np.tan(alpha_umb) * (y - sat_hor)
            if sat_ver <= umb_ver:
                um = True
    return um, pen
def check_shadow_canonical(r_host, r_sun, body = 'sun'):
    """conditionals if sat is in penumbra or umbra using conical shadow model
    from satellite orbits book
    Heavily assisted by Goekhan 
    Args:
        r_host (_type_): host pos
        r_sun (_type_): sun pos
        body (str, optional): body. TODO add moon. Defaults to 'sun'.

    Returns:
        um, pen: booleans regarding sat being in umbra and penumbra
    """    
    pen = False
    um = False
    if body=='sun':
        # TODO replace with floats
        alpha_umb = np.deg2rad(0.264121687)
        alpha_pen = np.deg2rad(0.269007205)
    elif body == 'moon':
        alpha_umb = np.deg2rad(0.265896)
        alpha_pen = np.deg2rad(0.267227)
    
    r_sun_norm = np.linalg.norm(r_sun[:3])
    r_host_norm = np.linalg.norm(r_host[:3])
    s_0 = (- np.reshape(r_host[:3], (1,3)) @ np.reshape(r_sun[:3], (3,1))) / r_sun_norm
    s_hor = np.linalg.norm(r_host[:3] - s_0)
    s_vert = np.sqrt(r_host_norm**2 - s_0**2)

    sin_f1 = (const.R_S + const.R_E) / r_sun_norm # penumbra shadow cone angle sine
    sin_f2 = (const.R_S - const.R_E) / r_sun_norm # umbra shadow cone angle sine
    # distances from Umbra and Penumbra vertices
    c1 = s_0 + const.R_E / sin_f1
    c2 = s_0 - const.R_E / sin_f2

    l1 = c1 * np.tan(np.arcsin(sin_f1))
    l2 = c2 * np.tan(np.arcsin(sin_f2))
    if np.dot(r_sun[:3], r_host[:3]) < 0:
        if s_vert <= l1:
            pen = True
            if s_vert <= np.abs(l2):
                um = True
    return um, pen

if __name__ == '__main__':
    import datetime as dt
    
    if 1:
        r_host = np.array([-4464.7e3, -5102e3, ])
        t_gps = 725824800.0	- t_conv.dt_gps2j2000tt()
        r_sun = compute_sun_vector_eci_better(t_gps)
        r_host = np.array([7370271.265829999,	0.0,	0.0,	-0.0,	128.35245607759342,	7353.307283899025])
        r_host = [r_sun/1e5, # NO
                  -r_sun/1e5, # YES
                  -r_sun/1e10, # ??
                  -r_sun/1e5 + [0, 0, 6e6*2] # NO
                  ]
        exp = [0, 1,0, 0]
        for ii, r_h in enumerate(r_host):
            um, pen = check_shadow(r_h, r_sun)

            print(f'rh = {r_h} \nUmbra {um}  Penumbra {pen}. Expected : {bool(exp[ii])}.')
        
    if 0:
        # Moon Vector Tests
        t_now = dt.datetime.now()
        # t_used = t_now
        t_gps = t_conv.utc2gws(t_now + dt.timedelta(hours = -1))
        t_used = dt.datetime(1994, 4, 28, 0, 0, 0)
        t_gps_example = t_conv.utc2gws(t_used, ls = 10)

        # t_gps_example = t_gps
        t_moon_ex = compute_moon_vector_eci(t_gps_example, t_ltravel=0,
                                            ls = 10)
        
        t_moon_true = np.array([
            -134240626,
            -311571590,
            -126693785
        ]) # m

        
        print(f'''
            t_now : {t_used}
            t_gps : {t_gps_example}
            r moon : {t_moon_ex}
            r_moon_err : {t_moon_true.flatten() - t_moon_ex.flatten()} m
            relative err : {(t_moon_true.flatten() - t_moon_ex.flatten())/t_moon_ex.flatten()*100} % 
            ''')
        
