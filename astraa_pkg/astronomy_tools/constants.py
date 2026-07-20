## WGS-84 defined constants 
# source https://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf
R_E = 6378137.0# [m]  Earth semi-major axis (equatorial radius)
R_E_polar = 6356752.314245 # [m], semi-minor axis (polar radius)
R_mesosphere = 100e3 # [m], rough height for Mesosphere atmoshperic laer

f_inv = 298.257223563 # inverse flattening
om = 7.2921151467e-5 # Earth angualr velocity [rad/s], used for GPS ephemeris processing
mu_e = 3.986004418e14 # m^3/s^2 Earth gravitational param
mu_e_gps = 3.986005e14 # corrected w/ GoGPS GPS's mu, used for GPS ephemeris processing
tau = 6.283185307179600 # 2 * pi
c = 299792458 # m/s speed of light
J2 = 1082.62668355e-6 # J2 gravity field coefficient for Earth
AU = 149597870.7e3 # m, Astronomical Unit # IAU 2012 definition
a_earth2moon = 348.748e3 #m
a_earth2sun = 149598023e3 # m
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
R_S = 695700e3 # m, Volumentric radius