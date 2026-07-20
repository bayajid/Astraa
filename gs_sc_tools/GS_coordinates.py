## setup all Ground station coordinates for ISS pass calculation
# Ground station coordinates
import numpy as np
gs_l3haris= [263.98, 33.02] # long, lat, GS coordinates. Long : {0:360}
gs_ober = [11.28, 48.08]  # oberpfaffenhofen OGS
gs_tene = [343.49, 28.3] # Tenerife 
gs_neme = [22.62, 37.7] # Nemeas, near Athens
gs_cret = [24.90, 35.21] # Crete Skinakas Observatory
gs_alme = [357.45, 37.23] # Almeria Caral Alto
gs_cal = [360-116.06, 34.23] # San Gabriel Mountains California Optical Communications Telescope Laboratory - https://ieeexplore-ieee-org.tudelft.idm.oclc.org/abstract/document/8357216
gs_names = ['L3 Harris, USA',
            'Oberpfaffenhofen, Germany',
            'Tenerife, Spain',
            'Nemeas, Greece',
            'Crete, Greece',
            'Almeria, Spain',
            'Cal, USA']
# Source for GS locations: 
gs_coordinates = [gs_l3haris,
                  gs_ober,
                  gs_tene,
                  gs_neme,
                  gs_cret,
                  gs_alme,
                  gs_cal
                ] # List of GS candidates. Each entry: [long, lat] deg
# visibility source: https://elib.dlr.de/55548/1/OLEO-DL_to_OGS_and_HAPs-IST07.pdf
mean_annual_vis = [
    0.7,
    0.45,
    .71,
    .74,
    .74,
    .64,
    .7
    ]
output_dict_full = {} # Data for each time step and GS
output_dict_overview = {} # data for each GS visibility
output_dict_overview['GS'] = {}    
gs_dict = output_dict_overview['GS']
for ii, gs_coord in enumerate(gs_coordinates):
    output_dict_overview['GS'][ii] = {}
    output_dict_overview['GS'][ii]['label'] = gs_names[ii]
    output_dict_overview['GS'][ii]['long/lat'] = gs_coord
    output_dict_overview['GS'][ii]['mean_annual_availability'] = mean_annual_vis[ii]
h_gs = 300
check_for_gs_visibility = 1
# GT
R_E = 6378.136e3 # Equatorial radius of Earth [m]
GM_E = 3.98600441e14 # Earth's gravitational parameter [m^3/s^2]
n_digits = 2 # number of decimal digits for results
eps_min = 10 # [deg], minimum elevation to avoid atmospheric distortion
eta_lct = [0, 84] # elevation FOV limits for LCT on ISS [deg] - [up, low]
phi_lct = [15, -15] # azimuth FOv limits for LCT on ISS [deg] - [left, right]
# ISS orbit (only handles circular orbits)
h_a = 418 # km
i_a = 51.64 # degrees
sin_rho = R_E / (R_E + h_a * 1e3) # earth angle calculation
rho = np.arcsin(sin_rho) # rad, observable Earth angle
## GS checks
sin_rho_gs = (R_E +h_gs*1e3) / (R_E + h_a*1e3) # gs earth angle, in case altitude is higher
rho_gs = np.arcsin(sin_rho_gs)