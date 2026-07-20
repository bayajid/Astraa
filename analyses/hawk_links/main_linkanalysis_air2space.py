## Main script to compute viewing angles and link times 
#%%
import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.conversions as conv
import paa_tools.paa_calculation as paa_calc
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as combplot
import pointing_calculations.conversion_pointing as ae
import pointing_calculations.ae_calculation as aecalc
import tudat_tools.tudat_converter as tud
import basic_tools.data_loading as bload
import astronomy_tools.constants as const
importlib.reload(modplot)
R_E = const.R_E # earth radius [m]
#%%
# 1) Semi-analytical aircraft stationary, only SC moving. Pass exactly overhead [skip for now?]
# 2) Actual simulated [lets go]
path_orbits = r'orbital_simulations\tudat_raw_states'
dat_sc = bload.open_dat(f'{path_orbits}\state_history.dat', print_cond=1)
#%% slice orbit
# ii_host = [1,2,3,4,5,6]
# ii_host = [7,8,9,10,11,12]
# ii_host = [7,8,9,10,11,12]
ii_host = [13, 14, 15, 16, 17, 18]
t_vec = dat_sc[:,0]
t_fromstart = t_vec - t_vec[0]
states_sc = dat_sc[:,ii_host]
r_host = states_sc[:,:3]
v_host = states_sc[:,3:]
# bra all 650-900 km??
print(f'Host satellite altitude : {(np.linalg.norm(r_host[0,:])-R_E)/1e3:.0f} km')
#%% Get aircraft flight path
tud_converter = tud.tudat_predictor()
kep_elem_sc = tud_converter.convert_cart2kepler(states_sc[0,:])
# 0 Semi-major axis (except if eccentricity = 1.0, then represents semilatus rectum)
# 1 Eccentricity
# 2 Inclination
# 3 Argument of periapsis
# 4 Longitude of ascending node
# 5 True anomaly
incl_sc = kep_elem_sc[2]


v_ac = 340 * 0.8 # m/s
h_ac = 5e3 # m 
# KEPLER ORBIT
incl_ac = incl_sc # incl [rad]
raan_ac = kep_elem_sc[4] # RAAN [rad]
w_ac = kep_elem_sc[3] # arg of perigee [rad]
r_ac = R_E + h_ac # semi-major axis [m]
e_ac = 1e-6 # eccentricity 
theta_ac = kep_elem_sc[5] + 60 / 57.3  # true anomaly [rad], offset 60 deg ahead from sc

T_ac = (np.pi*2*r_ac / v_ac) # orbital period [s]
print(f'Time to fly around earth : {T_ac/3600:.1f} hr with velocity {v_ac} m/s')
om_ac = np.pi*2/T_ac # angular velocity, rad/s
## Generate aircraft kepler states by propagating true anomaly by om_ac * dt
kep_elem_ac = np.zeros(states_sc.shape)
kep_elem_ac[0,:] = np.array([r_ac, e_ac, incl_ac, w_ac, raan_ac, theta_ac])
dt = t_vec[1] - t_vec[0]
for ii in range(kep_elem_ac.shape[0]):
    if ii > 0:
        kep_elem_ac[ii,:5] = kep_elem_ac[0,:5]
        # propagate true anomaly
        kep_elem_ac[ii,5] = kep_elem_ac[ii-1,5] + om_ac * dt

states_ac = np.zeros(kep_elem_ac.shape)
for ii in range(states_ac.shape[0]):
    states_ac[ii,:] = tud_converter.convert_kepler2cart(kep_elem_ac[ii,:])
    states_ac[ii,3:] = states_ac[ii,3:] / np.linalg.norm(states_ac[ii,3:]) * v_ac
#%% debug true anomaly check
debug = 0
if debug:
    f, ax = plt.subplots()
    # ax.plot(t_fromstart, kep_elem_sc[:,5]*57.3, label = 'SC, T = 90 min')
    ax.plot(t_fromstart, kep_elem_ac[:,5]*57.3, label = 'AC, T = 41 hrs')
    f.suptitle(f'Change in True Anomaly : {(kep_elem_ac[-1,5] - kep_elem_ac[0,5])*57.3:.1f} deg. x 41 = {(kep_elem_ac[-1,5] - kep_elem_ac[0,5])*57.3*41:.1f}')
    ax.set_ylabel('True anomaly [deg]')
    ax.set_ylim([0, 360])
    ax.set_xlabel('t [s]')
## debug 3d plot of arcs
if debug or 0:
    fig = plt.figure(figsize=(6,6), dpi=125)    
    ax = fig.add_subplot(111, projection='3d')
    fig, ax = modplot.add_earth(fig, ax)
    ax.set_title(f'SC and AC over earth trajectory. {t_fromstart[-1]/60:.1f} min')

    ax.scatter(states_sc[:, 6*0+1-1],
            states_sc[:, 6*0+2-1],
            states_sc[:, 6*0+3-1],
            label = 'SC orbit', c = 'r', s = 2)
    ax.scatter(states_ac[:, 6*0+1-1],
            states_ac[:, 6*0+2-1],
            states_ac[:, 6*0+3-1],
            label = 'AC flight path', c = 'b', s = 2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    ax.view_init(5, 20)
    plt.show()    
    # bplt.savefig(fig, name = '3d_sim_plot', save_folder = f'{output_path}')
#%% Compute from SC POV
importlib.reload(aecalc)
ae_sc2ac_full = aecalc.calc_ae_full(states_host=states_sc, 
                               states_target=states_ac,
                               R_atm = 0,
                               default_offset= 'down_along',
                               check_occultation = 0)
ae_sc2ac_full[:,:2] = np.rad2deg(ae_sc2ac_full[:,:2])
f, axs = combplot.plot_aer(t_fromstart, ae_sc2ac_full, setting = '', unit = 'deg', axlim = '')
axs[1].plot([0, 60], [60, 60], label = 'FOV limit')
axs[1].legend()
for ii, ax in enumerate(axs):
    ax.set_xlim([0, 30])
    if ii == 1:
        ax.set_ylim([0, 100])
        ax.set_yticks([0,15, 30, 45, 60, 75, 90])
    if ii == 2:
        # ax.set_yscale('log')
        # ax.set_yticks([1e5, 1e6, 1e7, 1e8 ])
        ax.set_ylim([1e5, 1e7])
        ax.set_yticks([0.01e7, 0.25e7, 0.5e7, 0.75e7, 1e7])
f.suptitle('SC to AC Viewing angles')

ae_ac2sc_full = aecalc.calc_ae_full(states_host=states_ac, 
                               states_target=states_sc,
                               R_atm = 0,
                               default_offset= 'up_along',
                               check_occultation = 0)
ae_ac2sc_full[:,:2] = np.rad2deg(ae_ac2sc_full[:,:2])
f, axs = combplot.plot_aer(t_fromstart, ae_ac2sc_full,
                           axlim='',
                            setting = '', unit = 'deg')
axs[1].plot([0, 60], [60, 60], label = 'FOV limit')
axs[1].legend()
for ii, ax in enumerate(axs):
    ax.set_xlim([0, 30])
    if ii == 1:
        ax.set_ylim([0, 100])
        ax.set_yticks([0,15, 30, 45, 60, 75, 90])
    if ii == 2:
        ax.set_ylim([1e5, 1e7])
        ax.set_yticks([0.01e7, 0.25e7, 0.5e7, 0.75e7, 1e7])
        # ax.set_yscale('log')
        # ax.set_yticks([1e5, 1e6, 1e7, 1e8 ])
f.suptitle('AC to SC FOV')
#%% Output, , Save

ii_vis_ac = [ii for ii, el in enumerate(ae_ac2sc_full[:,1]) if el >= 60]
ii_vis_sc = [ii for ii, el in enumerate(ae_sc2ac_full[:,1]) if el >= 60]

ii_common = [ii for ii in ii_vis_ac if ii in ii_vis_sc]

output_csv = pd.DataFrame.from_dict(
    {
        't_since_start_s' : t_fromstart[ii_common],
        'link_time_s' : t_fromstart[ii_common] - t_fromstart[ii_common[0]],
        'az_ac_to_sc_deg' : ae_ac2sc_full[ii_common,0],
        'el_ac_to_sc_deg' : ae_ac2sc_full[ii_common,1],
        'az_sc_to_ac_deg' : ae_sc2ac_full[ii_common,0],
        'el_sc_to_ac_deg' : ae_sc2ac_full[ii_common,1],
        'link_range_m' : ae_ac2sc_full[ii_common,2]
    }
)
output_csv.to_csv(r'outputs/tables/aircraft_spacraft_maxlink.csv', index = 0)
make_full_outputs = 1
if make_full_outputs:
    output_csv = pd.DataFrame.from_dict(
    {
        't_since_start_s' : t_fromstart,
        'link_time_s' : t_fromstart - t_fromstart[0],
        'az_ac_to_sc_deg' : ae_ac2sc_full[:,0],
        'el_ac_to_sc_deg' : ae_ac2sc_full[:,1],
        'az_sc_to_ac_deg' : ae_sc2ac_full[:,0],
        'el_sc_to_ac_deg' : ae_sc2ac_full[:,1],
        'link_range_m' : ae_ac2sc_full[:,2]
    }
)
output_csv.to_csv(r'outputs/tables/aircraft_spaceraft_60min.csv', index = 0)