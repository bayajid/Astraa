#%% IMPORTS
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
import astronomy_tools.astro_targets as where_sun
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec_op
import plotting_tools.basic_plotting as bplt
import prediction_methods.interpolators as interp

# path jazz
path_cwd = os.getcwd()
path_single_moon_csv = 'outputs\single_moon_outputs'

file_chosen = r'analyses\moon_tracking\32bitmoon1356587949.csv'
single_prec_data = pd.read_csv(file_chosen, header = None).values
#%%
dt_sec = single_prec_data[1,0] - single_prec_data[0,0]
# t_gps_0 = int(file_chosen[file_chosen.index('32bit')+9:-4])
t_length_days = (single_prec_data[-1,0] - single_prec_data[0,0] )/86400
t_vec_gps = single_prec_data[:,0]
t_gps_0 = t_vec_gps[0]
# dt_sec = 60
dt_0 = t_conv.gws2utc(t_gps_0)
t_gps_f = t_gps_0 + t_length_days*86400
# days
t_days_used = t_length_days
# t_days_used = 1
nrows = int(t_days_used * 86400 / dt_sec)
data_used = single_prec_data[:nrows,:]
t_vec = data_used[:,0]
make_full_run = 1
#%%
sf_obj = where_sun.body_fromsp(t_gps_0+t_conv.dt_gps2j2000tt())

if make_full_run:
    #
    err_placeholder_nocorrection = np.zeros((data_used.shape[0],5)) # t, pe [mrad], dr [m]
    for ii, t_gps_ii in enumerate(t_vec):
        dt_ii = t_gps_ii - t_gps_0
        r_moon_truth = sf_obj.get_sun(dt_ii, body = 'moon') # truth
        r_moon_single = single_prec_data[ii,1:]
        pe_ii = vec_op.calc_dot_angle(r_moon_truth, r_moon_single)
        dr_ii = r_moon_truth - r_moon_single
        err_placeholder_nocorrection[ii,:2] = [t_gps_ii, pe_ii*1e3]
        err_placeholder_nocorrection[ii,2:] = dr_ii
# hours
# update_interval = 6 * (dt_sec/3600) # 5 min
# update_interval = 36 * (dt_sec/3600)  # 30 min
update_interval = 3600  # 1 hr

# interp_option = 'constant'
interp_option = 'linear'

# update_interval = int(update_interval)
nr_updates = int(len(t_vec) * dt_sec/update_interval) 
t_gap = int(update_interval)
t_updates = [int(t_gps_0 + t_gap * ii) for ii in range(nr_updates)]
err_placeholder_1hr = np.zeros((data_used.shape[0],5+3)) # t, pe [mrad], dr [m], corr_dr
moon_vector_computed = np.zeros((data_used.shape[0],5)) # t_gps; r_moon; pe
upd_made = 0
interpolator = interp.we_interpolating_pos()
for ii, t_gps_ii in enumerate(t_vec):
    if int(t_gps_ii) in t_updates:
        r_correction = - err_placeholder_nocorrection[ii,2:]
        upd_made+=1
        if interp_option == 'constant':
            interpolator.get_const_interpolant(err_placeholder_nocorrection[ii,0], err_placeholder_nocorrection[ii,2:])
        elif interp_option == 'linear':
            ii_0 = ii
            ii_f = ii_0 + int(t_gap/dt_sec)
            try:
                interpolator.get_lin_interpolant(err_placeholder_nocorrection[ii_0,0],
                                                err_placeholder_nocorrection[ii_0,2:],
                                                err_placeholder_nocorrection[ii_f,0],
                                                err_placeholder_nocorrection[ii_f,2:],
                                                )
            except:
                print(f'out of data at {ii}, switching to constant')
                interpolator.get_const_interpolant(err_placeholder_nocorrection[ii,0], err_placeholder_nocorrection[ii,2:])
    r_correction = interpolator.interpolate(t_gps_ii)
    dt_ii = t_gps_ii - t_gps_0
    r_moon_truth = sf_obj.get_sun(dt_ii, body = 'moon')
    r_moon_single = single_prec_data[ii,1:] + r_correction
    pe_ii = vec_op.calc_dot_angle(r_moon_truth, r_moon_single)
    dr_ii = r_moon_truth - r_moon_single
    moon_vector_computed[ii,:2] = [t_gps_ii, pe_ii*1e3]
    moon_vector_computed[ii,2:] = r_moon_single

    err_placeholder_1hr[ii,:2] = [t_gps_ii, pe_ii*1e3]
    err_placeholder_1hr[ii,2:5] = dr_ii
    err_placeholder_1hr[ii,5:] = r_correction
if 1:
    sp_moon_corrected_df = pd.DataFrame(moon_vector_computed, columns = ['t_gps','pe_mrad','x', 'y', 'z'])
    sp_moon_corrected_df.to_csv(fr'outputs\tables\moon_corrections\SPmoon_{interp_option.upper()}_{update_interval:.3f}.csv', index = 0)
if 0:
    err_df = pd.DataFrame(err_placeholder_1hr, columns = ['t_gps','pe_mrad','dx', 'dy', 'dz', 'x_cor','y_cor','z_cor'])
    err_df.to_csv(fr'outputs\tables\moon_corrections\{interp_option.upper()}_{update_interval:.3f}.csv', index = 0)
print(f'Loop done. Total updates make : {upd_made} for interval of {update_interval} s in {t_days_used} days')
print(f'''Statistics - updates every {t_gap/60:.1f} min. {interp_option}
      3-sigma : {np.std(err_placeholder_1hr[:,1])*3:.2f} mrad'
      MAX : {np.max(err_placeholder_1hr[:,1]):.2f} mrad'
      ''')
#%%
# plot_x = 'hour'
importlib.reload(bplt)
plot_x = 'day'
n_days_plotted = t_days_used
n_hours_plotted = 2

t_fromstart_d = err_placeholder_1hr[:,0]
t_fromstart_d = (err_placeholder_1hr[:,0] - err_placeholder_1hr[0,0])/86400
t_fromstart_hr = (t_fromstart_d - t_fromstart_d[0])*24

if plot_x == 'hour':
    xlim = n_hours_plotted
    t_plotted = t_fromstart_hr
    unit = 'hr'
elif plot_x == 'day':
    xlim = n_days_plotted
    t_plotted = t_fromstart_d
    unit = 'day'

f, axs = plt.subplots(2)
ax = axs[0]
# ax.plot(t_plotted, err_placeholder_1hr[:,1])
ax.scatter(t_plotted, err_placeholder_1hr[:,1], s = 1)
ax.set_ylabel(f'PE [mrad]')
ax.grid('on')
ax.set_xlim([0, xlim])
ax.set_ylim([0,1])

ax = axs[1]
for ii in range(3):
    ax.plot(t_plotted, err_placeholder_1hr[:,ii+2], label = 'xyz'[ii])
    ax.set_ylabel(f'Pos. Error [m]')
    ax.set_xlim([0, xlim])
ax.set_xlabel(f't since start [{unit}]')
ax.legend()
ax.grid('on')
f.set_tight_layout('tight')
f.suptitle(f'Single-precision Moon vector on  {dt_0.year}-{dt_0.month}-{dt_0.day}, {interp_option.upper()} corrections every {update_interval/3600:.3f} h. 3sigma={np.std(err_placeholder_1hr[:,1])*3:.2f} mrad')
bplt.autosave(f, subfolder = 'MoonFixing')