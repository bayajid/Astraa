#-------------------------------------------------------------------##
#%%
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
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import attitude_tools.attitude_simulation as att_sim
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import pointing_calculations.ae_calculation as ae_calc
from scipy.interpolate import CubicSpline
import prediction_methods.interpolators as interp
import prediction_methods.j2propagator as j2prop
import analyses.attitude_predictions.attitude_prediction_utlities as att_pred
import attitude_tools.rotations as rot
import glob
from scipy.spatial.transform import Rotation as R

global flag 
flag = False
#%%
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def calculate_angles_between_vectors(vec1, vec2):
    """
    Calculate angles between two arrays of 3D vectors.
    
    Parameters:
    vec1, vec2: numpy arrays of shape (N, 3)
    
    Returns:
    angles: numpy array of shape (N,) containing angles in radians
    """
    # Calculate dot products for each pair of vectors
    dot_products = np.sum(vec1 * vec2, axis=1)
    
    # Calculate magnitudes for each vector
    mag1 = np.linalg.norm(vec1, axis=1)
    mag2 = np.linalg.norm(vec2, axis=1)
    
    # Calculate cosine of angles
    cos_angles = dot_products / (mag1 * mag2)
    
    # Clamp values to [-1, 1] to handle numerical errors
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Calculate angles in radians
    angles = np.arccos(cos_angles)
    
    return angles

def fix_quat_sign_scalar(q_prev, q):
    global flag
    flag = False
    # q_prev, q, dq are 4-element lists/tuples
    # dot = q_prev[0]*q[0] + q_prev[1]*q[1] + q_prev[2]*q[2] + q_prev[3]*q[3]
    # if dot < 0.0:
    #     q = [-q[0], -q[1], -q[2], -q[3], -q[4], -q[5], -q[6], -q[7]]
    #     flag = True

    dot = q_prev[1]*q[1] + q_prev[2]*q[2] + q_prev[3]*q[3]
    if dot < 0.0:
        q = [-q[0], -q[1], -q[2], -q[3], -q[4], -q[5], -q[6], -q[7]]
        flag = True
    return q

def normalize(q):
    return q / np.linalg.norm(q)

def slerp(q0, q1, t):
    q0, q1 = normalize(q0), normalize(q1)
    dot = np.dot(q0, q1)
    # if dot < 0:
    #     q1 = -q1
    #     dot = -dot
    if dot > 0.9995:  # almost linear
        return normalize((1 - t) * q0 + t * q1)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)

def squad(q0, q1, s0, s1, t):
    return slerp(
        slerp(q0, q1, t),
        slerp(s0, s1, t),
        2 * t * (1 - t)
    )

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def tangent_from_derivative(q, q_dot, dt):
    """
    Compute SQUAD tangent quaternion using q and q_dot from the message.
    Both are scalar-first [qw,qx,qy,qz].
    """
    q = normalize(q)
    q_inv = quat_conj(q)

    # small-step delta from derivative
    delta = quat_mul(q_inv, q_dot * dt)

    # vector part ~ rotation vector
    rotvec = delta[1:4]
    corr = R.from_rotvec(-0.25 * rotvec).as_quat()  # [x,y,z,w]
    corr = np.array([corr[3], corr[0], corr[1], corr[2]])  # convert to scalar-first
    return quat_mul(q, corr)

# scalar_first
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # scalar part
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

def resample_quaternion_messages(messages, dt_out=0.025):
    """
    messages: Nx9 array/list with format
              [t, qw, qx, qy, qz, dqw, dqx, dqy, dqz]
    dt_out: output step (default 0.025s = 40 Hz)

    Returns:
      new_times, new_data (M x 8 array with [qw, qx, qy, qz, dqw, dqx, dqy, dqz])
    """
    messages = np.asarray(messages)
    times = messages[:, 0]
    quats = messages[:, 1:5]
    qdots = messages[:, 5:9]

    new_times = np.arange(times[0], times[-1], dt_out)
    out_quats = []
    out_qdots = []

    for k in range(len(times) - 1):
        t0, t1 = times[k], times[k+1]
        q0, q1 = quats[k], quats[k+1]
        qd0, qd1 = qdots[k], qdots[k+1]
        dt = t1 - t0

        # tangents from derivatives
        s0 = tangent_from_derivative(q0, qd0, dt)
        s1 = tangent_from_derivative(q1, qd1, dt)

        # output times in this interval
        mask = (new_times >= t0) & (new_times < t1)
        local_times = (new_times[mask] - t0) / dt

        for tau in local_times:
            # Interpolate quaternion
            q_interp = squad(q0, q1, s0, s1, tau)
            out_quats.append(q_interp)
            
            # Interpolate derivative (linear interpolation for simplicity)
            qdot_interp = (1 - tau) * qd0 + tau * qd1
            out_qdots.append(qdot_interp)

    # Combine quaternions and derivatives
    out_quats = np.array(out_quats)
    out_qdots = np.array(out_qdots)
    new_data = np.column_stack([out_quats, out_qdots])  # [qw, qx, qy, qz, dqw, dqx, dqy, dqz]

    return new_times, new_data

## Load input data
quat_path = '/home/bkhan/Documents/Git/astropynaric/outputs/tables/rocketlab_quatpred/true_quatrocketlab_march.csv'
# quat_path = os.path.join(os.getcwd(), 'outputs/tables/rotate_all_axes_quatpred/true_quatrotate_all_axes.csv')
save_folder = os.path.join(os.getcwd(), 'outputs/tables/quat_test')
plot_dir = os.path.join(os.getcwd(), 'outputs/plots/quat_test')
os.makedirs(plot_dir, exist_ok=True)

## Interpolate to 5 ms
data_df = pd.read_csv(quat_path)  
propagators_enabled = 1
dt_req = 5e-3 # s, 50ms

#update_rates = [1, 2, 5, 10] # Hz
latencies = [0, 1, 2, 3, 4]
update_rates = [4]
latencies = [4]#latencies[:]

update_intervals = [np.round(1/ii,1) for ii in update_rates]
data_sliced = data_df.values
pe_calc = []
quat_pred = []
quat_true = []
latency_used = []
update_rate_used = []
vec_fromq = []
settings = []
vec_ref = [0, 1, 0]
neg_vec_fromq = []
cor_vec_fromq = []
vec_fromq_ham = []
neg_vec_fromq_ham = []
cor_vec_fromq_ham = []

#%%
t_sliced = data_sliced[:,0] 
# t_for_interp = t_sliced[(t_sliced >= 10) * (t_sliced <= t_sliced[-1]-10)]
t_for_interp = t_sliced # Time for prediction window
n_digits = 3
t_req = np.arange(t_sliced[0], t_sliced[-1]+dt_req, dt_req)
t_req = np.round(t_req, n_digits)
t_gps_interp = CubicSpline(t_sliced, data_sliced[:,0], axis = 0)

t_new, q_message = resample_quaternion_messages(data_df.values, dt_req)

#%%
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# Store vectors for each case
all_neg_vec = {}
all_vec_fromq = {}
all_cor_vec = {}

j = 0
while (j<3):
    # Reset vector lists for each iteration
    vec_fromq = []
    neg_vec_fromq = []
    cor_vec_fromq = []
    vec_fromq_ham = []
    neg_vec_fromq_ham = []
    cor_vec_fromq_ham = []
    
    q_bf_used = []
    q_bf_used = q_message.copy()  # Now contains [qw, qx, qy, qz, dqw, dqx, dqy, dqz]

    if j <1:
        q_bf_used[300:] = -q_bf_used[300:]
        settings = 'sign-changed'

    elif j == 1:
        q_bf_used[300:] = -q_bf_used[300:]
        settings = 'corrected'

    else:
        # q_bf_used = data_sliced[:,1:]
        settings = 'nominal'
        print('+ve')
        
    print(settings)
    # q_host_interp = CubicSpline(t_sliced, q_bf_used, axis = 0)
    q_host_interp = q_bf_used # CubicSpline(t_sliced, q_bf_used, axis = 0)

    
    for update_freq in (update_rates):
        for latency_selected in (latencies):
            update_freq_att_h = update_freq
            dt_gap_att_h = np.round(1/update_freq_att_h,3)
            dt_latency = dt_gap_att_h  * latency_selected
            t_update_arrival = np.round(np.arange(t_for_interp[0], t_for_interp[-1]+dt_gap_att_h, dt_gap_att_h),3)

            ## Adding LATENCY- receive attitude data after delay
            t_gps_interp_5ms =  t_new # t_gps_interp(t_req)
            
            # get true data
            # q_host_true_5ms = q_host_interp(t_req)
            # Ensure both arrays have the same length
            min_length = min(len(t_gps_interp_5ms), len(q_host_interp))
            t_gps_interp_5ms = t_gps_interp_5ms[:min_length]
            q_host_true_5ms = q_host_interp[:min_length]  # Match the length
            t_gps_pred_5ms = np.zeros(t_gps_interp_5ms.shape)
            q_host_pred_5ms = np.zeros((len(t_gps_interp_5ms), q_host_interp.shape[1]))  # Match dimensions
            
            # Debug: Check array sizes
            print(f"t_gps_interp_5ms size: {len(t_gps_interp_5ms)}")
            print(f"q_host_true_5ms size: {len(q_host_true_5ms)}")
            print(f"q_host_interp size: {len(q_host_interp)}")
            
            t_stamps_updates = t_update_arrival - dt_latency
            # data_full_att_h = q_host_interp(t_stamps_updates)
            data_full_att_h = q_host_interp# (t_stamps_updates)

            ii_next_att_h = 0
            quat_interp = interp.we_interpolating()
            for ii, t_ii in enumerate(t_gps_interp_5ms):
                if t_ii >= t_update_arrival[ii_next_att_h] and t_ii < t_update_arrival[-1]:
                    data_att_h = data_full_att_h[ii_next_att_h]
                    
                    if (j == 1) and (ii_next_att_h>0):                        
                        data_full_att_h [ii_next_att_h] = fix_quat_sign_scalar(data_full_att_h[ii_next_att_h-1], data_full_att_h[ii_next_att_h])
                        
                    data_att_h_held = data_att_h # Propagator-off case
                    # UPDATE INTERPOLANT
                    if ii_next_att_h >= 1:
                        
                        quat_interp.get_quad_interpolant(
                            t_both=t_stamps_updates[ii_next_att_h-1:ii_next_att_h+1],
                            r_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,:4],
                            v_both=data_full_att_h[ii_next_att_h-1:ii_next_att_h+1,4:],                
                        )
                        a=1
                    ii_next_att_h += 1
                else:
                    data_att_h = None
                if ii_next_att_h >=2:
                    if propagators_enabled:
                        data_att_interp = quat_interp.interpolate_flexible(t_ii)
                    else:
                        data_att_interp = data_att_h_held[:4]
                    quat_diff = data_att_interp - q_host_true_5ms[ii,:4]
                    a = 1
                else:
                    data_att_interp = [0,0,0,0]

                # store
                q_host_pred_5ms[ii,:4] = data_att_interp
                t_gps_pred_5ms[ii] = t_ii

                # if flag:
                #     print(f"{t_gps_pred_5ms[ii]}\t{q_host_pred_5ms[ii,:4]}")

                if j <1:
                    neg_vec_fromq_ham = rot.rotate_with_quat(vec_ref, q_host_pred_5ms[ii,:4], conj_switch = 0, h_q = 1).flatten()
                    neg_vec_fromq.append(neg_vec_fromq_ham)
                    #ax.quiver(0, 0, 0, neg_vec_fromq_ham[0], neg_vec_fromq_ham[1], neg_vec_fromq_ham[2], color='g', linestyle='--', alpha = 0.02, label='Rotated by -q')

                if j ==1:
                    cor_vec_fromq_ham = rot.rotate_with_quat(vec_ref, q_host_pred_5ms[ii,:4], conj_switch = 0, h_q = 1).flatten()
                    cor_vec_fromq.append(cor_vec_fromq_ham)
                    #ax.quiver(0, 0, 0, cor_vec_fromq_ham[0], cor_vec_fromq_ham[1], cor_vec_fromq_ham[2], color='g', linestyle='--', alpha = 0.02, label='Rotated by -q')
                else:
                    vec_fromq_ham = rot.rotate_with_quat(vec_ref, q_host_pred_5ms[ii,:4], conj_switch = 0, h_q = 1).flatten()
                    vec_fromq.append(vec_fromq_ham)
                    #ax.quiver(0, 0, 0, vec_fromq_ham[0], vec_fromq_ham[1], vec_fromq_ham[2], color='b',  linestyle='--', alpha = 0.02, label='Rotated by q')

                #ax.quiver(0, 0, 0, vec_ref[0], vec_ref[1], vec_ref[2], color='r',  linestyle='--', label='Original Vector')
            
            quat_pred.append(q_host_pred_5ms)
            latency_used.append(latency_selected)
            update_rate_used.append(update_freq)
            title_save = f'{settings}_quatpred_l{latency_selected}_u{update_freq}hz.csv'
            save_path = f'{save_folder}/{title_save}'
            
            result_df = pd.DataFrame.from_dict({
                't_s' : t_gps_pred_5ms, #t_gps_interp_5ms,                
                'q_pred_c' : q_host_pred_5ms[:,0],
                'q_pred_1' : q_host_pred_5ms[:,1],
                'q_pred_2' : q_host_pred_5ms[:,2],
                'q_pred_3' : q_host_pred_5ms[:,3],
            })
            result_df.to_csv(save_path, index=False)
            print(f'Saved {title_save}')
    
    # # Process vectors for this iteration
    neg_vec_fromq = np.array([x for x in neg_vec_fromq if not np.any(np.isnan(x))])
    vec_fromq = np.array([x for x in vec_fromq if not np.any(np.isnan(x))])
    cor_vec_fromq = np.array([x for x in cor_vec_fromq if not np.any(np.isnan(x))])
    
    # Store vectors for this case
    all_neg_vec[settings] = neg_vec_fromq
    all_vec_fromq[settings] = vec_fromq
    all_cor_vec[settings] = cor_vec_fromq

    
    j+=1


# The rest of the plotting code starts here
# Get vectors from the final iteration or use specific cases
neg_vec_fromq = all_neg_vec.get('sign-changed', np.array([]))
vec_fromq = all_vec_fromq.get('nominal', np.array([]))  
cor_vec_fromq = all_cor_vec.get('corrected', np.array([]))


#%%
# Original vector (red)
    
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# ax.legend(unique_labels.values(), unique_labels.keys())
plt_title = f"quatpred_l{latency_selected}_u{update_freq}hz"
# plt.savefig(os.path.join(plot_dir,f"{plt_title}_3D rotaion.png"), dpi=300, bbox_inches='tight')


# bplt.savefig(fig, plt_title, subfolder = 'quat_test',y_coord_tag = -4)

# figure,axs = plt.subplots(3,1, sharex=True)
# plt.suptitle('Difference in all 3 axes\nV rotated by q and -q')
# axs[0].plot(neg_vec_fromq[0]- vec_fromq[0])
# axs[1].plot(neg_vec_fromq[1]- vec_fromq[1])
# axs[2].plot(neg_vec_fromq[2]- vec_fromq[2])
# axs[0].set_ylabel('x')
# axs[1].set_ylabel('y')
# axs[2].set_ylabel('z')
# fname = f"{plt_title}_Diff_in all 3 axes"
# # bplt.savefig(figure,fname, subfolder = 'quat_test')
# plt.savefig(os.path.join(plot_dir,f"{plt_title}_Diff_in all 3 axes.png"), dpi=300, bbox_inches='tight')

figure,axs = plt.subplots(2,1, sharex=True)
plt.suptitle(f'{plt_title}_Pointing_error')

vec_array = np.tile([0, 1, 0], (neg_vec_fromq.shape[0],1))
cor_vec_array = np.tile([0, 1, 0], (cor_vec_fromq.shape[0],1))
pos_vec_array = np.tile([0, 1, 0], (vec_fromq.shape[0],1))

# dots_n = np.einsum('ij,ij->i', vec_array, normalize_vector(neg_vec_fromq))
# dots = np.einsum('ij,ij->i', vec_array, normalize_vector(vec_fromq))
# dots_e = np.einsum('ij,ij->i', normalize_vector(neg_vec_fromq), normalize_vector(vec_fromq))

axs[0].plot(calculate_angles_between_vectors(neg_vec_fromq, vec_array)*1e6,     linestyle = '--', alpha = 0.5, label = 'sign_swapped') # axs[0].plot(np.arccos(np.clip(dots_n,-1,1))*1e6)
axs[0].plot(calculate_angles_between_vectors(cor_vec_fromq, cor_vec_array)*1e6, linestyle = '--', alpha = 0.5, label = 'corrected')
axs[0].plot(calculate_angles_between_vectors(vec_fromq, pos_vec_array)*1e6,     linestyle = '--', alpha = 0.5, label = 'nominal') # axs[1].plot(np.arccos(np.clip(dots,-1,1))*1e6)
axs[1].plot(calculate_angles_between_vectors(vec_fromq, neg_vec_fromq)*1e6,     linestyle = '--', alpha = 0.5, label = 'sign_swapped') # axs[2].plot(np.arccos(np.clip(dots_e,-1,1))*1e6)
axs[1].plot(calculate_angles_between_vectors(vec_fromq, cor_vec_fromq)*1e6,     linestyle = '--', alpha = 0.5, label = 'corrected')

axs[0].set_ylabel('θ_VEC[µrad]')
# axs[].set_ylabel('θ_VEC[µrad]')
axs[1].set_ylabel('θ_diff[µrad]')
axs[1].legend()

fname = f"{plt_title}_PE"
# bplt.savefig(figure,fname, subfolder = 'quat_test')
plt.savefig(os.path.join(plot_dir,f"{plt_title}_Pointing_error.png"), dpi=300, bbox_inches='tight')

# figure,axs = plt.subplots(3,1, sharex=True)
# axs[0].plot(neg_vec_fromq[0], color='r')
# axs[1].plot(neg_vec_fromq[1], color='r')
# axs[2].plot(neg_vec_fromq[2], color='r')
# axs[0].set_ylabel('x')
# axs[1].set_ylabel('y')
# axs[2].set_ylabel('z')

# axs[0].plot(vec_fromq[0], color='g',linestyle='--')
# axs[1].plot(vec_fromq[1], color='g',linestyle='--')
# axs[2].plot(vec_fromq[2], color='g',linestyle='--')
# axs[0].legend('-+')
# plt.suptitle('Difference in rotated vectors')
# fname = f"{plt_title}_Diff_in rotated vec"
# # bplt.savefig(figure,fname, subfolder = 'quat_test')
# plt.savefig(os.path.join(plot_dir,f"{plt_title}_Diff_in rotated vec.png"), dpi=300, bbox_inches='tight')


interp_data = glob.glob(os.path.join(save_folder,"*_quatpred_*.csv"))
f, axs = plt.subplots(4,1)   
for i,file in enumerate (interp_data):
    filename = os.path.basename(file)
    i_text  = filename.split("_")[0]
    df = pd.read_csv(file)
    # if i == 0:
    #     i_text = 'Sign-changed'
    # elif i == 1:          
    #     i_text = 'Nominal'
    # else:
    #     i_text = 'Corrected'     
    
    for nrow in range(4):
        ax = axs[nrow]
        ax.plot(df.iloc[:,0].values, df.iloc[:,nrow+1], linestyle = "--", alpha = 0.5,  label = f"{i_text}")
        ax.set_ylabel(['q'+'c123'[nrow], 'qdot'+'c123'[nrow]][0])
        ax.grid('on')
    ax.legend()
    f.set_tight_layout('tight') 
    
plt.suptitle('Qaternion elements')  
plt.savefig(os.path.join(plot_dir,f"{plt_title}_Qaternion elements.png"), dpi=300, bbox_inches='tight')
    # fname = f"{i}_{plt_title}_"
    # bplt.savefig(f, fname, subfolder = 'quat_test')

nrow = 0
ncol = 0
f, axs = plt.subplots(4, 2)
for ncol in range(2):  
    data_df_ = data_df.iloc[:,1+ncol*4:(1+ncol)*4+1]
    for nrow in range(4):
        ax = axs[nrow, ncol]
        ax.plot(data_df.iloc[:,0], data_df_.iloc[:,nrow], color='C1')
        ax.set_ylabel(['q'+'c123'[nrow], 'qdot'+'c123'[nrow]][ncol])
        ax.grid('on')
f.set_tight_layout('tight')
# plt.savefig(os.path.join(plot_dir,f"{plt_title}_Original Q.png"), dpi=300, bbox_inches='tight')
fname = f"{plt_title}_Original Q"
bplt.savefig(f, fname,subfolder = 'quat_test', tag_option = 1,x_coord_tag= -8 )
plt.show()
#        vec_fromq_ham = rot.rotate_with_quat(vec_ref, quat_pred, conj_switch = 0, h_q = 1).flatten()

#print(vec_fromq- neg_vec_fromq)

# %%
