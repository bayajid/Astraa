# import numpy as np
# import matplotlib.pyplot as plt

# def naive_slerp(q0, q1, t):
#     dot = np.dot(q0, q1)
#     theta = np.arccos(np.clip(dot, -1.0, 1.0))
#     sin_theta = np.sin(theta)
#     if sin_theta == 0:
#         return q0.copy()
#     a = np.sin((1 - t) * theta) / sin_theta
#     b = np.sin(t * theta) / sin_theta
#     return a * q0 + b * q1

# def proper_slerp(q0, q1, t):
#     dot = np.dot(q0, q1)
#     q1_copy = q1.copy()
#     if dot < 0:
#         q1_copy = -q1_copy
#         dot = -dot
#     theta = np.arccos(np.clip(dot, -1.0, 1.0))
#     sin_theta = np.sin(theta)
#     if sin_theta == 0:
#         return q0.copy()
#     a = np.sin((1 - t) * theta) / sin_theta
#     b = np.sin(t * theta) / sin_theta
#     return a * q0 + b * q1_copy

# def rotate_vector(q, v):
#     w, x, y, z = q
#     conj = [w, -x, -y, -z]
#     def quat_mult(a, b):
#         a0, a1, a2, a3 = a
#         b0, b1, b2, b3 = b
#         return np.array([
#             a0*b0 - a1*b1 - a2*b2 - a3*b3,
#             a0*b1 + a1*b0 + a2*b3 - a3*b2,
#             a0*b2 - a1*b3 + a2*b0 + a3*b1,
#             a0*b3 + a1*b2 - a2*b1 + a3*b0
#         ])
#     v_quat = np.array([0, v[0], v[1], v[2]])
#     temp = quat_mult(q, v_quat)
#     result = quat_mult(temp, conj)
#     return result[1:]

# # Setup
# angle = 170 * np.pi / 180
# q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
# q1 = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])  # 170° around y
# q1_neg = -q1
# v0 = np.array([1, 0, 0])  # Test vector
# ts = np.linspace(-0.5, 1.5, 100)  # For smooth plot, with extrapolation

# # Compute angles for short path (proper SLERP, handles sign)
# angles_short = []
# for t in ts:
#     qt = proper_slerp(q0, q1, t)  # Same as proper_slerp(q0, q1_neg, t)
#     vt = rotate_vector(qt, v0)
#     ang = np.arctan2(vt[2], vt[0]) * 180 / np.pi
#     angles_short.append(ang)

# # Compute angles for long path (naive SLERP with flipped sign)
# angles_long = []
# for t in ts:
#     qt = naive_slerp(q0, q1_neg, t)
#     vt = rotate_vector(qt, v0)
#     ang = np.arctan2(vt[2], vt[0]) * 180 / np.pi
#     angles_long.append(ang)

# # Plot
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# axs[0].plot(ts, angles_short, label='Short path')
# axs[0].set_title('Proper SLERP (Short Arc, Sign Handled)')
# axs[0].set_xlabel('t (time parameter)')
# axs[0].set_ylabel('Rotation Angle (degrees)')
# axs[0].grid(True)
# axs[0].legend()

# axs[1].plot(ts, angles_long, label='Long path', color='orange')
# axs[1].set_title('Naive SLERP with Flipped Sign (Long Arc)')
# axs[1].set_xlabel('t (time parameter)')
# axs[1].set_ylabel('Rotation Angle (degrees)')
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.show()

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
    dot = q_prev[0]*q[0] + q_prev[1]*q[1] + q_prev[2]*q[2] + q_prev[3]*q[3]
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

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def resample_quaternion_messages(messages, dt_out=0.025):
    """
    messages: Nx8 array/list with format
              [t, qw, qx, qy, qz, dqw, dqx, dqy, dqz]
    dt_out: output step (default 0.025s = 40 Hz)

    Returns:
      new_times, new_quats (M x 4 array, scalar-first)
    """
    messages = np.asarray(messages)
    times = messages[:, 0]
    quats = messages[:, 1:5]
    qdots = messages[:, 5:9]

    new_times = np.arange(times[0], times[-1], dt_out)
    out_quats = []

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
            out_quats.append(squad(q0, q1, s0, s1, tau))

    return new_times, np.array(out_quats)

## Load input data
quat_path = os.path.join(os.getcwd(), 'outputs/tables/rocketlab_quatpred/true_quatrocketlab_march.csv')
# quat_path = os.path.join(os.getcwd(), 'outputs/tables/rotate_all_axes_quatpred/true_quatrotate_all_axes.csv')
save_folder = os.path.join(os.getcwd(), 'outputs/tables/quat_test')
plot_dir = os.path.join(os.getcwd(), 'outputs/plots/quat_test')
os.makedirs(plot_dir, exist_ok=True)

## Interpolate to 5 ms
data_df = pd.read_csv(quat_path)  
propagators_enabled = 1
dt_req = 0.01 #25e-3 # s, 50ms

#update_rates = [1, 2, 5, 10] # Hz
latencies = [0, 1, 2, 3, 4]
update_rates = [10]
latencies = [0]#latencies[:]

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
# positive
# q_bf_used = data_sliced[:,1:]
# negative


t_sliced = data_sliced[:,0] 
# t_for_interp = t_sliced[(t_sliced >= 10) * (t_sliced <= t_sliced[-1]-10)]
t_for_interp = t_sliced # Time for prediction window
n_digits = 3
t_req = np.arange(t_sliced[0], t_sliced[-1]+dt_req, dt_req)
t_req = np.round(t_req, n_digits)
t_gps_interp = CubicSpline(t_sliced, data_sliced[:,0], axis = 0)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

j = 0

while (j<3):
    q_bf_used = []
    q_bf_used = data_sliced[:,1:].copy()

    if j <1:
        # q_bf_used = -data_sliced[:,1:]
        # q_bf_used[100:] = data_sliced[100:,1:]

        
        # q_bf_used[0:60,:]  = q_bf_used[0:60,:]
        q_bf_used[100:] = -q_bf_used[100:]
        settings = 'negative'

    elif j == 1:
        q_bf_used[100:] = -q_bf_used[100:]
        settings = 'corrected'

    else:
        # q_bf_used = data_sliced[:,1:]
        settings = 'positive'
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
            t_gps_interp_5ms = t_gps_interp(t_req)
            
            # get true data
            # q_host_true_5ms = q_host_interp(t_req)
            q_host_true_5ms = q_host_interp
            t_gps_pred_5ms = np.zeros(t_gps_interp_5ms.shape)
            q_host_pred_5ms = np.zeros(q_host_true_5ms.shape)
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
                    ax.quiver(0, 0, 0, neg_vec_fromq_ham[0], neg_vec_fromq_ham[1], neg_vec_fromq_ham[2], color='g', linestyle='--', alpha = 0.02, label='Rotated by -q')

                if j ==1:
                    cor_vec_fromq_ham = rot.rotate_with_quat(vec_ref, q_host_pred_5ms[ii,:4], conj_switch = 0, h_q = 1).flatten()
                    cor_vec_fromq.append(cor_vec_fromq_ham)
                    ax.quiver(0, 0, 0, cor_vec_fromq_ham[0], cor_vec_fromq_ham[1], cor_vec_fromq_ham[2], color='g', linestyle='--', alpha = 0.02, label='Rotated by -q')
                else:
                    vec_fromq_ham = rot.rotate_with_quat(vec_ref, q_host_pred_5ms[ii,:4], conj_switch = 0, h_q = 1).flatten()
                    vec_fromq.append(vec_fromq_ham)
                    ax.quiver(0, 0, 0, vec_fromq_ham[0], vec_fromq_ham[1], vec_fromq_ham[2], color='b',  linestyle='--', alpha = 0.02, label='Rotated by q')

                ax.quiver(0, 0, 0, vec_ref[0], vec_ref[1], vec_ref[2], color='r',  linestyle='--', label='Original Vector')
            
            quat_pred.append(q_host_pred_5ms)
            latency_used.append(latency_selected)
            update_rate_used.append(update_freq)
            title_save = f'{settings}_quatpred_l{latency_selected}_u{update_freq}hz.csv'
            save_path = f'{save_folder}/{title_save}'
            result_df = pd.DataFrame.from_dict({
                't_s' : t_gps_interp_5ms,                
                'q_pred_c' : q_host_pred_5ms[:,0],
                'q_pred_1' : q_host_pred_5ms[:,1],
                'q_pred_2' : q_host_pred_5ms[:,2],
                'q_pred_3' : q_host_pred_5ms[:,3],
            })
            result_df.to_csv(save_path, index=False)
            print(f'Saved {title_save}')
    j+=1


neg_vec_fromq = np.array([x for x in neg_vec_fromq if not np.any(np.isnan(x))])
vec_fromq = np.array([x for x in vec_fromq if not np.any(np.isnan(x))])
cor_vec_fromq = np.array([x for x in cor_vec_fromq if not np.any(np.isnan(x))])



#%%
# Original vector (red)
    
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())
plt_title = f"quatpred_l{latency_selected}_u{update_freq}hz"
plt.savefig(os.path.join(plot_dir,f"{plt_title}_3D rotaion.png"), dpi=300, bbox_inches='tight')
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

figure,axs = plt.subplots(3,1, sharex=True)
plt.suptitle(f'{plt_title}_Pointing_error')

vec_array = np.tile([0, 1, 0], (neg_vec_fromq.shape[0],1))
cor_vec_array = np.tile([0, 1, 0], (cor_vec_fromq.shape[0],1))

pos_vec_array = np.tile([0, 1, 0], (vec_fromq.shape[0],1))

# dots_n = np.einsum('ij,ij->i', vec_array, normalize_vector(neg_vec_fromq))
# dots = np.einsum('ij,ij->i', vec_array, normalize_vector(vec_fromq))
# dots_e = np.einsum('ij,ij->i', normalize_vector(neg_vec_fromq), normalize_vector(vec_fromq))

axs[0].plot(calculate_angles_between_vectors(neg_vec_fromq, vec_array)*1e6, label = 'sign_swapped') # axs[0].plot(np.arccos(np.clip(dots_n,-1,1))*1e6)
axs[0].plot(calculate_angles_between_vectors(cor_vec_fromq, vec_array)*1e6, color = 'C1', label = 'corrected')
axs[1].plot(calculate_angles_between_vectors(vec_fromq, pos_vec_array)*1e6) # axs[1].plot(np.arccos(np.clip(dots,-1,1))*1e6)
axs[2].plot(calculate_angles_between_vectors(vec_fromq, neg_vec_fromq)*1e6, label = 'sign_swapped') # axs[2].plot(np.arccos(np.clip(dots_e,-1,1))*1e6)
axs[2].plot(calculate_angles_between_vectors(vec_fromq, cor_vec_fromq)*1e6, label = 'corrected')
axs[0].set_ylabel('θ_NEG_VEC[µrad]')
axs[1].set_ylabel('θ_VEC[µrad]')
axs[2].set_ylabel('θ_diff[µrad]')
plt.legend()

fname = f"{plt_title}_PE"
# bplt.savefig(figure,fname, subfolder = 'quat_test')
plt.savefig(os.path.join(plot_dir,f"{plt_title}_Pointing_error.png"), dpi=300, bbox_inches='tight')

figure,axs = plt.subplots(3,1, sharex=True)
axs[0].plot(neg_vec_fromq[0], color='r')
axs[1].plot(neg_vec_fromq[1], color='r')
axs[2].plot(neg_vec_fromq[2], color='r')
axs[0].set_ylabel('x')
axs[1].set_ylabel('y')
axs[2].set_ylabel('z')

axs[0].plot(vec_fromq[0], color='g',linestyle='--')
axs[1].plot(vec_fromq[1], color='g',linestyle='--')
axs[2].plot(vec_fromq[2], color='g',linestyle='--')
axs[0].legend('-+')
plt.suptitle('Difference in rotated vectors')
fname = f"{plt_title}_Diff_in rotated vec"
# bplt.savefig(figure,fname, subfolder = 'quat_test')
plt.savefig(os.path.join(plot_dir,f"{plt_title}_Diff_in rotated vec.png"), dpi=300, bbox_inches='tight')


interp_data = glob.glob(os.path.join(save_folder,"*_quatpred_*.csv"))
for i,file in enumerate (interp_data):
    df = pd.read_csv(file)
    f, axs = plt.subplots(4,1)        
    for nrow in range(4):
        ax = axs[nrow]
        ax.plot(df.iloc[:,0].values, df.iloc[:,nrow+1])
        ax.set_ylabel(['q'+'c123'[nrow], 'qdot'+'c123'[nrow]][0])
        ax.grid('on')
    f.set_tight_layout('tight') 
    plt.suptitle('Qaternion elements')   
    if i == 0:
        i_text = 'negative'
    else: 
        i_text = 'positive'
    plt.savefig(os.path.join(plot_dir,f"{i_text}_{plt_title}_.png"), dpi=300, bbox_inches='tight')
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
