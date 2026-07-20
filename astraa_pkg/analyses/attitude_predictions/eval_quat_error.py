# # script to numerically quanity the max angular difference between two quaternions
# # from test results of April 12 on orbit calibration
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os 
# import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# import pathlib
# import pandas as pd
# import scipy.io
# import importlib
# import os, sys
# sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
# import pandas as pd
# import random
# from basic_tools.vector_operations import calc_dot_angle
# import attitude_tools.conversions as conv
# import attitude_tools.rotations as rot
# import prediction_methods.error_generation as err_gen

# # generate a bunch of LOS for PE calculation
# nrows = int(1e3)
# los_all = err_gen.pos_err_gen([0,0,0], [1,1,1], nrows = nrows)

# q_mountingoffset_input = np.array([0.999657, -0.006996, 0.013992, 0.020988])

# q_resolved_1 = np.array([0.99967773,-0.00664017,0.01492263,0.01943363])
# q_resolved_2 = np.array([0.99967618,-0.00756482,0.01456293,0.01944792])



# def eval_prediction_pe(q_pred, q_true, ref_pt_vec):
#     ## Evaluate PE introduced by quaternion prediction method
#     print(f'Q true in : {q_true}')
#     print(f'Q resolved in : {q_pred}')
#     nrows_pred = ref_pt_vec.shape[0]
#     ref_pt_vec = ref_pt_vec / np.linalg.norm(ref_pt_vec, axis = 1).reshape([nrows_pred,1])
#     # placeholders
#     pe_all = np.zeros((nrows_pred, 1))
    
#     for ii, ref_vec in enumerate(ref_pt_vec):

#         los_true = rot.rotate_with_quat(ref_vec, q_true)
#         los_pred = rot.rotate_with_quat(ref_vec, q_pred, h_q = 1, conj_switch = 0).flatten()        
        
#         # prediction pointing error via dot product rule
#         pe_pred = calc_dot_angle(los_true, los_pred)*1e6 # [urad]
#         pe_all[ii] = pe_pred
#     return pe_all

# def get_pe_max(pe_all):
#     return np.abs(np.max(pe_all))

# pair_nr = 5

# if pair_nr == 1:
#     q_resolved = q_resolved_1
# elif pair_nr == 2:
#     q_resolved = q_resolved_2
# elif pair_nr == 3: # 1 vs 2
#     q_mountingoffset_input = np.array([0.9999772,-0.0039472,0.0047895,-0.0026544])
#     q_resolved = np.array([0.9999696,-0.0006953,0.0055769,-0.0054049])
# elif pair_nr == 4:# 1 vs 3
#     q_mountingoffset_input = np.array([0.9999772,-0.0039472,0.0047895,-0.0026544])
#     q_resolved = np.array([0.9999768,-0.0025490,0.0051537,-0.0036575])
# elif pair_nr == 5: # 1 vs 4
#     q_mountingoffset_input = np.array([0.9999772,-0.0039472,0.0047895,-0.0026544])
#     q_resolved = np.array([0.9999775,-0.0024551,0.0051533,-0.0035268])
    
# pe_max = get_pe_max(eval_prediction_pe(q_resolved, q_mountingoffset_input, los_all))

# print(f'Pair {pair_nr} -> PE = {pe_max/1e3:.1f} mrad')


#####---------------QUADRATIC-------------------####
# import numpy as np
# import matplotlib.pyplot as plt

# def quadratic_interpolation(q0, q1, q2, t):
#     # Quadratic interpolation of components: q(t) = (1-t)^2*q0 + 2*(1-t)*t*q1 + t^2*q2
#     a = (1 - t)**2
#     b = 2 * (1 - t) * t
#     c = t**2
#     q = a * q0 + b * q1 + c * q2
#     # Normalize to ensure valid quaternion
#     norm = np.sqrt(np.sum(q**2))
#     return q / norm if norm > 0 else q



# # Setup
# angle1 = 90 * np.pi / 180
# angle2 = 170 * np.pi / 180
# q0 = np.array([1.0, 0.0, 0.0, 0.0])  # 0°
# q1 = np.array([np.cos(angle1 / 2), 0.0, 0.0, np.sin(angle1 / 2)])  # 90° around z
# q2 = np.array([np.cos(angle2 / 2), 0.0, 0.0, np.sin(angle2 / 2)])  # 170° around z
# q1_neg = -q1
# v0 = np.array([1, 0, 0])  # Test vector
# ts = np.linspace(-0.5, 1.5, 100)  # For smooth plot, with extrapolation

# # Compute angles for original path
# angles_orig = []
# for t in ts:
#     qt = quadratic_interpolation(q0, q1, q2, t)
#     vt = rotate_vector(qt, v0)
#     ang = np.arctan2(vt[1], vt[0]) * 180 / np.pi
#     angles_orig.append(ang)

# # Compute angles for path with flipped middle quaternion
# angles_flipped = []
# for t in ts:
#     qt = quadratic_interpolation(q0, q1_neg, q2, t)
#     vt = rotate_vector(qt, v0)
#     ang = np.arctan2(vt[1], vt[0]) * 180 / np.pi
#     angles_flipped.append(ang)

# # Plot
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# axs[0].plot(ts, angles_orig, label='Original path')
# axs[0].set_title('Quadratic Interpolation (Original Middle Quaternion)')
# axs[0].set_xlabel('t (time parameter)')
# axs[0].set_ylabel('Rotation Angle (degrees)')
# axs[0].grid(True)
# axs[0].legend()

# axs[1].plot(ts, angles_flipped, label='Flipped path', color='orange')
# axs[1].set_title('Quadratic Interpolation (Flipped Middle Quaternion)')
# axs[1].set_xlabel('t (time parameter)')
# axs[1].set_ylabel('Rotation Angle (degrees)')
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.show()

###############--------------------Rotate Frame -----------------------------##########
import numpy as np
import matplotlib.pyplot as plt
import sys
import os, importlib
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import plotting_tools.plotting_utilities as plt_util
import plotting_tools.basic_plotting as bplt
from mpl_toolkits.mplot3d import Axes3D



importlib.reload(plt_util)
def quadratic_interpolation(q0, q1, q2, t):
    # Quadratic interpolation of components: q(t) = (1-t)^2*q0 + 2*(1-t)*t*q1 + t^2*q2
    a = (1 - t)**2
    b = 2 * (1 - t) * t
    c = t**2
    q = a * q0 + b * q1 + c * q2
    # Normalize to ensure valid quaternion
    norm = np.sqrt(np.sum(q**2))
    return q / norm if norm > 0 else q

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def normalize(q):
    return q / np.linalg.norm(q)

def quaternion_from_axis_angle(axis, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    axis = normalize_vector(axis)
    cos_half = np.cos(angle_rad / 2)
    sin_half = np.sin(angle_rad / 2)
    return np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector_by_quaternion(v, q):
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    temp = quaternion_multiply(q, v_quat)
    rotated = quaternion_multiply(temp, q_conj)
    return rotated[1:]

def rotate_vector(q, v):
    w, x, y, z = q
    conj = [w, -x, -y, -z]
    def quat_mult(a, b):
        a0, a1, a2, a3 = a
        b0, b1, b2, b3 = b
        return np.array([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0
        ])
    v_quat = np.array([0, v[0], v[1], v[2]])
    temp = quat_mult(q, v_quat)
    result = quat_mult(temp, conj)
    return result[1:]

plot_dir = os.path.join(os.getcwd(), 'outputs/plots/quat_test')

# Parameters
axis = np.array([0, 1, 0])  # Rotation axis (y-axis)
angle_deg = 45  # Rotation angle in degrees

# Create quaternion
q = quaternion_from_axis_angle(axis, angle_deg)
q = normalize(q)
print(f"Quaternion: {q}")

# negative quaternion
q_neg = -q
print(f"Negative Quaternion: {q_neg}")

angle1 = 90 * np.pi / 180
# angle2 = 170 * np.pi / 180
# q0 = np.array([1.0, 0.0, 0.0, 0.0])  # 0°
q1 = np.array([np.cos(angle1 / 2), 0.0, np.sin(angle1 / 2), 0.0])  # 90° around z
q1 = normalize(q1)

q1_neg = -q1
v0 = np.array([1, 0, 0])  # Test vector
ts = np.linspace(-0.5, 1.5, 100)  # For smooth plot, with extrapolation

# Compute angles for original path
# angles_orig = []
# for t in ts:
#     qt = quadratic_interpolation(q0, q1, q2, t)
#     vt = rotate_vector(qt, v0)
#     ang = np.arctan2(vt[1], vt[0]) * 180 / np.pi
#     angles_orig.append(ang)

# Original basis vectors
orig_X = np.array([1, 0, 0])
orig_Y = np.array([0, 1, 0])
orig_Z = np.array([0, 0, 1])

# Rotated basis vectors
rot_X = rotate_vector_by_quaternion(orig_X, q)
rot_Y = rotate_vector_by_quaternion(orig_Y, q)
rot_Z = rotate_vector_by_quaternion(orig_Z, q)


# Rotated basis vectors by neg quat
neg_rot_X = rotate_vector_by_quaternion(orig_X, q_neg)
neg_rot_Y = rotate_vector_by_quaternion(orig_Y, q_neg)
neg_rot_Z = rotate_vector_by_quaternion(orig_Z, q_neg)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Coordinate Frame Rotation by {angle_deg}° around {axis}')

# Original frame (red)
ax.quiver(0, 0, 0, orig_X[0], orig_X[1], orig_X[2], color='k', length = 3, label='Original X')
ax.quiver(0, 0, 0, orig_Y[0], orig_Y[1], orig_Y[2], color='k', length = 3, label='Original Y')
ax.quiver(0, 0, 0, orig_Z[0], orig_Z[1], orig_Z[2], color='k', length = 3, label='Original Z')

# Rotated frame (blue)
ax.quiver(0, 0, 0, rot_X[0], rot_X[1], rot_X[2], color='b', length = 1, label='Rotated X')
ax.quiver(0, 0, 0, rot_Y[0], rot_Y[1], rot_Y[2], color='b', length = 1, label='Rotated Y')
ax.quiver(0, 0, 0, rot_Z[0], rot_Z[1], rot_Z[2], color='b', length = 1, label='Rotated Z')

# Negative Rotated frame (orange)
ax.quiver(0, 0, 0, neg_rot_X[0], neg_rot_X[1], neg_rot_X[2], color='C1', length = 1, linestyle='dashed', label='Neg_Rotated X')
ax.quiver(0, 0, 0, neg_rot_Y[0], neg_rot_Y[1], neg_rot_Y[2], color='C1', length = 1, linestyle='dashed', label='Neg_Rotated Y')
ax.quiver(0, 0, 0, neg_rot_Z[0], neg_rot_Z[1], neg_rot_Z[2], color='C1', length = 1, linestyle='dashed', label='Neg_Rotated Z')


# Simplify legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())


figname = f"Coordinate frame rotation by Q and -Q"
bplt.savefig(fig, figname, subfolder= 'quat_test',   tag_option = 1,x_coord_tag= -8 )
plt.show()