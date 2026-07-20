## in this script, unit tests will be done for rotations
# ideally we have direct quaternion roatation of a vector
# this will be tested using basic single-axis rotations with expected outputs
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# import splines.quaternion
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import importlib
np.set_printoptions(3)
importlib.reload(rot)


test_case = 1


# 3-2-1 rotations
# input RPY angles [deg]
# and reference vector
# for test_case in [1,2,3,4,5,6,7]:
for test_case in [3]:    
    # output rotated vector via tested methods
    if test_case == 1:
        print(f'Test {test_case} - 45 deg rotation around Z')
        RPY = [0, 0, 45]
        vec_ref = [1, 0, 0]
        expected_output = [0.707, -0.707, 0]
    elif test_case == 2:
        print(f'Test {test_case} - -45 deg rotation around Z')
        RPY = [0, 0, -45]
        vec_ref = [1, 0, 0]
        expected_output = [0.707, 0.707, 0]
    elif test_case == 3:
        print(f'Test {test_case} - 60 deg rotation around y')
        RPY = [0, 60, 0]
        vec_ref = [1, 0, 0]
        expected_output = [1 * np.cos(np.deg2rad(60)), 0, 1 * np.sin(np.deg2rad(60))]
    elif test_case == 4:
        print(f'Test {test_case} - 90 around Y, 60 deg rotation around X')
        RPY = [60, 90, 0]
        vec_ref = [1, 0, 0]
        expected_output = [0, np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60))]
    elif test_case == 5:
        print(f'Test {test_case} - 180 Z; 90 Y; 270 X')
        RPY = [270, 90, 180]
        vec_ref = [1, 0, 0]
        expected_output = [0, 1, 0]
    elif test_case == 6: 
        print(f'Test {test_case} - 270 deg rotation around Z, 45 around X')
        RPY = [45, 0, 270]
        vec_ref = [-1, 0, 1]
        expected_output = [0, 0, np.sqrt(2)]
    elif test_case == 7: 
        print(f'Test {test_case} - 90 deg rotation around Z, 180 Y, 270 around X')
        RPY = [270, 180, 90]
        vec_ref = [1, 1, -1]
        expected_output = [-1,-1,-1]
    print(f'RPY input : {RPY}. Ref vec : {vec_ref}')
    dcm = conv.convert_ea2dcm(RPY, deg = 1)
    dcm_sp = R.from_matrix(dcm)
    ea_sp_flip = R.from_euler('ZYX', np.flip(RPY), degrees = True)
    ea_sp_noflip = R.from_euler('ZYX', RPY, degrees = True)
    ea_sp_negative = R.from_euler('ZYX', np.flip(RPY)*(-1), degrees = True)
    ea_sp_XYZ = R.from_euler('XYZ', RPY, degrees = True)
    
    # quaternion
    quat_own = conv.convert_ea2quat(RPY, deg = 1, ham_q = 0)
    quat_own_ham = conv.convert_ea2quat(RPY, deg = 1, ham_q = 1)
    print(f'Quat: Ham own :{quat_own_ham}')

    # vec_fromdcm_2 = vec_ref @ dcm
    vec_fromdcm = dcm @ vec_ref
    vec_from_spflip = ea_sp_flip.apply(vec_ref)
    vec_from_sp_noflip = ea_sp_noflip.apply(vec_ref)
    vec_from_sp_neg = ea_sp_negative.apply(vec_ref)
    vec_from_sp_XYZ = ea_sp_XYZ.apply(vec_ref)
    # vec_from_quat


    vec_fromq = rot.rotate_with_quat(vec_ref, quat_own, conj_switch = 0, h_q = 0).flatten()
    vec_fromq_ham = rot.rotate_with_quat(vec_ref, quat_own_ham, conj_switch = 0, h_q = 1).flatten()
    vec_fromq_conj = rot.rotate_with_quat(vec_ref, quat_own, conj_switch = 1).flatten()
    vec_fromq_ham_conj = rot.rotate_with_quat(vec_ref, quat_own_ham, conj_switch = 1, h_q = 1, reshuffle =1).flatten()
    print(f'''Output expected : -----{expected_output}-----\n
    Own EA -> Q vec Q_conj : {vec_fromq}. Pass - {np.linalg.norm(vec_fromq - np.array(expected_output))<1e-3}
    Own EA -> HAM; Q vec Q_conj : {vec_fromq_ham}. Pass - {np.linalg.norm(vec_fromq_ham - np.array(expected_output))<1e-3}
    Own EA -> Q_conj vec Q : {vec_fromq_conj}. Pass - {np.linalg.norm(vec_fromq_conj - np.array(expected_output))<1e-3}
    Own EA -> HAM; Q_conj vec Q w/ Switched Quat Mult Order : {vec_fromq_ham_conj}. Pass - {np.linalg.norm(vec_fromq_ham_conj - np.array(expected_output))<1e-3}
    Own EA -> DCM -> Rot : {vec_fromdcm}. Pass - {np.linalg.norm(vec_fromdcm - np.array(expected_output))<1e-3}

    SP + flip + neg : {vec_from_sp_neg}
    SP + noflip : {vec_from_sp_noflip}
    ''')
    # QUATERNION CONVERSIOn
    # Own [qvec; q0] : {quat_own.flatten()}
    # SP [qvec; q0] : {ea_sp_negative.as_quat()}
