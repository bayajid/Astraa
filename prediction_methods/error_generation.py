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
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import tudat_tools.tudat_converter as tudatconv
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
## Getting ECI to RSW rotation matrix
def pos_err_gen(err_mean, err_std, nrows = 1, frame_option = 'eci', ncols = None, s_host= None, seed_used = 1):
    """Function to generate position errors in ECI
    input mean and std for x/y/z components in pos (or vel)
    np.random will be used to sample random from a normal distribution
    and if needed, perform a RSW to ECI (in case RSW error components are provided)

    Args:
        err_mean (list/str): means of errors (3D-error or list of each axis)
        err_std err_mean (list/str): std of errors (3D-error or list of each axis)
        frame_option (str, optional): frame of input errror distribution. Defaults to 'eci'.
        s_host (array, optional): pos/vel of errors if RSW is used. Defaults to None.

    Returns:
        err_array (array) nrows x 3 array
    """        
    np.random.seed(seed_used)
    if ncols is None:
        ncols = np.round(err_std)
    err_dist = np.ones((nrows, ncols))
    if type(err_mean) != list:
        err_means = err_mean / 1.732
        err_means = np.array([err_means, err_means, err_means])

        err_stds = err_std / 1.732
        err_stds = np.array([err_stds, err_stds, err_stds])
    else:
        err_means = np.array(err_mean)
        err_stds = np.array(err_std)
   
    # get req random samples
    err_std_samples = np.random.randn(nrows, ncols)
    
    # calculate errors
    err_dist = err_dist * err_means
    err_dist = err_dist + err_std_samples * err_std
            
    # rotate
    # transform errors from rsw to ECI
    if frame_option == 'rsw':
        tudconv = tudatconv.tudat_predictor()
        # get RSW to ECI
        for ii, s_ii in enumerate(s_host):

            ROT_RSWfromECI = tudconv.calc_rotrsweci(r_h = s_host[ii,:3], v_h = s_host[ii,3:])
            # rotate to ECI
            err_dist[ii,:] = ROT_RSWfromECI.transpose() @ err_dist[ii,:]
    return err_dist
def add_errors_to_attitude(quat_true, std_err_att, mean_err_att = [0,0,0]):
    """function to generate attitude errors are compute the resulting quaternion

    Args:
        quat_eci2bf (array): array of attitude quaernions
        std_err_att (vector): std of attitude errors [rad]
        mean_err_att (vector): mean of attitude errors [rad]
    
    Returns:
        quat_with_error_ii, [err_att_host, quat_errors]
    """    
    
    err_att_host = pos_err_gen(mean_err_att, std_err_att, nrows = quat_true.shape[0])
    quat_with_error = np.zeros(quat_true.shape)
    quat_errors = np.zeros(quat_true.shape)
    for ii, err in enumerate(err_att_host):
        quat_error_ii = conv.convert_ea2quat(err, deg = 0)
        quat_with_error_ii = rot.multiply_quat_hamiltonian(quat_true[ii,:], quat_error_ii)
        quat_with_error[ii,:] = quat_with_error_ii.flatten()
        quat_errors[ii,:] = quat_error_ii
    return quat_with_error, [err_att_host, quat_errors]

def calculate_pmg_errors(ae_true, pmg_components_az, pmg_components_el):
    """Function to calculate delta Az/El terms according to
    the pointing Model extended (Kroll, 2013)

    Args:
        ae_true (array): true az/el values [rad]
        pmg_components_az (dict): dictionary with constant PM terms [rad]
        pmg_components_el (dict): dict with const PM el terms [rad]

    Returns:
        ae_with_pm, d_ae_pm: True azel and delta_azel PM terms [rad]
    """ 
    
    # get trig terms
    s_ae = np.sin(ae_true[:,:2])
    c_ae = np.cos(ae_true[:,:2])
    t_e = np.tan(ae_true[:,1])
    s_2a = np.sin(2*ae_true[:,0])
    s_3a = np.sin(3*ae_true[:,0])
    c_2a = np.cos(2*ae_true[:,0])
    c_3a = np.cos(3*ae_true[:,0])

    ae_with_pm_err = np.zeros((ae_true.shape[0], 2))
    d_ae_pm  = np.zeros((ae_true.shape[0], 2))
    
    for ii, ae_true_ii in enumerate(ae_true[:,:2]):
        c_a_ii = c_ae[ii,0]
        c_e_ii = c_ae[ii,1]
        s_a_ii = s_ae[ii,0]
        s_e_ii = s_ae[ii,1]
        t_e_ii = t_e[ii]        
        c_2a_ii = c_2a[ii]
        s_2a_ii = s_2a[ii]
        c_3a_ii = c_3a[ii]        
        s_3a_ii = s_3a[ii]        
        
        d_az_ii = pmg_components_az['aoff']  \
            + pmg_components_az['aan'] * s_a_ii * t_e_ii \
            + pmg_components_az['aae'] * c_a_ii * t_e_ii \
            + pmg_components_az['npae'] * t_e_ii \
            - pmg_components_az['bnp'] * c_e_ii \
            + pmg_components_az['aes'] * s_a_ii \
            + pmg_components_az['aec'] * c_a_ii
 
        d_el_ii = pmg_components_el['eoff'] \
            + pmg_components_el['ean'] * c_a_ii \
            + pmg_components_el['eae'] * s_a_ii \
            + pmg_components_el['eec'] * c_e_ii \
            + pmg_components_el['es2a'] * s_2a_ii \
            + pmg_components_el['ec2a'] * c_2a_ii \
            + pmg_components_el['es3a'] * s_3a_ii \
            + pmg_components_el['ec3a'] * c_3a_ii\
        # store errors
        d_ae_pm[ii,:] = [d_az_ii, d_el_ii]
        # store AE with PM terms
        ae_with_pm_ii = ae_true_ii + [d_az_ii, d_el_ii]
        ae_with_pm_err[ii,:] = ae_with_pm_ii
    return ae_with_pm_err, d_ae_pm
if __name__ == '__main__':
    err_mean = 50
    err_std = 30
    nrows = 100
    errors = pos_err_gen(err_mean, err_std, nrows)
    import matplotlib.pyplot as plt
    plt.hist(errors.flatten())
    