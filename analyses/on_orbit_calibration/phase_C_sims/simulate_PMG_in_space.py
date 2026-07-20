import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import pandas as pd
import sys, os
import json
# Measurement generation
# Basic setup - true values
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
path_azel = r'outputs\tables\pmg_sim_aefixed'

fname_t = 'pmg_tracking_fixedae_noerr1.csv'

az_loaded = pd.read_csv(f'{path_azel}/{fname_t}')
ae_track = az_loaded.values[:,[3,4]]
ae_comm = az_loaded.values[:,[1,2]]
az_meas = ae_track[:,0]
az_comm = ae_comm[:,0] 
el_meas = ae_track[:,1]
el_comm = ae_comm[:,1]

param_name = fr'{path_azel}/pmg_coefficients.json'
# az_true = np.concatenate((np.zeros(n_el), np.linspace(-170, 170, n_az)))  # True azimuth values
# el_true = np.concatenate((np.linspace(-30, 90, n_el), np.zeros(n_az)))  # True elevation values

# Define errors and corrections (example values)
with open(param_name ) as json_data:
    params = json.load(json_data)
    json_data.close()

uncertainty_oneSTDdev = 0.05/3

# Apply corrections to azimuth and elevation
# az_meas = az_true + params['aoff'] + params['aan'] * np.cos(np.radians(az_true)) + \
#              params['aae'] * np.sin(np.radians(az_true)) + np.random.randn(1,len(az_true)) * uncertainty_oneSTDdev
# el_meas = el_true + params['eoff'] + params['ean'] * np.cos(np.radians(el_true)) + \
#              params['eae'] * np.sin(np.radians(el_true)) + \
#              params['npae'] * np.sin(np.radians(2 * az_meas)) + params['bnp'] * np.cos(np.radians(2 * az_meas)) + \
#              params['eec'] + params['ees'] * np.sin(np.radians(el_true)) + \
#              params['ean2'] * np.sin(np.radians(2 * az_meas)) + params['eae2'] * np.cos(np.radians(2 * az_meas)) + \
#              params['ean3'] * np.sin(np.radians(3 * az_meas)) + params['eae3'] * np.cos(np.radians(3 * az_meas)) + np.random.randn(1,len(az_true)) * uncertainty_oneSTDdev


initial_params = np.array([1] * 15)  # Adjust based on your model

def cost_function(params, az_measured, el_measured, az_true, el_true):
    az_est = az_measured - params[0] - params[1] * np.cos(np.radians(az_measured)) - params[2] * np.sin(np.radians(az_measured))
    el_est = el_measured - params[3] - params[4] * np.cos(np.radians(el_measured)) - params[5] * np.sin(np.radians(el_measured)) - \
             params[6] * np.sin(np.radians(2 * az_est)) - params[7] * np.cos(np.radians(2 * az_est)) - \
             params[8] - params[9] * np.sin(np.radians(el_measured)) - \
             params[10] * np.sin(np.radians(2 * az_est)) - params[11] * np.cos(np.radians(2 * az_est)) - \
             params[12] * np.sin(np.radians(3 * az_est)) - params[13] * np.cos(np.radians(3 * az_est))

    # Calculate the error (difference) between estimated true values and actual true values
    error = np.sum((az_est - az_true) ** 2) + np.sum((el_est - el_true) ** 2)
    return error

res = minimize(cost_function, initial_params, args=(az_meas, el_meas, az_meas, el_meas),

               options={'maxiter': 10000})

estimated_params = res.x
print('Estimated Parameters:')
print(estimated_params)

# Real parameters
real_params = np.array(list(params.values()))

# Error in parameters
errParams = real_params - estimated_params
print('Error in params:')
print(errParams)

az_est = az_meas - estimated_params[0] - estimated_params[1] * np.cos(np.radians(az_meas)) - estimated_params[2] * np.sin(np.radians(az_meas))
el_est = el_meas - estimated_params[3] - estimated_params[4] * np.cos(np.radians(el_meas)) - estimated_params[5] * np.sin(np.radians(el_meas)) - \
        estimated_params[6] * np.sin(np.radians(2 * az_est)) - estimated_params[7] * np.cos(np.radians(2 * az_est)) - \
        estimated_params[8] - estimated_params[9] * np.sin(np.radians(el_meas)) - \
        estimated_params[10] * np.sin(np.radians(2 * az_est)) - estimated_params[11] * np.cos(np.radians(2 * az_est)) - \
        estimated_params[12] * np.sin(np.radians(3 * az_est)) - estimated_params[13] * np.cos(np.radians(3 * az_est))

# Error in estimation
errEst = np.sqrt((az_meas - az_est)**2 + (el_meas - el_est)**2) * np.pi/180 * 1e6
print('Error in estimation in µrad :')
print(errEst)


# Error in initial measurement
errMeas = np.std(np.sqrt((az_meas - az_est)**2 + (el_meas - el_est)**2)) * np.pi/180 * 1e6
print('3 standard deviation of measurement error in µrad :')
print(3*errMeas)

# Initial errors
print('3 standard deviation of inputted errors in µrad :')
print(3*uncertainty_oneSTDdev * np.pi/180 *1e6)

# Standard deviation of model residual (including noise)
print('3 standard deviation of estimation error in µrad :')
print(3*np.std(errEst))

