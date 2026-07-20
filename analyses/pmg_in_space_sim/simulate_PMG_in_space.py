import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

# Measurement generation
# Basic setup - true values
n_az = 25  # Number of azimuth only measurements
n_el = 13  # Number of elevation only measurements
az_true = np.concatenate((np.zeros(n_el), np.linspace(-170, 170, n_az)))  # True azimuth values
el_true = np.concatenate((np.linspace(-30, 90, n_el), np.zeros(n_az)))  # True elevation values

# Define errors and corrections (example values)
params = {'aoff': 3, 'aan': 0.1, 'aae': 0.05, 'npae': 0.02, 'bnp': 0.01, 'eoff': 2,
          'ean': 0.1, 'eae': 0.05, 'eec': 0.03, 'ees': 0.02, 'ean2': 0.01, 'eae2': 0.01,
          'ean3': 0.005, 'eae3': 0.005}

uncertainty_oneSTDdev = 0.05/3

# Apply corrections to azimuth and elevation
az_measured = az_true + params['aoff'] + params['aan'] * np.cos(np.radians(az_true)) + \
             params['aae'] * np.sin(np.radians(az_true)) + np.random.randn(1,len(az_true)) * uncertainty_oneSTDdev
el_measured = el_true + params['eoff'] + params['ean'] * np.cos(np.radians(el_true)) + \
             params['eae'] * np.sin(np.radians(el_true)) + \
             params['npae'] * np.sin(np.radians(2 * az_measured)) + params['bnp'] * np.cos(np.radians(2 * az_measured)) + \
             params['eec'] + params['ees'] * np.sin(np.radians(el_true)) + \
             params['ean2'] * np.sin(np.radians(2 * az_measured)) + params['eae2'] * np.cos(np.radians(2 * az_measured)) + \
             params['ean3'] * np.sin(np.radians(3 * az_measured)) + params['eae3'] * np.cos(np.radians(3 * az_measured)) + np.random.randn(1,len(az_true)) * uncertainty_oneSTDdev

# Plot the measured and true values
plt.figure()
plt.scatter(az_true, el_true, label='True Positions')
plt.scatter(az_measured, el_measured, label='Measured Positions')
plt.grid(True)
plt.legend()

# Optimization to estimate the parameters
initial_params = np.array([1] * 14)  # Adjust based on your model

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

res = minimize(cost_function, initial_params, args=(az_measured, el_measured, az_true, el_true),

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

az_est = az_measured - estimated_params[0] - estimated_params[1] * np.cos(np.radians(az_measured)) - estimated_params[2] * np.sin(np.radians(az_measured))
el_est = el_measured - estimated_params[3] - estimated_params[4] * np.cos(np.radians(el_measured)) - estimated_params[5] * np.sin(np.radians(el_measured)) - \
        estimated_params[6] * np.sin(np.radians(2 * az_est)) - estimated_params[7] * np.cos(np.radians(2 * az_est)) - \
        estimated_params[8] - estimated_params[9] * np.sin(np.radians(el_measured)) - \
        estimated_params[10] * np.sin(np.radians(2 * az_est)) - estimated_params[11] * np.cos(np.radians(2 * az_est)) - \
        estimated_params[12] * np.sin(np.radians(3 * az_est)) - estimated_params[13] * np.cos(np.radians(3 * az_est))

# Error in estimation
errEst = np.sqrt((az_true - az_est)**2 + (el_true - el_est)**2) * np.pi/180 * 1e6
print('Error in estimation in µrad :')
print(errEst)


# Error in initial measurement
errMeas = np.std(np.sqrt((az_measured - az_est)**2 + (el_measured - el_est)**2)) * np.pi/180 * 1e6
print('3 standard deviation of measurement error in µrad :')
print(3*errMeas)

# Initial errors
print('3 standard deviation of inputted errors in µrad :')
print(3*uncertainty_oneSTDdev * np.pi/180 *1e6)

# Standard deviation of model residual (including noise)
print('3 standard deviation of estimation error in µrad :')
print(3*np.std(errEst))

