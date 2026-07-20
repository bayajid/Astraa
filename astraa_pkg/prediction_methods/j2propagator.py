#%% Script to fully define a J2 propagator
# with the goal of having input : r0, v0, dt, t_req and output r,v for t_req
# with RK4 fixed-step numerical integrator
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.constants as const
def propagate_orbit(X, t_start = None, t_end = None, t_step = None, sp = 0):
    # function to predict state using RK4 integator
    # give t_start (t=0), t_end and t_Step (fixed)
    # uses J2
    
    if not sp:
        fct_integrate = integrate_rk4
        fct_j2 = calc_f_J2_separate
    else:
        # use single-precision functions
        fct_integrate = integrate_rk4_sp
        fct_j2 = calc_f_J2_separate_sp
    t_vec_prop = np.arange(t_start, t_end + t_step, t_step)
    
    x_1 = fct_integrate(fct_j2, X, t_vec_prop, t_step)[1]
    x_1 = np.hstack((t_vec_prop.reshape(len(t_vec_prop), 1), x_1))
    return x_1
def integrate_rk4(f_fct, x_0, t_vec_prop = None, t_step = None, return_both = 1):
    ## RK4 integration of the state
    # input state derivative function f_fct
    # initial state x_0
    # start, end times
    # and time-step (if None, will integrate in 1 step)
    # return both to return intermediate integration steps    
    nr_integration_steps = int(len(t_vec_prop))
    
    x_current = x_0
    x_out = np.zeros((nr_integration_steps, 6))
    for ii in range(nr_integration_steps):
        x_out[ii,:] = x_current
        x_1 = x_current
        k_1 = f_fct(x_current)
        
        x_2 = x_1 + k_1 * t_step/2
        k_2 = f_fct(x_2)

        x_3 = x_1 + k_2 * t_step/2
        k_3 = f_fct(x_3) 

        x_4 = x_1 + k_3 * t_step
        k_4 = f_fct(x_4)
        
        dx_rk4 = 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        x_updated = x_current + dx_rk4 * t_step

        # update
        x_current = x_updated
        # x_out[ii,:] = x_current
        # # store intermediate steps
        # k_factors = np.zeros((x_0.shape[0], 4))
        # k_factors[:,0] = k_1
        # k_factors[:,1] = k_2
        # k_factors[:,2] = k_3
        # k_factors[:,3] = k_4
        # update state
    if return_both:
        return x_current, x_out
    else:
        return x_current
def integrate_rk4_sp(f_fct, x_0, t_vec_prop = None, t_step = None, return_both = 1):
    ## RK4 integration of the state in single precision
    # input state derivative function f_fct
    # initial state x_0
    # start, end times
    # and time-step (if None, will integrate in 1 step)
    # return both to return intermediate integration steps    
    nr_integration_steps = int(len(t_vec_prop))
    
    x_current = np.float32(x_0)
    x_out = np.float32(np.zeros((nr_integration_steps, 6)))
    for ii in range(nr_integration_steps):
        x_out[ii,:] = np.float32(x_current)
        x_1 = np.float32(x_current)
        k_1 = np.float32(f_fct(x_current))
        
        x_2 = np.float32(x_1 + k_1 * t_step/2)
        k_2 = np.float32(f_fct(x_2))

        x_3 = np.float32(x_1 + k_2 * t_step/2)
        k_3 = np.float32(f_fct(x_3) )

        x_4 = np.float32(x_1 + k_3 * t_step)
        k_4 = np.float32(f_fct(x_4))
        
        dx_rk4 = np.float32(1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4))
        x_updated = np.float32(x_current + dx_rk4 * t_step)

        # update
        x_current = np.float32(x_updated)

    if return_both:
        return x_current, x_out
    else:
        return x_current    
def calc_f_J2_separate_sp(X):
    # function to calculate the state derivative 
    # considering J2 
    # input state [r, rdot, cdtr]
    # returns X_dot [rdot, rddot, 0]
    x_dot = np.zeros(X.shape[0])
    x_dot[:3] = X[3:6]    
    mu = np.float32(const.mu_e)
    J2 = np.float32(const.J2)
    r_e = np.float32(const.R_E)
    r = np.float32(np.linalg.norm(X[:3]))
    # Accelerations considering J2
    # cart_comp = X[0]
    # xddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((1 - 5*(X[2]/r)**2)*cart_comp/r))
    # cart_comp = X[1]
    # yddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((1 - 5*(X[2]/r)**2)*cart_comp/r))
    # cart_comp = X[2]
    # zddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((3 - 5*(X[2]/r)**2)*cart_comp/r))
    # Edited to new formulation frm https://space.stackexchange.com/questions/55441/orbit-propagator-with-j2-perturbation-has-larger-error-compared-with-simple-2-bo

    cart_comp = np.float32(X[0])
    xddot = np.float32(-mu / r**3 * cart_comp *(1 + 3/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2))
    cart_comp = np.float32(X[1])
    yddot = np.float32(-mu / r**3 * cart_comp *(1 + 3/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2))
    cart_comp = np.float32(X[2])
    zddot = np.float32(-mu / r**3 * cart_comp *(1 + 9/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2))
    x_dot[3:6] = [xddot, yddot, zddot]
    
    return x_dot
def calc_f_J2_separate(X):
    # function to calculate the state derivative 
    # considering J2 
    # input state [r, rdot, cdtr]
    # returns X_dot [rdot, rddot, 0]
    x_dot = np.zeros(X.shape[0])
    x_dot[:3] = X[3:6]    
    mu = const.mu_e
    J2 = const.J2
    r_e = const.R_E
    r = np.linalg.norm(X[:3])
    # Accelerations considering J2
    # cart_comp = X[0]
    # xddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((1 - 5*(X[2]/r)**2)*cart_comp/r))
    # cart_comp = X[1]
    # yddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((1 - 5*(X[2]/r)**2)*cart_comp/r))
    # cart_comp = X[2]
    # zddot = -mu / r**3 * cart_comp - (3/2*J2 * mu/r**2 * (r_e/r)**2 * ((3 - 5*(X[2]/r)**2)*cart_comp/r))
    # Edited to new formulation frm https://space.stackexchange.com/questions/55441/orbit-propagator-with-j2-perturbation-has-larger-error-compared-with-simple-2-bo

    cart_comp = X[0]
    xddot = -mu / r**3 * cart_comp *(1 + 3/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2)
    cart_comp = X[1]
    yddot = -mu / r**3 * cart_comp *(1 + 3/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2)
    cart_comp = X[2]
    zddot = -mu / r**3 * cart_comp *(1 + 9/2 * J2 * (r_e/r)**2 - 15/2 * J2 * (X[2] * r_e/r**2)**2)
    x_dot[3:6] = [xddot, yddot, zddot]
    
    return x_dot
if __name__ == '__main__':
    print('yo wadup')
    print('J2 propagator in the house')
    t0 = 694267200.0
    state_0 = [7027286.49681701,	1706.949891373594,	12145.579574706684,	-13.14539340431999,	1048.2167211056812,	7458.449519852215]
    state_0 = np.array(state_0)
    t_start = 0
    t_end = 10
    t_step = 1
    state_predicted = propagate_orbit(state_0, t_start= 0, t_end = 10, t_step = 1)[0]