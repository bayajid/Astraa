import numpy as np
#import plotting_tools.basic_plotting as pl
import os, sys
# Add parent directory to paths
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import analyses.sat_ground_contact_times.sat_to_ground_tools.gt_calc as gt
def simulate_ground_track(initial_cond, T_all, dt, use_j2 = True):
    ## Function to create an array of ground track topocentric coordinates
    ## after inputting a dictionary of initial conditions
    from gt_calc import calc_gt
    import numpy as np

    ## Unpack initial conditions
    a = initial_cond['a'] # semi-major axis [m]
    a_0 = initial_cond['az_0']
    lat_s = initial_cond['lat_0']
    phi_s = initial_cond['phi_s']
    i = initial_cond['i']
    n = initial_cond['n']
    # Calculate rotation rate of the orbit pole
    if use_j2:
        w_1 = -360 / 86164.1004 - 2.396288e9*(a*1e-3)**(-3.5) * np.cos(np.deg2rad(i))
    else:
        w_1 = -360 / 86164.1004
    T_range = np.arange(0, T_all+dt, dt)
    size = np.shape(T_range)
    
    ssp_array = np.zeros((size[0], 2)) # Long, Lat [deg]
    psi_array = np.zeros((size[0], 1)) # Heading angles [deg]
    
    for ii, tt in enumerate(T_range):
        res_deg = calc_gt(a_0, lat_s, phi_s, n, i, dt, w_1 = w_1)[1]
        ssp = [res_deg[2], res_deg[1]] # Lat/Long of SSP [deg]
        lat_s = ssp[0]
        a_0 = res_deg[0]
        phi_s = res_deg[3]
        psi_s = res_deg[5]        
        
        ## Store
        ssp_array[ii,:] = ssp
        psi_array[ii,:] = psi_s 
    return T_range, ssp_array, psi_array

# def calculate_fov():
    

    
def plot_gt(ssp_array, label_info):
    import matplotlib.pyplot as plt    
    fig, ax = plt.subplots()
    ax.plot(ssp_array[:,1], ssp_array[:,0], label = 'GT')
    fig.suptitle(label_info_scat['title'], fontsize = 12)
    ax.set_xlabel(label_info_scat['xlabel'], fontsize = 10)
    ax.set_ylabel(label_info_scat['ylabel'], fontsize = 10)
    ax.grid(True)
    # plt.show()
    
    ## Function to make a 2D plot of the GT
def calc_sc_nadir_coord(coord_ssp, coord_t, heading_ssp):
    """Function to calculate the spacecraft(terminal)-centered
    coordinates with respect to a target t (eg. ground station)
    using spherical geometry

    Args:
        coord_ssp (list): long, lat of SSP [deg]
        coord_t (list): long, lat of target (GS) [deg]
        heading_ssp (float): heading angle of satellite wrt North (CCW positive) [deg]

    Returns:
        lam - earth angle from SSP to GS [rad]
        phi_e - azimuth from SSP to GS w.r.t. North [rad]
        phi_h - azimuth from SSP to GS w.r.t. sat. velocity vector [deg]
    """    
    long_ssp = coord_ssp[0]
    lat_ssp = coord_ssp[1]
    
    long_t = coord_t[0]
    lat_t = coord_t[1]
    
    colat_ssp = np.deg2rad(gt.colat(lat_ssp, rad = 0)) # Colat of SSP [rad]
    colat_t = np.deg2rad(gt.colat(lat_t, rad = 0)) # Colat of SSP [rad]
    delta_long = np.deg2rad(long_ssp - long_t) # Difference in longitude btw SSP and T [rad]
    
    lam = np.arccos(np.cos(colat_t)*np.cos(colat_ssp) 
                    + np.sin(colat_t)*np.sin(colat_ssp)*np.cos(delta_long)) # Earth angle from Earth center to target [rad]
    
    phi_e = gt.acos2((np.cos(colat_t) - np.cos(colat_ssp)*np.cos(lam))
                     / (np.sin(colat_ssp)*np.sin(lam)), gt.fct_hem(delta_long)) # Az to target from SSP [rad]
    # check terminal's azimuth angle accounting for SSP heading angle
    phi_h = gt.mod360_deg(heading_ssp - np.rad2deg(phi_e)) # Terminal's az. WRT target [deg]
    # phi_h = gt.mod360_deg(heading_ssp + np.rad2deg(phi_e)) # Terminal's az. WRT target [deg]
    return lam, phi_e, phi_h
def calc_area_access_el(sin_rho, lam):
    """Function to calculate the area access elevation
    from GS to the satellite

    Args:
        sin_rho (float): sine of the angular radius of earth 
        lam (float): Earth central angle [rad]

    Returns:
        eta - nadir angle from SC to target [rad]
        eps - elevation angle from GS to SC [rad]
    """    
    # calculate area-access elevation (Elevation wrt horizon from GS)
    eta = np.arctan(sin_rho * np.sin(lam) 
                    / (1 - sin_rho*np.cos(lam))) # Nadir angle from SC to target [rad]
    eps = np.arccos(np.sin(eta) / sin_rho) # [rad], elevation from target to SC wrt horizon
    return eta, eps

# def check_gs_visibility()
def calc_target_point_coord(cos_lam, 
                            sin_lam, 
                            long_ssp,
                            lat_ssp,
                            cos_colat_ssp,
                            sin_colat_ssp,
                            cos_phi_e, 
                            phi_e,
                            hem_colat_ssp, 
                            ):
    # Function used to calculate target point coordinates based area access coordinates:
    # requires a prior calculation of the Earth angle, which depends on the SC viewing angle
    # and the spacecraft altitude
        #  cos_lam - cosine of the target Earth angle
        #  sin_lam - sine of the target Earth angle
        # long_ssp, lat_ssp - longitude and latitude of sub satellite point [deg]
        #  sin_colat_ssp - sine of colatitude of the sub satellite point
        #  cos_colat_ssp
        #  cos_phi_e - cosine of the azimuth of the target point relative to north, 
        #   CCW postive
        #  hem_colat_ssp - hemisphere function of the SSP colatitude (+1 if colat<180. -1 if colat>180)
    #  return [long_p, lat_p] [deg] target point geocentric coordinates
    cos_colat_p = (cos_lam * cos_colat_ssp + sin_lam * sin_colat_ssp * cos_phi_e)
    colat_p = np.arccos(cos_colat_p) # rad
    sin_colat_p = np.sin(colat_p) 
    cos_delta_long = np.round(((cos_lam - cos_colat_p * cos_colat_ssp) / 
                      (hem_colat_ssp * sin_colat_ssp * sin_colat_p )), 7)
    # return in Deg
    # geocentric longitude offset of target w.r.t. SSP. Positive westwards
    delta_long = np.rad2deg(np.arccos(cos_delta_long)) - 90 * (hem_colat_ssp - 1)
    # calculate 
    lat_p = 90 - np.rad2deg(colat_p)
    long_p = long_ssp - gt.fct_hem(phi_e, rad = 0) * delta_long
    return [long_p, lat_p]

def calc_fov_points(ssp_coord, 
                    heading_ssp, 
                    sin_rho, 
                    fov_el_bounds,
                    fov_az_bounds,
                    n_angle_points = 100):
    """
    

    Parameters
    ----------
    ssp_coord : [long, lat] for SSP [deg]
    heading_ssp : heading angle of SSP wrt N, positive CCW [deg]
        
    sin_rho : sine of earth angle
        
    fov_el_bounds : elevation bounds for intstrument FOV. [45,69.5] for LCT on ISS
    fov_az_bounds : azimuth bounds for instrument FOV wrt velocity vec. [-45, 45] for LCT on ISS
    n_angle_points : Number of points to approximate the FOV bounds. The default is 100.
    h_a : altitude of ground station [m]
    Returns
    -------
    [target_u, target_l, target_left, target_right]
    [upper, lower, left and right bounds] [long, lat] in deg
    

    """
    lat_ssp = ssp_coord[1]
    long_ssp = ssp_coord[0]
    
    phi_l = fov_az_bounds[0] # deg left side bound for sensor azimuth FOV
    phi_r = fov_az_bounds[1] # deg right side bound for sensor azimuth FOV
    
    # Convert SC-azimuth relative to heading angle
    # phi_h_l = phi_l + heading_ssp # deg 
    # phi_h_r = phi_r + heading_ssp # deg
    
    phi_h_l = heading_ssp - phi_l # deg 
    phi_h_r = heading_ssp - phi_r # deg
    
    eta_i = fov_el_bounds[0] # deg, inner bound for sensor elevation FOV
    eta_o = fov_el_bounds[1] # deg, inner bound for sensor elevation FOV
    
    # ranges of FOV angles
    phi_range = np.linspace(phi_h_l, phi_h_r, n_angle_points)
    eta_range = np.linspace(eta_i, eta_o, n_angle_points)
    
    size_phi = np.shape(phi_range)
    size_eta = np.shape(eta_range)
    # Upper and lower bound calculations
    # output placeholders
    target_u = np.zeros((size_phi[0], 2)) 
    target_l = np.zeros((size_phi[0], 2))
    
    # calculate constant upper and lower bound parameters
    sin_eta_o = np.sin(np.deg2rad(eta_o))
    cos_eps_o = sin_eta_o / sin_rho  # SC outer elevation angle
    eps_o = np.rad2deg(np.arccos(cos_eps_o))
    lambda_o = 90 - eta_o - eps_o # earth angle for outer FOV point

    sin_eta_i = np.sin(np.deg2rad(eta_i))
    cos_eps_i = sin_eta_i / sin_rho  # SC inner elevation angle
    eps_i = np.rad2deg(np.arccos(cos_eps_i))
    lambda_i = 90 - eta_i - eps_i # earth angle for inner FOV point
    
    # Inner and outer earth viewing angles
    cos_lam_o = np.cos(np.deg2rad(lambda_o))
    sin_lam_o = np.sin(np.deg2rad(lambda_o))
    cos_lam_i = np.cos(np.deg2rad(lambda_i))
    sin_lam_i = np.sin(np.deg2rad(lambda_i))
    
    # sub satellite point colatitude terms
    colat_ssp = gt.colat(lat_ssp, rad = 0)
    cos_colat_ssp = np.cos(np.deg2rad(colat_ssp))
    sin_colat_ssp = np.sin(np.deg2rad(colat_ssp))
    hem_colat_ssp = gt.fct_hem(colat_ssp, rad = 0) # hemisphere function of colat of ssp
    # calculate target point coordinates for upper and lower bounds
    for ii, phi_e in enumerate(phi_range):
        cos_phi_e = np.cos(np.deg2rad(phi_e))
        
        coord_p_i = calc_target_point_coord(cos_lam_i,
                                    sin_lam_i,
                                    long_ssp,
                                    lat_ssp,
                                    cos_colat_ssp,
                                    sin_colat_ssp,
                                    cos_phi_e, 
                                    phi_e,
                                    hem_colat_ssp, 
                                    )
        coord_p_o = calc_target_point_coord(cos_lam_o,
                                    sin_lam_o,
                                    long_ssp,
                                    lat_ssp,
                                    cos_colat_ssp,
                                    sin_colat_ssp,
                                    cos_phi_e, 
                                    phi_e,
                                    hem_colat_ssp, 
                                    )
        ## store target point bounds
        target_u[ii,:] = coord_p_o
        target_l[ii,:]  = coord_p_i
        
    
    # Left and right bounds calculations
    target_left = np.zeros((size_eta[0], 2))
    target_right = np.zeros((size_eta[0], 2))
    # constant parameters
    cos_phi_h_left = np.cos(np.deg2rad(phi_h_l))
    cos_phi_h_right = np.cos(np.deg2rad(phi_h_r))
    for ii, eta_ii in enumerate(eta_range):
        # Get req angles
        # calculate required earth angles
        sin_eta_ii = np.sin(np.deg2rad(eta_ii))
        cos_eps_ii = sin_eta_ii / sin_rho  # SC outer elevation angle
        eps_ii = np.rad2deg(np.arccos(cos_eps_ii))
        lambda_ii = 90 - eta_ii - eps_ii # earth angle for outer FOV point

        cos_lam_ii = np.cos(np.deg2rad(lambda_ii))
        sin_lam_ii = np.sin(np.deg2rad(lambda_ii))

        coord_p_left = calc_target_point_coord(cos_lam_ii,
                                    sin_lam_ii,
                                    long_ssp,
                                    lat_ssp,
                                    cos_colat_ssp,
                                    sin_colat_ssp,
                                    cos_phi_h_left,
                                    phi_h_l,
                                    hem_colat_ssp, 
                                    )
        coord_p_right = calc_target_point_coord(cos_lam_ii,
                                    sin_lam_ii,
                                    long_ssp,
                                    lat_ssp,
                                    cos_colat_ssp,
                                    sin_colat_ssp,
                                    cos_phi_h_right, 
                                    phi_h_r,
                                    hem_colat_ssp, 
                                    )
        ## store target point bounds
        target_left[ii,:] = coord_p_left
        target_right[ii,:]  = coord_p_right           
    return [target_u, target_l, target_left, target_right]

def calc_vis_area_point(ssp_coord, 
                    heading_ssp, 
                    sin_rho, 
                    n_angle_points = 100):
    """
    Function to calculate the visible area from the satellite

    Parameters
    ----------
    ssp_coord : [long, lat] for SSP [deg]
    heading_ssp : heading angle of SSP wrt N, positive CCW [deg]
        
    sin_rho : sine of earth angle
        
    n_angle_points : Number of points to approximate the FOV bounds. The default is 100.
    h_a : altitude of ground station [m]
    Returns
    -------
    [target_u, target_l, target_left, target_right]
    [upper, lower, left and right bounds] [long, lat] in deg
    

    """
    lat_ssp = ssp_coord[1]
    long_ssp = ssp_coord[0]
    
    eta_o = np.rad2deg(np.arcsin(sin_rho))    
    # ranges of FOV angles
    phi_range = np.linspace(0, 360, n_angle_points)
    
    vis_area = np.zeros((len(phi_range), 2))
    
    # calculate constant upper and lower bound parameters
    sin_eta_o = np.sin(np.deg2rad(eta_o))
    cos_eps_o = sin_eta_o / sin_rho  # SC outer elevation angle
    eps_o = np.rad2deg(np.arccos(cos_eps_o))
    lambda_o = 90 - eta_o - eps_o # earth angle for outer FOV point
    sin_lam_o = np.sin(np.deg2rad(lambda_o))
    cos_lam_o = np.cos(np.deg2rad(lambda_o))
    # sub satellite point colatitude terms
    colat_ssp = gt.colat(lat_ssp, rad = 0)
    cos_colat_ssp = np.cos(np.deg2rad(colat_ssp))
    sin_colat_ssp = np.sin(np.deg2rad(colat_ssp))
    hem_colat_ssp = gt.fct_hem(colat_ssp, rad = 0) # hemisphere function of colat of ssp
    # calculate target point coordinates for upper and lower bounds
    for ii, phi_e in enumerate(phi_range):
        cos_phi_e = np.cos(np.deg2rad(phi_e))
        
        coord_p_i = calc_target_point_coord(cos_lam_o,
                                    sin_lam_o,
                                    long_ssp,
                                    lat_ssp,
                                    cos_colat_ssp,
                                    sin_colat_ssp,
                                    cos_phi_e, 
                                    phi_e,
                                    hem_colat_ssp, 
                                    )
        ## store target point bounds
        vis_area[ii,:] = coord_p_i
        
    
    # Left and right bounds calculations
    return vis_area
def calc_required_fov(simulated_ground_track,
                      req_coord_gs,
                      eta_lct,
                      phi_lct,
                      sin_rho,
                      rho,
                      check_visiblity = True,
                      t_vec_used = None,
                      ):
    ## Function to generate the FOV bounds 
    # when providing the satellite ground track function output,
    # sensor elevation FOV limits [deg]
    # sensor azimuth FOV limits [deg]
    # sine of earth angle
    # earth angle [rad]
    # check_visibility - conditional to limit FOV computations
    # Set to False when want to get FOV coordinates regardless of GS location.
    # t_vec_usd - time vector if it is desired to limit the analysed datapoints
    # outputs a dictionary with time keys
    # and FOV bound whenever the GS is "observable" by ISS (not when its in the limited FOV)
    output_dict = {}
    
    # Unpack simulated ground_track outputs
    t_vec, lat_gt, long_gt= simulated_ground_track[0], simulated_ground_track[1][:,0], simulated_ground_track[1][:,1]
    heading_angle_gt = simulated_ground_track[-1]
    if t_vec_used != None:
        ind_used = [ii for ii in range(len(t_vec)) if t_vec[ii] in t_vec_used]
        t_vec = t_vec_used
        lat_gt = lat_gt[ind_used]
        long_gt = long_gt[ind_used]
        heading_angle_gt = heading_angle_gt[ind_used]
    
    # Calculate constant terms
    colat_gs = 90 - req_coord_gs[1]
    cos_colat_gs = np.cos(np.deg2rad(colat_gs))
    sin_colat_gs = np.sin(np.deg2rad(colat_gs))
    for ii, t in enumerate(t_vec):
        output_dict[t] = {}
        output_dict[t]['ssp'] = [long_gt[ii], lat_gt[ii]]
        output_dict[t]['heading'] = heading_angle_gt[ii]
        if check_visiblity: # Compute satellite FOV if GS is within observable area
            # calculate earth angle from SSP to GS to check if FOV needs to be computed
            colat_ssp = 90 - lat_gt[ii]
            delta_long = np.abs(long_gt[ii] - req_coord_gs[0])
            cos_colat_ssp = np.cos(np.deg2rad(colat_ssp))
            sin_colat_ssp = np.sin(np.deg2rad(colat_ssp))
            cos_delta_long = np.cos(np.deg2rad(delta_long))
            
            lam = np.arccos(cos_colat_gs*cos_colat_ssp + sin_colat_gs * sin_colat_ssp * cos_delta_long)
            
            fov_visible = calc_vis_area_point(ssp_coord = output_dict[t]['ssp'],
                                heading_ssp = output_dict[t]['heading'],
                                sin_rho = sin_rho
                )
            output_dict[t]['vis_area'] = fov_visible
            if lam < rho: # GS is observable from SSP
                output_dict[t]['observable'] = True
                fov_bounds = calc_fov_points(ssp_coord = output_dict[t]['ssp'],
                                    heading_ssp = output_dict[t]['heading'],
                                    sin_rho = sin_rho,
                                    fov_el_bounds = eta_lct,
                                    fov_az_bounds = phi_lct
                    )
                fov_visible = calc_vis_area_point(ssp_coord = output_dict[t]['ssp'],
                                    heading_ssp = output_dict[t]['heading'],
                                    sin_rho = sin_rho
                    )
                output_dict[t]['vis_area'] = fov_visible
                output_dict[t]['fov_up'] = fov_bounds[0]
                output_dict[t]['fov_lo'] = fov_bounds[1]
                output_dict[t]['fov_l'] = fov_bounds[2]
                output_dict[t]['fov_r'] = fov_bounds[3]
                output_dict[t]['fov_bounds'] = fov_bounds
            else:
                output_dict[t]['observable'] = False
                output_dict[t]['fov_up']  = None
                output_dict[t]['fov_lo']  = None
                output_dict[t]['fov_l']  = None
                output_dict[t]['fov_r']  = None
                output_dict[t]['fov_bounds']  = None
        else: # Compute satellite FOV for each point
            fov_visible = calc_vis_area_point(ssp_coord = output_dict[t]['ssp'],
                                heading_ssp = output_dict[t]['heading'],
                                sin_rho = sin_rho
                )
            output_dict[t]['vis_area'] = fov_visible
            output_dict[t]['observable'] = True
            fov_bounds = calc_fov_points(ssp_coord = output_dict[t]['ssp'],
                                heading_ssp = output_dict[t]['heading'],
                                sin_rho = sin_rho,
                                fov_el_bounds = eta_lct,
                                fov_az_bounds = phi_lct
                )
            output_dict[t]['fov_up'] = fov_bounds[0]
            output_dict[t]['fov_lo'] = fov_bounds[1]
            output_dict[t]['fov_l'] = fov_bounds[2]
            output_dict[t]['fov_r'] = fov_bounds[3]
            output_dict[t]['fov_bounds'] = fov_bounds
    return output_dict
def find_gs_in_fov(output_dict, 
                   req_coord_gs):
    # Function used to evaluate the number and length of FOV-GS passes. Input the output_dict
    # generated using the ground_track_FOV_calc code
    # and the required GS coordinates [long, lat]
    # Output format: {pass index}->tsart; t_end; length; ii_start; ii_end
    observation_dict = {}
    n_observed_passes = 0
    ii_pass_prev = 0 # index of previous GS pass
    for ii, t in enumerate(output_dict.keys()):
        t_chosen = t
        chosen_dict = output_dict[t_chosen]
        if chosen_dict['observable']:
            bounds_chosen_up = chosen_dict['fov_up']
            bounds_chosen_lo = chosen_dict['fov_lo']
            bounds_chosen_l = chosen_dict['fov_l']
            bounds_chosen_r = chosen_dict['fov_r']
            # check if any bounds signs change
            # upper

            # transform GS and FOV coordinates to avoid sign changes
            # if sign_swap:
            if 0: # Not sure if needed- check whether FOV is discontinuous. 
                sign_swap = 0 # conditional to check if long/lat change signs within the FOV bounds
                if np.sign(np.max(bounds_chosen_up[:,0])) != np.sign(np.min(bounds_chosen_up[:,0])):
                    sign_swap = 1
                elif np.sign(np.max(bounds_chosen_up[:,1])) != np.sign(np.min(bounds_chosen_up[:,1])):
                    sign_swap = 1
                    # lower
                elif np.sign(np.max(bounds_chosen_lo[:,0])) != np.sign(np.min(bounds_chosen_lo[:,0])):
                    sign_swap = 1
                elif np.sign(np.max(bounds_chosen_lo[:,1])) != np.sign(np.min(bounds_chosen_lo[:,1])):
                    sign_swap = 1
                    # left
                elif np.sign(np.max(bounds_chosen_l[:,0])) != np.sign(np.min(bounds_chosen_l[:,0])):
                    sign_swap = 1
                elif np.sign(np.max(bounds_chosen_l[:,1])) != np.sign(np.min(bounds_chosen_l[:,1])):
                    sign_swap = 1
                    
                elif np.sign(np.max(bounds_chosen_r[:,0])) != np.sign(np.min(bounds_chosen_r[:,0])):
                    sign_swap = 1
                elif np.sign(np.max(bounds_chosen_r[:,1])) != np.sign(np.min(bounds_chosen_r[:,1])):
                    sign_swap = 1
                if sign_swap:
                    # print('Shifting coordinates by 360 deg')
                    long_gs_analyzed = req_coord_gs[0] + 360
                    lat_gs_analyzed = req_coord_gs[1]+ 360
                    bounds_transf_l = bounds_chosen_l + 360
                    bounds_transf_r = bounds_chosen_r +360
                    bounds_transf_up = bounds_chosen_up + 360
                    bounds_transf_lo = bounds_chosen_lo + 360
            else:
                long_gs_analyzed = req_coord_gs[0]
                lat_gs_analyzed = req_coord_gs[1]
                bounds_transf_l = bounds_chosen_l
                bounds_transf_r = bounds_chosen_r
                bounds_transf_up = bounds_chosen_up
                bounds_transf_lo = bounds_chosen_lo
                # # calculate earth angle and elevation angles ToDo: Ended here!!
                # 
                
            ## check for intersections with each bound
            count_az = 0
            count_el = 0
            if long_gs_analyzed > np.max(bounds_transf_l[:,0]):
                count_az +=1
            if long_gs_analyzed > np.max(bounds_transf_r[:,0]):
                count_az +=1
            if lat_gs_analyzed > np.max(bounds_transf_up[:,1]):
                count_el +=1
            if lat_gs_analyzed > np.max(bounds_transf_lo[:,1]):
                count_el +=1  
                
            if count_el == 1 and count_az == 1:
                gs_pass = True
            else:
                ## check again for intersections with each bound
                count_az = 0
                count_el = 0
                if long_gs_analyzed < np.min(bounds_transf_l[:,0]):
                    count_az +=1
                if long_gs_analyzed < np.min(bounds_transf_r[:,0]):
                    count_az +=1
                if lat_gs_analyzed < np.min(bounds_transf_up[:,1]):
                    count_el +=1
                if lat_gs_analyzed < np.min(bounds_transf_lo[:,1]):
                    count_el +=1  
                    
                if count_el == 1 and count_az == 1:
                    gs_pass = True
                else:
                    gs_pass = False
            if gs_pass:
                ii_pass = ii # Update pass index
                if ii_pass == ii_pass_prev+1:
                    pass_continues = True
                else:
                    pass_continues = False
                if not pass_continues:
                    n_observed_passes +=1
                    observation_dict[n_observed_passes] = {}
                    observation_dict[n_observed_passes]['t_start'] = t
                    observation_dict[n_observed_passes]['ii_start'] = ii
                    observation_dict[n_observed_passes]['ii_end'] = ii
                    observation_dict[n_observed_passes]['t_end'] = t
                    observation_dict[n_observed_passes]['length'] = t - observation_dict[n_observed_passes]['t_start']
                else: # pass continues
                    observation_dict[n_observed_passes]['t_end'] = t
                    observation_dict[n_observed_passes]['ii_end'] = ii
                    observation_dict[n_observed_passes]['length'] = t - observation_dict[n_observed_passes]['t_start']
                ii_pass_prev = ii_pass  # Update prev pass index
    return observation_dict 
def calculate_gs_visibility(gs_dict, 
                        simulated_ground_track, 
                        rho,
                        sin_rho,
                        eps_min,
                        eta_lct,
                        phi_lct,
                        dt_used,
                        R_E = 6378e3,
                        gs_used = None, 
                        check_gs_visibility = 1,
                        n_digits = 2):
    """Function to check if the ground station is visible 
    from the provided ground track points and spacecraft's altitude, 
    terminal's FOV limitations and GS elevation

    Args:
        gs_dict (dict): dictionary of ground station data (coordinates, names)
        simulated_ground_track (array): array output by the simulate_ground_track function
        rho (float): max angular radius of observable earth
        sin_rho (float): sin of rho
        eps_min (float): minimum GS elevation threshold [deg]
        eta_lct (list): Terminal's FOV elevation limits (45 to 70) [deg]
        phi_lct (list): Terminal's FOV azimuth limits (-45 to 45) [deg]
        R_E (float, optional): Earth radius [m]. Defaults to 6378e3.
        gs_used (list/int, optional): Indexes from gs_dict to limit considered ground stations. Defaults to None.
        check_gs_visibility (bool, optional): Whether to check if GS is visible before computing other anglse. Defaults to 1.
        n_digits (int, optional): Decimal places for result outputs. Defaults to 2.

    Returns:
        gs_dict, output_dict_full, output_dict_raw
    """    
    if gs_used == None: # Go over all ground stations
        gs_used = gs_dict.keys()
        print(f'Chosen every ground station - {len(gs_used)} total.')
    elif type(gs_used) == int:
        print(f'Chosen ground station:\n{gs_dict[gs_used]}')
        gs_used = [gs_used] # put into list to loop over
    
    # Unpack simulated_ground_track array
    t_vec, long_gt, lat_gt = simulated_ground_track[0], simulated_ground_track[1][:,0], simulated_ground_track[1][:,1]
    heading_angle_gt = simulated_ground_track[-1]

    output_dict_full = {} # Data for each time step and GS
    # Iterate over time vector
    for ii, t in enumerate(t_vec):
        output_dict_full[t] = {}
        
        n_observable_gs = 0 # obsevable groudn station at each time counter (considering all limitations)
        n_visible_gs = 0 # visible GS at each time (within 0-90 deg elevation and 0:360 azimuth of GS)
        n_fov_gs = 0 # visible GS from the SC-centered terminal's limited FOV
        
        lat_ssp = lat_gt[ii]
        long_ssp = long_gt[ii]
        heading_ssp = heading_angle_gt[ii][0]
        
        ## Store ground track coordinates and heading angle
        output_dict_full[t]['coord_gt'] = [long_ssp, lat_ssp]
        output_dict_full[t]['heading'] = heading_ssp
        output_dict_full[t]['gs'] = {} # Storing GS visibility data
        
        for jj, ind_gs in enumerate(gs_used): # Loop over every GS and check for visibility conditions
            gs_coord = gs_dict[ind_gs]['long/lat']
            output_dict_full[t]['gs'][jj] = {}
            long_p = gs_coord[0]
            lat_p = gs_coord[1]
            output_dict_full[t]['gs'][jj]['coord_t'] = [long_p, lat_p]
            output_dict_full[t]['gs'][jj]['coord_gt'] = np.round(output_dict_full[t]['coord_gt'], n_digits)
            # Check for visibility
            is_gs_visible = 0
            is_above_el_limit = 0
            
            # Calculate earth angle and azimuth angles
            # from satellite to ground station (Orbit Design and Constellation management, ch 9.1)
            lam, phi_e, phi_h = calc_sc_nadir_coord([long_ssp, lat_ssp], gs_coord, heading_ssp) 
            # phi_h - azimuth wrt heading
            # phi_e - az wrt north
            phi_h = np.round(phi_h,n_digits) # round to 2 digits
            
            if check_gs_visibility:
                if lam < np.pi/2 - rho: # checks if earth angle is above maximum earth angle
                    is_gs_visible = 1
                    n_visible_gs += 1
                else:
                    is_gs_visible = 0
            else:
                is_gs_visible = 1 # Set to visible if not checking
            output_dict_full[t]['gs'][jj]['is_visible'] = bool(is_gs_visible)
                
            if is_gs_visible:
                # Check if GS is in FOV
                eta, eps = np.rad2deg(calc_area_access_el(sin_rho, lam)) # sc-nadir elevation and gs-elevation [deg]
                # Check if elevation is above the minimum threshold
                is_above_el_limit = 0
                if eps > eps_min:
                    is_above_el_limit = 1
                output_dict_full[t]['gs'][jj]['is_above_el'] = bool(is_above_el_limit)
                # Calculate slant range
                D = R_E * np.sin(lam) / np.sin(np.deg2rad(eta)) # slant range [m]
                
                is_in_fov = False
                if eta < np.max(eta_lct) and eta > np.min(eta_lct):
                    # within Terminal El limits. Convert Az to + -180:180 deg
                    if phi_h > 180:
                        phi_h = phi_h - 360
                    if phi_h > np.min(phi_lct) and phi_h < np.max(phi_lct):
                        # within terminal azimuth limits
                        is_in_fov = True
                        n_fov_gs += 1
                output_dict_full[t]['gs'][jj]['is_in_fov'] = is_in_fov
                if is_above_el_limit and is_in_fov: # Add to observable GS counter
                    n_observable_gs += 1
                    is_observable = True
                else:
                    is_observable = False
                output_dict_full[t]['gs'][jj]['is_observable'] = is_observable
                phi_h = np.round(phi_h, n_digits)
                eta = np.round(eta, n_digits)
                eps = np.round(eps, n_digits)
                D = np.round(D, n_digits)
            else:
                eta, eps, D = None, None, None
                output_dict_full[t]['gs'][jj]['is_in_fov'] = bool(0)
                output_dict_full[t]['gs'][jj]['is_above_el'] = bool(0)
                output_dict_full[t]['gs'][jj]['is_observable'] = bool(0)
            output_dict_full[t]['gs'][jj]['earth_angle'] = np.round(np.rad2deg(lam),n_digits)
            output_dict_full[t]['gs'][jj]['sc_azimuth'] = phi_h
            output_dict_full[t]['gs'][jj]['sc_elevation'] = eta
            output_dict_full[t]['gs'][jj]['gs_elevation'] = eps
            output_dict_full[t]['gs'][jj]['slant_range'] = D
        output_dict_full[t]['n_observable_gs'] = n_observable_gs
        output_dict_full[t]['n_visible_gs'] = n_visible_gs
        output_dict_full[t]['n_fov_gs'] = n_fov_gs
    
    # Get time vector of when a GS is observable
    t_vec_obs = []
    for t in output_dict_full.keys():
        n_gs = output_dict_full[t]['n_observable_gs']
        n_vis = output_dict_full[t]['n_visible_gs']
        n_fov = output_dict_full[t]['n_fov_gs']
        # if n_gs > 0 or n_vis>0 or n_fov>0:
        if n_gs > 0:
            # print(f't={t} observ={n_gs}, vis = {n_vis}, fov = {n_fov}')
            t_vec_obs.append(t)
            
    # Process pass times and visibility occurences for the analysed GS
    
    for ii, ind_gs in enumerate(gs_used): # Loop over every GS and check for visibility conditions
        gs = gs_dict[ind_gs]['long/lat']
        gs_dict[ind_gs]['passes'] = {} # Add another key to fill pass data
        output_dict_raw = {} # dict to output raw results
        
        n_total_passes = 0
        jj_pass_prev = 0
        total_pass_time = 0
        for jj, t in enumerate(output_dict_full.keys()):
            output_dict_current = output_dict_full[t]['gs'][ii]
            output_dict_raw[jj] = {}
            output_dict_raw[jj] = output_dict_current
            output_dict_raw[jj]['t'] = t
            if output_dict_current['is_observable']:
                gs_pass = True
            else:
                gs_pass = False
            
            if gs_pass: # check if currently passing GS
                jj_pass = jj
            
                if jj_pass == jj_pass_prev +1: # Check if pass is continual or new
                    pass_continue = True
                else:
                    pass_continue = False
                
                if not pass_continue: # new pass
                        # Check what the previous' pass max GS elevation was
                    if n_total_passes > 0:
                        ii_pass_0 = gs_dict[ind_gs]['passes'][n_total_passes]['ii_start']
                        ii_pass_end = gs_dict[ind_gs]['passes'][n_total_passes]['ii_end']
                        el_list = []
                        for ii_pass in np.arange(ii_pass_0, ii_pass_end+1, 1):
                            el_list.append(output_dict_raw[ii_pass]['gs_elevation'])
                    
                        el_max = np.max(el_list)
                        gs_dict[ind_gs]['passes'][n_total_passes]['gs_elevation_max'] = el_max
                    n_total_passes += 1 # Register next pass
                    gs_dict[ind_gs]['passes'][n_total_passes] = {}
                    gs_dict[ind_gs]['passes'][n_total_passes]['t_start'] = t
                    gs_dict[ind_gs]['passes'][n_total_passes]['t_end'] = t
                    gs_dict[ind_gs]['passes'][n_total_passes]['ii_start'] = jj
                    gs_dict[ind_gs]['passes'][n_total_passes]['ii_end'] = jj
                    gs_dict[ind_gs]['passes'][n_total_passes]['length'] = t - gs_dict[ind_gs]['passes'][n_total_passes]['t_start']
                    
                else:
                    ## DOESNT FUNCTION YET, EXCLUDE (visible time calculation)
                    # for backstep in range(500):
                    #     if output_dict_full[t-backstep*10]['gs'][0]['is_visible']:
                    #         continue
                    #     else:
                    #         t_start_vis = t-(backstep+1)*10
                    #         break
                    # for backstep in range(500):
                    #     try:
                    #         if output_dict_full[t+backstep*10]['gs'][0]['is_visible']:
                    #             continue
                    #         else:
                    #             t_end_vis = t+(backstep-1)*10
                    #             break
                    #     except:
                    #         t_end_vis = t+(backstep-1)*10
                    # gs_dict[ind_gs]['passes'][n_total_passes]['length_vis'] = t_end_vis - t_start_vis
                    gs_dict[ind_gs]['passes'][n_total_passes]['ii_end'] = jj
                    gs_dict[ind_gs]['passes'][n_total_passes]['t_end'] = t
                    gs_dict[ind_gs]['passes'][n_total_passes]['length'] = t - gs_dict[ind_gs]['passes'][n_total_passes]['t_start']
                    total_pass_time += gs_dict[ind_gs]['passes'][n_total_passes]['length']
                jj_pass_prev = jj_pass  # Update prev pass index
                gs_dict[ind_gs]['total_passes'] = n_total_passes
                gs_dict[ind_gs]['total_pass_time'] = total_pass_time
                # gs_dict[ind_gs]['effective_pass_time'] = total_pass_time * gs_dict[ind_gs]['mean_annual_availability']
                ii_pass_0 = gs_dict[ind_gs]['passes'][n_total_passes]['ii_start']
                ii_pass_end = gs_dict[ind_gs]['passes'][n_total_passes]['ii_end']
                el_list = []
                for ii_pass in np.arange(ii_pass_0, ii_pass_end+1, 1):
                    el_list.append(output_dict_raw[ii_pass]['gs_elevation'])
            
                el_max = np.max(el_list)
                gs_dict[ind_gs]['passes'][n_total_passes]['gs_elevation_max'] = el_max
    
    return gs_dict, output_dict_full,  output_dict_raw
