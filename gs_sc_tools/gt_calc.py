import numpy as np
deg2rad = np.pi/180 # conversion factor from degrees to radians
rad2deg = 180/np.pi 
## Globals
R_E = 6378.136e3 # Equatorial radius of Earth [m]
D = 86164.1004 # Length of siderial day, s

# %% Functions
def calc_heading_fromgt(t_vec, long, lat):
    # calculate heading angle usign spherical geometry
    # input long, lat in [deg]
    # return heading angles in [rad]
    # uses theory in Orbit Design and Constellation Management book (2001) Appendix A7.3
    heading_angles = np.zeros((len(t_vec)-1, 2))
    
    # FROM APPENDIX A7-3 of ODCM book - side ANGLE side problem
    for ii, t in enumerate(heading_angles):
        long_0 = long[ii]
        long_1 = long[ii+1]
        lat_0 = lat[ii]
        lat_1 = lat[ii+1]


        b = colat(lat_1)
        a = colat(lat_0)
        C = long_1 - long_0
        a, b, C = np.deg2rad(a), np.deg2rad(b), np.deg2rad(C)
        c = acos2(np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b) * np.cos(C), fct_hem(C))
        b1 = -acos2((np.cos(b) - np.cos(a) * np.cos(c)) / (np.sin(a) * np.sin(c)), fct_hem(b))
        # heading_angles[ii,1] = mod360(np.rad2deg(b1) + 180, rad = 0)
        heading_angles[ii,1] = 2*np.pi + b1
        heading_angles[ii,0] = t_vec[ii]
    return heading_angles

def fct_hem(ang, rad = 1):
	# the hemisphere function. Outputs -1 (ang in III or IV) or 1 (ang in I or II)
	if not rad: # check if angle is in radian
		ang = ang * deg2rad
	if ang<0 or ang>2*np.pi:
		ang = mod360(ang)
	if ang<np.pi:
		return 1
	else:
		return -1
def mod360(ang, rad = 1):
	# function to make wrap an angle 0 and 2 pi
    # returns in radians
	if not rad: # check if angle is in radian
		ang = ang * deg2rad
	if ang<0:
		return ang + 2 * np.pi
	elif ang>2*np.pi:
		return ang%(2*np.pi)
	else:
		return ang
	
def mod360_deg(ang):
    if ang < 0:
        return ang + 360
    elif ang >= 360:
        return ang%(360)
    else:
        return ang
    
def acos2(val, hem):
	if val>1:
		val = 1
	elif val<-1:
		val = -1
	return np.arccos(val) * hem
	
def colat(el, rad = 0):
	if not rad:
		return 90 - el
	else:
		return np.pi/2 - el

def SSA(s1, s2, a):
	'''
	Parameters
	----------
	s1 : f
		first side (colat of S). [rad]
	s2 : f
		second side (angular dist between S and P) [rad]
	a : f
		angle between s1 and s2 [rad].

	Returns
	-------
	cos_s3 : f
		cosine of the last side.
		np.cos(s1) * np.cos(s2) + np.sin(s1) * np.sin(s2) * np.cos(a)
	'''
	cos_s3 = np.cos(s1) * np.cos(s2) + np.sin(s1) * np.sin(s2) * np.cos(a)
	cos_s3 = round(cos_s3, 4)
	return cos_s3
def SSS(s1, s2, s3):
	'''
	Parameters
	----------
	s1 : colat2
		side 1 [rad].
	s2 : colat1
		side 2 [rad].
	s3 : rho2
		side 3 [rad]. (first one in frac)

	Returns
	-------
	cos_a : 
		cosine of the angle between s1 and s2
		(np.cos(s3) - np.cos(s2) * np.cos(s1)) / (np.sin(s2) * np.sin(s1))
	'''
	cos_a = (np.cos(s3) - np.cos(s2) * np.cos(s1)) / (np.sin(s2) * np.sin(s1))
	cos_a = round(cos_a, 4)
	return cos_a

def SSS_da(rho2_f, rho1_f, lat1_f):
    
	'''
	# Function to calculate cos_da in the dual-axis problem

	Parameters
	----------
	rho2_f : P-S
		arc length.
	rho1_f : lat of S
		.
	lat1_f : TYPE
		az of P.

	Returns
	-------
	cos of da. (change in azimuth of P about C)
	(np.cos(rho2_f) - np.cos(rho1_f) * np.sin(lat1_f)) / (np.sin(rho1_f) * np.cos(lat1_f))
	'''
	cosda_f = (np.cos(rho2_f) - np.cos(rho1_f) * np.sin(lat1_f)) / (np.sin(rho1_f) * np.cos(lat1_f))
	return cosda_f

def calc_swath_points(gt_coord, h, psi, eta, print_cond = 0):
            
    """Function to transform ground track coordinates into the left and right ground swath point coordinates

    Args:
        gt_coord (list): lat and long of gt point [deg]
        h (float): altitude [m]
        psi (float): heading angle rel. to North [deg]
        eta (float): nadir angle from SC vector to target point at the SC point
    Returns:
        P_l, P_r, swath width [deg, deg, m]
    """            
    # Test with Scatterometer specs
    # h = 720e3
    ## Finding P coordinates based on SC frame
    psi = 11/6 * np.pi # Fake input heading angle
    eta = np.deg2rad(eta) # Deg, viewing angle
    sc_colat = np.deg2rad(90 - gt_coord[0])
    
    sin_rho = R_E / (R_E + h)
    cos_eps = np.sin(eta)/sin_rho
    eps = np.arccos(cos_eps)
    lam = np.pi/2 - eps - eta 

    # phi_e_l = psi - 2*np.pi + np.pi/2 # 90 deg azimuth from heading angle
    phi_e_l = psi - 2*np.pi # 0 deg azimuth from heading angle
    p_colat_l = np.arccos(np.cos(lam)*np.cos(sc_colat) + np.sin(lam)*np.sin(sc_colat)*np.cos(phi_e_l))
    # print(np.rad2deg(p_colat_l))
    delta_long_l = np.arccos((np.cos(lam)-np.cos(p_colat_l)*np.cos(sc_colat))/(fct_hem(sc_colat)*np.sin(sc_colat)*np.sin(p_colat_l))) - np.pi/2 * (fct_hem(sc_colat)-1)
    long_p_l = gt_coord[1] - np.rad2deg(delta_long_l) # longitude of target [deg]
    p_geo_l = [90 - np.rad2deg(p_colat_l), np.rad2deg(mod360(long_p_l , rad = 0))]
    ## Right side
    phi_e_r = psi - 2*np.pi + np.pi*3/2 # 270 deg azimuth from heading angle
    p_colat_r = np.arccos(np.cos(lam)*np.cos(sc_colat) + np.sin(lam)*np.sin(sc_colat)*np.cos(phi_e_r))
    # print(np.rad2deg(p_colat_r))
    delta_long_r = np.arccos((np.cos(lam)-np.cos(p_colat_r)*np.cos(sc_colat))/(fct_hem(sc_colat)*np.sin(sc_colat)*np.sin(p_colat_r))) - np.pi/2 * (fct_hem(sc_colat)-1)
    long_p_r = gt_coord[1] + np.rad2deg(delta_long_r) # longitude of target [deg]
    p_geo_r = [90 - np.rad2deg(p_colat_r), np.rad2deg(mod360(long_p_r, rad = 0))]
    swath_width = lam*R_E*2
    if print_cond:
        print(f'''
            Calculated 2d angles
            heading angle : {np.rad2deg(psi):.3f} deg
            nadir angle : {np.rad2deg(eta):.3f} deg
            elevation angle : {np.rad2deg(eps):.3f} deg
            Earth angle : {np.rad2deg(lam):.3f} deg
            Swath width : {swath_width/1e3} km
            ''')
        print(f'SC geo coordinates: {gt_coord} \nLeft swath: {p_geo_l}\nRight swath: {p_geo_r}\nphi_e_l:{np.rad2deg(phi_e_l)} phi_e_r:{np.rad2deg(phi_e_r)}')
    return p_geo_l, p_geo_r, swath_width
def calc_gt(a_0,
            d_0,
            phi_s_0,
            n,
            i,
            t,
            rho_2 = 90,
            w_1 = -360 / 86164.1004):
    """Function to calculate the ground track coordinates of the sub-satellite point S
    using a dual-axis spiral, including the J2 perturbation for a (currently) circular orbit
    

    Args:
        a_0 (float): initial longitude of orbital pole [deg]
        d_0 (float): initial latitude of sub-sat point [deg]
        phi_s_0 (float): az from ascending node to S [deg] ? 
        n (float): angular rate of sat [rad/s]
        i (inclination)
        t (float): time or time-step [s]
        w_1 (float): angular velocity of O [deg/s] ToDo Remove?
        rho_2 (float): angular distance from O to S (90 deg) ToDo Remove?
        use_j2 (bool): conditional to use J2 terms effect on the rotation
    Returns:
        a_o, a_s, d_s, phi_s, n_g, psi
    """    
        
    #n = calculate orbital rate
    a_o = a_0 + w_1 * t # deg, longitude of orbit pole
    phi_s = phi_s_0 + n*t # deg, az of S about Orbital pole rel to North
    # phi_s = 270 + phi_s_0 + n*t # deg, az of S about O rel to North
    
    # Convert all parameters to radians
    orbital_params = [w_1, i, n, d_0, rho_2, phi_s, w_1, a_o]
    orbital_params_rad = np.deg2rad(orbital_params) 
    w_1, i, n, d_0, rho_2, phi_s, w_1, a_o = orbital_params_rad
    a_o = mod360(a_o)
    delta_a = acos2(-np.tan(d_0)/np.tan(i), -fct_hem(phi_s)) # Change in longitude of S
    # Euler axis parameters
    d_e = np.arctan((w_1 + n*np.cos(i))/(n*np.sin(i))) # rad
    rho_e = np.arccos(np.sin(d_e)*np.sin(d_0) + np.cos(d_e)*np.cos(d_0)*np.cos(delta_a))
    w_e = n * np.sin(i) / np.cos(d_e) # rot rate about E [rad/s]
    d_psi = acos2((np.sin(d_e)-np.cos(rho_e)*np.sin(d_0))/(np.sin(rho_e)*np.cos(d_0)), fct_hem(delta_a))
    # Results
    a_s = mod360(a_o + delta_a) # rad, longitude of S
    d_s = np.pi/2 - np.arccos(np.sin(i)*np.cos(phi_s)) # rad, latitude of S
    n_g = w_e * np.sin(rho_e) # ground track velocity of S
    psi = mod360(d_psi - np.pi/2)
    results = [a_o, a_s, d_s, phi_s, n_g, psi]
    
    results_deg = np.rad2deg(results)
    if results_deg[1]<-1:
        print('HAPPENING STOP STOP STOP')
        asd = 123
    return results, results_deg
def analyse_earth_coverage(req_coord, initial_cond, T_all = 6200, d_eta = 1, dt = 30, d_grid = 1, w_1 = None, gt_daily = None):
    from old_simulate_ground_track import convert_gt_to_swath, simulate_ground_track
    import numpy as np
    """Function to analyse the daily ground coverage of the

    Args:
        req_coord (list): lat_min, lat_max, long_min, long_max [deg]
        T_all (int, optional): Simulation time. Defaults to 6200.
        d_eta (int, optional): Step in nadir angle. Defaults to 1.

    Returns:
        cov_grid - matrix of covered and uncovered lat/long grid
        cov_perc - percentage of grid covered
    """    
    n_digits_grid = -int(np.log10(d_grid))
    ## Create mesh-grid of global-coverage
    lat_arange = np.arange(-90, 90+d_grid, d_grid)
    long_arange  = np.arange(0, 359+d_grid, d_grid)
    ## Create bounds for required coverage
    lat_min, lat_max = req_coord[0], req_coord[1]
    long_min, long_max = req_coord[0+2], req_coord[1+2]
    lat_arange_req_cov = np.arange(req_coord[0], req_coord[1]+d_grid, d_grid)
    long_arange_req_cov  = np.arange(req_coord[0+2], req_coord[1+2]+d_grid, d_grid)
    num_lat = np.size(lat_arange_req_cov)
    num_long = np.size(long_arange_req_cov)
    grid_blank = np.zeros((num_lat, num_long))
    grid_blank = {}
    grid_blank['grid'] = {}
    for lat in lat_arange:
        grid_blank['grid'][lat] = {}
        for long in long_arange:
            grid_blank['grid'][lat][long] = 0
    grid_blank['total'] = num_lat*num_long
    grid_blank['covered'] = 0
    ## Calculate ground-track for a day
    if gt_daily == None: # GT can be re-used for single satellite coverage analysis
        gt_daily = simulate_ground_track(initial_cond, T_all, dt = 30)
    if w_1 != None: # ToDo can remove this nad w_1 as input
        gt_daily = simulate_ground_track(initial_cond, T_all, dt = 30)
    ssp_daily = gt_daily[0]
    psi_daily = gt_daily[1]
    ## Calculate swath coordinates
    swath_arr_lat_l, swath_arr_lat_r, swath_arr_long_l, swath_arr_long_r = convert_gt_to_swath(ssp_daily, psi_daily, initial_cond, swath_dec = n_digits_grid)
    ## Loop through lat/long grid and check off covered 
    for ii in range(np.shape(swath_arr_lat_l)[0]): # for each row
        lat_row_l = swath_arr_lat_l[ii,:]
        long_row_l = swath_arr_long_l[ii,:]
        lat_row_r = swath_arr_lat_l[ii,:]
        long_row_r = swath_arr_long_l[ii,:]
        for jj in range(np.shape(swath_arr_long_l)[1]): # for each column
            ## Take covered lat and long ranges
            lat_l = lat_row_l[jj]
            long_l = long_row_l[jj]
            lat_r = lat_row_r[jj]
            long_r = long_row_r[jj]
            # Fil in
            grid_blank['grid'][lat_l][long_l] = 1
            grid_blank['grid'][lat_r][long_r] = 1
    
    # Check number of filled grids
    tot = 0
    for lat in grid_blank['grid'].keys():
        if lat >= lat_min and lat <= lat_max:
            for long in grid_blank['grid'][lat].keys():
                if long >= long_min and long <= long_max:
                    if grid_blank['grid'][lat][long] == 1:
                        tot += 1
    grid_blank['covered'] = tot
    cov_grid = grid_blank
    cov_perc = grid_blank['covered']/ grid_blank['total']*100
    return cov_grid, cov_perc
def dict_2_array(cov_grid):
    import numpy as np
    ## Input cov_grid['grid']
    # Output X - longitude range
    # Y - latitude range
    # Z - array of cov/not cov (rows - lat, cols - long)
    num_lat = len(cov_grid.keys())
    lat = list(cov_grid.keys())[-1]
    num_long  = len(cov_grid[lat].keys())
    vals = np.zeros((num_lat, num_long))
    for ii, lat in enumerate(cov_grid.keys()):
        row = cov_grid[lat]
        for jj, long in enumerate(row.keys()):
            vals[ii,jj] = row[long]
    x = list(cov_grid[lat])
    y = list(cov_grid.keys())
    return x, y, vals
def make_ground_track_plot(long, lat, req_coord_all, plot_glossary = None):
    ## Function to plot the ground track. 
    # Input ground track lists: long, lat and GS coordinates [long, lat] 
    # all in [deg]
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.set_xlim([0, 359])
    ax.set_ylim([-90, 90])
    x_ticks = np.arange(0, 390, 30)
    y_ticks = np.arange(-90, 100, 10)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.scatter(long, lat, c = 'b', s = 1, label = 'Ground track')
    for req_coord in req_coord_all:
        plt.scatter(req_coord[0], req_coord[1], c = 'r', s = 10)
    plt.scatter(req_coord[0], req_coord[1], c = 'r', s = 10, label = 'GS locations')
    if plot_glossary != None:
        xlabel = plot_glossary['xlabel']
        ylabel = plot_glossary['ylabel']
        title =  plot_glossary['title']
        
        ax.set_xlabel(xlabel, fontsize = 12)
        ax.set_ylabel(ylabel, fontsize = 12)
        ax.grid()
        fig.suptitle(title, fontsize = 16)
    ax.legend()
    return fig, ax
def make_lat_long_plot(x, y, vals, req_coord, plot_glossary = None):
    # xy - lists of lat and longitude covered by satellite
    # req coord - GS location [lat, long] deg
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, vals)
    ax.set_xlim([0, 359])
    ax.set_ylim([-90, 90])
    x_ticks = np.arange(0, 390, 30)
    y_ticks = np.arange(-90, 100, 10)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    if plot_glossary != None:
        xlabel = plot_glossary['xlabel']
        ylabel = plot_glossary['ylabel']
        title =  plot_glossary['title']
        
        ax.set_xlabel(xlabel, fontsize = 12)
        ax.set_ylabel(ylabel, fontsize = 12)
        fig.suptitle(title, fontsize = 16)
    plt.scatter([req_coord[0], req_coord[1]], c = 'r', label = 'GS location')
    ax.legend()
    return fig, ax
    
if __name__=='__main__':
    if 1:
    
        long =[-1.47, -0.75]
        lat = [43.29, 43.62]
        t_vec = [0,10]
        
        heading = calc_heading_fromgt(t_vec, long, lat)
    if 0:
        R_E = 6378.136e3 # Equatorial radius of Earth [m]
        GM_E = 3.98600441e14 # Earth's gravitational parameter [m^3/s^2]
        ## Orbital params
        i = 87
        h = 900e3
        a = R_E+h
        n = np.rad2deg(np.sqrt(GM_E/a**3))
        T_orbit = int(360/n)*2
        ## Fill in initial conditions to calculate GT
        initial_conditions_scat = {}
        initial_conditions_scat['a'] = a
        initial_conditions_scat['h'] = h
        initial_conditions_scat['az_0'] = 1
        initial_conditions_scat['lat_0'] = -i
        initial_conditions_scat['phi_s'] = 180
        initial_conditions_scat['i'] = i
        initial_conditions_scat['n'] = n
        initial_conditions_scat['eta'] = 42
        
        req_coord = [-45, 45, 0, 359] # Lat min, Lat max, Long min, Long max
        
        a = analyse_earth_coverage(req_coord, initial_conditions_scat, T_all = 6200*13, d_eta = 1, dt = 60, d_grid = 1)
        b = dict_2_array(a[0]['grid'])
        make_lat_long_plot(b[0], b[1], b[2], req_coord)
        
    if False:
        from SphericalTriangle import *
        import matplotlib.pyplot as plt
        dt = 60
        GM_E = 3.98600441e14 # Earth's gravitational parameter [m^3/s^2]
        R_E = 6378.136e3 # Equatorial radius of Earth [m]
        h = 900e3 #m
        a = R_E+h
        n = np.rad2deg(np.sqrt(GM_E/a**3))
        phi_s_0 = 180 # Az from ascending node to S
        i = 70 # deg
        dt = 1 # 
        s_0 = [1e-5, -i] #Initial sub-sat point location [long, lat]
        a_o0 = 1e-5
        T_all = int(6200/10) # Number of seconds
        p_arr, s_arr = np.zeros((T_all, 2)), np.zeros((T_all, 2)) # placeholders to store S and P coordinates
        s = s_0 # sub-satellite point coordinates
        phi_s = phi_s_0
        # Including J2 term
        if True:
            ## Coordinate transformatio
            sc_geo = [10, 39] # Lat, long of SSP
            p_geo = [2, 20] # Lat, long of target point
            delta_long = np.deg2rad(sc_geo[1] - p_geo[1]) # rad, diff in longitude
            # Colatitutes in rad
            sc_colat = np.deg2rad(90 - sc_geo[0])
            p_colat = np.deg2rad(90 - p_geo[0])
            lam = np.arccos(np.cos(p_colat)*np.cos(sc_colat) + np.sin(p_colat)*np.sin(sc_colat)*delta_long) # Earth angle from Earth center to target
            phi_e = acos2((np.cos(p_colat) - np.cos(sc_colat)*np.cos(lam))/ (np.sin(sc_colat)*np.sin(lam)), fct_hem(delta_long)) # Az to target from SSP
            sin_rho = R_E / (R_E + h)
            lam_0 = np.arccos(sin_rho) # Central Earth angle
            eta = np.arctan(sin_rho * np.sin(lam) / (1 - sin_rho*np.cos(lam))) # Nadir angle from SC to target
            cos_epsilon = np.sin(eta)/sin_rho
            D = R_E * (sin_rho/np.sin(eta))
            
            
        if False:
            use_j2 = True
            if use_j2:
                w_1 = -360 / 86164.1004 - 2.396288e9*(a*1e-3)**(-3.5) * np.cos(np.deg2rad(i))
            else:
                w_1 = -360 / 86164.1004
            for ii in range(T_all):
                res_deg = calc_gt(a_o0, s[1], phi_s, n, i, dt)[1]
                s = [res_deg[1], res_deg[2]]
                a_o0 = res_deg[0]
                phi_s = res_deg[3]
                s_arr[ii,:] = s
            plt.figure(4)
            plt.plot(s_arr[:,0], s_arr[:,1])
            makenice('Long-Lat plot', 'Longitude [deg]', 'Latitude [deg]', label = 0)
            # savefig_geom('az-el', 8)
            plt.show()