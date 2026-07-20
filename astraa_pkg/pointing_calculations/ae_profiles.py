import numpy as np
def get_ae_for_pmg(el_lims = [85, -55],
                   az_lims = [175, -175],
                   nr_el_steps = 5,
                   ang_rate = 0.4,
                   t_vec = None,
                   ):
    # Az/EL that follows the PMG pointing setup
    # fixed el angle and vary over entire azimuth range

    az_lim_0 = az_lims[1]
    e_0 =el_lims[0]
    e_1 =el_lims[1]
    el_values = np.linspace(e_0, e_1, nr_el_steps)
    el_values = np.append(el_values, el_values[-1])
    if t_vec is None:
        dt = 1
        t_vec = np.arange(0,5*3600,dt)
    else:
        dt = t_vec[1] - t_vec[0]

    ae_profile = np.zeros((t_vec.shape[0],2))

    rotate_az = 1
    rotate_el = 0

    ae_0 = [az_lim_0, e_0]
    az_lim_curr = az_lim_0

    el_ii = 1
    el_lim_curr = el_values[el_ii]
    ae_profile[0,:] = ae_0
    az_rotate_dir = np.sign(az_lim_curr * (-1))
    el_rotate_dir = -1

    a_0 = ae_0[0]
    e_0 = ae_0[1]

    # Loop over required az/el changes
    for ii, t in enumerate(t_vec):
        a_1 = a_0
        e_1 = e_0
        if rotate_az:
            a_1 = a_1 + az_rotate_dir * ang_rate * dt

            if np.abs(a_1) >= np.abs(az_lim_curr):
                rotate_az = 0
                # a_1 = az_lim_curr
                rotate_el = 1
                az_lim_curr = az_lim_curr * (-1) # TODO will need to change if diff
                # limits are used
                az_rotate_dir = - az_rotate_dir
            # go from az lim to az lim
        if rotate_el:
            e_1 = e_0 + el_rotate_dir * ang_rate * dt

            if e_1 <= el_lim_curr:
                rotate_az = 1
                rotate_el = 0
                el_ii +=1
                # e_1 = el_lim_curr
                try:
                    el_lim_curr = el_values[el_ii]
                except:
                    ae_profile = ae_profile[:ii]
                    break

        # store
        ae_profile[ii,:] = [a_1, e_1]
        a_0 = a_1
        e_0 = e_1

    return ae_profile, ii
