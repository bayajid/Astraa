% Script to compute sun angle in Azimuth/elevation for given
% host, target states and host quaternions. First from ECI to BF (spacecraft attitude)
% Second from BF to GF (mounting offset)
% Author : Kipras
make_targetsat_pointing_test = 0;
make_sun_pointing_test = 1; % KP 23-05-05 Mounting offset added as 2nd qutaernion.
full_test = 1;
partial_rsun_eci_test = 1;
% pointing test cases not set up yet (TODO)
make_pointing_tests = 0;
test_case = 1;
if make_pointing_tests
    if test_case == 1
        'Co-planar link'    
        states_host = [-2.35e+06  1.22e+05  6.99e+06 -6.97e+03 -4.08e+01 -2.34e+03];
        states_target = [-5.15e+06  9.21e+04  5.28e+06 -5.27e+03 -8.96e+01 -5.13e+03];
        t_gps_in = 1325635148.816;
        attitude_host = [ 0.    0.16 -0.01 -0.99  0.    0.    0.    0.  ];
        % Az, El [rad], expected to target
        ae_expected_2target = [ 0.   -0.22];
    elseif  test_case == 2
        % Cross-plane link
        'Cross-plane link'
        states_host = [-6.93e+06  1.52e+06  2.01e+06 -2.52e+03 -4.15e+03 -5.52e+03];
        states_target = [-6.76e+06 -2.84e+06  8.29e+05  1.11e+03 -4.36e+03 -5.82e+03];
        t_gps_in = 1325635148.816;
        attitude_host = [ 0.18  0.54 -0.26 -0.78  0.    0.    0.    0.  ];
        
        ae_expected_2sun= [ 0.   0]; % Az, El [rad], expected to Sun (TODO)
        % Az, El [rad], expected to target
        ae_expected_2target = [ 0.7  -0.31];
    end
end
%% Outputs - host to target sat pointing
if make_targetsat_pointing_test 
    [az, el, slant] = ae_calculator(states_host, states_target, attitude_host, mounting_ofset);
    fprintf('Target Satellite direction. Calculated Az %f rad, el %f rad, \nslant  %.3e m \n', az, el, slant)
    fprintf('Target Satellite direction. EXPECTED Az %f rad, el %f rad \n', ae_expected_2target)
end

%% Outputs - host to SUN pointing

if make_sun_pointing_test 
    if full_test
        if test_case == 1
            t_gps_in = 1367320722; % Fitting May 5, 1:19:00 PM
            % Expected outputs [deg] Az wrt North, West positive. Elevation up pos.
            az_exp = 176.9916051;
            el_exp = 58.12955837;
            attitude_host = [0.333027958	0.347016514	-0.866678706	0.132438438];
            mounting_offset = [0 1 0 0];
            % ECI states of Mynaric HQ Gilching at given time
            states_host = [3095497.429	2938316.218	4739522.916	-214.2668797	224.9424613	0.487494905];
            
            pos_sun = transpose(where_sun(t_gps_in)); % Earth - Sun vector
        
            [az, el, slant] = ae_calculator(states_host, pos_sun, attitude_host, mounting_offset);
        elseif test_case == 2
            t_gps_in = 1367339982; % Fitting May 5, 1:19:00 PM
            % Expected outputs [deg] Az wrt North, West positive. Elevation up pos.
            az_exp = 84.76;
            el_exp = 17.41;
            attitude_host = [ 0.16817803  0.82520669 -0.43767561  0.31494463];
            mounting_offset = [0 1 0 0];
            % ECI states of Mynaric HQ Gilching at given time
            states_host = [-2.37627696e+06  3.52863131e+06  4.75194201e+06 -2.57313230e+02 -1.74066669e+02  5.82835117e-01];
            
            pos_sun = transpose(where_sun(t_gps_in)); % Earth - Sun vector
%             pos_sun =[1.07411937e+11 9.72016341e+10 4.21353881e+10];
            [az, el, slant] = ae_calculator(states_host, pos_sun, attitude_host, mounting_offset);
        end
        fprintf('SUN direction.\nCalculated Az %f deg, el %f deg\n', rad2deg(az), rad2deg(el))
        fprintf('EXPECTED Az %f deg, el %f deg %.3e m \n', az_exp, el_exp)
    elseif partial_rsun_eci_test
        % inputs
        t_gps = 1367564382;
        % Outputs
        r_sun_exp_maths = [1.02725291e+11 1.01496138e+11 4.39970357e+10];
        r_sun_exp_skyfield = [1.02714302e+11 1.01502519e+11 4.39994785e+10];
        error_expected = 84; % urad

        r_sun_matlab = where_sun(t_gps);

        pe_sunvec = calc_pe(r_sun_exp_skyfield, r_sun_matlab)

    end
end