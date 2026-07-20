% Script to compute moon angle in Azimuth/elevation for given
% host, target states and host quaternions. First from ECI to BF (spacecraft attitude)
% Second from BF to GF (mounting offset)
% Author : Kipras

use_single_precision = 0;
if use_single_precision
    feature('SetPrecision', 24)
end
full_test = 0;
partial_rmoon_test = 1;
% pointing test cases not set up yet (TODO)
test_case = 1;
 
if partial_rmoon_test
    % inputs
    t_gps = 1362862782.0;
    % Outputs    
    r_moon_exp_skyfield = [-5.88731422e+07 -3.29028757e+08 -1.69397197e+08];
    error_expected = 1.7; % mrad
%     if use_single_precision
    r_moon_matlab_single = where_moon(t_gps, 1);
    
    r_moon_matlab_double = where_moon(t_gps, 0);
    
    % expected PE - 1.734 mrad
    pe_moonvec_dbl = calc_pe(r_moon_exp_skyfield, r_moon_matlab_double)
    pe_moonvec_sgl = calc_pe(r_moon_exp_skyfield, r_moon_matlab_single)

end