function [az,el,slant] = ae_calculator(states_host,states_target,attitude_host, mounting_offset)
%AE_CALCULATOR Summary of this function goes here
%   Function to compute Azimuth and Elevation pointing angles
% for an input host state [position, velocity] ECI, m, m/s
% target state [position, velocity] ECI, m, m/s
% Quaternion, rate, scalar-first from ECI to terminal-frame
% return azimuth, elevation, slant range in rad, rad, m

% Unpack and shape inputs
r_host = states_host(1:3);
r_target = states_target(1:3);
quat_eci2bf = attitude_host(1:4);
quat_bf2gf = mounting_offset(1:4);

% ECI Line of sight [m]
los_eci = r_target - r_host;

% Body-frame Line of sight [m] using body-frame attitude 
los_bf = qrotate(los_eci, quat_eci2bf);

% Global-frame Line of sight [m] using mounting offset
los_gf = qrotate(los_bf, quat_bf2gf);

% Convert cartesian LOS to Azimuth/Elevation/Slant-range
[az, el] = convert_los_to_ae(los_gf);
slant = norm(los_gf);
    
end

