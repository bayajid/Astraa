function [az,el] = convert_los_to_ae(los)
%CONVERT_LOS_TO_AE Summary of this function goes here
%   Convert line of sight in cartesian components to pointing angles
% azimuth and elevation [rad]
% in the global terminal frame
% wraps azimuth to +/-pi

if los(1) > 0
    az = atan(los(2) / los(1));
else
    az = pi + atan(los(2) / los(1));
end
if az > pi
    az = az - 2 * pi;
end

el = asin(los(3) / norm(los));

end

