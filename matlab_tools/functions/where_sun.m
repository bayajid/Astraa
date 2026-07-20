function [r_sun_eci_j2000] = where_sun(GPS_TIME)
% Author: BKhan
% INPUT:
% GPS_TIME = GPS TIME in [sec]
%
% OUTPUT:
% a = 1x3 Position vector towards Sun in ECI(J2000) frame
%  -----------------------------%%
% CONSTANTS
global GPS_start_jd
GPS_start_jd = 2444244.5;   % Julian day at GPS start time 1980-01-06T12:00:00
light_time = 299792.458 ;   % Light travel time [km/s]
E2sun = 149597870.7;        % Earth Sun distance [km]

a = zeros(1,3);
time_tt_jd = ((GPS_TIME - (E2sun/light_time) + 51.84) /604800)*7 + GPS_start_jd;
tt = ((GPS_TIME + 51.84) /604800)*7 + GPS_start_jd; % tt - Terrestrial Time

% Julian centuries 
ttt = (tt - 2451545.0)/ 36525.0;

% Julian centuries
time = (time_tt_jd - 2451545.0)/ 36525.0;

% meanLonitudeDegrees
mean_long = rem((280.460 + 36000.771 * time),360.0);   % in degrees
% meanAnomalyRadians
mean_anomaly = rem(deg2rad(357.5291092 + 35999.05034* time), 2*pi);
if(mean_anomaly < 0.0)
    mean_anomaly = 2*pi + mean_anomaly;
end

% eclipticLongitudeRadians
lambd = mean_long + 1.914666471 * sin(mean_anomaly) + 0.019994643 * sin(2 * mean_anomaly); % in deg
lambd = deg2rad(rem(lambd, 360.0)); % rad
% eclipticObliquityRadians
epsilon = deg2rad(23.439291 - 0.0130042 * time);  % in radians
% MagnitudeSunVectorKM
magr = 149597870.7*(1.000140612  - 0.016708617 *cos(mean_anomaly)-0.000139589 *cos( 2.0 *mean_anomaly )) ;    % in m

% Sun vector in ECI_MOD
a(1) = magr*(cos((lambd)));
a(2) = magr*(cos((epsilon))* sin((lambd)));
a(3) = magr*(sin((epsilon))* sin((lambd)));

% Sun vector in ECI_J2000
r_sun_eci_j2000 = MOD2ECI(a, ttt);
end

