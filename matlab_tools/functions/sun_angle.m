% Author: BKhan
%% TEST example
clc,clear all;
global GPS_start_jd;
GPS_start_jd = 2444244.5;  % Julian day at GPS start time 1980-01-06T12:00:00
format long;

%time = ['1994-4-2T12:00:00']; 
%fprintf(time);
mag = 149597870.7;  % AU to KM
%time_gps =  827971182;%4.49323209e+08;
% time_gps = 1325030348.816; %For TUDAT data	
time_gps = 1277695847.816; % New TUDAT data
%sun_vec_vallado = mag*[0.9771945, 0.1924424, 0.0834308]; % Sun Vector in ECI_MOD
astropy = mag*[0.17455569255634798 -0.8879012354260074 -0.38490260752313427];%[0.9775306601615426, 0.19106889542206248, 0.08283562777630926];%
% TUDAT = mag*[0.17760976 -0.90291345 -0.39141037]; 	
TUDAT = [-2.75554193e10  1.37233200e11  5.94879708e+10];

[a] = sun_vector(time_gps);


fprintf('\nSUN_VECTOR_IN_ECI_J2000[km]:JD_GPS: %11.9f, %11.9f, %11.9f',a);
%fprintf('\nSUN_VECTOR_VALLADO_ECI_MOD[km]:     %11.9f, %11.9f, %11.9f',sun_vec_vallado);
fprintf('\nSUN_VECTOR_IN_ASTROPY_J2000[km]:    %11.9f, %11.9f, %11.9f',astropy);
fprintf('\nSUN_VECTOR_IN_TUDAT_J2000[km]:    %11.9f, %11.9f, %11.9f',TUDAT);
%theta = acos(dot(a,sun_vec_vallado)/(norm(a)*norm(sun_vec_vallado)));
theta_AP = acos(dot(a,astropy)/(norm(a)*norm(astropy)));
theta_TD = acos(dot(a,TUDAT)/(norm(a)*norm(TUDAT)));
%fprintf("\n\nDifference calc[ECI_MOD], vs Vallado: %f [µradians]",theta*1e6);
fprintf("\nDifference calc[ECI_J2000], vs Astropy: %f [µradians]",theta_AP*1e6);
fprintf("\nDifference calc[ECI_J2000], vs TUDAT: %f [µradians]",theta_TD*1e6);
% END TEST
%% ------- SUN ANGLE CALCULATOR-------%%
function [reci]= MOD2ECI(r, ttt)
[prec] = precess(ttt);
reci = prec * r';
end
function[prec] = precess(ttt)
factor = pi / (180.0*3600.0);

zeta  = (2306.2181*ttt + 0.30188*(ttt)^2 + 0.017998*(ttt)^3);
theta = (2004.3109*ttt - 0.42665*(ttt)^2 - 0.041833*(ttt)^3);
z     = (2306.2181*ttt + 1.09468*(ttt)^2 + 0.018203*(ttt)^3);

zeta = factor*zeta;
theta = factor*theta;
z = factor*z;

coszeta  = cos(zeta);
sinzeta  = sin(zeta);
costheta = cos(theta);
sintheta = sin(theta);
cosz     = cos(z);
sinz     = sin(z);
% ----------------- form matrix  mod to j2000 -----------------
prec(1,1) =  coszeta * costheta * cosz - sinzeta * sinz;
prec(1,2) =  coszeta * costheta * sinz + sinzeta * cosz;
prec(1,3) =  coszeta * sintheta;
prec(2,1) = -sinzeta * costheta * cosz - coszeta * sinz;
prec(2,2) = -sinzeta * costheta * sinz + coszeta * cosz;
prec(2,3) = -sinzeta * sintheta;
prec(3,1) = -sintheta * cosz;
prec(3,2) = -sintheta * sinz;
prec(3,3) =  costheta;
end

function [r_sun_eci_j2000] = sun_vector(GPS_TIME)
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
magr = 149597870.7*(1.000140612  - 0.016708617 *cos(mean_anomaly)-0.000139589 *cos( 2.0 *mean_anomaly ));    % in Km

% Sun vector in ECI_MOD
a(1) = magr*(cos((lambd)));
a(2) = magr*(cos((epsilon))* sin((lambd)));
a(3) = magr*(sin((epsilon))* sin((lambd)));

% Sun vector in ECI_J2000
r_sun_eci_j2000 = MOD2ECI(a, ttt);
end
