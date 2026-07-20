function [r_moon_eci_j2000] = where_moon(GPS_TIME, use_single)
    % Author: Kipras paliusis
    % INPUT:
    % GPS_TIME = GPS TIME in [sec]
    %
    % OUTPUT:
    % a = 1x3 Position vector towards Moon in ECI(J2000) frame
    %  -----------------------------%%
    % CONSTANTS
    
    GPS_start_jd = 2444244.5;   % Julian day at GPS start time 1980-01-06T12:00:00
    % seconds from Earth to Moon
    light_travel_time = 1.27; 
    R_E = 6378137; % [m] Earth equatorial radius
    a = zeros(1,3);
    % Time for apparent positions
    time_tt_jd = ((GPS_TIME - light_travel_time + 51.84) /604800)*7 + GPS_start_jd;
    % Time for true positions/rotations
    tt = ((GPS_TIME + 51.84) /604800)*7 + GPS_start_jd; % tt - Terrestrial Time
    
    % Julian centuries 
    ttt = (tt - 2451545.0)/ 36525.0;
    
    % Julian centuries    
    time = (time_tt_jd - 2451545.0)/ 36525.0;
    if use_single
        time = single(time);
        ttt = single(ttt);
    end
    % Moon longitude [deg] lambda_moon
    moon_longitude_deg = 218.32 + 481267.883 * time;
    moon_anomaly_deg = 1/3600*(485868.249036 + 1717915923.2178*time + 31.8792*time*time + .051635*time*time*time - 0.00024470*time*time*time*time);
    moon_anomaly_rad = deg2rad(moon_anomaly_deg);

    sun_anomal_deg = 357.5291092+35999.05034*time;
    sun_anomal_rad = deg2rad(sun_anomal_deg);
    
    mean_elong_sun_deg = 1/3600*(1072260.70369 + 1602961601.2090*time - 6.3706*time*time +0.006593*time*time*time -0.00003169*time*time*time*time);
    mean_elong_sun_rad = deg2rad(mean_elong_sun_deg);
    % mean argument of latitude of the moon [deg] u_Moon
    moon_mean_arg_latitude_deg = 1/3600*(335779.526232 + 1739527262.8478*time - 12.7512*time*time - 0.001037*time*time*time + 0.00000417*time*time*time*time);
    moon_mean_arg_latitude_rad = deg2rad(moon_mean_arg_latitude_deg);
    % eclipticLongitudeRadians
    moon_ecl_longitude_deg = moon_longitude_deg + 6.29 * sin(moon_anomaly_rad)  - 1.27 * sin(moon_anomaly_rad - 2*mean_elong_sun_rad) + 0.66 * sin(2*mean_elong_sun_rad) + 0.21 * sin(2*moon_anomaly_rad) - 0.19 * sin(sun_anomal_rad) - 0.11 * sin(2*moon_mean_arg_latitude_rad);
    moon_ecl_longitude_rad = deg2rad(moon_ecl_longitude_deg);
    
    % ecliptic latitude of the moon [deg]
    phi_ecliptic_moon_deg = 5.13 * sin(moon_mean_arg_latitude_rad) + 0.28 * sin(moon_anomaly_rad + moon_mean_arg_latitude_rad) - 0.28 * sin(moon_mean_arg_latitude_rad - moon_anomaly_rad) - 0.17 * sin(moon_mean_arg_latitude_rad - 2 * mean_elong_sun_rad);
    phi_ecliptic_moon_rad = deg2rad(phi_ecliptic_moon_deg);
    
    obliquity_ecliptic_deg = 23.439 - 0.0000004 * time;
    obliquity_ecliptic_rad = deg2rad(obliquity_ecliptic_deg);
    
    % Parallax
    eta_moon_deg = 0.9508 + 0.0518*cos(moon_anomaly_rad) + 0.0095 * cos(moon_anomaly_rad - 2 * mean_elong_sun_rad)+ 0.0078 * cos(2 * mean_elong_sun_rad) + 0.0028 * cos(2 * moon_anomaly_rad);    
    eta_moon_rad = deg2rad(eta_moon_deg);

    r_moon_m = R_E / sin(eta_moon_rad);
    s_i = cos(phi_ecliptic_moon_rad) *  cos(moon_ecl_longitude_rad);
    s_j = cos(obliquity_ecliptic_rad) * cos(phi_ecliptic_moon_rad) * sin(moon_ecl_longitude_rad) - sin(obliquity_ecliptic_rad) * sin(phi_ecliptic_moon_rad);
    s_k = sin(obliquity_ecliptic_rad) * cos(phi_ecliptic_moon_rad) * sin(moon_ecl_longitude_rad) + cos(obliquity_ecliptic_rad) * sin(phi_ecliptic_moon_rad);
    a(1) = s_i * r_moon_m;
    a(2) = s_j * r_moon_m;
    a(3) = s_k * r_moon_m;
    % Sun vector in ECI_J2000
    r_moon_eci_j2000 = MOD2ECI(a, ttt);
    end
    
    