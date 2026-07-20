%% Script to generate and output single-precision moon vectors
% into a CSV with the desired start date and time-resolution

%2023-01-01 00:00:00
% t_gps_0 = 1356559182; % 
% dt_sec = 60;
% t_length_days = 31;

% Matching TUDAT simulation time
t_gps_0 = 725824800.0 + 630763148.81;
dt_sec = 10;
t_length_days = 7;

t_gps_f = t_gps_0 + t_length_days * 86400;

time_loop = t_gps_0:dt_sec:t_gps_f;

output_placeholder = zeros(length(time_loop),4); % t_gps; x; y; z
output_placeholder(:,1) = time_loop;


for ii = 1:length(time_loop)
    t_ii = time_loop(ii);
    moon_vector = where_moon(t_ii, 1);    
    output_placeholder(ii, 2:4) = moon_vector;
end
writematrix(output_placeholder, append('32bitmoon', int2str(t_gps_0),'.csv'))