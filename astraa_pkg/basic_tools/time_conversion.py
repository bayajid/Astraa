import datetime as dt
from datetime import timezone, timedelta
import numpy as np
from skyfield.api import load

def dt_j2000tt2gps(date_gw0_gps = '1980-01-06 00:00:00', date_j2000_tt = '2000-01-01 12:00:00'):
    # does the opposite of 
    # Function to convert [s] since GPS WEEK 0 to [s] since J2000 IN Terrestrial Time    
    # t_0_gps = t_0_j2000_tt + dt_j2000tt2gps_s
    dt_GPS2TT = t_sys_conv(0)['dt_GPS2TT']
    dt_gw0_gps = dt.datetime.strptime(date_gw0_gps, '%Y-%m-%d %H:%M:%S') 
    dt_gw0_tt = dt_gw0_gps + dt.timedelta(seconds= dt_GPS2TT)
    dt_j2000_tt = dt.datetime.strptime(date_j2000_tt, '%Y-%m-%d %H:%M:%S') 
    dt_j2000tt2gps = (dt_j2000_tt - dt_gw0_tt)
    dt_j2000tt2gps_s = dt_j2000tt2gps.days*86400 + dt_j2000tt2gps.seconds + dt_j2000tt2gps.microseconds/1e6 # [s]
    return dt_j2000tt2gps_s

def dt_gps2j2000tt(date_gw0_gps = '1980-01-06 00:00:00', date_j2000_tt = '2000-01-01 12:00:00'):
    # Function to convert [s] since GPS WEEK 0 to [s] since J2000 IN Terrestrial Time
    # [s] from GPS to TERESTIAL TIME
    # t_0_j2000_tt = t_0_gps + gps_t_2j2000_tt
    dt_GPS2TT = t_sys_conv(0)['dt_GPS2TT']
    dt_gw0_gps = dt.datetime.strptime(date_gw0_gps, '%Y-%m-%d %H:%M:%S') 
    dt_gw0_tt = dt_gw0_gps + dt.timedelta(seconds= dt_GPS2TT)
    
    dt_j2000_tt = dt.datetime.strptime(date_j2000_tt, '%Y-%m-%d %H:%M:%S') 
    dt_gps2j2000tt = (dt_gw0_tt - dt_j2000_tt)
    dt_gps2j2000ttt_s = dt_gps2j2000tt.days*86400 + dt_gps2j2000tt.seconds + dt_gps2j2000tt.microseconds/1e6
    return dt_gps2j2000ttt_s
    # function to convert time in [s] since GPS start time : '1980-01-06 00:00:00' GPS/UTC
    # to [s] since J2000 (TT) L 2000-01-01 12:00:00

def t_sys_conv(leap_s = 18):
    # return dict of time differences [s]
    # leap_s - leap seconds from BRDC file, which gives leap seconds since 1980. 18 as of 2016-2023
    # time differences given in the form: t_X = t_Y + dt_X2Y
    # dt_X2Z = dt_X2Y + dt_Y2Z
    # dt_X2Y = -dt_Y2X
    dt_TAI2GPS = -19 # s, constant difference
    dt_UTC2GPS = leap_s # leap seconds, input from gps broadcast ephem or IERS Bulletin C
    dt_TAI2TT = 32.184 # s, constant difference
    dt_UTC2TAI = dt_UTC2GPS - dt_TAI2GPS
    dt_GPS2TT =  - (-dt_TAI2TT + dt_TAI2GPS)
    conv_dict = {
        'dt_TAI2GPS' : dt_TAI2GPS,
        'dt_UTC2TAI' : dt_UTC2TAI,
        'dt_UTC2GPS' : dt_UTC2GPS,
        'dt_TAI2TT' : dt_TAI2TT,
        'dt_GPS2TT' : dt_GPS2TT
    }
    return conv_dict

def gws2utc(gw, date_gw0 = '1980-01-06 00:00:00', ls = 18):
    # function to convert GPS seconds to seconds since J2000
    dt_gw0 = dt.datetime.strptime(date_gw0, '%Y-%m-%d %H:%M:%S') # date at GPS Week 0
    # GPS date at input GPS Week [s]
    dt_gw = dt_gw0 + dt.timedelta(seconds = gw)
    # UTC date at input GPS week -offset by leap seconds since 1980 (9 at 1980. +18 sinec 2016)
    dt_utc = dt_gw + dt.timedelta(seconds = ls)
    return dt_utc

def dt2jd(datetime):
    # convert dt.datetime object into julian date
    # requires datetime library
    tt = datetime.timetuple()
    jd_date = 367*tt.tm_year - int(7/4*(tt.tm_year + int((tt.tm_mon+9)/12) )) + int(275*tt.tm_mon/9) + tt.tm_mday + 1721013.5 + ((tt.tm_sec/60 + tt.tm_min)/60 + tt.tm_hour)/24 # IN GPS TIME
    return jd_date

def jd2dt(jd):
    # convert julian date to grgorian date based on Algo 22
    # from Valado2013
    # Input Julian date 
    l_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    t_1900 = (jd - 2415019.5)/365.25
    yy = 1900 + int(t_1900) # year
    l_yy = int((yy - 1900 - 1)*.25) # leap years
    days = (jd - 2415019.5) - ((yy - 1900)*365 + l_yy)
    if days < 1:
        yy -=1
        l_yy = int((yy - 1900 - 1)*.25) # leap years
        days = (jd - 2415019.5) - ((yy - 1900)*365 + l_yy)
    if yy%4 == 0:
        l_month[1] = 29 # leap year
    day_of_year = int(days)
    for ii, lmo in enumerate(l_month):
        sum_lmonth = sum(l_month[:ii])
        if sum_lmonth+1>day_of_year:
            sum_lmonth = sum(l_month[:ii-1])
            break
        else:
            ii+=1 # accounts for December
    month = ii
    day = day_of_year - sum_lmonth
    tau = (days - day_of_year)*24
    hh = int(tau)
    mm = int(np.round((tau - hh)*60, 8))
    ss = int((tau - hh - mm/60)*3600)
    if (tau - hh - mm/60)*3600 > 0:
        micross = int(((tau - hh - mm/60)*3600)%1*1e6)
    else:
        micross = 0

    return dt.datetime(yy, month, day, hh, mm, ss)    

def jd2jc(jd):
    # convert Julian Days to Julian Centuries
    # according to 3-42 of Vallado 2013 (page184)
    # Julian Centuries in teh same time convention as the Julian Date (eg JD in UT1 results in JC in UT1)
    jc = (jd - 2451545)/36525
    return jc

def tgps2jc(t_gps_s):
    # convert gps time [s] since '1980-01-06 00:00:00'
    # to Julian Centuries
    # according to P183 of Vallado 2013 (page184)
    jd_gps0 = 2444244.5 # Julian date corresponding to t_gps 0
    jd_gps_curr = jd_gps0 + t_gps_s/86400 # julian date of current GPS time
    jc_gps_curr = jd2jc(jd_gps_curr)
    return jc_gps_curr

def utc2gws(date_utc, date_gw0 = '1980-01-06 00:00:00', ls = 18):
    # function to convert GPS seconds to seconds since J2000
    
    dt_gw0 = dt.datetime.strptime(date_gw0, '%Y-%m-%d %H:%M:%S') # date at GPS Week 0
    datetime_gw = date_utc + dt.timedelta(seconds = ls) # GPS time 
    # GPS date at input GPS Week [s]
    # UTC date at input GPS week -offset by leap seconds since 1980 (9 at 1980. +18 sinec 2016)
    time_delta = datetime_gw-dt_gw0
    gps_s = time_delta.seconds + time_delta.days*86400 + time_delta.microseconds / 1e6

    return gps_s    

def sky_utc2gps(utc_dt):
    """
    Input: UTC datetime
    """
    gps_epoch = dt.datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

    # Use current known leap seconds (update manually when new leap seconds announced)
    leap_seconds = 18

    gps_seconds = (utc_dt - gps_epoch).total_seconds() + leap_seconds

    return gps_seconds

def sky_gps2utc(gps_seconds, leap_seconds=18):
    """
    Convert GPS seconds since 1980-01-06 to a timezone-aware UTC datetime.
    
    Parameters:
    - gps_seconds: float, seconds since GPS epoch (1980-01-06 00:00:00 UTC)
    - leap_seconds: int, number of leap seconds to subtract (default 18)
    
    Returns:
    - datetime.datetime in UTC timezone
    """
    gps_epoch = dt.datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    
    # Subtract leap seconds to get UTC time delta
    utc_delta = timedelta(seconds=gps_seconds - leap_seconds)
    
    utc_time = gps_epoch + utc_delta
    return utc_time

def gps_to_j2000(gps_seconds):
    """
    Convert GPS time (seconds since January 6, 1980, 00:00:00 UTC) to J2000 time
    (seconds since January 1, 2000, 11:58:55.816 UTC).
    
    Args:
        gps_seconds (float): Seconds since GPS epoch (January 6, 1980, 00:00:00 UTC).
    
    Returns:
        float: Seconds since J2000 epoch (January 1, 2000, 11:58:55.816 UTC).
    """
    # Unix timestamp of GPS epoch (Jan 6, 1980, 00:00:00 UTC)
    gps_epoch_unix = 315964800.0
    
    # Unix timestamp of J2000 epoch (Jan 1, 2000, 11:58:55.816 UTC)
    j2000_epoch_unix = 946727935.816
    
    # Convert GPS time to Unix time
    unix_time = gps_seconds + gps_epoch_unix
    
    # Convert to seconds since J2000
    j2000_time = unix_time - j2000_epoch_unix
    
    return j2000_time

def j2000_to_gps(j2000_seconds):
    """
    Convert J2000 time (seconds since January 1, 2000, 11:58:55.816 UTC) to GPS time
    (seconds since January 6, 1980, 00:00:00 UTC).
    
    Args:
        j2000_seconds (float): Seconds since J2000 epoch (January 1, 2000, 11:58:55.816 UTC).
    
    Returns:
        float: Seconds since GPS epoch (January 6, 1980, 00:00:00 UTC).
    """
    # Unix timestamp of GPS epoch (Jan 6, 1980, 00:00:00 UTC)
    gps_epoch_unix = 315964800.0
    
    # Unix timestamp of J2000 epoch (Jan 1, 2000, 11:58:55.816 UTC)
    j2000_epoch_unix = 946727935.816
    
    # Convert J2000 time to Unix time
    unix_time = j2000_seconds + j2000_epoch_unix
    
    # Convert to seconds since GPS epoch
    gps_time = unix_time - gps_epoch_unix
    
    return gps_time


if __name__ == '__main__':
    print(f'conversion from GPS [s] to [s] since J2000 TT')
    print(dt_gps2j2000tt())
    print(f'Reverse: {dt_j2000tt2gps()}')

    print("-----------------------------")

    # Example for t_gps calculation
    t_now = dt.datetime.utcnow() # UTC time 
    t_sky = load.timescale().now().utc_datetime()
    t_gps_now = utc2gws(t_now)
    t_gps_sky = sky_utc2gps(t_sky)
    t_utc_sky = sky_gps2utc(t_gps_sky)
    t_j2000_now = gps_to_j2000(t_gps_now)
    print(f"{t_now} UTC : {t_sky} UTC_SKY")
    print(f'{t_now} UTC -> GPS sec {t_gps_now}')
    
    print(f'{t_sky} UTC_SKY -> SKY_GPS sec {t_gps_sky}')
    print(f'{t_gps_sky} SKY_GPS -> SKY_UTC {t_utc_sky}')
    print(f'{t_now} UTC -> J2000 sec {t_j2000_now}')
    print(f'{t_j2000_now} J2000 -> GPS sec {j2000_to_gps(t_j2000_now)} -> UTC {gws2utc(t_gps_now)}')