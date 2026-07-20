#%%
import numpy as np 
latlong_us = [48.09109136005601, 11.309158562668255]
latlong_hut = [48.087172976752484, 11.3103405925428130]

latlong_us_legal = [48.09146991218012, 11.308811854325517]
latlong_probeam = [48.09119081983069, 11.306418417926585]

latlong_us_buildingcorner = [48.09120652198673, 11.309009263692023]
latlong_neighbor_buildingcorner = [48.09100287418912, 11.308889190951106]
# spot = 'probeam'
# spot = 'hut'
spot = 'bell'

if spot  == 'probeam':
    latlong_us_used = latlong_us_legal
    latlong_target_used = latlong_probeam
elif spot == 'hut':
    latlong_us_used = latlong_us
    latlong_target_used = latlong_hut
elif spot == 'bell':    
    latlong_us_used = latlong_us_buildingcorner
    latlong_target_used = latlong_neighbor_buildingcorner

delta_long = latlong_target_used[1] - latlong_us_used[1]
delta_lat = latlong_target_used[0] - latlong_us_used[0]

heading_from_north = -np.rad2deg(np.arctan2(     delta_long , delta_lat))
print(f'Az towards {spot.upper()} : {heading_from_north:.2f} deg W pos ')