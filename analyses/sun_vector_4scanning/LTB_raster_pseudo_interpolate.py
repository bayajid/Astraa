# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:53:25 2022

@author: Martin Brechtelsbauer, Miguel Machin, 


-----------------

STEPS
1. open the GUI and find the best CPA position
2. set CPA position in the set_cpa_position(el,azi) in URAD in the line 256
3. run "sudo ./setup_can.sh"
4. run python LTB_rastercPA.py
-----------------
"""

# import can
# import cantools
import time
#from pprint import pprint
# from termcolor import colored
from datetime import datetime

import numpy as np
import csv

grid_params = {'area_x': 20000, #Set in urad
            'area_y': 20000,     #Set
            'step':  3000,        #Set
            'offset_x': 0,       #Set
            'offset_y': 0,       #Set
            'interval': 0.5,
            'loop': 1}

circle_params = {'radius': 400}

quadrants = {'A': [-2000,7000],
            'B': [-9000,-2000],
            'C': [-2000,-7000],
            'D': [5000,2000],
            'E': [20000,0],
            'F': [-20000,0],
            'G': [0,-20000],
            'H': [20000,0]
}

operation_modes = { "RESERVED": 0x00,
                    "STANDBY": 0x01,
                    "SETUP": 0x02,
                    "ACQ1A": 0x03,
                    "ACQ1B": 0x04,
                    "ACQ2": 0x05,
                    "FINEACQ": 0x06,
                    "COMM": 0x07,
                    "STOP": 0x08,
                    "PREPARE": 0x09
                    }

terminal_modes = { "STANDBY": 0x00,
                   "READY": 0x01,
                   "SHUTDOWN": 0x02,
                   "OPERATIONAL": 0x03,
                   "SELFCHECK": 0x04,
                   "MANUAL": 0x05,
                   "ERROR": 0x06
                   }

def spiral(rep):
    for _ in range(rep):
        dot = 2000
        d = dot * 0.05
        for i in range(dot): 
            t = i / d * np.pi
            x = (1 + 0.1*t) * np.cos(t) * 120
            y = (1 + 0.1*t) * np.sin(t) * 120
            yield x, y
        
def circle(rep):
    for _ in range(rep):
        for theta in np.arange(0, 2 * np.pi, 0.05):
            x = circle_params['radius'] * np.sin(theta)
            y = circle_params['radius'] * np.cos(theta)
            yield x, y
        
def grid():
    r1 = range(-grid_params['area_x'], grid_params['area_x'] + grid_params['step'], grid_params['step'])
    r2 = range(-grid_params['area_y'], grid_params['area_y'] + grid_params['step'], grid_params['step'])
    for i in range (0, grid_params['loop']):      
        # r1.reverse()
        r1 = r1[::-1]
        for x in r1:      
            # r2.reverse()
            r2 = r2[::-1]
            for y in r2:               
                yield x + grid_params['offset_x'], y + grid_params['offset_y']
def individual_quadrants():
    for x,y in quadrants.values():
        yield(x,y)


class mk3can:
    def __init__(self, can_interface = 'can0', dbc_file = 'CondorMk3.dbc'):
        print("__init__")
        self.db = cantools.database.load_file(dbc_file)
        self.can_bus = can.interface.Bus(can_interface, bustype='socketcan', bitrate=1000000)  
        self.message = ''
        self.data = ''
        self.fpa = {"tip": 0, "tilt": 0}
        self.fpa_cur = {"tip": 0, "tilt": 0}
        self.paa = {"tip": 0, "tilt": 0}
        self.paa_cur = {"tip": 0, "tilt": 0}
        self.tsp_op1 = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.tsp_op2 = {"E": 0, "F": 0, "G": 0, "H": 0}
        self.tsp_xy = {"x": 0, "y": 0}
        self.ftc_status = 0
        self.cpa = {"el":0, "az":0}
        self.cpa_cur = {"el":0, "az":0}

        
    def __del__(self):
        print("__del__")
        self.notifier.stop()
        self.can_bus.shutdown()
        
    def _can_receive(self, data):
        try:
            signals = self.db.decode_message(data.arbitration_id, data.data, decode_choices = False)
            message = self.db.get_message_by_frame_id(data.arbitration_id)
            # print(message.name)
        except:
            if data.arbitration_id != 0:
                pass # print(colored("{}: Message could not be decoded".format(hex(data.arbitration_id)), 'red'))
        else:   
            if message.name == "FTC_Status":
                self.ftc_status = list(signals.values())[3]
                # print(list(signals.values())[3])
            elif message.name == "FTC_PositionNom":
                self.fpa_cur = list(signals.values())[0:2]
                self.paa_cur = list(signals.values())[2:5]
                # print(data.name)
            elif message.name == "TSP_OpticalPower1":       
                self.tsp_op1 = signals.values()
            elif message.name == "TSP_OpticalPower2":       
                self.tsp_op2 = signals.values()
                # print(list(signals.values()))
            elif message.name == "TSP_Status":
                self.tsp_xy = list(signals.values())[5:7]
                # print((signals.values()))
            elif message.name == "CTC_PointingNom":
                self.cpa_cur = list(signals.values())[0:2]
            elif message.name == "EMC_Diagnostic":
                emc_var = list(signals.values())[0:9]
            elif message.name == "EDFA_Status1":
                #self.edfa = list(signals.values())[1:2]
                self.edfa = list(signals.values())[6:7]
           
    def _can_send(self):
        try:
            message = can.Message(arbitration_id = self.message.frame_id, data = self.data, is_extended_id = False)
        except:
            print(colored("Error encoding CAN message", 'red'))
                
        try:
            self.can_bus.send(message)
        except:
            print(colored("Error sending CAN message", 'red'))
        else: 
            pass
            # print(str(hex(self.message.frame_id) + ": " + self.message.name))
            
    def start_receive(self):
        print(colored("Receive notifier started", 'green'))
        self.notifier = can.Notifier(self.can_bus, [self._can_receive])
        time.sleep(0.2)
        
    def stop_receive(self):
        print(colored("Receive notifier stopped", 'green'))
        self.notifier.stop()
        
    def set_cpa_position(self, el, az):
        self.cpa['el'] = el
        self.cpa['az'] = az
        self.message = self.db.get_message_by_name("OC_CpaManualPointing")
        self.data = self.message.encode({'oc_el_manual_pointing': el, 
                                         'oc_az_manual_pointing': az})  
        self._can_send()
        
    def set_fpa_position(self, tip, tilt):
        self.fpa['tip'] = tip
        self.fpa['tilt'] = tilt
        self.message = self.db.get_message_by_name("OC_FsmManualPointing")
        self.data = self.message.encode({'oc_fpa_tip_manual_pointing': tip, 
                                         'oc_fpa_tilt_manual_pointing': tilt, 
                                         'oc_paa_tip_manual_pointing': self.paa['tip'], 
                                         'oc_paa_tilt_manual_pointing': self.paa['tilt']})  
        self._can_send()
        
    def set_paa_position(self, tip, tilt):
        self.paa['tip'] = tip
        self.paa['tilt'] = tilt
        self.message = self.db.get_message_by_name("OC_FsmManualPointing")
        self.data = self.message.encode({'oc_fpa_tip_manual_pointing': self.fpa['tip'], 
                                         'oc_fpa_tilt_manual_pointing': self.fpa['tilt'], 
                                         'oc_paa_tip_manual_pointing': tip, 
                                         'oc_paa_tilt_manual_pointing': tilt})  
        self._can_send()

    def set_terminal_mode(self, terminal_mode, can_selector):
        if type(terminal_mode) == str and terminal_mode.upper() in terminal_modes:
            terminal_mode = terminal_modes[terminal_mode.upper()]
        elif type(terminal_mode) == int and terminal_mode in terminal_modes:
            pass
        else:
            assert False, "Terminal mode does not exist"
                                   
        self.message = self.db.get_message_by_name("OC_SystemControl")
        self.data = self.message.encode({'oc_can_selector': can_selector, 
                                         'oc_terminal_mode': terminal_mode, 
                                         'oc_osa_id': 0,
                                         'oc_terminal_role': 0
                                         })   
        self._can_send()
        
    def init_fts(self): 
        cb.set_terminal_mode("STANDBY", 1)
        time.sleep(1)
        cb.set_fpa_position(0, 0)
        cb.set_paa_position(0, 0)
        time.sleep(1)        
        cb.set_terminal_mode("READY", 1)
        time.sleep(1)
        cb.set_terminal_mode("MANUAL", 1)
        time.sleep(1)
        




if __name__ == "__main__":

    import datetime as dt
    import pandas as pd
    import numpy as np
    from time import process_time
    from scipy.interpolate import interp1d
    ### KP:MAY 30 UPDATES FROM HERE
       
    # cb = mk3can()
    # cb.start_receive()
    # cb.init_fts()

    g = grid()
    # cb.set_cpa_position(0,0)        #Initial position
    # cb.set_fpa_position(0,0)
    time.sleep(5) # KP: moved time-tracking to after sleep command
    # Load csv
    path_to_ae = "ae_gs2sun.csv"
    ae_df = pd.read_csv(path_to_ae)
    ae_vals = ae_df.iloc[:,2:].values
    # KP May 30 : New file to track ae_gs2sun.csv reference time!
    path_to_tinfo = 'ref_time.csv'
    t_info_df = pd.read_csv(path_to_tinfo)
    # Auto-parse start time from ref_time.csv
    month_ref = t_info_df['month_used'].values[0]
    day_ref =  t_info_df['day_used'].values[0] 
    hour_ref = t_info_df['h_start'].values[0]
    minute_ref = 0 # KP: OK as long as ref time is also from minute=0
    t_ref = dt.datetime(2023, 5, day_ref, hour_ref, minute_ref, 0)
    # current time
    t_now = dt.datetime.now() 
    # tiem difference
    t_difference = (t_now - t_ref)
    t_res_sun_angles = t_info_df['t_res'].values[0]
    print(f'''Sun-pointing angle data:
    month start {month_ref}
    day start {day_ref}
    hour start {hour_ref}
    time res [s]: {t_res_sun_angles}
          ''')
    seconds_since_start = int(t_difference.seconds) + t_difference.microseconds/1e6 # seconds
    # round
    n_digits_used = len(str(t_res_sun_angles))-2 # 3 digits for 5 ms
    seconds_since_start = np.round(seconds_since_start, n_digits_used)
    ii_start = int(seconds_since_start/t_res_sun_angles)

    print(f'''Runtime:{t_now} SLICING INSPECTION
        Reference data time {t_ref}
        Time difference between now and ref start time: {seconds_since_start} s
        for dt = {t_res_sun_angles} s, start row : {ii_start}''')
    
    # get start row
    print(f'Automatically chosen row : {ae_df.iloc[ii_start,:]}')
    ae_sun_used = ae_vals[ii_start:, :]
    t_since_start = ae_df.iloc[:,0] - ae_df.iloc[0,0] # Time vector since start [s]
    t_since_start = t_since_start.values
    az_offsets  = ae_sun_used[:,0] - ae_sun_used[0,0] # azimuth vector since start, from 0
    el_used = ae_sun_used[:,1] # elevation vector since start, actual value
    az_used  = ae_sun_used[:,0]
    ## Jun 1st - Making sun az/el interpolator 
    spline_length = 30 # minutes
    ii_end = np.where(t_since_start < spline_length*60)[0][-1]
    
    az_interpolant = interp1d(t_since_start[:ii_end], az_used[:ii_end])
    el_interpolant = interp1d(t_since_start[:ii_end], el_used[:ii_end])
    use_interpolant = 1
    ## Jun 1st - Interpolant ready

    logfile = open('logfile_CPA_Cisil.log', 'w')
    logfile.writelines('"time";"CPA_El_cmd";"CPA_Az_cmd";"cpa_az";"cpa_el";"tip_pos";"tilt_pos";"tip_paa";"tilt_paa";"tsp_x";"tsp_y";"A";"B";"C";"D";"E";"F";"G";"H"\n')
    
   
    # cb = mk3can()
    # cb.start_receive()
    # cb.init_fts()

    g = grid()
    # cb.set_cpa_position(0,0)        #Initial position
    # cb.set_fpa_position(0,0)
    # time.sleep(5)

    row = 0
    wait_time = 0.02 # terminal wait time
    expected_nr_rows = 500e3
    storage_placeholder = np.zeros((int(expected_nr_rows), 9)) # 9 columns
    counter_update_step = t_res_sun_angles / wait_time # number of CTS updates per given sun-angle
    row_updates = int(np.round(wait_time / t_res_sun_angles,0))
    t_start = time.process_time()
    t_since_start = 0
    
    time_prerun = dt.datetime.now()
    wait_time_homing = 0.5
    print(f'Prerun time :{t_start}')
    
    
    ##
    time.sleep(wait_time_homing)
    
    t_posthoming = time.process_time()
    t_since_start += wait_time_homing
    print(f'Post homing time :{t_posthoming}')
    print(f'Post homing time + sleep time:{t_posthoming+wait_time_homing}')
    t_now = time.process_time() 

    for ii, (x, y) in enumerate(g):
        ## Jun 1 INTERPOLANT changes
        if use_interpolant:         
            t_since_start += wait_time
            el_sun = el_interpolant(t_since_start)
            az_sun = az_interpolant(t_since_start)
        else:
            el_sun = el_used[row]
            az_sun = az_used[row]        
            if ii >= counter_update_step and ii%counter_update_step == 0:
                row +=row_updates

        x_in = x+el_sun
        y_in = y+az_sun
        # for jj in range(100):
        #     am = jj**10
        ## Jun 1 INTERPOLANT changes DONE
        # cb.set_cpa_position(x_in, y_in)

        # store into placeholder
        storage_placeholder[ii,:3] = [ii*wait_time, x_in, y_in]
        t_prev = t_now
        t_now = time.process_time() 
        t_execution = t_now - t_prev
        t_sleep = wait_time - t_execution
        # t_since_start +=t_sleep
        print(f'sleeping time : {t_sleep} Execution time : {t_execution}. t since start : {t_since_start}')
        time.sleep(t_sleep)
    
    if 0:
        print(f'Run done. Processing and saving logs')
        print_freq = 200
        for jj, row in enumerate(storage_placeholder):
            
            datetime_then = time_prerun + dt.timedelta(seconds = row[0])
            x_in = row[1]
            y_in =  row[2]
            cpa_cur = row[3]
            fpa_cur = row[4]
            paa_cur =  row[5]
            tsp_xy = row[6]
            tsp_op1 = row[7]
            tsp_op2 = row[8]
            logfile.writelines(datetime_then + "; " + str(x_in) + "; " + str(y_in) + "; " + "; ".join(list(map(str, cpa_cur))) + "; "+ "; ".join(list(map(str, fpa_cur))) + "; " + "; ".join(list(map(str, paa_cur))) + "; " + "; ".join(list(map(str, tsp_xy))) +  "; " + "; ".join(list(map(str, tsp_op1)))+ ";" + "; ".join(list(map(str, tsp_op2))) + "\n")
            logfile.flush()
            
            if jj >= print_freq and jj%print_freq == 0:
                print("Ctr {} Position: {}, {}".format(ii, np.rad2deg(x_in*1e-6), np.rad2deg(y_in*1e-6)))
            
            if jj >= ii:
                print(f'Log processing finished. Rows recorded : {jj}')
                break

        cb.set_cpa_position(0,0)
        cb.set_fpa_position(0,0)

        print("Code Execution Finished")

        cb.stop_receive()
    logfile.close()
            

