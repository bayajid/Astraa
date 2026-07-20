import numpy as np
import pandas as pd
def log2df(logname, parent_path = '', add_power_sum = 1):
    if 'scan' in logname:
        scan_df = pd.read_csv(f'{parent_path}/{logname}', delimiter=";")
    elif 'log' not in logname:
        data = np.genfromtxt(f'{parent_path}/{logname}', delimiter = ';', skip_header=4)
        scan_df = pd.DataFrame(data = data[:,1:], columns = ["t_gps","can_time[ms]","az_exp","el_exp","cpa_az","cpa_el","tip_pos","tilt_pos","tip_paa","tilt_paa","tsp_x","tsp_y","A","B","C","D","E","F","G","H",'op_mode','beamDet'])
    else:
        scan_df = pd.read_csv(f'{parent_path}/{logname}', delimiter=";")        
        fct_cmd = 1e-6
    if add_power_sum:
        power_sum = scan_df['A']+scan_df['B']+scan_df['C']+scan_df['D']+scan_df['E']+scan_df['F']+scan_df['G']+scan_df['H'] 
        scan_df['power_sum'] = power_sum

    
    return scan_df