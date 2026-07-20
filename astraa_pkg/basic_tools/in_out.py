import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform
import subprocess
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
import zipfile
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

def make_n_save(fname, data, t_vec = None,data_cols = None, main_folder = r'outputs/tables', subfolder = None, open_folder = 1):
    """Generic robust function to create csv's of arrays


    Args:
        fname (string): filename w/o .csv
        t_vec (vector): time vector
        data (array): array of data
        data_cols (list, optional): list of column names. Defaults to None.
        main_folder (string, optional): Default save folder. Defaults to r'outputs/tables'.
        subfolder (additional folder, optional): Defaults to None.
    """
    if type(t_vec)!=type(None):
        n_cols = data.shape[1]+1
    else:
        n_cols = data.shape[1]
    output_data = np.zeros((data.shape[0], n_cols))

    if type(t_vec) != type(None):
        output_data[:,0] = t_vec
        output_data[:,1:] = data
    else:
        output_data = data
    print(f'Output shape : {output_data.shape}')
    # FOlder jazz
    if type(subfolder) != type(None):
        full_folder = f'{main_folder}/{subfolder}'
        try:
            os.mkdir(full_folder)
            print(f'Created {full_folder}')
        except:
            print(f'Could not create {full_folder}')
    else:
        full_folder = main_folder

    
    full_path = fr'{full_folder}/{fname}.csv'

    if type(data_cols) == type(None):
        print('Giving automatic column labels')
        data_cols = [f'col{ii}' for ii in range(data.shape[1]+1)]
    
    output_df = pd.DataFrame(data = output_data, columns = data_cols)
    output_df.to_csv(full_path, index = 0)
    print(f'Saved {fname} to {full_folder}')
    print(output_df.head(), '\n',output_df.tail())
    if open_folder:
        if platform.system() == 'Windows':
            os.startfile(os.path.realpath(full_folder))
        else:
            subprocess.run(["xdg-open", full_folder])
    return output_df

def save_dict_2_csv(dict, fname, main_folder = r'outputs/tables', subfolder = None, open_folder = 1):
    if type(subfolder) != type(None):
        full_folder = f'{main_folder}/{subfolder}'
        try:
            os.mkdir(full_folder)
            print(f'Created {full_folder}')
        except:
            print(f'Could not create {full_folder}')
    else:
        full_folder = main_folder

    
    full_path = fr'{full_folder}/{fname}.csv'   
    output_df = pd.DataFrame.from_dict(dict)
    output_df.to_csv(full_path, index = 0)
    print(f'Saved {fname} to {full_folder}')
    print(output_df.head(), '\n',output_df.tail())
    if open_folder:
        if platform.system() == 'Windows':
            os.startfile(os.path.realpath(full_folder))
        else:
            subprocess.run(["xdg-open", full_folder])
    return output_df
def save_azel(t_gps, 
              s_h, 
              s_t, 
              q_eci2bf, 
              q_mo, 
              ae, 
              ncols = 32,
              paa = None,
              fname = 'ptcalc_inout_default',
              subfolder = 'pointing_calculation_io',
              full_folder = None,
              pos_unit = 'm',
              open_folder = 1,
              zip_name = '',
              make_zip = 1
              ):
    """function to save az/el and pointing calculation inputs to a csv
    with 1 value per column

    Args:
        t_gps (_type_): s
        s_h (_type_): m, m/s
        s_t (_type_): m, m/s
        q_eci2bf (_type_): quat, quat_dot, eci2body
        q_mo (_type_): quat, quat_dot, body2lct (mounting offset)
        ae (_type_): az el [rad]
        ncols (int) : number of colkumns of outputs
        paa (array): [rad] Optional PAA angle columns (az, el). Defaults to None
        fname (str, optional): no .csv needed. Defaults to 'ptcalc_inout_default'.
        subfolder (str, optional): _description_. Defaults to 'pointing_calculation_io'.
        full_folder (_type_, optional): _description_. Defaults to None.
    """    
    # TODO perhaps its useful to use the TMTC defined labels?
    if type(full_folder) == type(None):
        full_folder = fr'outputs/{subfolder}'

    full_path = fr'{full_folder}/{fname}.csv'

    try:
        os.mkdir(full_folder)
    except:
        pass

    # 0 t_gps; 
    # 1-4 aer [rad, rad, m], 
    # 4-10 s h [m, m/s],
    # 10-16 s t [m, m/s],
    # 16-24 q, qdot, ECI 2 bf (body-frame satellite attitude)
    # 24-32 q, qdot, Mounting offset
    # 33, 34 PAA
    
    if pos_unit == 'm':
        pos_factor = 1
    elif pos_unit == 'km':
        pos_factor = 1e-3
    
    if type(paa) != type(None):
        ncols = ncols + 2
    output_data = np.zeros((s_h.shape[0], ncols)) 
    
    output_data[:,[0]] = t_gps.reshape((s_h.shape[0], 1))
    i_aer = 1
    output_data[:,i_aer : i_aer + ae.shape[1]] = ae
    output_data[:,i_aer+2] = output_data[:,i_aer+2] * pos_factor
    i_4 = 4
    output_data[:,i_4:i_4+s_h.shape[1]] = s_h * pos_factor
    i_10 = 10
    output_data[:,i_10:i_10+s_t.shape[1]] = s_t * pos_factor
    i_16 = 16
    output_data[:,i_16:i_16+q_eci2bf.shape[1]] = q_eci2bf
    i_24 = 24
    output_data[:,i_24:i_24+q_mo.shape[1]] = q_mo
    columns = ['t_gps',
               'az_lct_rad',
               'el_lct_rad',
               'r_lct',
               'x_h',
               'y_h',
               'z_h',
               'vx_h',
               'vy_h',
               'vz_h',
               'x_t',
                'y_t',
                'z_t',
                'vx_t',
                'vy_t',
                'vz_t',
                'q1_eci2bf',
                'q2_eci2bf',
                'q3_eci2bf',
                'q4_eci2bf',
                'qdot1_eci2bf',
                'qdot2_eci2bf',
                'qdot3_eci2bf',
                'qdot4_eci2bf',
                'q1_mo',
                'q2_mo',
                'q3_mo',
                'q4_mo',
                'qdot1_mo',
                'qdot2_mo',
                'qdot3_mo',
                'qdot4_mo'
               ]
    if type(paa) != type(None):
        output_data[:,-2:] = np.array(paa).reshape((len(paa),2))
        columns.append('paa_az_rad')
        columns.append('paa_el_rad')


    df_tosave = pd.DataFrame(data = output_data, columns = columns)
    try:
        df_tosave.to_csv(full_path, index = 0)
        print(f'Saved {fname} to {full_folder}')
        if make_zip:
            with zipfile.ZipFile(fr'{full_folder}/{zip_name}', 'w') as zipf:
                zipf.write(full_path, arcname = os.path.basename(full_path) )
                zipf.write(fr'{full_folder}/ref_time.csv', arcname = os.path.basename(fr'{full_folder}/ref_time.csv'))
            pass
        if open_folder:
            try:
                if platform.system() == 'Windows':
                    os.startfile(os.path.realpath(full_folder))
                else:
                    subprocess.run(["xdg-open", full_folder])
            except:
                print(f'Failed to open {full_folder}')

        return 1, df_tosave
    except:
        # os.mkdir()
        print(f'Failed to save {fname} to {full_folder}')
        return 0, None
    
import json

def save_dct2json(dict, name, main_folder = r'outputs/tables', 
                  subfolder = None, 
                  open_folder = 1, 
                  print_cond=1):
    """Saving dictionaries into JSON files

    Args:
        dict (dict): Dictionary to be saved
        name (str): name of file to be saved (w/o file extension)
        folder (path): complete or relative path to save the file
        print_cond (int, optional): Conditional to print whether saves were successfull. Defaults to 1.
    """    
    if print_cond:
        print(f'Attempting to save {name}.')
    try:
        if subfolder is not None:
            full_folder = f'{main_folder}/{subfolder}'
        else:
            full_folder = main_folder
        try:
            os.mkdir(full_folder)
        except:
            print(f'Could not create {full_folder}')
            with open(f'{full_folder}/{name}.json', "w") as json_to_save:
                json.dump(dict, json_to_save, indent = 4)

                print(f'Created {full_folder}')
        if print_cond:
            print(f'{name} saved succesfully.')
    except Exception as e:
        if print_cond:
            print(e)
            print(f'Failed to save {name}.')

if __name__ == '__main__':
    pass