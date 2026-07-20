## Functions used to open raw files
import numpy as np
import os
import json

def load_dat_like_csv(file_path, nrows=None, delim=" ", skip_first_row=False, dtype=float):
    with open(file_path, "r") as f:
        data_raw = f.readlines()

    # Handle first row skip
    start = 1 if skip_first_row else 0

    # Split lines into lists of values
    rows = []
    for jj, line in enumerate(data_raw[start:]):
        if nrows is not None and jj >= nrows:
            break
        parts = line.strip().split(delim)
        if parts:  # ignore empty lines
            rows.append(parts)

    # Convert to array like CSV
    data_content = np.array(rows, dtype=dtype)

    return data_content

def open_dat(folder = '', 
            nrows = None,             
            fname_states = 'state_history.dat',
            fname_params = 'simulation_parameters.json',
            base_path = None,
            skip_first_row = False,
            first_row = 0, 
            delim = ',',
            print_cond = 0,
            data_type = float):
    # Function to load a .dat file. Returns array of the values
    data_loaded = 0
    data_read = 0
    data_content = None
    sim_parameters = None
    
    if base_path == None:
        path_states_full = os.path.normpath(os.path.join(folder, fname_states))
        path_params_full = os.path.normpath(os.path.join(folder, fname_params))
    else:
        path_states_full = os.path.normpath(os.path.join(base_path, folder, fname_states))
        path_params_full = os.path.normpath(os.path.join(base_path, folder, fname_params))

    try: ## Load states
        data_raw = open(path_states_full).readlines()
        data_loaded = 1

        if data_type == float:
            data_content = load_dat_like_csv(
                        path_states_full,
                        # nrows=1000,
                        delim="\t",   # or "\t" if tab-delimited
                        skip_first_row=False
                    )
            # if skip_first_row:
            #     data_content = np.array([i.strip().split(delim) for i in data_raw[first_row:][1:]]).astype(data_type)
            # else: # Jun-09 KP: Doesnt work. Just load everything
            #     if type(nrows) == type(None):
            #         data_content = np.array([i.strip().split(delim) for i in data_raw]).astype(data_type)
            #     else:
            #         data_content = np.array([dat.strip().split(delim) for jj, dat in enumerate(data_raw) if jj < nrows]).astype(data_type)

            # data_read = 1
        print(f'Loaded {path_states_full}. Shape : {data_content.shape}')
    except Exception as e:
        print(f'Failed to load states. Path tried: {path_states_full}')
        print(f'Error: {str(e)}')
        return None, None
        
    try: ## Load simulation parameters
        with open(path_params_full, 'r') as j:
            sim_parameters = json.load(j)
        print(f'Loaded {path_params_full}.')
    except Exception as e:
        print(f'Failed to load parameters. Path tried: {path_params_full}')
        print(f'Error: {str(e)}')
        return None, None

    return data_content, sim_parameters