## Functions used to open raw files
import numpy as np
import os
def open_dat(filename = 0, 
            nrows = None,
             folder_path = '',
             skip_first_row = False,
             first_row = 0, 
             delim = '\t',
             print_cond = 0,
             data_type = float):
    # Function to load a .dat file. Returns array of the values
    import os
    data_loaded = 0
    data_read = 0
    try: # Choosing the correct file
        if folder_path != '':
            filename_n_path = os.path.normpath(folder_path + "\\" + filename)
        else:
            filename_n_path = filename # in case file with entire path is provided
        data_raw = open(filename_n_path).readlines()
        data_loaded = 1

        # , 'modified on', time.ctime(os.path.getmtime(filename_n_path))[:3], ',', time.ctime(os.path.getmtime(filename_n_path))[8:-4])
        if data_type == float:
            if skip_first_row:
                data_content = np.array([i.strip().split(delim) for i in data_raw[first_row:][1:]]).astype(data_type)
            else: # Jun-09 KP: Doesnt work. Just load everything
                if type(nrows) == type(None):
                    data_content = np.array([i.strip().split(delim) for i in data_raw]).astype(data_type)
                else:
                    data_content = np.array([dat.strip().split(delim) for jj, dat in enumerate(data_raw) if jj < nrows]).astype(data_type)

            data_read = 1

        # else: # Load ancillary data
        #     data_content = [i.strip().split(delim) for i in data_raw]
        #     data_read = 1
        if print_cond:
            print("Read file ", filename, ' shape:', data_content.shape)
        # data_raw.close()
        return data_content
    except:
        if print_cond:
            if not data_loaded:
                print('Unable to load %s' % filename_n_path)
            elif not data_read:
                print(f'Loaded but unable to read {filename_n_path}')