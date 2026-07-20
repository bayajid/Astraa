import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path = r'/home/kpaliusis/Documents/propagator_evalboard_test_feb2025/OHOCC_82_py'
files_all = os.listdir(path)
input_files = [file for file in files_all if 'inputs' in file]

ii = 3
loaded_inputs = pd.read_csv(f'{path}/{input_files[ii]}')
loaded_inputs = loaded_inputs.values[:,3:]
loaded_input_end = np.nonzero(np.average(loaded_inputs,axis=1)==1)[0][0]
loaded_inputs = loaded_inputs[:,:loaded_input_end]
print(f'Loaded file: {input_files[ii]}')
r_h_updates = loaded_inputs[:,0:7]
r_t_updates = loaded_inputs[:,7:14]
att_h_updates = loaded_inputs[:,14:18]

def get_upd_sent(updates):
    update_rows = updates[np.nonzero(updates[:,0] != 0)[0],:]
    t_0 = update_rows[0,0]
    update_rows[:,0] = update_rows[:,0] - t_0
    update_rows = update_rows[np.nonzero(update_rows[:,0] >= 0)[0],:]
    return update_rows
    
att_upd = get_upd_sent(att_h_updates)
rt_upd = get_upd_sent(r_t_updates)
rh_upd = get_upd_sent(r_h_updates)

f, ax = plt.subplots()
ax.plot(rt_upd[:,0], rt_upd[:,1]/rt_upd[:,1]*1.5, 'o-', label = 'target')
ax.plot(rh_upd[:,0], rh_upd[:,1]/rh_upd[:,1]*1, 'o-', label = 'r_h')
ax.plot(att_upd[:,0], att_upd[:,1]/att_upd[:,1]*2, 'o-', label = 'att_h')
ax.set_ylabel('upd [-]')
ax.set_xlabel('t since start [s]')
ax.grid('on')
ax.legend()
ax.set_xlim([0,25])
f.suptitle(f'Target updates sent, {input_files[ii]}')
# ax.set_xlim([0,40])
# target_updates = 
# upd_sent[0,9:16] = upd_pos_t
# upd_sent[0,2:9] = upd_pos_h
# upd_sent[0,16:26] = upd_att_h