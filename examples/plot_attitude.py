import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = r"/home/bkhan/Documents/Git/astropynaric/examples/output_data/tables/rotate_all_pred084_quatpred"
file  = "swapped_true_quatrotate_all_pred084_roll5.0_pitch10.0_yaw45.0_1Hz_3s.csv"
file2 = "true_quatrotate_all_pred084_roll5.0_pitch10.0_yaw45.0_1Hz_3s.csv"

df1 = pd.read_csv(os.path.join(directory, file),sep=',')
df2 = pd.read_csv(os.path.join(directory, file2),sep=',')

plt.figure()
# plt.plot(df1['time'], df1 ['q_true_w'])
# plt.plot(df1['time'], df1 ['q_true_x'])
# plt.plot(df1['time'], df1 ['q_true_y'])
plt.plot(df1['time'], df1 ['q_true_z'])

# plt.plot(df2['time'], df2['q_true_w'])
# plt.plot(df2['time'], df2 ['q_true_x'])
# plt.plot(df2['time'], df2 ['q_true_y'])
plt.plot(df2['time'], df2 ['q_true_z'])

plt.show()
