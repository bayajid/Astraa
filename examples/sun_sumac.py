import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"/home/bkhan/Downloads/sumac_data_sun.csv"
df = pd.read_csv(csv_path)


plt.figure()
# plt.plot(df["CTC.LosPredAzimuth"], label = "CTC.LosPredAzimuth")
# plt.plot(df["CTC.LosSunAzimuth"], label = "CTC.LosSunAzimuth")
#plt.plot(df["CTC.LosSunAzimuth"] - df["CTC.LosPredAzimuth"], label = "Diff")
# plt.plot(df["CTC.PRED_AZ"], label = "CTC.PRED_AZ")
plt.plot(df["CTC.SUN_AZ"]-df["CTC.LosSunAzimuth"], label = "diff")

plt.legend()
plt.show()