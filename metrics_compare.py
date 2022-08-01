import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.animation as animation

filepath = '/Volumes/Seagate-AA/Thesis/data_new/mc_3d_unet_{0}.h5'
filepath_raw = '/Volumes/Seagate-AA/Thesis/data_raw/headneck_3d_new.h5'
fig = plt.figure()
new_df = []
test_model1 = pd.read_csv('/Volumes/Seagate-AA/Thesis/result_1.csv')

# Loop through patients
for patients in tqdm(range(31, 32)):
    images = []
    for i in range(1, 21):
        with h5py.File(filepath.format(i), 'r') as f:
            images.append(f['mc_output'][patients])
    uncertainty_map = np.stack(images, axis=-1).std(axis=-1)
    #make_dataframe(patients, uncertainty_map)