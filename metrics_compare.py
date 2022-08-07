import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tqdm import tqdm
from mask_stats import compute_evaluations_for_mask_pairs
import os
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

filepath = '/Volumes/Seagate-AA/Thesis/data_new/mc_3d_unet_{0}.h5'
filepath_raw = '/Volumes/Seagate-AA/Thesis/data_raw/headneck_3d_new.h5'
fig = plt.figure()
new_df = []
test_model1 = pd.read_csv('/Volumes/Seagate-AA/Thesis/result_1.csv')


def make_dataframe(patient_id, patient_uncertainity_map, new_dice, UR_dice):
    patient_score = {'patient': patient_id,
                     'mean_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].mean(),
                     'min_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].min(),
                     'max_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].max(),
                     'tumour_volume': ground_truths[patient_id].sum(),
                     'tumour_volume_predicted': np.round(mean_prediction).sum(),
                     'new_dice': new_dice,
                     'error_dice': UR_dice
                     }

    new_df.append(patient_score)


# Load raw images and ground truths
with h5py.File(filepath_raw.format(1), 'r') as rf:
    test_folds = ['fold_11', 'fold_12', 'fold_13']
    raw_images = []
    ground_truths = []
    for test_fold in test_folds:
        for i in range(len(rf[test_fold]['input'])):
            raw_images.append(rf[test_fold]['input'][i])
            ground_truths.append(rf[test_fold]['target'][i])

# Loop through patients
for patients in tqdm(range(0, 40)):
    images = []
    #feature_importance = []
    for i in range(1, 21):
        with h5py.File(filepath.format(i), 'r') as f:
            images.append(f['mc_output'][patients])
            #feature_importance.append(f['gb_output'][patients])
    uncertainty_map = np.stack(images, axis=-1).std(axis=-1)
    _uncertainty_map = uncertainty_map.copy()
    _uncertainty_map[_uncertainty_map == 0] = np.nan
    _uncertainty_map[_uncertainty_map < np.nanmean(_uncertainty_map) + 3*(np.nanstd(_uncertainty_map))] = 0
    _uncertainty_map[_uncertainty_map > np.nanmean(_uncertainty_map) + 3*(np.nanstd(_uncertainty_map))] = 1
    #uncertainty_feature = np.stack(feature_importance, axis=-1).std(axis=-1)
    mean_prediction = np.stack(images, axis=-1).mean(axis=-1)
    error_region = np.abs(np.round(mean_prediction) - ground_truths[patients])
    plot_rows = 1
    plot_columns = 4

    true_eval, pred_eval = compute_evaluations_for_mask_pairs(ground_truths[patients], np.round(mean_prediction))
    dice_new = np.nanmean(true_eval['overall'])

    true_eval, pred_eval = compute_evaluations_for_mask_pairs(error_region, _uncertainty_map)
    error_dice = np.nanmean(true_eval['overall'])

    make_dataframe(patients, uncertainty_map, dice_new, error_dice)

df = pd.DataFrame(new_df)
df.to_csv("/Volumes/Seagate-AA/Thesis/uncertainty_output/patient_uncertainty.csv", index=False)
