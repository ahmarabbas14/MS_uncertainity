import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tqdm import tqdm
import os
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

filepath = '/Volumes/Seagate-AA/Thesis/data_new/mc_3d_unet_{0}.h5'
filepath_raw = '/Volumes/Seagate-AA/Thesis/data_raw/headneck_3d_new.h5'
fig = plt.figure()
new_df = []
test_model1 = pd.read_csv('/Volumes/Seagate-AA/Thesis/result_1.csv')


def make_dataframe(patient_id, patient_uncertainity_map):
    patient_score = {'patient': patient_id,
                     'mean_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].mean(),
                     'min_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].min(),
                     'max_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].max(),
                     'tumour_volume': ground_truths[patient_id].sum(),
                     'tumour_volume_predicted': np.round(mean_prediction[patient_id]).sum(),
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
for patients in tqdm(range(20, 21)):
    images = []
    feature_importance = []
    for i in range(1, 21):
        with h5py.File(filepath.format(i), 'r') as f:
            images.append(f['mc_output'][patients])
            feature_importance.append(f['gb_output'][patients])
    uncertainty_map = np.stack(images, axis=-1).std(axis=-1)
    _uncertainty_map = uncertainty_map.copy()
    _uncertainty_map[_uncertainty_map == 0] = np.nan
    _uncertainty_map[_uncertainty_map < np.nanmean(_uncertainty_map) + 3*(np.nanstd(_uncertainty_map))] = 0
    uncertainty_feature = np.stack(feature_importance, axis=-1).std(axis=-1)
    mean_prediction = np.stack(images, axis=-1).mean(axis=-1)
    error_region = np.abs(np.round(mean_prediction) - ground_truths[patients])
    plot_rows = 1
    plot_columns = 4

    # make_dataframe(patients, uncertainty_map)

    # first check the shape (173, 191, 265) --> continue
    # ims = []
    for j in tqdm(range(173)):
        # im = plt.imshow(image[:, j, :], 'gray')
        # _uncertainty_map = uncertainty_map.copy()
        # _uncertainty_map[_uncertainty_map < np.mean(_uncertainty_map)] = np.nan
        plt.figure(figsize=[50,50])
        # plt.suptitle(
        #     f'DICE~{test_model1["f1_score"][patients]:.8f}\n'
        #     f'Avg~{uncertainty_map.mean():.8f} - Max~{uncertainty_map.max():.8f}\n'
        #     f'Slice Avg~{uncertainty_map[j, :, :].mean():.8f} Slice Max~{uncertainty_map[j, :, :].max():.8f}\n'
        # )

        # Plot PET channel with ground truth and prediction
        plt.subplot(plot_rows, plot_columns, 1)
        plt.title('PET/CT', fontsize=50)
        plt.contour(ground_truths[patients][j, :, :, 0], 1, levels=[0.5], colors='blue')
        plt.contour(mean_prediction[j, :, :, 0], 1, levels=[0.5], colors='red')
        plt.imshow(raw_images[patients][j, :, :, 1], 'gray')
        plt.axis('off')


        # Plot Error Region
        plt.subplot(plot_rows, plot_columns, 2)
        plt.title('Error region', fontsize=50)
        plt.imshow(raw_images[patients][j, :, :, 1], 'gray')
        plt.imshow(error_region[j, :, :], cmap=ListedColormap(['black', 'red']), alpha=0.5)
        plt.axis('off')


        # Plot Uncertainty in prediction
        plt.subplot(plot_rows, plot_columns, 3)
        plt.title('Uncertainty map', fontsize=50)
        plt.imshow(raw_images[patients][j, :, :, 1], 'gray')
        plt.imshow(_uncertainty_map[j, :, :], alpha=0.5)
        plt.axis('off')


        # Plot Uncertainty in PET/CT feature importance
        plt.subplot(plot_rows, plot_columns, 4)
        plt.title('Uncertainty in feature importance', fontsize=40)
        plt.imshow(uncertainty_feature[j, :, :, 0])
        plt.imshow(uncertainty_feature[j, :, :, 1], alpha=0.5)
        plt.axis('off')


        plt.subplots_adjust(wspace=0.1, hspace=0.1)


        # Save Images
        os.makedirs(
            os.path.dirname('/Volumes/Seagate-AA/Thesis/uncertainty_output/images_0_1/Patient{0}/'.format(patients)),
            exist_ok=True)
        plt.savefig(
            '/Volumes/Seagate-AA/Thesis/uncertainty_output/images_0_1/Patient{0}/Patient{0}_{1}.png'.format(patients, j),
            bbox_inches='tight')

    # Save animation
    # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
    #                                 repeat_delay=1000)
    # ani.save('/Volumes/Seagate-AA/Thesis/uncertainty_output/Patient_{0}.mp4'.format(patients))

# df = pd.DataFrame(new_df)
# df.to_csv("/Volumes/Seagate-AA/Thesis/uncertainty_output/patient_positive_uncertainty.csv", index=False)
