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


def make_dataframe(patient_id, patient_uncertainity_map):
    patient_score = {'patient': patient_id,
                     'mean_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].mean(),
                     'min_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].min(),
                     'max_uncertainty': patient_uncertainity_map[patient_uncertainity_map.nonzero()].max(),
                     'tumour_volume': ground_truths[patient_id].sum()}

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
for patients in tqdm(range(31, 32)):
    images = []
    for i in range(1, 21):
        with h5py.File(filepath.format(i), 'r') as f:
            images.append(f['mc_output'][patients])
    uncertainty_map = np.stack(images, axis=-1).std(axis=-1)
    mean_prediction = np.stack(images, axis=-1).mean(axis=-1)
    #make_dataframe(patients, uncertainty_map)


    # first check the shape (173, 191, 265) --> continue
    # ims = []
    for j in tqdm(range(173)):
        # im = plt.imshow(image[:, j, :], 'gray')
        # _uncertainty_map = uncertainty_map.copy()
        # _uncertainty_map[_uncertainty_map < np.mean(_uncertainty_map)] = np.nan
        plt.figure()
        plt.suptitle(
            f'DICE~{test_model1["f1_score"][patients]:.8f}\n'
            f'Avg~{uncertainty_map.mean():.8f} - Max~{uncertainty_map.max():.8f}\n'
            f'Slice Avg~{uncertainty_map[j, :, :].mean():.8f} Slice Max~{uncertainty_map[j, :, :].max():.8f}\n'
        )

        # Plot Ground truth and prediction
        plt.subplot(1, 2, 1)
        # plt.imshow(raw_images[patients][j, :, :, 0], 'gray')
        plt.contour(ground_truths[patients][j, :, :, 0], 1, levels=[0.5], colors='blue')
        plt.contour(images[0][j, :, :, 0], 1, levels=[0.5], colors='red')
        plt.contour(mean_prediction[j, :, :, 0], 1, levels=[0.5], colors='yellow')
        # im2 = plt.imshow(uncertainty_map[:, j, :])
        plt.imshow(uncertainty_map[j, :, :])

        # Plot PET channel
        plt.subplot(1, 2, 2)
        plt.imshow(raw_images[patients][j, :, :, 1], 'jet')

        # ims.append([im2])

        # Save Images
        os.makedirs(os.path.dirname('/Volumes/Seagate-AA/Thesis/uncertainty_output/images/Patient{0}/'.format(patients)),
                    exist_ok=True)
        plt.savefig(
            '/Volumes/Seagate-AA/Thesis/uncertainty_output/images/Patient{0}/Patient{0}_{1}.png'.format(patients, j))

    # Save animation
    # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
    #                                 repeat_delay=1000)
    # ani.save('/Volumes/Seagate-AA/Thesis/uncertainty_output/Patient_{0}.mp4'.format(patients))

# df = pd.DataFrame(new_df)
# df.to_csv("/Volumes/Seagate-AA/Thesis/uncertainty_output/patient_positive_uncertainty.csv", index=False)
