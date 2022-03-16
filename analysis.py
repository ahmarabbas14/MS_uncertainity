import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import matplotlib.animation as animation

filepath = '/Volumes/Seagate-AA/Thesis/data/mc_3d_unet_{0}.h5'
fig = plt.figure()
images = []
for i in range(10, 15):
    with h5py.File(filepath.format(i), 'r') as f:
        images.append(f['mc_output'][0])
uncertainty_map = np.stack(images, axis=-1).std(axis=-1)

# first check the shape (173, 191, 265) --> continue
with h5py.File(filepath.format(1), 'r') as f:
    image = f['mc_output'][0]
ims = []
for j in tqdm(range(191)):
    # plt.imshow(image[:, j, :], 'gray')
    # _uncertainty_map = uncertainty_map.copy()
    # _uncertainty_map[_uncertainty_map < np.mean(_uncertainty_map)] = np.nan
    im = plt.imshow(uncertainty_map[:, j, :])
    ims.append([im])

    # Save Images
    # plt.savefig('/Volumes/Seagate-AA/Thesis/uncertainty_output/Patient0_{0}.mp4'.format(j))

# Save animation
ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                repeat_delay=1000)
ani.save('/Volumes/Seagate-AA/Thesis/uncertainty_output/Patient_{0}.mp4'.format(0))
