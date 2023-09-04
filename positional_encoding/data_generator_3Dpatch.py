import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit
from torch.utils.data import Dataset


class Dataset_3D(Dataset):
    def __init__(self, ):
        self.input_images, self.target_images, self.coord = build_dataset()
        self.input_images = np.expand_dims(self.input_images, axis=1)
        self.target_images = np.expand_dims(self.target_images, axis=1)

    def __getitem__(self, index):
        inputs, target, coord = self.input_images[index], self.target_images[index], self.coord[index]
        return inputs, target, coord

    def __len__(self):
        return len(self.input_images)


@jit(nopython = True)
def boundary_filter(img):
    # only the boundary will remain
    boundary = np.zeros(img.shape, dtype=np.int8)
    for s in range(len(img)):
        index = np.where(img[s] == 1)
        for i in range(len(index[0])):
            x, y = index[0][i], index[1][i]
            if np.sum(img[s, x-1:x+2, y-1:y+2]) < 9: # check the 3*3 kernel
                boundary[s, x, y] = 1
    return boundary


@jit(nopython = True)
def patchify(img, boundary, sampling):
    b_index = np.where(boundary == 1)
    padding = np.zeros((32, 2048, 2048), dtype=np.int8)
    patches = np.zeros((len(sampling), 64, 64, 64), dtype=np.int8)
    img = np.concatenate((padding, img, padding), axis=0)
    coord = []
    for i, sample in enumerate(sampling):
        s, x, y = b_index[0][sample]+32, b_index[1][sample], b_index[2][sample]
        patches[i] = img[s-32:s+32, x-32:x+32, y-32:y+32]
        coord.append([s//15, x//16, y//16])
    return patches, coord


def build_dataset(path='/work/310613060/two_stage/Position_Encoding/Align_Model'):
    # img_folders = os.listdir(path)

    # dice for dataset
    # dataset = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
    dataset = [4,11,16]
    dataset_seed = random.sample(dataset, k=1)
    dataset_seed = dataset_seed[0]
    # dataset_seed = random.randint(0, 5)
    # dataset_seed = 21
    
    if dataset_seed == 0:
        img_folders = ['_0001_P', '_0001_G']
    elif dataset_seed == 1:
        img_folders = ['_0002_P', '_0002_G'] 
    elif dataset_seed == 2:
        img_folders = ['_0003_P', '_0003_G']
    elif dataset_seed == 3:
        img_folders = ['_0004_P', '_0004_G']
    elif dataset_seed == 4:
        img_folders = ['_0005_P', '_0005_G'] 
    elif dataset_seed == 5:
        img_folders = ['_0006_P', '_0006_G']
    elif dataset_seed == 6:
        img_folders = ['_0007_P', '_0007_G']
    elif dataset_seed == 7:
        img_folders = ['_0008_P', '_0008_G'] 
    elif dataset_seed == 8:
        img_folders = ['_0009_P', '_0009_G']
    elif dataset_seed == 9:
        img_folders = ['_0010_P', '_0010_G']
    elif dataset_seed == 10:
        img_folders = ['_0011_P', '_0011_G'] 
    elif dataset_seed == 11:
        img_folders = ['_0012_P', '_0012_G']
    elif dataset_seed == 12:
        img_folders = ['_0013_P', '_0013_G']
    elif dataset_seed == 13:
        img_folders = ['_0014_P', '_0014_G']
    elif dataset_seed == 14:
        img_folders = ['_0015_P', '_0015_G']
    elif dataset_seed == 15:
        img_folders = ['_0016_P', '_0016_G']
    elif dataset_seed == 16:
        img_folders = ['_0017_P', '_0017_G']

    else:
        pass
    # deal with input images first
    print(img_folders)
    folder_path = os.path.join(path, img_folders[0])
    img_names = os.listdir(folder_path)
    img_names.sort(key=lambda x:int(x[:-4])) # sorting images
    img_paths = [os.path.join(folder_path, _) for _ in img_names]

    array_3d = np.zeros((len(img_paths), 2048, 2048), dtype=np.int8)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img = np.clip(np.array(img), 0, 1) # binary
        array_3d[i] = img.astype('int8')
    boundary_mask = boundary_filter(array_3d)

    # slice-pixels distribution
    # cdf = boundary_mask.sum(axis=-1).sum(axis=-1)
    # plt.plot(np.arange(boundary_mask.shape[0]), cdf)
    # plt.xlabel("slice", fontsize=16)
    # plt.ylabel("num. pixels", fontsize=16)
    # plt.show()
    # plt.close()

    # uniform sampling
    # sampling = np.random.choice(a=np.sum(boundary_mask), size=10000, replace=False, p=None)
    sampling_low = np.random.choice(a=np.sum(boundary_mask[:200]), size=5000, replace=False, p=None)

    # sampling more in the lower slice
    sampling_high = np.random.choice(a=np.sum(boundary_mask[200:]), size=5000, replace=False, p=None)
    sampling_high = sampling_high + np.sum(boundary_mask[:200])
    sampling = np.concatenate([sampling_low, sampling_high], axis=0)
    input_patches, coord = patchify(array_3d, boundary_mask, sampling) # (10000, 64, 64, 64)
    # image_show(input_patches[5002])

    # deal with target images
    folder_path = os.path.join(path, img_folders[1])
    img_names = os.listdir(folder_path)
    img_names.sort(key=lambda x:int(x[:-4])) # sorting images
    img_paths = [os.path.join(folder_path, _) for _ in img_names]
    array_3d = np.zeros((len(img_paths), 2048, 2048), dtype=np.int8)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img = np.clip(np.array(img), 0, 1) # binary
        array_3d[i] = img.astype('int8')
    target_patches, coord = patchify(array_3d, boundary_mask, sampling) # (10000, 64, 64, 64)
    coord = np.array(coord)

    return input_patches, target_patches, coord


def image_show(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        # ax = plt.gca()
        # ax.add_patch(patches.Rectangle((1465, 1046), 128, 128, facecolor='red', fill=True))
        plt.axis('off')
        plt.show()
        plt.close()
    else:
        for i in range(img.shape[0]):
            plt.imshow(img[i, : ,:], cmap='gray')
            # plt.axis('off')
            plt.savefig(str(i)+'.png')
            # plt.show()
            plt.close()


# build_dataset()
# d_set = Dataset_3D()
# image_show(d_set[100][0][32])
# image_show(d_set[100][1][32])