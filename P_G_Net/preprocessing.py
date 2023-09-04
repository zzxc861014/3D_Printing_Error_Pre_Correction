import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit


@jit(nopython = True)
def boundary_filter(img):
    boundary = img.copy()
    index = np.where(img == 1)
    for i in range(len(index[0])):
        x, y = index[0][i], index[1][i]
        if np.sum(img[x-1:x+2, y-1:y+2]) == 9:
            boundary[x, y] = 0
    return boundary

@jit(nopython = True)
def patchify(img, boundary, sampling):
    b_index = np.where(boundary == 1)
    patchs = np.zeros((len(sampling), 128, 128))
    for i, sample in enumerate(sampling):
        x, y = b_index[0][sample], b_index[1][sample]
        patchs[i] = img[x-64:x+64, y-64:y+64]
    return patchs


def random_patchs(img):
    prob = np.clip(np.sum(img)/1e6, 0.01, 1)
    num_patchs = int(100*prob)
    boundary = boundary_filter(img)
    image_show(boundary)
    if np.max(boundary) == 0:
        return None
    sampling = np.random.choice(a=np.sum(boundary), size=num_patchs, replace=False, p=None)
    patchs = patchify(img, boundary, sampling)
    # image_show(patchs)
    return patchs


def image_show(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        ax = plt.gca()
        ax.add_patch(patches.Rectangle((1465, 1046), 128, 128, facecolor='red', fill=True))
        plt.axis('off')
        plt.show()
        plt.close()
    else:
        for i in range(img.shape[0]):
            plt.imshow(img[i, : ,:], cmap='gray')
            plt.axis('off')
            plt.show()
            plt.close()


def main():
    path = 'raw_data'
    img_folders = os.listdir(path)

    for folder in img_folders:
        folder_path = os.path.join(path, folder)
        img_paths = [os.path.join(folder_path, _) for _ in os.listdir(folder_path) if _.endswith('.png')]
        npy_file = np.zeros((len(img_paths), 2000, 2000))
        patch_id = 0
        print(len(img_paths))
        for img_path in img_paths:
            img = Image.open(img_path)
            img = np.clip(np.array(img), 0, 1) # binary
            # image_show(img)
            patchs = random_patchs(img)
            if patchs is None:
                continue
            if patch_id + len(patchs) < 10000:
                npy_file[patch_id:patch_id+len(patchs), :, :] = patchs
                patch_id += len(patchs)
            else:
                break
        valid_slice = int(np.sum(np.max(npy_file.reshape(10000, -1), axis=1)))
        print(valid_slice)
        np.save(folder+'.npy', npy_file[:valid_slice, :, :].astype(bool))


if __name__ == '__main__':
    main()