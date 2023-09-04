import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import torch
from numba import jit

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


def full_braces(folder_path='/work/310613060/Model_Slice_2048_2048/_0017_P/'):
    img_names = os.listdir(folder_path)
    img_names.sort(key=lambda x:int(x[:-4])) # sorting images
    img_paths = [os.path.join(folder_path, _) for _ in img_names]

    array_3d = np.zeros((len(img_paths), 2048, 2048), dtype=np.int8)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img = np.clip(np.array(img), 0, 1) # binary
        array_3d[i] = img.astype('int8')
    boundary_mask = boundary_filter(array_3d)
    n_pixels = np.sum(boundary_mask)
    print('pixels at boundary:', n_pixels)
    return array_3d, boundary_mask, n_pixels

@jit(nopython = True)
def patchify(img, b_index, sampling):
    padding = np.zeros((32, 2048, 2048), dtype=np.int8)
    patches = np.zeros((len(sampling), 64, 64, 64), dtype=np.int8)
    img = np.concatenate((padding, img, padding), axis=0)
    for i, sample in enumerate(sampling):
        x, y, z = b_index[0][sample]+32, b_index[1][sample], b_index[2][sample]
        patches[i] = img[x-32:x+32, y-32:y+32, z-32:z+32]
    return np.expand_dims(patches, axis=1)

@jit(nopython = True)
def reconstruction(reconstructed_braces, patches, b_index, mini_set):
    for i, sample in enumerate(mini_set):
        x, y, z = b_index[0][sample]+32, b_index[1][sample], b_index[2][sample]
        local_patch = reconstructed_braces[x-32:x+32, 0, y-32:y+32, z-32:z+32]
        reconstructed_braces[x-32:x+32, 0, y-32:y+32, z-32:z+32] = local_patch + patches[i, 0]
        reconstructed_braces[x-32:x+32, 1, y-32:y+32, z-32:z+32] += 1
    return reconstructed_braces

@jit(nopython = True)
def boundary_remove(reconstructed_braces, b_index, sample):
    for i, point in enumerate(sample):
        x, y, z = b_index[0][point]+32, b_index[1][point], b_index[2][point]
        reconstructed_braces[x-32:x+32, y-32:y+32, z-32:z+32] = 0
    return reconstructed_braces

def main():
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('/work/310613060/two_stage/Cross_Validation_Dice/P_G/checkpoints16-17/generator_1500.pth')
    model.to(device)
    model.eval()
    threshold = 0.5

    braces, boundary, n_pixels = full_braces()
    b_index = np.where(boundary == 1)
    del boundary

    batch_size = 30
    interval = 100
    samples = np.arange((n_pixels-1)//interval)
    print(len(samples)//batch_size+1)
    reconstructed_braces = np.zeros((braces.shape[0]+64, 2, braces.shape[1], braces.shape[2]), dtype=np.int8)
    reconstructed_braces[32:-32, 0, :, :] = braces
    reconstructed_braces = boundary_remove(reconstructed_braces, b_index, np.arange((n_pixels-1)//interval)*interval)
    print('cleanse finished')

    start_time = time.time()
    with torch.no_grad():
        for batch in range(len(samples)//batch_size+1):
            mini_set = samples[batch*batch_size:(batch+1)*batch_size]*interval
            patches = patchify(braces, b_index, mini_set)
            patches = torch.from_numpy(patches).cuda().float()
            print(patches.shape)
            output = model(patches)
            output = output.cpu().numpy()
            output[output>=threshold] = 1
            output[output<threshold] = 0
            output.astype(int)
            reconstructed_braces = reconstruction(reconstructed_braces, output, b_index, mini_set)
    end_time = time.time()
    consumption = end_time-start_time
    print('it takes %d mins %d secs'%(consumption//60, consumption%60))

    print('reconsturction finished; voting procedure starts')
    img, votes = reconstructed_braces[32:-32, 0, :, :], reconstructed_braces[32:-32, 1, :, :] 
    print(np.max(votes), np.min(votes[votes>1]))
    del reconstructed_braces
    votes[votes==0] = 1
    img = np.divide(img, votes)
    img[img>=0.5] = 1
    img[img<0.5] = 0
    for i in range(len(img)):
        im = Image.fromarray(img[i]*255)
        im = im.convert('L')
        im.save('/work/310613060/two_stage/Cross_Validation_Dice/Master_thesis_image/P_G/_0017_/'+str(i).zfill(3)+'.png')
        # plt.imshow(img[i], 'gray')
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.savefig('./testing_image/'+str(i).zfill(3)+'.png',bbox_inches='tight',pad_inches=0.0)
        # plt.close()


if __name__ == '__main__':
    main()

