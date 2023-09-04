import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_generator_3Dpatch import Dataset_3D
from vox2vox import Generator, Discriminator, G_train, D_train
from posi_encoding import PositionalEncodingPermute1D

if torch.cuda.is_available():
    print(" -- GPU is available -- ")
if torch.distributed.is_available():
    print(" -- parallel calculation is available -- ")


def get_args_parser():
    parser = argparse.ArgumentParser('vox2vox training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1600, type=int)
    parser.add_argument('--channels', default=16, type=int)
    parser.add_argument('--lamb', default=100, type=int)
    # optimizer setting
    parser.add_argument('--lr_G', default=1e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    # parallel setting
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--nproc_per_node', default=4, type=int)
    return parser
    
def Position_encoding_1D(coord):
    p_enc_per_1d_s = PositionalEncodingPermute1D(10)
    p_enc_per_1d_xy = PositionalEncodingPermute1D(10)

    array = np.zeros((len(coord),10,30))
    array = p_enc_per_1d_s(torch.from_numpy(array))
    array_s = torch.zeros(len(coord),10,1)

    for i, index in enumerate(coord[:,0]):
        array_s[i,:,0] = array[i,:,index]  

    array = np.zeros((len(coord),10,128))
    array = p_enc_per_1d_xy(torch.from_numpy(array))
    array_x = torch.zeros(len(coord),10,1)
    for i, index in enumerate(coord[:,1]):
        array_x[i,:,0] = array[i,:,index]  

    array_y = torch.zeros((len(coord),10,1))
    for i, index in enumerate(coord[:,2]):
        array_y[i,:,0] = array[i,:,index] 

    array = torch.cat((array_s, array_x, array_y), dim=1)
    array = array.unsqueeze(2)
    array = array.unsqueeze(3)

    return array

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
        
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('num_gpus=', num_gpus)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    training_set = Dataset_3D()
    sampler = torch.utils.data.distributed.DistributedSampler(training_set)
    train_loader = DataLoader(training_set, 
                              batch_size=args.batch_size, 
                              num_workers=0, drop_last=True 
                              , sampler=sampler
                              )
    G = Generator(1, 1, args.channels).cuda()
    G = nn.SyncBatchNorm.convert_sync_batchnorm(G).cuda()
    G = DistributedDataParallel(G, broadcast_buffers=False)
    D = Discriminator(1, 1, args.channels).cuda()
    D = nn.SyncBatchNorm.convert_sync_batchnorm(D).cuda()
    D = DistributedDataParallel(D, broadcast_buffers=False)

    criterion_DiceBCE = DiceBCELoss()
    criterion_Dice = DiceLoss()
    criterion_BCE = nn.BCELoss()
    criterion_L1 = nn.MSELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.9, 0.999))

    G.train()
    D.train()
    D_Loss, G_Loss = [], []
    n_minibatch = (len(training_set)//args.batch_size)
    """
    # saving test
    torch.save(G, '/work/310613060/two_stage/Position_Encoding/checkpoints/generator_0.pth')
    # torch.save(D, './checkpoints/discriminator_0.pth')
    testing_image = torch.unsqueeze(torch.tensor(training_set[100][0]), 0)
    g = G(testing_image.cuda().float())
    save_image(g[0, 0, 32, :, :], '/work/310613060/two_stage/Position_Encoding/checkpoints/G0.png')
    """
    # start training
    for epoch in range(args.epochs):
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0
        sampler.set_epoch(epoch)
        for inputs, targets, coords in train_loader:
            batch += 1
            coord = Position_encoding_1D(coords)
            D_losses.append(D_train(D, G, inputs, targets, coord, criterion_BCE, optimizer_D, args.lamb))
            G_losses.append(G_train(D, G, inputs, targets, coord, criterion_Dice, criterion_L1, optimizer_G))
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            print('epoch:[%d / %d] batch:[%d / %d] loss_d= %.3f  loss_g= %.3f' % 
                 (epoch + 1, args.epochs, batch, n_minibatch, d_l, g_l))
            torch.cuda.empty_cache()
        D_Loss.append(d_l)
        G_Loss.append(g_l)
        if (epoch+1) % 10 == 0:
            torch.save(G, '/work/310613060/two_stage/Position_Encoding/checkpoints_PG_128x128/generator_'+str(epoch+1)+'.pth')
            # torch.save(D, './checkpoints/discriminator_'+str(epoch+1)+'.pth')
            testing_image = torch.unsqueeze(torch.tensor(training_set[100][0]), 0)
            print(testing_image.shape)
            coord = torch.unsqueeze(torch.tensor(training_set[100][2]), 0)
            print(coord.shape)
            coord = Position_encoding_1D(coord)
            g = G(testing_image.cuda().float(), coord.cuda().float())
            save_image(g[0, 0, 32, :, :], '/work/310613060/two_stage/Position_Encoding/checkpoints_PG_128x128/G'+str(epoch+1)+'.png')

        # generate new dataset
        training_set = Dataset_3D()
        sampler = torch.utils.data.distributed.DistributedSampler(training_set)
        train_loader = DataLoader(training_set, 
                                batch_size=args.batch_size, 
                                num_workers=0, drop_last=True, 
                                sampler=sampler
                                )

    # training profile
    plt.plot(np.arange(args.epochs), D_Loss, label='Discriminator Losses')
    plt.plot(np.arange(args.epochs), np.array(G_Loss) / 100, label='Generator Losses / 100')
    plt.legend()
    plt.savefig('loss_PG_concate_channel_128x128.png')
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)