import torch
import torch.nn as nn
import numpy as np


# 3D version of UnetGenerator
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        """
        :param in_ch: input channel
        :param out_ch: output channel
        :param ngf: number of generator's first conv filters
        """
        super(Generator, self).__init__()

        # U-Net encoder
        # each layer, image size/=2
        # 64 * 64
        self.en1 = nn.Sequential(
            nn.Conv3d(in_ch, ngf, kernel_size=4, stride=2, padding=1),
        )
        # 32 * 32 * 32
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 2)
        )
        # 16 * 16 * 16
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 4)
        )
        # 8 * 8 * 8
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 8)
        )
        # 4 * 4 * 4
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 8)
        )
        # 2 * 2 * 2
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
        )

        # U-Net decoder
        # skip-connect: output of previous layer+symmetric conv layer
        # 1 * 1 * 1（input）
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 2 * 2 * 2
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 4 * 4 * 4
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 4),
            nn.Dropout(p=0.5)
        )
        # 8 * 8 * 8
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 2),
            nn.Dropout(p=0.5)
        )
        # 16 * 16 * 16
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf),
            nn.Dropout(p=0.5)
        )
        # 32 * 32 * 32
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ngf * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, X):
        """
        :param X: input 3D array
        :return: output of discriminator
        """
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)

        # Decoder
        de1_out = self.de1(en6_out)
        de1_cat = torch.cat([de1_out, en5_out], dim=1)  # cat by channel
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en4_out], 1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en3_out], 1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en2_out], 1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en1_out], 1)
        de6_out = self.de6(de5_cat)
        return de6_out


# PatchGAN
class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, ndf=64):
        """
        :param in_ch: input channel
        :param ndf: number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()

        # 64 * 64 * 64（input）
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_ch + out_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32 * 32
        self.layer2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 16 * 16 * 16
        self.layer3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8 * 8 * 8
        self.layer4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 7 * 7 * 7
        self.layer5 = nn.Sequential(
            nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 6 * 6 * 6（output size）

    def forward(self, X):
        """
        :param X: input image
        :return: output of discriminator
        """
        layer1_out = self.layer1(X)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        return layer5_out


# training discrimimator
def D_train(D: Discriminator, G: Generator, X, Y, BCELoss, optimizer_D, device='cuda:0'):
    """
    :param D: Discriminator
    :param G: Generator
    :param X: input images
    :param Y: target images
    :param BCELoss: loss function
    :param optimizer_D: discriminator optimizer
    :return: loss of discriminator
    """
    x = X.cuda().float() 
    y = Y.cuda().float() 
    xy = torch.cat([x, y], dim=1)  # stack on dim=1
    D.zero_grad()
    # real image
    D_output_r = D(xy).squeeze()
    soft_target = np.random.uniform(low=0.9, high=1.0, size=D_output_r.shape)
    D_real_loss = BCELoss(D_output_r, torch.from_numpy(soft_target).cuda().float())
    # fake image
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.shape).cuda())
    # back propagation
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


# training generator
def G_train(D: Discriminator, G: Generator, X, Y, DiceLoss, L1, optimizer_G, lamb=100, device='cuda:0'):
    """
    :param D: Discriminator
    :param G: Generator
    :param X: input images
    :param Y: target images
    :param BCELoss: loss function
    :param L1: l1 regularization
    :param optimizer_G: generator optimizer
    :param lamb: weight of l1 regularization
    :return: loss of generator
    """
    x = X.cuda().float() 
    y = Y.cuda().float()   

    G.zero_grad()
    # fake image
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    G_Dice_loss = DiceLoss(D_output_f, torch.ones(D_output_f.size()).cuda())
    G_L1_Loss = L1(G_output, y)
    # back propagation
    G_loss = G_Dice_loss + lamb * G_L1_Loss
    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item()