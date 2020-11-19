import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import pytorch3d

from holoprogan_arch import PG, PD
from torch.utils.data import DataLoader
import random
import numpy as np
import math

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()
num_epochs = 200
z_dim = 128
lr = 0.0001
image_size = 64
image_channels = 3
beta_1 = 0.5
beta_2 = 0.999
latent_lambda = 0.

class HoloPGAN:
    def __init__(self, depthScale0=128):
        self.batch_size = 32
        self.gen_losses = []
        self.disc_losses = []
        self.alpha = 0
        self.depthOtherScales = []
        self.gnet_orig = G(im_chan=image_channels, depthScale = depthScale).to(device)
        self.gnet_orig.block

        self.gnet_orig = self.gnet_orig.apply(weights_init)

        self.dnet_orig = D(im_chan=image_channels, depthScale = depthScale0).to(device) 
        self.dnet_orig = self.dnet_orig.apply(weights_init)
    
    def getNetG(self):
        gnet = PG(im_chan=image_channels, depthScale = depthScale0).to(device)

        for depth in self.depthOtherScales:
            gnet.addScale(depth)

        if self.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):
        dnet = PD(im_chan=image_channels, depthScale = depthScale0).to(device) 
        self.disc = self.disc.apply(weights_init)

        for depth in self.depthOtherScales:
            dnet.addScale(depth)

        if self.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)
            
        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.dnet_orig.module.parameters()),
                          betas=[0, 0.99], .0001)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.gnet_orig.module.parameters()),
                          betas=[0, 0.99], .0001)

    def addScale(self, depthNewScale):
        self.netG = self.gnet_orig.module
        self.netD = self.gnet_orig.module

        self.netG.addScale(depthNewScale)
        self.netD.addScale(depthNewScale)

        self.depthOtherScales.append(depthNewScale)

        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        r"""
        Update the blending factor alpha.
        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.gnet_orig.module.setNewAlpha(newAlpha)
        self.dnet_orig.module.setNewAlpha(newAlpha)

        self.alpha = newAlpha

    def getSize(self):
        r"""
        Get output image size (W, H)
        """
        return self.gnet_orig.module.getOutputSize()