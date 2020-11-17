import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import CelebA, MNIST
import matplotlib.pyplot as plt
import pytorch3d

from layers import RigidBodyTransformation, AdaIN, ProjUnit, GenBlock3d, GenBlock2d, spectral_norm
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

# GENERATOR
class G(nn.Module):
    def __init__(self, im_chan=3, gf_dim=64):
      super(G, self).__init__()
      self.gf_dim = gf_dim

      batch_size = 32
      # starting constant
      xstart = (torch.randn((self.gf_dim * 8, 4, 4, 4), dtype=torch.float, device=device) - 0.5 ) / 0.5
      nn.init.xavier_uniform_(xstart.data, 1.)
      xstart = xstart.repeat(batch_size, 1, 1, 1, 1)

      self.xstart = nn.Parameter(xstart)
      self.xstart.requires_grad = True

      self.block0 = GenBlock3d(512, 512, use_conv = False, use_adain=True)
      self.block1 = GenBlock3d(self.gf_dim * 8, 256, stride = 2, use_adain=True)
      self.block2 = GenBlock3d(256, 128, stride = 2, use_adain=True)
      self.block3 = RigidBodyTransformation(16, 16)
      self.block4 = GenBlock3d(128, 64, stride = 1)
      self.block5 = GenBlock3d(64, 64, stride = 1)
      self.block6 = ProjUnit(16 * 64, 512)
      self.block7 = GenBlock2d(512, 256)
      self.block8 = GenBlock2d(256, 64)
      self.block9 = GenBlock2d(64, im_chan, isLastLayer = True)

    def forward(self, z, in_view):
      # Not sure whether we use block0 or not
      x = self.block0(self.xstart, z)
      x = self.block1(x, z)
      x = self.block2(x, z)
      # Rigid body 3d Transformation
      x = self.block3(x, in_view)
      # 3d conv blocks
      x = self.block4(x, z)
      x = self.block5(x, z) 
      # proj 
      x = self.block6(x)  
      # 2d conv blocks
      x = self.block7(x, z)  
      x = self.block8(x, z)  
      x = self.block9(x, z)  
      return x

class D(nn.Module):
  def __init__(self, im_chan=3, df_dim=64):
      super(D, self).__init__()
      self.df_dim = df_dim
      # self.block0 = nn.Sequential(nn.Conv2d(im_chan, df_dim, kernel_size = 3, stride = 1, padding=1),
      #                             nn.LeakyReLU())
      self.block1 = DBlock(3, 128)
      self.block2 = DBlock(128, 256)
      self.block3 = DBlock(256, 512)
      self.pred = nn.Sequential(nn.Flatten(),
                                  nn.Linear(8 * 8 * 512, 1))
      self.encode = nn.Sequential(nn.Flatten(),
                                  nn.Linear(8 * 8 * 512, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 128),
                                  nn.Tanh())

  def forward(self, input):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        # 3d conv blocks
        self.batch_size = input.shape[0]
        # h0 = self.block0(input)
        h1 = self.block1(input)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        #Returning logits to determine whether the images are real or fake
        h4 = self.pred(h3)
        encode = self.encode(h3)
        return torch.sigmoid(h4), h4, encode
