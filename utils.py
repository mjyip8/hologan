from itertools import chain
import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.datasets import CelebA, MNIST
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

model_dir = './models'
result_dir = './results'

# SAMPLING FUNCTIONS
def sampling_Z(batch_size = 32, z_dim = 128):
    return torch.tensor(np.random.normal(0, 1, (batch_size,z_dim)), dtype=torch.float, device=device)

def generate_random_rotation_translation(batch_size, elevation_low=-70, elevation_high=70, azimuth_low=-90, azimuth_high=90):
    params = np.zeros((batch_size, 6))
    column = np.arange(0, batch_size)
    azimuth = np.random.randint(azimuth_low, azimuth_high, (batch_size)).astype(np.float) * math.pi / 180.0
    temp = np.random.randint(elevation_low, elevation_high, (batch_size))
    elevation = temp.astype(np.float) * math.pi / 180.0
    params[column, 0] = azimuth
    params[column, 1] = elevation
    params[column, 2] = 1.0
    return params

def generate_rotation_imgs(gen, batch_size, z_batch, epoch, elevation_low=-70, elevation_high=70, azimuth_low=-90, azimuth_high=90, num = 5):
    elev = np.linspace(elevation_low, elevation_high, num)
    azim = np.linspace(azimuth_low, azimuth_high, num)
    idx = 0
    with torch.no_grad():
        for elevation in elev:
          for azimuth in azim:
            params = np.zeros((batch_size, 6)).astype(np.float32)
            column = np.arange(0, batch_size)
            params[column, 0] = azimuth.astype(np.float) * math.pi / 180
            params[column, 1] = elevation.astype(np.float) * math.pi / 180
            params[column, 2] = 1.0

            x_fake = gen(z_batch, params)
            save_image(x_fake,
                       "%s/epoch%d_{0:06d}.png".format(idx) % (result_dir), epoch)
            idx += 1

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def show_tensor_images(image_tensor, num_images=32, size=(1, image_size, image_size)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
