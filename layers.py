import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import numpy as np
import math
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

# rendering components
import pytorch3d
from pytorch3d.renderer.cameras import (
    camera_position_from_spherical_angles,
    look_at_rotation   
)

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# 3D RIGID BODY TRARNSFORMATION
class RigidBodyTransformation(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, size = 64, new_size = 128, inplace: bool = False):
        super(RigidBodyTransformation, self).__init__()
        self.inplace = inplace
        self.size = size
        self.new_size = new_size
        self.batch_size = 32

    def forward(self, input: Tensor, view_in) -> Tensor:
        self.batch_size = input.shape[0]
        self.n_channels = input.shape[4]
        affine = self.rotate_around_grid_centroid(view_in)
        target = self.rotation_resampling(input, affine)
        return target

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def rotate_around_grid_centroid(self, view_in):
        # Calculating the transformation matrix first, Pytorch3d has right handed coordinate system
        azim = view_in[:,0]
        elev = view_in[:,1]
        scale = view_in[:,2]
        C = camera_position_from_spherical_angles(1., elev, azim - .5 * math.pi, degrees = False, device = device)
        R = look_at_rotation(C, device=device)
        mat = torch.eye(3, device=device)
        mat[0, 0] = -1.
        mat[2, 2] = -1.
        R = torch.matmul(R, mat)
        affine_mat = torch.zeros((R.shape[0], 3, 4), device=device)
        affine_mat[:, :,0:3] = R
        return affine_mat

    def rotation_resampling(self, input: Tensor, affine) -> Tensor:
        grid = nn.functional.affine_grid(affine, input.size())
        return nn.functional.grid_sample(input, grid, padding_mode='zeros')

# ADAIN AND PROJECTION UNIT
class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''

    def __init__(self, channels, w_dim, is2d = True):
        super().__init__()
        self.is2d = is2d
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True) if is2d else nn.InstanceNorm3d(channels, affine=True)
        self.style_scale_transform = nn.Sequential(nn.Linear(w_dim, channels), nn.ReLU())
        self.style_shift_transform = nn.Sequential(nn.Linear(w_dim, channels), nn.ReLU())

    def forward(self, image, z):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(z)
        style_shift = self.style_shift_transform(z)
        if self.is2d:
            style_scale = style_scale[:,:,None,None]
            style_shift = style_scale[:,:]
            transformed_image = normalized_image * style_scale + style_shift
        else:
            style_scale = style_scale[:,:,None,None,None]
            style_shift = style_shift[:,:,None,None,None]
            transformed_image = normalized_image * style_scale + style_shift
        return transformed_image

class ProjUnit(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
          nn.Conv2d(input_channels, output_channels, kernel_size=1),
          nn.LeakyReLU()
        )

    def forward(self, input):
        x = input.permute(0, 1, 4, 2, 3).reshape([input.shape[0], -1, input.shape[2], input.shape[3]])
        return self.net(x)

# GENERATOR 2D AND 3D BLOCKS
class GenBlock3d(nn.Module):
    def __init__(self, input_channels, output_channels, z_dim=128,
                 stride = 1, use_conv = True, use_adain = False):
        super(GenBlock3d, self).__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=0 if stride == 2 else 1)
        self.map = nn.Sequential(
            nn.Linear(z_dim, output_channels),
            nn.ReLU()
        )
        self.use_adain = use_adain
        self.adain = AdaIN(output_channels, output_channels, is2d = False)
        self.act = nn.LeakyReLU()
        self.use_conv = use_conv

    def forward(self, image, z):
        x = image 
        if self.use_conv:
            x = self.conv(image)
            x = x[:,:, 0:image.shape[2] * self.stride, 0:image.shape[3] * self.stride, 0:image.shape[4] * self.stride]
        if (self.use_adain):
            map = self.map(z)
            x = self.adain(x, map)
        return self.act(x)

class GenBlock2d(nn.Module):
    def __init__(self, input_channels, output_channels, z_dim=128, isLastLayer = False, use_activation=True):
        super(GenBlock2d, self).__init__()

        if (isLastLayer == False):
            self.stride = 2
            self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 4, stride = self.stride, padding = 0)
            self.adain = AdaIN(output_channels, z_dim, is2d=True)
            if (use_activation):
                self.act = nn.LeakyReLU()
            self.use_adain = True
        else:
            self.stride = 1
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 4, stride = self.stride, padding = 2)
            self.act = None
            if (use_activation):
                self.act = nn.Tanh()
            self.use_adain = False

    def forward(self, image, w = None):
        x = image 
        x = self.conv(image)
        x = x[:,:, 0:image.shape[2] * self.stride, 0:image.shape[3] * self.stride]
        x = self.adain(x, w) if self.use_adain else x
        return self.act(x) if self.act != None else x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding=1))
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.rl = nn.LeakyReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.instance_norm(x)
        return self.rl(x)

    
from torch.autograd import Variable

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module