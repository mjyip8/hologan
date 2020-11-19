import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from torchvision import datasets
import matplotlib.pyplot as plt
from hologan_arch import G, D
from progan_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d

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
class PG(nn.Module):
    def __init__(self, im_chan=image_channels, depthScale = depthScale0):
        super(ProG, self).__init__()
        self.gf_dim = gf_dim

        batch_size = 32
        self.scalesDepth = [depthScale0]
        self.gen = G(im_chan=im_chan, default=False)
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        # Initialize the scale 0
        self.initFormatLayer(dimLatent)
        self.dimOutput = dimOutput
        self.groupScale0 = GenBlock2d(depthScale0, depthScale0, use_activation = False)
        self.toRGBLayers.append(GenBlock2d(depthScale0, output_channels, isLastLayer = True))

        self.alpha = 0
        self.leakyRelu = torch.nn.LeakyReLU()
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()
        self.generationActivation = nn.Tanh()
        self.depthScale0 = depthScale0

    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def addScale(self, depthNewScale):
        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale,
                                                    depthNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
                                                    3, padding=1,
                                                    equalized=True,
                                                    initBiasToZero=self.initBiasToZero))
        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                                                self.dimOutput,
                                                1,
                                                equalized=True,
                                                initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        self.alpha = alpha

    
    def forward(self, z, in_view):
        # Not sure whether we use block0 or not
        x = self.gen(z, in_view)  

        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            x = self.leakyRelu(convLayer(x))

        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](x)
            y = Upscale2d(y)

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            x = Upscale2d(x)
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        return x


class PD(nn.Module):
    def __init__(self, disc, depthScale0):
        super(ProD, self).__init__()
        
        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()

        # Initialize the last layer
        self.decisionLayer = D(im_chan=3, default=False)

        # Layer 0
        self.groupScaleZero = DBlock(depthScale0, 256)
        self.fromRGBLayers.append(DBlock(3, depthScale0))

        # Initalize the upscaling parameters
        self.alpha = 0
        self.leakyRelu = torch.nn.LeakyReLU()

    def addScale(self, depthNewScale):

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        self.scaleLayers.append(DBlock(depthNewScale, depthNewScale))
        self.fromRGBLayers.append(DBlock(3, depthNewScale))

    def setNewAlpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, z):
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.fromRGBLayers[- 2](y)
        x = self.fromRGBLayers[-1](x)

        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2
        for layer in reversed(self.scaleLayers):
            x = layer(x)
            x = nn.AvgPool2d((2, 2))(x)
            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1-self.alpha) * x

            shift -= 1

        return self.decisionLayer(x)
