import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm.auto import tqdm
import random
import numpy as np
import math
import PIL.Image as Image
from hologan import HoloGAN


# CONSTANTS
batch_size = 32
crop_size = 150
re_size = 64


def load_data():
	offset_height = (218 - crop_size) // 2
	offset_width = (178 - crop_size) // 2
	crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Lambda(crop),
	     transforms.ToPILImage(),
	     transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
	     transforms.ToTensor(),
	     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

	celeba_data = datasets.ImageFolder('./data_faces', transform=transform)
	data_loader = DataLoader(celeba_data,batch_size=batch_size,shuffle=True)
	return data_loader


data_loader = load_data()
hg = HoloGAN(data_loader)
hg.train()
