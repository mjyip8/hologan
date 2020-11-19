import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST
import matplotlib.pyplot as plt
import pytorch3d

from hologan_arch import G, D
from utils import sampling_Z, weights_init, generate_random_rotation_translation, generate_rotation_imgs, show_tensor_images
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
latent_lambda = 12.

class HoloGAN:
    def __init__(self, data_loader):
        self.gen = G(im_chan=image_channels).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.disc = D(im_chan=image_channels).to(device) 
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.gen = self.gen.apply(weights_init)
        self.disc = self.disc.apply(weights_init)
        self.data_loader = data_loader
        self.batch_size = 32
        self.gen_losses = []
        self.disc_losses = []

    def train(self):
        # Training loop here
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        display_step = 5000
        save_step = 5
        
        # WHOLE TRAINING PROCESS
        torch.cuda.empty_cache()
        for epoch in range(num_epochs):
            pbar = tqdm(range(len(self.data_loader)))
            gen_loss = []
            disc_loss = []
            for i, data in enumerate(self.data_loader):
                if (data[0].shape[0] == self.batch_size):
                    g_loss, d_loss, batch_z, fake_images, data = self.train_on_batch(data)
                    gen_loss.append(g_loss)
                    disc_loss.append(d_loss)

                    if i % display_step == 0:
                        show_tensor_images(fake_images.detach(), "fake_" + str(epoch) + str(i) + ".png", num_images=self.batch_size)
                        show_tensor_images(data.detach(), "real_" + str(epoch) + str(i) + ".png", self.batch_size)
                    if i == np.ceil(float(len(self.data_loader.dataset)) / self.batch_size) - 2:
                        generate_rotation_imgs(self.gen, self.batch_size, batch_z, epoch + 1)
                    pbar.update(1)
            
            pbar.close()
            if (epoch+1) % save_step == 0:
                self.gen_losses.append(np.mean(gen_loss))
                self.disc_losses.append(np.mean(disc_loss))
                print(f"Epoch {epoch + 1}: gen_loss: {self.gen_losses[-1]}; disc_loss: {self.disc_losses[-1]}")
                # SAVE MODEL HERE
                dd = {}
                dd['g'] = self.gen.state_dict()
                dd['d'] = self.disc.state_dict()
                dd['optim_gen'] = self.gen_opt.state_dict()
                dd['optim_gen'] = self.disc_opt.state_dict()
                dd['epoch'] = epoch + 1
                torch.save(dd, "%s/%i.pkl" % ("./models", epoch+1))

    # TRAINING FOR ONE EPOCH
    def train_on_batch(self, data):
        cur_step = 0
        data = data[0]

        batch_z = sampling_Z()
        view_in = generate_random_rotation_translation(self.batch_size)
        data = data.to(device)

        # UPDATE DISCRIMINATOR FIRST
        self.disc_opt.zero_grad()
        fake_images = self.gen(batch_z, view_in)
        disc_pred_fake, disc_logits_fake, q_fake = self.disc(fake_images.detach())
        disc_pred_real, disc_logits_real, _ = self.disc(data)
        
        # injecting some randomness 
        fake_labels = torch.zeros_like(disc_logits_fake) 
        real_labels = torch.ones_like(disc_logits_real)     
        if random.random() < .15:  
            fake_labels = torch.ones_like(disc_logits_fake) 
            real_labels = torch.zeros_like(disc_logits_real)
        disc_fake_loss = nn.functional.binary_cross_entropy_with_logits(disc_logits_fake, fake_labels)
        disc_real_loss = torch.nn.functional.binary_cross_entropy_with_logits(disc_logits_real, real_labels) 
        q_loss = latent_lambda * torch.mean(torch.square(batch_z - q_fake))
        disc_loss = disc_fake_loss + disc_real_loss + q_loss
        disc_loss.backward()
        self.disc_opt.step()

        # UPDATE GENERATOR
        self.gen_opt.zero_grad()
        fake_images = self.gen(batch_z, view_in).cuda()
        d_pred_fake, disc_logits_fake, q_fake = self.disc(fake_images.detach())
        gen_loss = nn.functional.binary_cross_entropy_with_logits(disc_logits_fake, torch.ones_like(disc_logits_fake))      
        q_loss = latent_lambda * torch.mean(torch.square(batch_z - q_fake))
        gen_loss = gen_loss + q_loss
        gen_loss.backward()
        self.gen_opt.step()

        return gen_loss.item(), disc_loss.item(), batch_z, fake_images, data