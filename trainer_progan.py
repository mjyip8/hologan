import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import pytorch3d

from arch import G, D
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
latent_lambda = 25.


data_loader = load_data()

class HoloPGANTrainer:
    def __init__(self, data_loader):
        self.model = HoloPGAN()
        self.netD = self.model.getNetD()
        self.netG = self.nodel.getNetG()
        self.optimizerD = self.model.getOptimizerD()
        self.optimizerG = self.model.getOptimizerG()
        
        # TRAINING CONFIG STUFF
        self.maxEpochsAtScale = [2, 2] #[20, 30]
        self.imageScales = [64, 128]
        self.depthScales = [128, 64]
        self.alphaNJumps = [0, 4] #[0, 300]
        self.alphaSizeJumps = [0, 32]
        
    def updateDatasetForScale(scale):
        if (scale == 64):
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
            folder_name = "./data_faces"

            celeba_data = datasets.ImageFolder('./data_faces', transform=transform)
            data_loader = DataLoader(celeba_data,batch_size=batch_size,shuffle=True)
            return data_loader
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]) 
            celeba_data = datasets.ImageFolder("./data_faces_" + str(scale), transform=transform)

            data_loader = DataLoader(celeba_data,batch_size=batch_size,shuffle=True)
            return data_loader
        
        
    def train(self):
        n_scales = len(self.depthScales)

        for idx in range(len(self.maxEpochsAtScale)):
            scale = self.imageScales[idx]
            self.data_loader = self.updateDatasetForScale(scale)

            for epoch in range(self.maxEpochsAtScale[idx]):
                shiftIter = 0
if self.startIter > 0:
    shiftIter = self.startIter
    self.startIter = 0
                
shiftAlpha = 0
while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
shiftAlpha += 1
                
                print("TRAINING EPOCH " + str(epoch)):
                pbar = tqdm(range(len(data_loader)))

                gen_loss = []
                disc_loss = []
                for i, data in enumerate(data_loader):
                    if (data[0].shape[0] == self.batch_size):
                        g_loss, d_loss, batch_z, fake_images, data = self.train_on_batch(data)
                        gen_loss.append(g_loss)
                        disc_loss.append(d_loss)

                    if i % display_step == 0:
                        show_tensor_images(fake_images.detach(), num_images=self.batch_size)
                        show_tensor_images(data.detach(), self.batch_size)
                    if i == np.ceil(float(len(self.data_loader.dataset)) / batch_size) - 1:
                        generate_rotation_imgs(gen, self.batch_size, batch_z, epoch + 1)
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
                torch.save(dd, "%s/%i.pkl" % ("./holopgan_models/models_" + str(scale), epoch+1, scale))
            self.model.addScale(depthScales[idx + 1])


    # TRAINING FOR ONE EPOCH
    def train_on_batch(self, data):
        cur_step = 0
        data = data[0]

        batch_z = sampling_Z()
        view_in = generate_random_rotation_translation(self.batch_size)
        data = data.to(device)

        # UPDATE DISCRIMINATOR FIRST
        self.disc_opt.zero_grad()
        fake_images = self.model.netG(batch_z, view_in)
        disc_pred_fake, disc_logits_fake, q_fake = self.netD(fake_images.detach())
        disc_pred_real, disc_logits_real, _ = self.netD(data)
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