# import os
# from tqdm import tqdm
import visdom
import torch
# import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid  # , save_image
# from solver import Solver
from model import BetaVAE_H, BetaVAE_B
# from dataset import return_data
# import torchvision
import cv2
import math
import numpy as np
import time
from torchvision import datasets, transforms

device = torch.device('cpu')
state_dict = torch.load('snr4', map_location=device)
net = BetaVAE_H(z_dim=32, nc=3)
net.load_state_dict(state_dict['model_states']['net'], False)
encoder = net.encoder
decoder = net.decoder

encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

path = 'E:\etavae\CelebA'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), ])
Celeba_dataset = datasets.ImageFolder(path, transform=transform)
data_loader = torch.utils.data.DataLoader(Celeba_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                          drop_last=True)



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def awgn(input, SNR):
    snr = 10 ** (SNR / 10.0)
    Si = torch.mean(torch.square(input))
    Ni = Si / snr
    noise = torch.randn(input.shape) * torch.sqrt(Ni)

    return input + noise

x1, label = iter(data_loader).next()
x = x1.to(device)
print(x.shape)
distributions = encoder(x)

mu = distributions[:, :32]
logvar = distributions[:, 32:]
z = reparametrize(mu, logvar)
# print(z)

z_moise = awgn(z,10)



x_recon = decoder(z_moise)
x_hat = F.sigmoid(x_recon)

viz = visdom.Visdom()
viz.images(x, nrow=8, win='x', opts=dict(title='x'))
viz.images(x_hat, nrow=8, win=f'x2', opts=dict(title='x2'))

# print(z)
# z[0, 2] = 3
# print(z)
# z[25] 负 白
# x_recon = decoder(z)
# x_hat = F.sigmoid(x_recon)
#
# viz = visdom.Visdom()
# viz.images(x, nrow=8, win='x', opts=dict(title='x'))
# viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='20'))


'''计算PSNR'''


def psnr(img1, img2):
    mse = np.mean(np.square(img1 - img2))
    return 20 * math.log10(1 / math.sqrt(mse))


'''PSNR'''
img_in = torch.squeeze(x)
img_out = torch.squeeze(x_hat)
img_in = img_in.numpy()
img_re = img_out.detach().numpy()
PSNR = psnr(img_in, img_re)

print(f'PSNR:{PSNR}')