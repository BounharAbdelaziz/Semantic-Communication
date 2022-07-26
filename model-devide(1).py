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
state_dict = torch.load('1500000', map_location=device)   #保存和加载数据模型
net = BetaVAE_H(z_dim=32, nc=3)
net.load_state_dict(state_dict['model_states']['net'],False)  #用于将训练的参数权重加载到新的模型中
encoder = net.encoder
decoder = net.decoder

encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

path = 'E:\Ebetavae\CelebA'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), ])
Celeba_dataset = datasets.ImageFolder(path, transform=transform)
data_loader = torch.utils.data.DataLoader(Celeba_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                          drop_last=True)#更改输入数据个数
data_loader_1 = torch.utils.data.DataLoader(Celeba_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False,
                                          drop_last=True)#更改输入数据个数
# print(data_loader)
# print(data_loader_1)
def reparametrize(mu, logvar):#vae系数
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def awgn(input_user, SNR):
    snr = 10 ** (SNR / 10.0)
    Si = torch.mean(torch.square(input_user))   #返回带有输入元素平方的新张量input加入
    Ni = Si / snr
    noise = torch.randn(input_user.shape) * torch.sqrt(Ni)
    #print(input)
    return input_user + noise

def awgn2(input_interference, SNR):
    snr = 10 ** (SNR / 10.0)#噪声db转换
    Si = torch.mean(torch.square(input_interference))   #返回带有输入元素平方的新张量input加入
    Ni_inter1 = (1/2*Si/10*snr)
    noise = torch.randn(input_interference.shape) * torch.sqrt(Ni_inter1)
    #print(input)
    return input_interference

# def awgn3(input_interference, SNR):
#     snr = 10 ** (SNR / 10.0)#噪声db转换
#     Si = torch.mean(torch.square(input_interference))   #返回带有输入元素平方的新张量input加入
#     Ni_inter2 = (1/2*Si / 10*snr)
#     noise = torch.randn(input_interference.shape) * torch.sqrt(Ni_inter2)
#     #print(input)
#     return input_interference

x1, label = iter(data_loader).next()
x = x1.to(device)
distributions = encoder(x)#输入图片

x2, label = iter(data_loader_1).next()
x2 = x2.to(device)
distributions_2 = encoder(x2) #输入图片


mu = distributions[:, :32]#vae参数
logvar = distributions[:, 32:]
z = reparametrize(mu, logvar)                     #发射端1

mu = distributions_2[:, :32]                      #vae参数
logvar = distributions_2[:, 32:]
z1 = reparametrize(mu, logvar)                    #发射端2

z_moise = awgn(z,10)
z_interference1=awgn2(z1,10)                     #用户1来自用户2的干扰
z_moise =z_moise + z_interference1               #用户1接收端信号

z_moise_1 = awgn(z1,10)
z_interference2=awgn2(z,10)                     #用户2来自用户1的干扰
z_moise_1 = z_moise_1 + z_interference2          #用户2接收信号端


x_recon = decoder(z_moise)
     #解卷积
x_hat = F.sigmoid(x_recon)                     #激活函数

x_recon_1 = decoder(z_moise_1)   #解卷积
x_hat_1 = F.sigmoid(x_recon_1)   #激活函数


viz = visdom.Visdom()
viz.images(x, nrow=8, win='user1_input', opts=dict(title='user1_input'))
viz.images(x_hat, nrow=8, win=f'user1_output', opts=dict(title='user1_output'))
viz.images(x2, nrow=8, win='user2_input', opts=dict(title='user2_input'))
viz.images(x_hat_1, nrow=8, win=f'user2_output', opts=dict(title='user2_output'))

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


'''PSNR'''  #峰值信噪比，越大越好40左右最好
img_in = torch.squeeze(x)
img_out = torch.squeeze(x_hat)
img_in = img_in.numpy()
img_re = img_out.detach().numpy()
PSNR = psnr(img_in, img_re)

print(f'PSNR:{PSNR}')



img_in_1 = torch.squeeze(x2)
img_out_1 = torch.squeeze(x_hat_1)
img_in_1 = img_in_1.numpy()
img_re_1 = img_out_1.detach().numpy()
PSNR_1 = psnr(img_in_1, img_re_1)

print(f'PSNR:{PSNR_1}')