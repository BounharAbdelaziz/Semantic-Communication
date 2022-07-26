import os
from tqdm import tqdm
import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid  # , save_image
from solver import Solver
from model import BetaVAE_H, BetaVAE_B
from dataset import return_data
import torchvision
from torchvision import datasets, transforms

state_dict = torch.load('E:/betavae/checkpoints/main/1000000')
# state_dict = torch.load("E:/betavae//1500000")
# state_dict = torch.load('F:/Beta-VAE-master/checkpoints/celeba_H_beta10_z32（SNR4）/1500000')
net = BetaVAE_H(z_dim=32, nc=3)
net.load_state_dict(state_dict['model_states']['net'])
encoder = net.encoder
decoder = net.decoder
device = torch.device('cpu')
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

# path = 'E:/data/CelebA'
data = "E:/betavae/CelebA/img_ali_celeba"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), ])
Celeba_dataset = datasets.ImageFolder(path, transform=transform)
data_loader = torch.utils.data.DataLoader(Celeba_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False,
                                          drop_last=True)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


x, label = iter(data_loader).next()
x = x.to(device)
# x = x[10]
distributions = encoder(x)
mu = distributions[:, :32]
logvar = distributions[:, 32:]
z = reparametrize(mu, logvar)
# z[0, 31] = -3
# 7 backgroundcolor
# 8 backgroundcolor
# 8 xingbie

print(z)
x_recon = decoder(z)
x_hat = F.sigmoid(x_recon)

viz = visdom.Visdom()
viz.images(x, nrow=4, win='x_new', opts=dict(title='x_new'))
viz.images(x_hat, nrow=4, win='x_hat20', opts=dict(title='21'))
