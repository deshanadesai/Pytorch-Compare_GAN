from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from collections import defaultdict
import pdb
import torch.distributions as tdist

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm



def gmm_sampler(num_samples, num_mixtures, mean, cov, mix_coeffs):
    z = np.random.multinomial(num_samples, mix_coeffs)

    samples = np.zeros(shape=[num_samples, len(mean[0])])
    target = np.zeros(shape=[num_samples])
    
    i_start = 0
    data = []
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
            mean=np.array(mean)[i, :],
            cov=np.diag(np.array(cov)[i, :]),            
            size=z[i])
        
        target[i_start:i_end] = i

        for j in range(i_start,i_end):
            data.append({"x":samples[j],"class":target[j]})
        i_start = i_end
   
    return data


class SynthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_set_len, train_set):
        self.train_set_len = train_set_len
        self.train_set = train_set
    
    def __len__(self):
        return self.train_set_len  

    def __getitem__(self, idx):
        return self.train_set[idx]
        

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

try:
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.outf+"/images", exist_ok=True)
    os.makedirs(opt.outf+"/models", exist_ok=True)
except OSError:
    pass
    
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
 
# Define parameters of the dataset

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

inputdim=2
num_mixtures = 2
radius = 2.0
std = 0.02
thetas = np.linspace(0, 2 * np.pi, num_mixtures + 1)[:num_mixtures]
xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
mix_coeffs = [1./num_mixtures for i in range(num_mixtures)]
mean=tuple(zip(xs, ys))
cov=tuple([(std, std)] * num_mixtures)



batch_size = opt.batchSize
train_set_len = 10000
train_set = gmm_sampler(train_set_len, num_mixtures, mean, cov, mix_coeffs)

synthdataset = SynthDataset(train_set_len, train_set)
dataloader = DataLoader(synthdataset, batch_size=batch_size,
                        shuffle=True, num_workers=4) 

# change to whatever loss is needed.
adversarial_loss = torch.nn.CrossEntropyLoss()
adversarial_loss.to(device)

bce_loss = torch.nn.BCELoss()
bce_loss.to(device)

# KL
# def g_loss(d_fake_score):
#     g_loss_kl = -torch.mean(torch.exp(d_fake_score-1))
#     return g_loss_kl

# def d_loss(d_real,d_fake):
#     return -(torch.mean(d_real) - torch.mean(torch.exp(d_fake-1)))


# Reverse KL
def g_loss(d_fake_score):
    g_loss_kl = -torch.mean(-1-d_fake_score)
    return g_loss_kl

def d_loss(d_real,d_fake):
    return -(torch.mean(-torch.exp(d_real)) - torch.mean(-1-d_fake))




class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()   
        
        
        self.c1 = nn.Linear(opt.nz, 128)
        self.c1_bn = nn.BatchNorm1d(128)
        self.c1_relu = nn.ReLU(True)
        self.c2 = nn.Linear(128, 128)
        self.c2_bn = nn.BatchNorm1d(128)
        self.c2_relu = nn.ReLU(True)
        self.c3 = nn.Linear(128, 2)
        self.c3_tanh = nn.Tanh()    
        
    def forward(self, input_1):
        output = self.c1_relu(self.c1_bn(self.c1(input_1)))
        output = self.c2_relu(self.c2_bn(self.c2(output)))
        output = self.c3(output) # removed tanh because we dont want bounding
        return output

        
class D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(D, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.sigmoid

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)
      

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
netD = D(2,256,1)
netD.to(device)
netD.apply(weights_init)  
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))



netG = G()
netG.to(device)
netG.apply(weights_init)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

meanmatrix = np.matrix(mean)
import matplotlib.tri as tri
import matplotlib.mlab as mlab

def plot(points, title):
    plt.clf()
    
    for i, sample in enumerate(dataloader):
        inp = np.array(sample['x'])
        target = np.array(sample['class'])
        x = np.reshape(inp, [inp.shape[0], inputdim])
        plt.scatter(x[:,0],x[:,1],color='r',alpha=0.5,s=1)
    # set axes range
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)    
    
    
    
    
    
    xcoord = points[:, 0]
    ycoord = points[:, 1]
    zcoord = xcoord * np.exp(-xcoord**2 - ycoord**2)
#     ngridx = 600
#     ngridy = 600
#     xi = np.linspace(-3.1, 3.1, ngridx)
#     yi = np.linspace(-3.1, 3.1, ngridy)    
#     zi = mlab.griddata(xcoord, ycoord, zcoord, xi, yi, interp='linear')

    plt.scatter(points[:,0], points[:, 1], s=10, c='b')
#     triang = tri.Triangulation(xcoord, ycoord)
#     plt.tricontour(xcoord, ycoord, zcoord, 15, linewidths=0.5, colors='k')
#     plt.tricontourf(xcoord, ycoord, zcoord, 15,
#                 norm=plt.Normalize(vmax=abs(zi).max(), vmin=-abs(zi).max()))
#     plt.colorbar()
#     plt.scatter([meanmatrix[:, 0]], [meanmatrix[:, 1]], s=100, c='r', alpha=0.5)
    plt.title(title.replace("_"," "))
    plt.ylim(-3, 3)
    plt.xlim(-3, 3)
    plt.savefig(opt.outf+'/'+title)
    plt.close()




# plotting true dist:

plt.clf()
plt.grid(linestyle=':', linewidth='0.5', color='black')

color = ['r','g','b','y','k']
dims1 = 0
dims2 = 1

for i, sample in enumerate(dataloader):
    inp = np.array(sample['x'])
    target = np.array(sample['class'])
    x = np.reshape(inp, [inp.shape[0], inputdim])
    plt.scatter(x[:,0],x[:,1],color='r',s=1)
# set axes range
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig('truedist.png')


samples = []


def g_sample():
    with torch.no_grad():
        gen_input = torch.randn(batch_size*10, nz, device=device)
        g_fake_data = netG(gen_input)
        return g_fake_data.cpu().numpy()


    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

for epoch in tqdm(range(opt.niter)):
      for i, sample in enumerate(dataloader):
            points = Variable(sample['x'].type(Tensor))
            targets = Variable((sample['class']).type(LongTensor), requires_grad = False)        
            batch_size = points.size(0)

            z = torch.randn(batch_size, nz, device=device)

            valid = Variable(Tensor(points.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(points.size(0), 1).fill_(0.0), requires_grad=False)

            real_points = Variable(points.type(Tensor), requires_grad = False) 

            # Update G

            optimizerG.zero_grad()
            gen_points = netG(z)
            output_d = netD(gen_points)
#             gloss = bce_loss(output_d, valid)
            gloss = g_loss(output_d)
            gloss.backward()
            optimizerG.step()


            # Update D

            optimizerD.zero_grad()
            output_d_fake = netD(gen_points.detach())
            output_d_real = netD(real_points)
#             dloss_fake = bce_loss(output_d_fake, fake) 
#             dloss_real = bce_loss(output_d_real, valid)
#             dloss = (dloss_fake+dloss_real)/2.
            dloss = d_loss(output_d_real, output_d_fake)
            dloss.backward()
            optimizerD.step()


#             print("[%d/%d] [%d/%d] [G loss: %f] [D loss: %f (R) %f (F) %f]" % (epoch, opt.niter, i, len(dataloader), gloss.item(), dloss.item(), dloss_real.item(), dloss_fake.item()))

            if i % 10==0:
                # display points
                g_fake_data = g_sample()
                samples.append(g_fake_data)
                plot(g_fake_data, title='Iteration_{}'.format(epoch*len(dataloader)+i))              



