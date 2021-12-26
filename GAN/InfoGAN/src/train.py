import model
import param
import utils

import model
import param
import utils

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import os

GEN_PATH = "./gen_weight_data.pt"
DIS_PATH = "./dis_weight_data.pt"


#train data
trans_function = torchvision.transforms.Compose([
                                                 torchvision.transforms.Scale((28,28)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(0.5, 0.5),
])

train_dataset = torchvision.datasets.MNIST(root = 'MNIST/processed/training.pt', train = True, download = True, transform = trans_function)
test_dataset = torchvision.datasets.MNIST(root = 'MNIST/processed/training.pt', train = False, download = True, transform = trans_function)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, shuffle = True)


#model
generator = model.Generator(param.z_dim, param.cate, param.con)
discriminator = model.Discriminator(param.c, param.h, param.w)


#load weight
if os.path.isfile(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH))
else:
    generator.apply(utils.init_weight)

if os.path.isfile(DIS_PATH):
    discriminator.load_state_dict(torch.load(DIS_PATH))
else:
    discriminator.apply(utils.init_weight)


#train
G_optim = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
D_optim = optim.Adam(discriminator.parameters(), lr = 0.0001, betas=(0.5, 0.999))


criterion = nn.BCELoss()
criterion2 = nn.MSELoss()

for _ in range(param.epochs):
  for idx, data in enumerate(train_loader):
    if len(data[0]) < param.batch_size:
      break

    sample_noise, sample_c1, sample_c2 = utils.generate_noise(param.batch_size, param.z_dim, param.cate, param.con)
    fake_img = generator(sample_noise)

    #fake_input
    d_fake_img = fake_img.detach()
    fake_d, c1, c2 = discriminator(d_fake_img)
    print(fake_d[0])

    #real_input
    real_img = data[0].view(param.batch_size, 1, 28,28)
    real_d, real_c1, real_c2 = discriminator(real_img)
    cate = F.one_hot(data[1], num_classes = 10)

    #target
    real = torch.ones((param.batch_size, 1))
    fake = torch.zeros((param.batch_size, 1))

    #mutual loss
    m1 = criterion(real_c1, cate.float())
    m2 = criterion2(c2, sample_c2)
    mutual = m1 + m2

    #loss d
    real_loss = criterion(real_d, real)
    fake_loss = criterion(fake_d, fake)
    D_loss = real_loss + fake_loss + param.lda * mutual

    D_optim.zero_grad()
    D_loss.backward(retain_graph=True)
    D_optim.step()


    #---------------------------update G-----------------------------
    sample_noise, sample_c1, sample_c2 = utils.generate_noise(param.batch_size, param.z_dim, param.cate, param.con)
    fake_img = generator(sample_noise)
    fake_img =fake_img.view(param.batch_size, 1, 28, 28)
    fake_d,c1,c2 = discriminator(fake_img)

    #mutual loss
    m1 = criterion(c1, sample_c1)
    m2 = criterion2(c2, sample_c2)
    m = m1 + m2

    G_loss = criterion(fake_d, real)
    G_loss = G_loss + param.lda * m

    G_optim.zero_grad()
    G_loss.backward(retain_graph=True)
    G_optim.step()

    print('epochs: ',_,' idx: ',idx, ' D_loss: ', D_loss.item(), ' loss2: ', G_loss.item())


  # save weight
  torch.save(generator.state_dict(), 'gen_weight_data.pt')
  torch.save(discriminator.state_dict(), 'dis_weight_data.pt')