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


#MNIST
trans_function = torchvision.transforms.Compose([
                                                 torchvision.transforms.Scale((28,28)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(0.5, 0.5)
])

train_dataset = torchvision.datasets.MNIST(root = 'MNIST/processed/training.pt', train = True, download = True, transform = trans_function)
test_dataset = torchvision.datasets.MNIST(root = 'MNIST/processed/training.pt', train = False, download = True, transform = trans_function)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, shuffle = True)


#model
generator = model.Generator(param.z_dim, param.label_dim, param.c, param.h, param.w)
discriminator = model.Discriminator(param.label_dim, param.c, param.h, param.w)


#load_weight
if os.path.isfile(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH))
else:
    generator.apply(utils.init_weight)

if os.path.isfile(DIS_PATH):
    discriminator.load_state_dict(torch.load(DIS_PATH))
else:
    discriminator.apply(utils.init_weight)


# train
G_optim = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
D_optim = optim.Adam(discriminator.parameters(), lr = 0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for i in range(param.epochs):
    for idx, data in enumerate(train_loader):
        label = data[1]
        data = data[0]

        if len(data) != param.batch_size:
            break

        z = utils.generate_noise(param.batch_size, param.z_dim)
        y = F.one_hot(torch.tensor(label), num_classes = 10)

        fake_img = generator(z, y.float())

        # fake_input
        fake_d = discriminator(fake_img, y.float())  # batch_size by 1

        # real_input
        real_img = data.view(-1, 1, 28, 28)
        real_d = discriminator(real_img, y.float())

        real_y = torch.ones((param.batch_size, 1))
        fake_y = torch.zeros((param.batch_size, 1))

        # loss d
        real_loss = criterion(real_d, real_y)  # -torch.mean(log(read_d) * reaL_y + log(1-real_d)*(1-real_y))
        fake_loss = criterion(fake_d, fake_y)  # -torch.mean(log(fake_d) * fake_y + log(1-fake_d)*(1-fake_y))
        loss = real_loss + fake_loss

        D_optim.zero_grad()
        loss.backward()
        D_optim.step()

        # update G
        z = utils.generate_noise(param.batch_size, param.z_dim)
        fake_img = generator(z, y.float())

        fake_d = discriminator(fake_img, y.float())
        loss2 = criterion(fake_d, real_y)  # (log(fake_d) minimize == log(1-fake_d) maximize)

        G_optim.zero_grad()
        loss2.backward()
        G_optim.step()

        print('epochs: ', i, ' idx: ', idx, ' loss1: ', loss, ' loss2: ', loss2)

    # save weight
    torch.save(generator.state_dict(), 'gen_weight_data.pt')
    torch.save(discriminator.state_dict(), 'dis_weight_data.pt')