import param

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, label_dim, c, h, w):
        super(Generator, self).__init__()

        self.z = z_dim
        self.L = label_dim

        self.c = c
        self.h = h
        self.w = w

        self.z_layer = nn.Linear(self.z, 200)
        self.L_layer = nn.Linear(self.L, 56)
        self.drop1 = nn.Dropout(param.dropout)
        self.drop2 = nn.Dropout(param.dropout)

        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

        def forward(self, input):
            return self.model(input).view(-1, 1, 28, 28)

    def forward(self, z, y):
        pre_z = F.relu(self.z_layer(z))
        pre_z = self.drop1(pre_z)

        pre_L = F.relu(self.L_layer(y))
        pre_L = self.drop2(pre_L)

        x = torch.cat([pre_z, pre_L], dim = 1)
        return self.model(x).view(-1, self.c, self.h, self.w)


class Discriminator(nn.Module):
    def __init__(self, label_dim, c, h, w):
        super(Discriminator, self).__init__()

        self.L = label_dim
        self.c = c
        self.h = h
        self.w = w

        self.x_layer = nn.Linear(self.c * self.h * self.w, 824)
        self.L_layer = nn.Linear(self.L, 200)
        self.drop1 = nn.Dropout(param.dropout)
        self.drop2 = nn.Dropout(param.dropout)

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward(self, img, label):
        img_flat = img.view(-1, 28 * 28)

        pre_x = F.relu(self.x_layer(img_flat))
        pre_x = self.drop1(pre_x)

        pre_L = F.relu(self.L_layer(label))
        pre_L = self.drop2(pre_L)

        x = torch.cat([pre_x, pre_L], dim = 1)
        d = self.model(x)

        return d