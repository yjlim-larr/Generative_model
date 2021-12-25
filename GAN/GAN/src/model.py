import torch
import torch.nn as nn
import torch.functional as f

class Generator(nn.Module):
    def __init__(self, z_dim, c, h, w):
        super(Generator, self).__init__()

        self.z = z_dim
        self.c = c
        self.h = h
        self.w = w

        self.model = nn.Sequential(
            nn.Linear(self.z, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, self.c * self.h * self.w),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input).view(-1, 1, 28, 28)



class Discriminator(nn.Module):
    def __init__(self, c, h, w):
        super(Discriminator, self).__init__()

        self.c = c
        self.h = h
        self.w = w

        self.model = nn.Sequential(
            nn.Linear(self.c * self.h * self.w, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x_flat = img.view(-1, 28 * 28)
        d = self.model(x_flat)
        return d
