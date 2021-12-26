import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, z_dim, cate, con):
        super(Generator, self).__init__()

        self.z = z_dim
        self.cate = cate
        self.con = con

        self.model = nn.Sequential(
            nn.Linear(74, 128, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

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


class Discriminator(nn.Module):
    def __init__(self, c, h, w):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(c * h * w, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # D
        self.D = nn.Linear(256, 1, bias=True)

        # for Q.
        self.Q1 = nn.Linear(256, 128, bias=True)
        self.B2 = nn.BatchNorm1d(128)

        # c1, c2
        self.Q2 = nn.Linear(128, 10, bias=True)  # c_i
        self.Q3 = nn.Linear(128, 2, bias=True)  # c_j

    def forward(self, img):
        img = img.view(-1, 28 * 28)
        x = self.model(img)

        d = F.sigmoid(self.D(x))

        x = self.B2(F.leaky_relu(self.Q1(x)))
        c1 = F.softmax(self.Q2(x), dim=1)  # batch_size by 10
        c2 = torch.normal(mean=self.Q3(x), std=torch.tensor(0.01))

        return d, c1, c2