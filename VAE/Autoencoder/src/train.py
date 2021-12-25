import utils
import model
import param

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import os


VAE_PATH = "./vae_weight_data.pt"


#train_data_MNIST
transfunction = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((28,28)),
                                                torchvision.transforms.ToTensor(),
                                               # torchvision.transforms.Normalize(0.5, 0.5)
])

train_dataset = torchvision.datasets.MNIST(root = './MNIST/processed/training.pt', train = True, transform=transfunction, download = True)
test_dataset = torchvision.datasets.MNIST(root = './MNIST/processed/training.pt', train = True, transform=transfunction, download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128, shuffle = True)



#model
vae = model.VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)

#load_weight
if os.path.isfile(VAE_PATH):
    vae.load_state_dict(torch.load(VAE_PATH))
else:
    vae.apply(utils.init_weight)



#train
criterion = nn.BCELoss()
vae_optim = optim.Adam(vae.parameters(), lr = param.lr)

for i in range(param.epochs):
    for idx, (data, C) in enumerate(train_loader):
        if len(data) < param.batch_size:
            break

        recon_batch, mu, log_var = vae(param.sampling, data)
        BCE = F.binary_cross_entropy(recon_batch.view(-1, 784), data.view(-1, 784), reduction='sum')

        #Categorize
        [a,b] = mu.size()
        C = torch.tensor(C).view(-1, 1)
        C = C.expand(-1, b)

        #Regularization
        KLD = -0.5 * torch.sum(1 + log_var - (C-mu).pow(2) - log_var.exp())

        #cal loss
        loss = BCE + KLD

        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()

        print('epochs: ', i, 'idx: ', idx, 'RECON: ', BCE, 'loss: ', loss)

    #save weight
    torch.save(vae.state_dict(), 'vae_weight_data.pt')
