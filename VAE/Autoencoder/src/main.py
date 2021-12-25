import model
import param
import utils

import torch
import torchvision

import os
import matplotlib.pyplot as plt


VAE_PATH = "./vae_weight_data.pt"


#data load
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


#evaluation
ToPIL = torchvision.transforms.ToPILImage()
with torch.no_grad():
    for idx, data in enumerate(test_loader):
      if len(data[0]) < param.batch_size:
          break

      recon_batch, mu, log_var = vae(param.sampling, data[0])
      sigma = torch.exp(0.5 * log_var)

      z = torch.randn((param.batch_size, param.z_dim));
      z = mu + torch.mul(z, sigma)
      fake_img = vae.decoder(z)

      ToPIL = torchvision.transforms.ToPILImage()
      fig = plt.figure()

      for i in range(0, 10):
          img = ToPIL(fake_img[i].view(1,28,28))
          subplot = fig.add_subplot(2,10,i+1)
          subplot.imshow(img)

          img = ToPIL(data[0][i])
          subplot = fig.add_subplot(2, 10, i + 11)
          subplot.imshow(img)

      plt.show()
      break
"""
with torch.no_grad():
    z = torch.randn((2,2))
    fake_img = decoder(z.float())
    ToPIL = torchvision.transforms.ToPILImage()
    img = ToPIL(fake_img[0])
    plt.imshow(img)
    plt.show()
"""