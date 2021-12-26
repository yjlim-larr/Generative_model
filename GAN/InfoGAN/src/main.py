import model
import param
import utils

import torch
import torch.nn.functional as F
import torchvision

import os
import matplotlib.pyplot as plt



GEN_PATH = "./gen_weight_data.pt"
DIS_PATH = "./dis_weight_data.pt"


#model
generator = model.Generator(param.z_dim, param.cate, param.con)
discriminator = model.Discriminator(param.c, param.h, param.w)


#load_weight
if os.path.isfile(GEN_PATH):
    generator.load_state_dict(torch.load(GEN_PATH))
else:
    generator.apply(utils.init_weight)

if os.path.isfile(DIS_PATH):
    discriminator.load_state_dict(torch.load(DIS_PATH))
else:
    discriminator.apply(utils.init_weight)



#evaluation
ToPIL = torchvision.transforms.ToPILImage()
with torch.no_grad():
    sample_noise, sample_c1, sample_c2 = utils.generate_noise(param.batch_size, param.z_dim, param.cate, param.con)
    fake_img = generator(sample_noise)

    fake_img = generator(sample_noise)
    sample_c1 = torch.argmax(sample_c1, dim = 1)

    ToPIL = torchvision.transforms.ToPILImage()
    fig = plt.figure()

    for i in range(0, 10):
        img = ToPIL(fake_img[i].view(1, 28, 28))
        subplot = fig.add_subplot(1, 10, i + 1)
        subplot.set_title(str(sample_c1[i].item())) #category
        subplot.imshow(img)

    plt.show()