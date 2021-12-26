import torch
import torch.nn as nn

def generate_noise(batch_size, dim, cate, con):
    # noise == batch_size * (dim + cate + con)
    uniform_dis = torch.distributions.uniform.Uniform(-1, 1)

    l = []
    for _ in range(cate):
        l.append(1 / cate)

    one_hot_cate = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor(l))

    sample_c1_list = []
    for i in range(batch_size):
        c1 = one_hot_cate.sample().view(cate)
        sample_c1_list.append(c1)

    c1 = torch.stack(sample_c1_list)
    c1.requires_grad = True

    c2 = uniform_dis.sample((batch_size, con))
    c2.requires_grad = True

    z = torch.randn((batch_size, dim))
    z = torch.cat((z, c1, c2), dim=1)

    return z, c1, c2

def init_weight(model):
  if isinstance(model, nn.Linear):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0.01)

  if isinstance(model, nn.Conv2d):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0.01)