import torch
import torch.nn as nn
import torch.optim as optim

def init_weight(model):
  if isinstance(model, nn.Linear):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0.01)

  if isinstance(model, nn.Conv2d):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0.01)


def generate_noise(batch_size, z_dim):
  return torch.randn((batch_size, z_dim))
