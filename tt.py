from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)

for batch_idx, (data, target) in enumerate(train_loader):
    print (data)
    data = data * 255
    data = data.numpy().astype('uint8')
    plt.imshow(data.squeeze(),cmap = 'Greys')
    plt.axis("off")
    plt.show()
    break