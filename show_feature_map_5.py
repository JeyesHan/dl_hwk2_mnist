from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from question5 import Net as MyNet


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(0,1/torch.sqrt(m.weight.data[0]/2))
        if m.bias.requires_grad:
            m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self,dims):
        super(ResidualBlock, self).__init__()
        self.dims = dims
        self.encoder = nn.Sequential(
            nn.Conv2d(self.dims, self.dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dims, self.dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.dims)
        )

    def forward(self, x):
        return F.relu(x + self.encoder(x))

model = MyNet()
#print (model)
if args.cuda:
    model.cuda()

def show_feature_map(x,size,rows,colums,name):#assume x is 1*20*24*24 size=24 rows=4 colums=5
    res = np.zeros((size * rows, size * colums))
    for i in range(rows):
        for j in range(colums):
            temp = np.squeeze(x[:, colums * i + j,:,:])
            res[i*size:(i+1)*size,j*size:(j+1)*size] = temp
    #print (res)
    #plt.imshow(res,cmap = 'Greys')
    plt.imshow(res, cmap='gray')
    plt.title("{}".format(name))
    plt.axis("off")
    plt.show()

def get_features1_hook(self, input, output):
    fm1 = output.long().cpu()
    fm1 = fm1.data.numpy()
    show_feature_map(fm1, 24, 4, 4, "conv1 feature map")

def get_features2_hook(self, input, output):
    fm1 = output.long().cpu()
    fm1 = fm1.data.numpy()
    show_feature_map(fm1, 12, 4, 8, "conv2 feature map")

def get_features3_hook(self, input, output):
    fm1 = output.long().cpu()
    fm1 = fm1.data.numpy()
    show_feature_map(fm1, 12, 4, 8, "conv3 feature map")

model.load_state_dict(torch.load("net-epoch-90-0.9951.pth"))
model.train(False)

i = 1
for data, target in test_loader:
    #print (data)
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    if i<=7:
        i+=1
        continue
    handle1 = model.encoder1[1].register_forward_hook(get_features1_hook)
    handle1 = model.encoder1[4].register_forward_hook(get_features2_hook)
    handle2 = model.residual_blocks[1].encoder[4].register_forward_hook(get_features3_hook)
    output = model(data)  # first conv
    break

