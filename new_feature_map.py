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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)	#1*28*28 -> 20*24*24->20*12*12
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)	#20*12*12 -> 50*8*8 -> 50*4*4
        self.conv3 = nn.Conv2d(50,500,kernel_size=4)    #500*1*1
        self.conv4 = nn.Conv2d(500,10,kernel_size=1)    #10*1*1
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        x = F.max_pool2d(self.conv1(x),2)
        x = F.max_pool2d(self.conv2(x),2)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.squeeze(x)
        return F.log_softmax(x)#, dim=1)

model = Net()
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



model.load_state_dict(torch.load("net-epoch-20_baseline.pth"))
model.train(False)

i = 1
for data, target in test_loader:
    #print (data)
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    if i==1:
        i+=1
        continue
    feature_map_conv1 = model.conv1(data)                   #first conv
    feature_map_conv2 = F.max_pool2d(feature_map_conv1, 2)
    feature_map_conv2 = model.conv2(feature_map_conv2)      #second conv
    feature_map_conv3 = F.max_pool2d(feature_map_conv2,2)
    feature_map_conv3 = model.conv3(feature_map_conv3)      #third conv
    break

fm1 = feature_map_conv1.long().cpu()
fm2 = feature_map_conv2.long().cpu()
fm3 = feature_map_conv3.long().cpu()

fm1 = fm1.data.numpy()
fm2 = fm2.data.numpy()
fm3 = fm3.data.numpy()


show_feature_map(fm1,24,4,5,"conv1 feature map")
show_feature_map(fm2,8,5,10,"conv2 feature map")
show_feature_map(fm3,1,20,25,"conv3 feature map")
