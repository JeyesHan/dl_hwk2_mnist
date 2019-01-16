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
from PIL import Image
from question5 import Net as newNet

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)	#1*28*28 -> 20*24*24->20*12*12
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)	#20*12*12 -> 50*8*8 -> 50*4*4
        self.conv3 = nn.Conv2d(50,500,kernel_size=4)    #500*1*1
        self.conv4 = nn.Conv2d(500,10,kernel_size=1)    #10*1*1

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x),2)
        x = F.max_pool2d(self.conv2(x),2)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.squeeze(x)
        return F.log_softmax(x)#, dim=1)

def predict(picture_name):
    img = Image.open(picture_name)
    img_new = img.convert('L')
    img_new = np.array(img_new) * 1.0 / 255
    img_new = torch.FloatTensor(img_new)
    img_new = 1 - img_new.unsqueeze(0).unsqueeze(0)
    #print (img_new)
    img_new = (img_new - 0.1307) / 0.3081

    if args.cuda:
        img_new = img_new.cuda()
    img_new = Variable(img_new, volatile=True)
    output = model(img_new)
    pred = np.argmax(output.cpu().data.numpy(),0)
    print ("the predicted number is {}".format(pred))

model = newNet()
if args.cuda:
    model.cuda()
#model.load_state_dict(torch.load("net-epoch-20_baseline.pth"))
model.load_state_dict(torch.load("net-epoch-90-0.9951.pth"))
model.train(False)

predict("numbers/1.png")
predict("numbers/2.png")
predict("numbers/3.png")
predict("numbers/4.png")
predict("numbers/7.png")
predict("numbers/difficult_3.png")
predict("numbers/difficult_9_1.png")
predict("numbers/difficult_9_2.png")



