from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(32),
                       transforms.RandomCrop(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                        #transforms.Normalize((0.1307,), (1,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                        #transforms.Normalize((0.1307,), (1,))
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(128, 10)
        #self.fc2 = nn.Linear(50, 10)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),    #1*28*28 -> 64*24*24
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, padding=1, bias=False),    #64*24*24 ->128*12*12
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks= nn.Sequential(
                ResidualBlock(32),
                ResidualBlock(32)
            )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, padding=1, bias=False),   #128*12*12 -> 256*6*6
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),   #256*6*6 -> 512*3*3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, bias=False),   #512*3*3 -> 512*1*1
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.apply(init_weights)

    def forward(self, x):
        image_feature = self.encoder1(x)
        image_feature_fuse = self.residual_blocks(image_feature)
        image_texture = self.encoder2(image_feature_fuse).view(-1,128)
        x = F.relu(self.fc(image_texture))
        #x = F.dropout(x,0.8,training=self.training)
        x = torch.squeeze(x)
        return F.log_softmax(x)#, dim=1)

model = Net()
if args.cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(args.momentum,0.999))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return loss.data[0]

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (1. * correct) / len(test_loader.dataset)


if __name__ == '__main__':
    loss = []
    val_acc = []
    test_acc = []

    #model.load_state_dict(torch.load("net-epoch-100_baseline.pth"))
    for epoch in range(1, args.epochs + 1):
        loss.append(train(epoch))
        #model.load_state_dict(torch.load("net-epoch-20_baseline.pth"))
        model.train(False)
        #print("yes:")
        test_acc_temp = test()
        test_acc.append(test_acc_temp)
        if(test_acc_temp == max(test_acc)):
            torch.save(model.state_dict(), "net-epoch-"+str(epoch)+"-"+str(test_acc_temp)+".pth")

    #torch.save(model.state_dict(), "net-epoch-200_baseline.pth")
    print (loss)
    print (test_acc)

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    plt.subplot(121);
    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss decline")

    plt.subplot(122);
    plt.plot(test_acc)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title("test accuracy")
    plt.show()
