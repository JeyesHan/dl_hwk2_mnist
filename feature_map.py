from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model_save_load import load_ckpt,save_ckpt
from main import Net,test

print("start")
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
                    #help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(args.seed)
#if args.cuda:
    #torch.cuda.manual_seed(args.seed)

print("here")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
print("hello0")
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


test_model = Net()
print("hello1")
#modules = [model]
#load_ckpt(modules, "net-epoch-20_baseline", load_to_cpu=False)

test_model.load_state_dict(torch.load("net-epoch-20_baseline.pth"))
test_model.train(False)

print("hello2")
if args.cuda:
    test_model.cuda()
print("hello3")
test_model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = test_model(data)
    test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
