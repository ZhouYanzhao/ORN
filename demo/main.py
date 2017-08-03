# Oriented Response Network https://arxiv.org/pdf/1701.01833 
# Authored by Yanzhao Zhou, Qixiang Ye, Qiang Qiu and Jianbin Jiao 
# Project Page: http://yzhou.work/ORN
# PyTorch MNIST-Variants Demo 
from __future__ import print_function
import math
import numbers
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.autograd import Variable
from orn.modules import ORConv2d
from orn.functions import oralign1d

# Training settings
parser = argparse.ArgumentParser(description='ORN.PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use-arf', action='store_true', default=False,
                    help='upgrading to ORN')
parser.add_argument('--orientation', type=int, default=8, metavar='O',
                    help='nOrientation for ARFs (default: 8)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# custom transform
class RandomRotate(object):
    """Rotate the given PIL.Image counter clockwise around its centre by a random degree 
    (drawn uniformly) within angle_range. angle_range is a tuple (angle_min, angle_max). 
    Empty region will be padded with color specified in fill."""
    def __init__(self, angle_range=(-180,180), fill='black'):
        assert isinstance(angle_range, tuple) and len(angle_range) == 2 and angle_range[0] <= angle_range[1]
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.angle_range = angle_range
        self.fill = fill

    def __call__(self, img):
        angle_min, angle_max = self.angle_range
        angle = angle_min + random.random() * (angle_max - angle_min)
        theta = math.radians(angle)
        w, h = img.size
        diameter = math.sqrt(w * w + h * h)
        theta_0 = math.atan(float(h) / w)
        w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
        h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
        pad = math.ceil(max(w_new - w, h_new - h) / 2)
        img = ImageOps.expand(img, border=int(pad), fill=self.fill)
        img = img.rotate(angle, resample=Image.BICUBIC)
        return img.crop((pad, pad, w + pad, h + pad))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(32),
                       RandomRotate((-180, 180)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.Scale(32),
                       RandomRotate((-180, 180)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, use_arf=False, nOrientation=8):
        super(Net, self).__init__()
        self.use_arf = use_arf
        self.nOrientation = nOrientation
        if use_arf:
            self.conv1 = ORConv2d(1, 10, arf_config=(1,nOrientation), kernel_size=3)
            self.conv2 = ORConv2d(10, 20, arf_config=nOrientation,kernel_size=3)
            self.conv3 = ORConv2d(20, 40, arf_config=nOrientation,kernel_size=3, stride=1, padding=1)
            self.conv4 = ORConv2d(40, 80, arf_config=nOrientation,kernel_size=3)
        else:   
            self.conv1 = nn.Conv2d(1, 80, kernel_size=3)
            self.conv2 = nn.Conv2d(80, 160, kernel_size=3)
            self.conv3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(320, 640, kernel_size=3)
        self.fc1 = nn.Linear(640, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        if self.use_arf:
            x = oralign1d(x, self.nOrientation)
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net(args.use_arf, args.orientation)
print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adadelta(model.parameters())
best_test_acc = torch.Tensor(1).zero_()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    test_acc = 100. * correct / len(test_loader.dataset)
    if test_acc > best_test_acc[0]:
        best_test_acc[0] = test_acc
        print('best test accuracy: {:.2f}%'.format(best_test_acc[0]))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

print('best test accuracy: {:.2f}%'.format(best_test_acc[0]))

# save parameters
torch.save(model.state_dict(), 'model.pt7')