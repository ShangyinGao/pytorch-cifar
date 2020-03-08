'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import math
import argparse
from tensorboardX import SummaryWriter

from models import *
from utils import progress_bar, get_lr

import pdb


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_adaptive', action='store_true')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--bs_train', default=256, type=int, help='bs for traning')
parser.add_argument('--bs_test', default=100, type=int, help='bs for esting')
parser.add_argument('--optim', default='SGD', type=str, help='optimizer')
parser.add_argument('--scheduler', default='multistep', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--total_epoch', default=200, type=int)
parser.add_argument('--adder_v', default='v1', type=str)
parser.add_argument('--net', default='addernet', type=str)
args = parser.parse_args()
print(f'args:\n{args}')

writer = SummaryWriter()
print(f'writer.logdir: {writer.logdir}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
assert start_epoch < 350, f'start_epoch {start_epoch} bigger than 350'

# Data
print('==> Preparing data..')

if args.dataset == 'mnist':
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.bs_train, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.bs_test, shuffle=True, num_workers=2)
elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs_test, shuffle=False, num_workers=2)
else:
    raise NotImplementedError

print(f'trainloader length: {len(trainloader)}')
print(f'testloader length: {len(testloader)}')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
if args.net.lower() == 'resnet':
    net = ResNet18()
elif args.net.lower() == 'addernet':
    net = resnet20(adder_v=args.adder_v)
else: 
    raise NotImplementedError

# print(net)

net = net.to(device)

## add graph 
# dummy_input = torch.randn(16, 3, 32, 32)
# writer.add_graph(net, (dummy_input, ), True)
# writer.close()
# exit()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optim.SGD([{"params" : param} for param in net.parameters()], lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    raise NotImplementedError

if args.scheduler == 'multistep':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,150], gamma=0.1)
elif args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
else:
    raise NotImplementedError

train_idx = 0
test_idx = 0
global_lr = args.lr

eta = 1e-4

def check_gradient_norm(model, optimizer):
    global global_lr
    layer_idx = 0
    if 'resnet' in model.__class__.__name__.lower():
        key = ['conv', 'weight']
    else:
        key = ['adder']
    for (name, param), optim_param in zip(model.named_parameters(), optimizer.param_groups):
        # print(f'name: {name}, param: {param.shape}')
        if all([x in name for x in key]) and not name.startswith('conv1') and 'downsample' not in name:
            # print(f'{key}_layer {layer_idx}: {param.shape}')
            # print(f'{layer_idx}: {name} norm: {param.grad.norm()}')
            grad_norm = param.grad.norm()
            writer.add_scalar(f'grad_check/{layer_idx}', grad_norm, train_idx)

            if key == ['adder'] and args.lr_adaptive:
                lr = global_lr * eta * math.sqrt(param.grad.numel()) / (grad_norm + 1e-3)
                optim_param['lr'] = lr
                writer.add_scalar(f'lr/{layer_idx}', lr, train_idx)

            layer_idx += 1

# Training
def train(epoch):
    global train_idx, global_lr
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.scheduler == 'cosine':
            scheduler.step(epoch + batch_idx / len(trainloader))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        ## check gradient
        # torch.nn.utils.clip_grad_value_(net.parameters(), 1)
        check_gradient_norm(net, optimizer)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # ## 
        global_lr = get_lr(optimizer)
        curr_acc = correct / total
        writer.add_scalar('train/loss', loss, train_idx)
        writer.add_scalar('train/acc', curr_acc, train_idx)
        writer.add_scalar('train/lr', global_lr, train_idx)
        train_idx += 1

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
            % (train_loss/(batch_idx+1), 100.*curr_acc, correct, total, global_lr))

def fake_train(epoch):
    global train_idx
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch + batch_idx / len(trainloader))
        curr_lr = get_lr(optimizer)
        optimizer.step()
        writer.add_scalar('train/lr', curr_lr, train_idx)
        train_idx += 1
        progress_bar(batch_idx, len(trainloader), 'batch_idx: %d'
            % ( batch_idx))

def test(epoch):
    global best_acc, test_idx
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            ##
            curr_acc = correct / total
            writer.add_scalar('test/loss', test_loss, test_idx)
            writer.add_scalar('test/acc', curr_acc, test_idx)
            test_idx += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*curr_acc, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, args.total_epoch):
    train(epoch)
    test(epoch)
    # fake_train(epoch)
    if args.scheduler == 'multistep':
        scheduler.step()

writer.close()
