'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_kernel_op

import pdb


class BasicBlock(nn.Module):
    def __init__(self, adder2d, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = adder2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = adder2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                adder2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use adder2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, adder2d, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = adder2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = adder2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                adder2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class PreActBlockFullAdd(PreActBlock):
    def __init__(self, adder2d, in_planes, planes, stride=1):
        super(PreActBlockFullAdd, self).__init__(adder2d, in_planes, planes, stride)
        self.fc1 = adder2d(planes, planes//16, kernel_size=1)
        self.fc2 = adder2d(planes//16, planes, kernel_size=1)
        self.fc_bn1 = nn.BatchNorm2d(planes//16) 
        self.fc_bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc_bn1(self.fc1(w)))
        w = torch.sigmoid(self.fc_bn2(self.fc2(w)))
        # Excitation
        out = out * w

        out += shortcut
        return out 


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, **kwargs):
        super(SENet, self).__init__()
        self.in_planes = 16 ## 64

        self.adder2d = get_kernel_op(kwargs)
        input_channels = 3 if kwargs['dataset'].lower() == 'cifar10'  else 1
        first_conv = kwargs.get('first_conv')
        self.fc_conv = kwargs.get('fc_conv')
        self.fake = nn.Conv2d(1, 1, 1)

        if first_conv:
            self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False) ## 64
        else:
            self.conv1 = self.adder2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False) ## 64

        self.bn1 = nn.BatchNorm2d(16) ## 64
        self.layer1 = self._make_layer(block,  16, num_blocks[0], stride=1) ## 64
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  ## 128
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  ## 256
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if self.fc_conv:
            # self.linear = nn.Linear(64, num_classes) ## 512
            # self.bn2 = nn.Sequential()
            self.linear = nn.Conv2d(64, num_classes, 1)
        else:
            # self.linear = self.adder2d(64, num_classes, 1) ## 512
            self.linear = self.adder2d(64, num_classes, 1) ## 512

        self.bn2 = nn.BatchNorm2d(num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.adder2d, self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8) ## 4

        # if self.fc_conv:
        #     out = out.view(out.size(0), -1)

        # out = self.linear(out)
        # out = self.bn2(out)

        # if self.fc_conv:
        #     return out
        # else:
        #     return out.view(out.shape[0], -1)

        out = self.linear(out)
        out = self.bn2(out)

        return out.view(out.shape[0], -1)

def SENet18(**kwargs):
    return SENet(PreActBlock, [3, 3, 3], **kwargs)  ## [2,2,2,2]
    # return SENet(PreActBlockFullAdd, [3, 3, 3], **kwargs)  ## [2,2,2,2]


def test():
    net = SENet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
