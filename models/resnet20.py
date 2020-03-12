# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# import adder
import torch
import torch.nn as nn

from utils import get_kernel_op

import pdb


def conv3x3(adder2d, in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, adder2d, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(adder2d, inplanes, planes, stride = stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(adder2d, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdderNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, **kwargs):
        super(AdderNet, self).__init__()

        self.adder2d = get_kernel_op(kwargs)
        input_channels = 3 if kwargs['dataset'].lower() == 'cifar10'  else 1
        first_conv = kwargs.get('first_conv')
        fc_conv = kwargs.get('fc_conv')
        # self.fake = nn.Conv2d(1, 1, 1)

        self.inplanes = 16

        if first_conv:
            self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = self.adder2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)

        if fc_conv:
            self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        else:
            self.fc = self.adder2d(64 * block.expansion, num_classes, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_classes)
        
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, self.adder2d):
                nn.init.kaiming_normal_(m.adder, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


         
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.adder2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.adder2d, inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.adder2d, inplanes = self.inplanes, planes = planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return AdderNet(BasicBlock, [3, 3, 3], **kwargs)
   

def test():
    net = resnet20()
    print(net)
    print('\n'+'#'*40)
    y = net(torch.randn(16,3,32,32))
    print(y.size())

if __name__ == "__main__":
    test()

