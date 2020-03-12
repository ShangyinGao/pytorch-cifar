'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math

from .addop import my_cdist_op
from utils import print_tensor_shape

import types
import pdb


##
## default: False, 1
use_my_cdist = True
global_p = 1


def adder2d_function(X, W, stride=1, padding=0, p=1, debug=False):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    if use_my_cdist:
        assert p == 1, 'my_cdist only support p eq. 1'

    if use_my_cdist:
        # print('{0: <20}\t{1}'.format(f'name: W', f'shape: {W.shape}'))
        # print('{0: <20}\t{1}'.format(f'name: X', f'shape: {X.shape}'))
        # print('{0: <20}\t{1}'.format(f'name: W_col', f'shape: {W_col.shape}'))
        # print('{0: <20}\t{1}'.format(f'name: X_col', f'shape: {X_col.transpose(0, 1).shape}'))
        out = my_cdist_op.apply(X_col.transpose(0,1).contiguous(), W_col)
    else:
        out = -torch.cdist(W_col,X_col.transpose(0,1).contiguous(),p)

    # if p == 2:
    #     out = -out

    if debug:
        if p == 1:
            print(f'out[0]_should: {-torch.norm(X_col[:, 0]-W_col[0], 1)}')
        elif p == 2:
            print(f'out[0]_should: {torch.norm(X_col[:, 0]-W_col[0], 2)}')
        print(f'out_real: {out[0, 0]}')
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    if debug:
        print_tensor_shape(dir(), locals())
    
    return out

    
class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.kaiming_normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding, p=global_p)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output

    def extra_repr(self):
        return f'{"v1".upper()}, {self.input_channel} {self.output_channel}, '+\
                 f'kenrel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias}'
    
    
def test():
    print('\n'+'#'*40)
    X = torch.randn(16, 16, 32, 32)
    W = torch.randn(16, 16, 3, 3)
    print(f'X.shape: {X.shape}\nW.shape: {W.shape}')
    debug = True
    out_l1 = adder2d_function(X, W, padding=1 ,debug=debug)
    out_l2 = adder2d_function(X, W, padding=1, p=2, debug=debug)
    print(f'out_l1.shape: {out_l1.shape}')
    print(f'out_l2.shape: {out_l2.shape}')

if __name__ == "__main__":
    test()


