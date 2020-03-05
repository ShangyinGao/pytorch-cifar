import torch

from utils import print_tensor_shape

import pdb


class my_cdist_op(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        # assert bias == False, 'bias not supproted'
        ctx.save_for_backward(input, weight)
        out = -torch.cdist(input, weight, p=1)
        # print(out.shape)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        input_unsqueeze = torch.unsqueeze(input, 2).expand(input.shape[0], input.shape[1], weight.shape[0]).permute(1, 0, 2)
        weight_unsqueeze = torch.unsqueeze(weight, 2).expand(weight.shape[0], weight.shape[1], input.shape[0]).permute(1, 2, 0) 

        input_weight_delta = input_unsqueeze - weight_unsqueeze

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(input_weight_delta, grad_output.t()).sum(2).permute(1, 0)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(input_weight_delta.permute(0, 2, 1), grad_output).sum(2).permute(1, 0)

        # print('*'*50)
        # print_tensor_shape(dir(), locals())
        # print('*'*50)
        # print(ctx.needs_input_grad)

        return grad_input, grad_weight

# class my_cdist(torch.nn.Module):
#     def __init__(self):
#         super(my_cdist, self).__init__()

#     def forward(self, input, weight):
#         return my_cdist_op.apply(input, weight)
