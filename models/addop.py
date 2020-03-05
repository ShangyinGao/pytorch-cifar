import torch

class Adder(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass