import torch

# from utils import print_tensor_shape

import pdb


class my_cdist_op(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        # assert bias == False, 'bias not supproted'
        ctx.save_for_backward(input, weight)
        out = -torch.cdist(input, weight, p=1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        input_unsqueeze = torch.unsqueeze(input, 2).expand(input.shape[0], input.shape[1], weight.shape[0]).permute(1, 0, 2)
        weight_unsqueeze = torch.unsqueeze(weight, 2).expand(weight.shape[0], weight.shape[1], input.shape[0]).permute(1, 2, 0) 
        grad_output_unsqueeze = torch.unsqueeze(grad_output, 2).expand(grad_output.shape[0], grad_output.shape[1], input.shape[1]).permute(2, 0, 1)

        input_weight_delta = input_unsqueeze - weight_unsqueeze

        if ctx.needs_input_grad[0]:
            grad_input = torch.mul(-input_weight_delta, grad_output).sum(2).permute(1, 0)
            grad_input = torch.nn.functional.hardtanh(grad_input)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.mul(input_weight_delta, grad_output).sum(1).permute(1, 0)

        # print('*'*50)
        # print_tensor_shape(dir(), locals())
        # print('*'*50)
        # print(ctx.needs_input_grad)

        return grad_input, grad_weight


class my_cdist(torch.nn.Module):
    def __init__(self, weight_size):
        super(my_cdist, self).__init__()
        # self.fun = my_cdist_op()
        self.weight = torch.nn.Parameter(torch.randn(*weight_size))

    def forward(self, input):
        return my_cdist_op.apply(input, self.weight)


class torch_cdist(torch.nn.Module):
    def __init__(self, p, weight_size):
        super(torch_cdist, self).__init__()
        self.p = p
        self.weight = torch.nn.Parameter(torch.randn(*weight_size))

    def forward(self, input):
        return -torch.cdist(input, self.weight, p = self.p)


def _check_gradient():
    size_a, size_b, size_c = 3, 5, 4 # 128, 576, 64
    input_size = (size_a, size_b)
    weight_size = (size_c, size_b)

    input = torch.randn(*input_size, requires_grad=True) # [a, b]
    input_1 = input.clone().detach().requires_grad_(True)
    input_2 = input.clone().detach().requires_grad_(True)

    cdist = my_cdist(weight_size)
    cdist_1 = torch_cdist(1, weight_size)
    cdist_2 = torch_cdist(2, weight_size)

    out = cdist(input)
    out_1 = cdist_1(input_1)
    out_2 = cdist_2(input_2)
    
    out.data = out_2.data

    loss = torch.sum(out)
    loss_1 = torch.sum(out_1)
    loss_2 = torch.sum(out_2)


    loss.backward()
    loss_1.backward()
    loss_2.backward()

    for p, p1, p2 in zip(cdist.parameters(), cdist_1.parameters(), cdist_2.parameters()):
        gp = p.grad
        gp_1 = p1.grad
        gp_2 = p2.grad
        pdb.set_trace()


if __name__ == "__main__":
    _check_gradient()