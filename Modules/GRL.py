import torch

class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)        
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None

class GRL(torch.nn.Module):
    def __init__(self, alpha= 1.0):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return Func.apply(x, torch.FloatTensor([self.alpha]).to(device= x.device))