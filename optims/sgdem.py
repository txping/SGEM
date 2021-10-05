import torch
from torch.optim import Optimizer

class SGDEM(Optimizer):
    r"""Implements AEGDM algorithm.
    It has been proposed in `AEGDM: Adaptive Gradient Decent with Energy and Momentum`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        c (float, optional): term added to the original objective function (default: 1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _SGDEM: Adaptive Gradient Decent with Energy and Momentum:
    """

    def __init__(self, params, lr=0.1, c=1.0, momentum=0.9, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, c=c, momentum=momentum, weight_decay=weight_decay)

        super(SGDEM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDEM, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Make sure the closure is defined and always called with grad enabled
        closure = torch.enable_grad()(closure)
        loss = closure()

        for group in self.param_groups:
            if not 0.0 < loss+group['c']:
                raise ValueError("c={} does not satisfy f(x)+c>0".format(group['c']))

            lr = group['lr']
            c = group['c']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            sqrtloss = torch.sqrt(loss.detach() + c)

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError('SGDEM does not support sparse gradients')

                state = self.state[param]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param, memory_format = torch.preserve_format)
                    state['energy'] = sqrtloss * torch.ones_like(param, memory_format = torch.preserve_format)

                exp_avg = state['exp_avg']
                energy = state['energy']

                state['step'] += 1

                # weight decay
                if weight_decay != 0:
                    grad = grad.add(param, alpha = weight_decay)

                # exponential moving average of gradient values
                exp_avg.mul_(momentum).add_(grad, alpha = 1 - momentum)
                exp_avg = exp_avg / (1 - momentum ** state['step'])

                # transformed momentum
                tsf_exp_avg = exp_avg / (2 * sqrtloss)

                energy.div_(1 + 2 * lr * tsf_exp_avg ** 2)
                param.addcmul_(energy, tsf_exp_avg, value = -2 * lr)

        return loss
