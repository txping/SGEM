import torch
from torch.optim import Optimizer

class AEGD(Optimizer):
    r"""Implements AEGD algorithm.
    It has been proposed in `AEGD: Adaptive Gradient Decent with Energy`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        c (float, optional): term added to the original objective function (default: 1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        aegdw (boolean, optional): whether to use the AEGDW variant of this algorithm
            (arxiv.org/abs/1711.05101) (default: False)

    .. _AEGD: Adaptive Gradient Decent with Energy:
    """

    def __init__(self, params, lr=0.1, final_lr=0.1, gamma=1e-3, c=1.0,
                 momentum=0, dampening=0, weight_decay=0,
                 exp_avg=False, nesterov=False, aegdw=False, clip=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, final_lr=final_lr, gamma=gamma, c=c, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay,
                        exp_avg=exp_avg, nesterov=nesterov, aegdw=aegdw, clip=clip)

        super(AEGD, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AEGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('exp_avg', False)
            group.setdefault('nesterov', False)
            group.setdefault('aegdw', False)
            group.setdefault('clip', False)

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

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            if not 0.0 < loss+group['c']:
                raise ValueError("c={} does not satisfy f(x)+c>0".format(group['c']))

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            exp_avg = group['exp_avg']
            nesterov = group['nesterov']
            aegdw = group['aegdw']
            clip = group['clip']
            gamma = group['gamma']
            #final_lr = group['final_lr']
            #lr = group['lr']
            c = group['c']

            # Evaluate g(x)=(f(x)+c)^{1/2}
            sqrtloss = torch.sqrt(loss.detach() + c)

            for p in group['params']:
                if p.grad is None:
                    continue
                df = p.grad
                if df.is_sparse:
                    raise RuntimeError('AEGD does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['energy'] = sqrtloss * torch.ones_like(p)
                    state['buf'] = torch.zeros_like(p)

                step = state['step']
                energy = state['energy']
                buf = state['buf']

                step += 1

                # Evaluate dg/dx = (df/dx) / (2*g(x))
                dg = df / (2 * sqrtloss)

                # Update energy
                energy.div_(1 + 2 * group['lr'] * dg ** 2)

                # Perform weight decay / L_2 regularization on g(x)
                if aegdw:
                    p.mul_(1 - group['lr'] * weight_decay)
                else:
                    dg = dg.add(p, alpha=weight_decay)

                if momentum != 0:
                    if exp_avg:
                        buf.mul_(momentum).add_(dg, alpha=1 - momentum)
                        buf = buf / (1 - momentum ** step)
                    else:
                        buf.mul_(momentum).add_(dg, alpha=1 - dampening)

                    if nesterov:
                        dg = dg.add(buf, alpha=momentum)
                    else:
                        dg = buf

                if clip:
                    final_lr = group['final_lr'] * group['lr'] / base_lr
                    lower_bound = final_lr * (1 - 1 / (gamma * step + 1))
                    upper_bound = final_lr * (1 + 1 / (gamma * step))
                    step_size = torch.full_like(energy, 2*group['lr'])
                    step_size.mul_(energy).clamp_(lower_bound, upper_bound).mul_(dg)
                    p.add_(-step_size)
                else:
                    p.addcmul_(energy, dg, value=-2 * group['lr'])

        return loss
