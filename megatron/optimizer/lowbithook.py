""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from .adamw import AdamW

class LowbitHook(Optimizer):
    def __init__(self, params, base_optimizer=AdamW, **kwargs):
        defaults = dict(**kwargs)
        super(LowbitHook, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avg_errs = []
            ref_points = []


            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg_err'] = torch.zeros_like(p)
                    state['refer_point'] = torch.zeros_like(p)
                    
                    
                exp_avg_errs.append(state['exp_avg_err'])
                ref_points.append(state['refer_point'])
            step_size = group['lr'] * group.get('lr_decay', 1.0)
            pre_lr = group.get('pre_lr', step_size)    
            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avg_errs=exp_avg_errs,
                ref_points=ref_points,
                local_eta=1.0,
                lr_ratio=pre_lr / step_size
            )
            self._multi_tensor_step(**kwargs)
            group['pre_lr'] = step_size
        return loss

    @torch.no_grad()
    def _multi_tensor_step(self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avg_errs: List[Tensor],
        ref_points: List[Tensor],
        *,
        local_eta: float,
        lr_ratio: float
    ):
        if len(params) == 0:
            return

        torch._foreach_add_(grads, ref_points, alpha=local_eta)
        self.base_optimizer.step()
        h_g_sgn = [torch.sign(t) for t in grads]
        torch._foreach_zero_(ref_points)
        torch._foreach_add_(ref_points, grads)

        torch._foreach_abs_(grads)
        torch._foreach_mul_(grads, 4.0)
        torch._foreach_add_(grads, local_eta**2)
        torch._foreach_sqrt_(grads)
        torch._foreach_add_(grads, local_eta)
        
        torch._foreach_mul_(ref_points, 2.0)
        torch._foreach_div_(ref_points, grads)
        torch._foreach_mul_(ref_points, ref_points)
        torch._foreach_mul_(ref_points, h_g_sgn)
        
        torch._foreach_mul_(exp_avg_errs, lr_ratio)