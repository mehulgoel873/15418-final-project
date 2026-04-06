"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, p in enumerate(self.params):
            if i not in self.u:
                self.u[i] = (1 - self.momentum) * (
                    p.grad.detach() + self.weight_decay * p.data
                )
            else:
                self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * (
                    p.grad.detach() + self.weight_decay * p.data
                )
            p.data = p.data - self.lr * (self.u[i])
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, p in enumerate(self.params):
            g_t = p.grad.detach() + self.weight_decay * p.data

            if i in self.m:
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * (g_t)
            else:
                self.m[i] = (1.0 - self.beta1) * (g_t)

            if i in self.v:
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * ((g_t) ** 2)
            else:
                self.v[i] = (1.0 - self.beta2) * ((g_t) ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)

            p.data = p.data - self.lr * (m_hat / ((v_hat**0.5) + self.eps))
        ### END YOUR SOLUTION
