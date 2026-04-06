"""The module."""

from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape(
                    (out_features,)
                ),
                device=device,
                dtype=dtype,
            )
        else:
            self.bias = init.zeros(out_features, 1, device=device, dtype=dtype).reshape((out_features,))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(X.shape) > len(self.weight.shape):
            new_shape = X.shape[:-2] + self.weight.shape
            temp = ops.matmul(X, ops.broadcast_to(self.weight, new_shape))
        else:
            temp = ops.matmul(X, self.weight)
        return temp + ops.broadcast_to(self.bias.reshape((1, -1)), temp.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = X.shape[0]
        M = math.prod(X.shape[1:])
        return X.reshape((N, M))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules: list[Module] = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C = logits.shape
        y_one_hot = init.one_hot(C, y, device=logits.device, dtype=logits.dtype)
        logsumexp_N = ops.logsumexp(logits, axes=(1,))
        softmaxloss_N = logsumexp_N - ops.summation(logits * y_one_hot, axes=1)
        out = ops.summation(softmaxloss_N) / softmaxloss_N.shape[0]
        return out
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype
        )
        self.running_mean = init.zeros(1, dim, device=device, dtype=dtype)
        self.running_var = init.ones(1, dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean_C, var_C = self.running_mean, self.running_var
        N, C = x.shape
        if self.training:
            mean_C = ops.reshape(ops.summation(x, axes=0) / N, (1, -1))
            var_C = ops.reshape(
                (ops.summation((x - ops.broadcast_to(mean_C, x.shape)) ** 2, axes=0))
                / N,
                (1, -1),
            )

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean_C
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var_C
        norm_x_N_C = (x - ops.broadcast_to(mean_C, x.shape)) / (
            (ops.broadcast_to(var_C, x.shape) + self.eps) ** 0.5
        )
        return ops.broadcast_to(self.weight, x.shape) * norm_x_N_C + ops.broadcast_to(
            self.bias, x.shape
        )

        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w_N = Parameter(
            init.ones((dim), device=device, dtype=dtype), device=device, dtype=dtype
        )
        self.b_N = Parameter(
            init.zeros((dim), device=device, dtype=dtype), device=device, dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C = x.shape
        assert C == self.dim
        assert len(x.shape) == 2
        assert x.dtype == self.w_N.dtype == self.b_N.dtype
        mean = ops.summation(x, axes=1) / self.dim
        mean = mean.reshape((N, 1))
        variance = (
            ops.summation((x - ops.broadcast_to(mean, x.shape)) ** 2, axes=1)
        ) / self.dim

        variance = variance.reshape((N, 1))
        norm_x_N_C = (x - ops.broadcast_to(mean, x.shape)) / (
            (ops.broadcast_to(variance, x.shape) + self.eps) ** 0.5
        )
        return ops.broadcast_to(
            self.w_N.reshape((1, self.dim)), norm_x_N_C.shape
        ) * norm_x_N_C + ops.broadcast_to(self.b_N.reshape((1, self.dim)), x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            rand_var = init.randb(
                *x.shape, p=(1 - self.p), device=x.device, dtype=x.dtype
            )
            return (x * rand_var) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = self.fn(x)
        assert out.shape == x.shape
        return out + x
        ### END YOUR SOLUTION
