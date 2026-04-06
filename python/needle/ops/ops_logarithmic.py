from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = tuple((1,)) if axes is None else axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        c = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_exp = array_api.exp(Z - array_api.broadcast_to(c, Z.shape))
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        Z_log = array_api.log(Z_sum)
        logexp = Z_log + c
        return Z - array_api.broadcast_to(logexp, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = LogSumExp(self.axes)
        logsumexp_value = lse(node.inputs[0])
        grad_logsumexp = lse.gradient(summation(out_grad, axes=1), logsumexp_value)
        return out_grad - grad_logsumexp
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSoftmax(axes=axes)(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if isinstance(axes, int):
            axes = tuple([axes])
        assert axes is None or isinstance(
            axes, tuple
        ), f"LogSumExp axes of type {type(axes)}"
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        c = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_exp = array_api.exp(Z - array_api.broadcast_to(c, Z.shape))
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        Z_log = array_api.log(Z_sum)

        ret_shape = list(Z_log.shape)
        if self.axes is None:
            ret_shape = (1,)
        else:
            assert len(self.axes) == 1, "LogSumExp only accross one axis"
            ret_shape = ret_shape[: self.axes[0]] + ret_shape[self.axes[0] + 1 :]
        return (Z_log + c).reshape(ret_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]

        axes = self.axes if self.axes is not None else tuple(range(len(x.shape)))
        cur_shape = list(out_grad.shape)
        for axis in sorted(axes):
            cur_shape.insert(axis, 1)
        out_grad = reshape(out_grad, tuple(cur_shape))
        out_grad = broadcast_to(out_grad, x.shape)

        log_softmax = logsoftmax(x, axes=self.axes)
        return out_grad * exp(log_softmax)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
