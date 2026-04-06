"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

import math


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs[0], node.inputs[1]
        log_x = log(x)
        out_a, out_b = out_grad * (y * (x ** (y - 1))), out_grad * (x**y) * log_x
        assert out_a.dtype == x.dtype
        assert out_b.dtype == y.dtype
        return out_a, out_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        out = out_grad * (self.scalar * (x ** (self.scalar - 1)))
        assert out.dtype == x.dtype
        return out
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        out_a, out_b = out_grad / b, out_grad * a * (-1 * (b ** (-2)))
        assert out_a.dtype == a.dtype
        assert out_b.dtype == b.dtype
        return out_a, out_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        out = out_grad * (1 / self.scalar)
        assert out.dtype == a.dtype
        return out
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = [i for i in range(len(a.shape))]
        if self.axes is not None:
            axes[self.axes[0]] = self.axes[1]
            axes[self.axes[1]] = self.axes[0]
        else:
            axes[-2], axes[-1] = axes[-1], axes[-2]

        out = array_api.transpose(a, axes=axes)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert out_grad.dtype == node.inputs[0].dtype
        return out_grad.transpose(axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not a.is_compact():
            a = a.compact()
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        assert math.prod(out_grad.shape) == math.prod(a.shape)
        assert out_grad.dtype == a.dtype
        return out_grad.reshape(a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        n_diff = len(self.shape) - len(a.shape)
        input_shape = (1,) * n_diff + a.shape
        axis = []
        for i, in_dim in enumerate(input_shape):
            if in_dim == 1:
                axis.append(i)

        if axis:
            out = out_grad.sum(axes=tuple(axis)).reshape(a.shape)
            assert out.dtype == a.dtype
            return out
        else:
            assert out_grad.dtype == a.dtype
            return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = tuple([axes])
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        out_shape = list(out_grad.shape)
        if not self.axes:
            out_shape = [1 for _ in a.shape]
        else:
            for axis in self.axes:
                out_shape.insert(axis, 1)
        out = broadcast_to(out_grad.reshape(out_shape), a.shape)
        assert out.dtype == a.dtype
        return out
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad

        grad_a = grad_a.sum(
            axes=tuple(i for i in range(len(grad_a.shape) - len(a.shape)))
        )
        grad_b = grad_b.sum(
            axes=tuple(i for i in range(len(grad_b.shape) - len(b.shape)))
        )

        assert grad_a.dtype == a.dtype
        assert grad_b.dtype == b.dtype
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        assert out_grad.dtype == node.inputs[0].dtype
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        out = out_grad * (Tensor(1) / a)
        assert out.dtype == a.dtype
        return out
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        out = out_grad * exp(a)
        assert out.dtype == a.dtype
        return out
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.inputs[0].detach().numpy()
        mask = (data > 0).astype(node.inputs[0].dtype)

        mask_tensor = Tensor(mask, device=node.inputs[0].device, requires_grad=False)

        return multiply(out_grad, mask_tensor)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * ((-1 * (tanh(a) ** 2)) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0
        new_shape = list(args[0].shape)
        new_shape.insert(0, len(args))
        out: array_api.ndarray = array_api.full(
            shape=new_shape, fill_value=0, device=args[0].device, dtype=args[0].dtype
        )
        for i in range(len(args)):
            out[i] = args[i]
        permute_arr = list(i + 1 for i in range(len(new_shape[1:])))
        permute_arr.insert(self.axis, 0)
        return out.permute(tuple(permute_arr))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        access_idx = [
            0 if i == self.axis else slice(None, None, None)
            for i in range(len(A.shape))
        ]
        out_shape = tuple(list(A.shape[: self.axis]) + list(A.shape[self.axis + 1 :]))
        out = []
        for i in range(A.shape[self.axis]):
            access_idx[self.axis] = i
            A_split = A[tuple(access_idx)].compact().reshape(out_shape)
            out.append(A_split)
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Flip(self.axes)(out_grad)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: array_api.ndarray):
        ### BEGIN YOUR SOLUTION
        new_shapes = []
        new_idxs = []
        for i in range(a.ndim):
            if i in self.axes:
                new_shapes.append(a.shape[i] * (self.dilation + 1))
                new_idxs.append(slice(None, None, self.dilation + 1))
            else:
                new_shapes.append(a.shape[i])
                new_idxs.append(slice(None, None, None))

        out = array_api.full(
            shape=new_shapes, fill_value=0, dtype=a.dtype, device=a.device
        )
        out[tuple(new_idxs)] = a
        return out

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shapes = []
        new_idxs = []
        for i in range(a.ndim):
            if i in self.axes:
                assert a.shape[i] % (self.dilation + 1) == 0
                new_shapes.append(a.shape[i] // (self.dilation + 1))
                new_idxs.append(slice(None, None, self.dilation + 1))
            else:
                new_shapes.append(a.shape[i])
                new_idxs.append(slice(None, None, None))
        # print(a.shape, new_shapes, new_idxs, self.dilation)
        assert a[tuple(new_idxs)].shape == tuple(new_shapes)
        return a[tuple(new_idxs)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B: array_api.ndarray):
        ### BEGIN YOUR SOLUTION
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        A_padded = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )
        inner_dim = K * K * C_in
        A_im2_col = array_api.im2_col(A_padded, K, self.stride).compact()
        A_reshaped = A_im2_col.reshape((-1, inner_dim))
        if not B.is_compact():
            B = B.compact()
        out = A_reshaped @ B.reshape((-1, C_out))
        return out.reshape(
            (
                N,
                (H - K + 1 + 2 * self.padding) // self.stride,
                (W - K + 1 + 2 * self.padding) // self.stride,
                C_out,
            )
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K = W.shape[0]
        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        W_flipped = flip(W, axes=(0, 1))
        W_T = transpose(W_flipped, axes=(2, 3))

        # print(X.shape, out_grad.shape, W_T.shape)
        X_grad = conv(out_grad, W_T, padding=2 * (K // 2) - self.padding)
        # print(X_grad.shape)
        assert (
            X_grad.shape == X.shape
        ), f"X, X_grad shapes inequal, X shape: {X.shape}, X_grad shape: {X_grad.shape}"

        X_T = transpose(X, axes=(0, 3))
        out_grad_T_T = transpose(
            transpose(out_grad, (0, 1)),
            (1, 2),
        )
        W_grad = conv(X_T, out_grad_T_T, padding=self.padding, stride=1)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))
        return X_grad, W_grad

        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)
