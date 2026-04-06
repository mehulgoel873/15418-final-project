"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=in_channels * kernel_size**2,
                fan_out=out_channels * kernel_size**2,
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels,
                ),
                device=device,
                dtype=dtype,
            )
        )

        interval = 1 / np.sqrt(self.in_channels * self.kernel_size**2)
        if bias:
            self.bias = Parameter(
                init.rand(
                    self.out_channels,
                    low=-1 * interval,
                    high=1 * interval,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        xT = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))

        padding = math.ceil((self.kernel_size - 1) / 2.0)
        x_conv = ops.conv(xT, self.weight, stride=self.stride, padding=padding)

        _, _, _, C_out = x_conv.shape
        assert C_out == self.out_channels

        if self.bias is not None:
            x_conv += ops.broadcast_to(
                ops.reshape(self.bias, (1, 1, 1, self.out_channels)), x_conv.shape
            )
        return ops.transpose(ops.transpose(x_conv, (2, 3)), (1, 2))
        ### END YOUR SOLUTION
