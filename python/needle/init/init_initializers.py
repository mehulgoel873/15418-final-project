import math
from .init_basic import *
from typing import Any, Optional


GAINS = {"relu": math.sqrt(2)}


def xavier_uniform(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(*(fan_in, fan_out), low=-1 * a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(*(fan_in, fan_out), mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    shape: Optional[tuple[int]] = None,
    **kwargs: Any
) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = GAINS[nonlinearity] * math.sqrt(3.0 / fan_in)
    # print(**kwargs)
    if shape:
        return rand(*shape, low=-1 * bound, high=bound, **kwargs)
    else:
        return rand(*(fan_in, fan_out), low=-1 * bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(
    fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any
) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = GAINS[nonlinearity] * math.sqrt(1.0 / fan_in)
    return randn(*(fan_in, fan_out), mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
