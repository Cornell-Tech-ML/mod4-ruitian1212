from typing import Tuple

from . import operators
from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand
from .fast_ops import FastOps


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    out = input.contiguous().view(batch, channel, height, new_width, kw)
    out = (
        out.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return out, new_height, new_width


# TODO: Implement for Task 4.3.


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Args:
    ----
        input (Tensor): batch x channel x height x width
        kernel (Tuple[int, int]): height x width of pooling

    Returns
    -------
        Tensor: Pooled tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = input.mean(4)

    return out.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


class Max(Function):
    """A class to implement the forward and backward function of max operation"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Output the maximum value of a certain tensor input in the certain dimension"""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Output the argmax value of a certain tensor in a certain dimension"""
        input, dim = ctx.saved_values
        max_values = max_reduce(input, int(dim.item()))
        argmax_values = max_values == input
        return (argmax_values * grad_output, 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Return the tensor of max values of a certain input in a certain dimension"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Return the tensor of the softmax version of the original tensor"""
    exp_input = input.exp()
    softmax_score = exp_input / exp_input.sum(dim=dim)
    return softmax_score


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Return the log of the softmax as a tensor using LSE trick."""
    max_value = max(input, dim)
    term1 = (input - max_value).exp().sum(dim=dim).log()
    return input - term1 - max_value


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Args:
    ----
        input (Tensor): batch x channel x height x width
        kernel (Tuple[int, int]): height x width of pooling

    Returns
    -------
        Tensor: Pooled tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = max(input, 4)

    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Randomly drop out values in the input tensor"""
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
