import torch
from torch import nn
from torch.autograd import Function
from torch import Tensor
import torch.nn.functional as F

# class LowerBound(Function):
#     @staticmethod
#     def forward(self, inputs, bound):

#         b = torch.ones(inputs.size()) * bound
#         # b = b.to(inputs.device)  # , device=inputs.device
#         self.save_for_backward(inputs, b)
#         return torch.max(inputs, b)

#     @staticmethod
#     def backward(self, grad_output):
#         inputs, b = self.saved_tensors

#         pass_through_1 = inputs >= b
#         pass_through_2 = grad_output < 0

#         pass_through = pass_through_1 | pass_through_2
#         return pass_through.type(grad_output.dtype) * grad_output, None


# class GDN(nn.Module):
#     """Generalized divisive normalization layer.
#     y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
#     """

#     def __init__(
#         self,
#         channels,
#         inverse=False,
#         beta_min=1e-6,
#         gamma_init=0.1,
#         reparam_offset=2**-18,
#     ):
#         super(GDN, self).__init__()
#         self.inverse = inverse
#         self.beta_min = beta_min
#         self.gamma_init = gamma_init
#         self.reparam_offset = torch.tensor([reparam_offset])

#         self.build(channels)

#     def build(self, channels):
#         self.pedestal = self.reparam_offset**2
#         self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
#         self.gamma_bound = self.reparam_offset

#         # Create beta param
#         beta = torch.sqrt(torch.ones(channels) + self.pedestal)
#         self.beta = nn.Parameter(beta)

#         # Create gamma param
#         eye = torch.eye(channels)
#         g = self.gamma_init * eye
#         g = g + self.pedestal
#         gamma = torch.sqrt(g)
#         self.gamma = nn.Parameter(gamma)

#     def forward(self, inputs):
#         unfold = False
#         if inputs.dim() == 5:
#             unfold = True
#             bs, channels, d, w, h = inputs.size()
#             inputs = inputs.view(bs, channels, d * w, h)

#         _, channels, _, _ = inputs.size()

#         # Beta bound and reparam
#         beta = LowerBound.apply(self.beta, self.beta_bound)
#         beta = beta**2 - self.pedestal

#         # Gamma bound and reparam
#         gamma = LowerBound.apply(self.gamma, self.gamma_bound)
#         gamma = gamma**2 - self.pedestal
#         gamma = gamma.view(channels, channels, 1, 1)

#         # Norm pool calc
#         norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
#         norm_ = torch.sqrt(norm_)

#         # Apply norm
#         if self.inverse:
#             outputs = inputs * norm_
#         else:
#             outputs = inputs / norm_

#         if unfold:
#             outputs = outputs.view(bs, channels, d, w, h)
#         return outputs


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)
