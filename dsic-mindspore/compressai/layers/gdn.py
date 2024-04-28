# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import mindspore as ms
import mindspore as mindspore
import mindspore.ops as ops
import mindspore.nn as nn
ms.set_context(device_target="GPU")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from mindspore import Tensor
# from torch import Tensor

from compressai.ops.parametrizers import NonNegativeParametrizer

__all__ = ["GDN", "GDN1"]


class GDN(nn.Cell):
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
        beta = ops.ones(in_channels, mindspore.float32)
        beta = self.beta_reparam.init(beta)
        self.beta = ms.Parameter(beta, name="beta", requires_grad=True)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * ops.eye(in_channels, in_channels, mindspore.float32)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = ms.Parameter(gamma, name="gamma", requires_grad=True)

    def construct(self, x: Tensor) -> Tensor: #x (N,Cin,Hin,Win) 
        _, C, _, _ = x.shape

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape((C, C, 1, 1)) #(out_channels, in_channels,kH,kW)
        norm = ops.conv2d(x ** 2, gamma, pad_mode='pad')
        norm = ops.bias_add(norm, beta)

        if self.inverse:
            norm = ops.Sqrt()(norm)
        else:
            norm = ops.Rsqrt()(norm)

        out = ops.mul(x, norm) #逐元素相乘，存在广播机制

        return out


class GDN1(GDN):
    r"""Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}

    """

    def construct(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.shape

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape((C, C, 1, 1))
        norm = ops.conv2d(ops.abs(x), gamma, pad_mode='pad')
        norm = ops.bias_add(norm, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = ops.mul(x, norm)

        return out
