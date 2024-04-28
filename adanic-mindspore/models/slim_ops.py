# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from compressai.ops.parametrizers import NonNegativeParametrizer

class AdaConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=None,
                 group=1,
                 has_bias=True,
                 padd_mode='pad',
                 M_mapping=None,
                 in_shape_static=False,
                 out_shape_static=False):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]
        if 1 < len(in_channels_list) != len(out_channels_list) > 1:
            assert M_mapping is not None  # for (N -> M), (M -> N)
        super(AdaConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            group=group,
            has_bias=has_bias,
            pad_mode=pad_mode)
        assert self.group in (1, self.out_channels) or len(in_channels_list) == len(out_channels_list) == 1, \
            'only support regular conv, pwconv and dwconv'
        if padding == None:
            padding = ((self.stride[0] - 1) + self.dilation[0] * (
                    self.kernel_size[0] - 1)) // 2
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_inc = self.in_channels if self.in_chn_static else None
        self.running_outc = self.out_channels if self.out_chn_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = self.group
        self.mode = 'largest'
        # self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(np.array(self.in_channels_list)).float().cuda()
        self.out_channels_list_tensor = torch.from_numpy(np.array(self.out_channels_list)).float().cuda()
        self.M_to_M = M_mapping is not None and len(in_channels_list) == len(out_channels_list)
        self.M_mapping = M_mapping
        self.in_shape_static = in_shape_static
        self.out_shape_static = out_shape_static

    def forward(self, x):
        # if self.prev_channel_choice is None:
        #     self.prev_channel_choice = self.channel_choice
        if self.M_to_M:
            self.channel_choice = self.M_mapping[self.channel_choice]  # for (M -> M) layer

        if 1 < len(self.in_channels_list) < len(self.out_channels_list):
            self.running_inc = self.in_channels_list[self.M_mapping[self.channel_choice]]  # for (M -> N) layer
        elif not self.in_chn_static:
            self.running_inc = self.in_channels_list[self.channel_choice]

        if self.in_shape_static:
            x = x[:, :self.running_inc]

        # hard slimming (on inference)
        if not self.in_chn_static:
            self.running_inc = x.size(1)
        if not self.out_chn_static:
            if 1 < len(self.out_channels_list) < len(self.in_channels_list):
                self.channel_choice = self.M_mapping[self.channel_choice]  # for (N -> M) layer
            self.running_outc = self.out_channels_list[self.channel_choice]
        weight = self.weight[:self.running_outc, :self.running_inc] # [out, in, h, w]
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        if self.group in (1, self.out_channels):
            self.running_groups = 1 if self.group == 1 else self.running_outc
        else:
            self.running_groups = self.group
        # self.prev_channel_choice = None
        self.channel_choice = -1
        ret = ops.conv2d(x,
                        weight,
                        pad_mode='pad',
                        self.padding,
                        self.stride,
                        self.dilation,
                        self.running_groups)
        if self.out_shape_static:
            N, C, H, W = ret.shape
            out = torch.zeros(N, self.out_channels_list[-1], H, W, device=ret.device)
            out[:, :C] = ret
            return out
        return ret

        
class AdaConvTranspose2d(nn.Conv2dTranspose):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=None,
                 group=1,
                 has_bias=True,
                 M_mapping=None,
                 in_shape_static=False):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]
        if 1 < len(in_channels_list) != len(out_channels_list) > 1:
            assert M_mapping is not None  # for (N -> M), (M -> N)
        super(AdaConvTranspose2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            group=group,
            has_bias=has_bias)
        assert self.group in (1, self.out_channels) or len(in_channels_list) == len(out_channels_list) == 1, \
            'only support regular conv, pwconv and dwconv'
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.channel_choice = -1  # dynamic channel list index
        self.in_chn_static = len(in_channels_list) == 1
        self.out_chn_static = len(out_channels_list) == 1
        self.running_inc = self.in_channels if self.in_chn_static else None
        self.running_outc = self.out_channels if self.out_chn_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_group = self.group
        self.mode = 'largest'
        # self.prev_channel_choice = None
        self.in_channels_list_tensor = ms.Tensor.from_numpy(np.array(self.in_channels_list)).float()
        self.out_channels_list_tensor = ms.Tensor.from_numpy(np.array(self.out_channels_list)).float()
        self.M_to_M = M_mapping is not None and len(in_channels_list) == len(out_channels_list)
        self.M_mapping = M_mapping
        self.in_shape_static = in_shape_static

    def forward(self, x):
        # if self.prev_channel_choice is None:
        #     self.prev_channel_choice = self.channel_choice
        if self.M_to_M:
            self.channel_choice = self.M_mapping[self.channel_choice]
        # print("Choice: {}".format(self.channel_choice))
        if 1 < len(self.in_channels_list) < len(self.out_channels_list):
            self.running_inc = self.in_channels_list[self.M_mapping[self.channel_choice]]  # for (M -> N) layer
        elif not self.in_chn_static:
            self.running_inc = self.in_channels_list[self.channel_choice]

        if self.in_shape_static:
            x = x[:, :self.running_inc]

        # hard slimming (on inference)
        if not self.in_chn_static:
            self.running_inc = x.size(1)
        if not self.out_chn_static:
            if 1 < len(self.out_channels_list) < len(self.in_channels_list):  # for (N -> M) layer
                self.channel_choice = self.M_mapping[self.channel_choice]
            self.running_outc = self.out_channels_list[self.channel_choice]

        weight = self.weight[:self.running_inc, :self.running_outc]
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        if self.group in (1, self.out_channels):
            self.running_groups = 1 if self.group == 1 else self.running_outc
        else:
            self.running_groups = self.group
        # self.prev_channel_choice = None
        self.channel_choice = -1
        return ops.Conv2DTranspose(out_channel=self.out_channels,
                                    kernel_size=self.kernel_size,
                                    mode=1,
                                    pad_mode='pad',
                                    pad=self.padding,
                                    stride=self.stride,
                                    dilation=self.dilation,
                                    group=self.running_groups)
                                    (x, weight, ops.shape(x))
                


class LowerBound(nn.Cell):
    def construct(self, inputs, bound):
        #b = ops.ones_like(inputs) * bound
        #ctx.save_for_backward(inputs, b)
        return ops.maximum(inputs, bound)  # 这里逐元素比较两个a,b的大小，bound不是dim
    
    def bprop(self, inputs, bound, out, grad_output):
        #inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= bound
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.astype(grad_output.dtype) * grad_output, None


class GDN(nn.Cell):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = ops.sqrt(ops.ones(ch)+self.pedestal)
        self.beta = ms.Parameter(beta, name='beta', requires_grad=True)

        # Create gamma param
        eye = ops.eye(ch, ch, ms.float32)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = ops.sqrt(g)

        self.gamma = ms.Parameter(gamma, name='gamma', requires_grad=True)
        self.pedestal = self.pedestal

    def construct(self, inputs):
        unfold = False
        if inputs.ndim == 5:
            unfold = True
            bs, ch, d, w, h = inputs.shape
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.shape

        # Beta bound and reparam
        self.beta_bound = ops.ones_like(self.beta) * self.beta_bound
        beta = LowerBound()(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        self.gamma_bound = ops.ones_like(self.gamma) * self.gamma_bound
        gamma = LowerBound()(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.Conv2d(ch, ch, kernel_size=1, weight_init=gamma, bias_init=beta, has_bias=True)(inputs**2)  # no bias
        norm_ = ops.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class SlimGDNPlus(GDN):
    def __init__(
        self,
        in_channels_list: list,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        if isinstance(in_channels_list, int):
            self.in_channels_list = [in_channels_list]
        else:
            self.in_channels_list = in_channels_list
        super().__init__(self.in_channels_list[-1], inverse, beta_min, gamma_init)
        self.affine_s = ms.Parameter(ms.Tensor(ms.ops.ones(2, len(self.in_channels_list))), name='affine_s')  # 2 for gamma, beta
        self.affine_b = ms.Parameter(ms.Tensor(ms.ops.Zeros(2, len(self.in_channels_list))), name='affine_b')  # 2 for gamma, beta
        self.running_inc = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        running_inc = x.shape[1]
        assert running_inc in self.in_channels_list, "Unsupported number of input channel (available: {})".format(self.in_channels_list)
        idx = self.in_channels_list.index(running_inc)
        gamma = self.gamma_reparam(self.affine_s[0, idx] * self.gamma[:running_inc, :running_inc] + self.affine_b[0, idx])
        gamma = gamma.reshape(running_inc, running_inc, 1, 1)
        beta = self.beta_reparam(self.affine_s[1, idx] * self.beta[:running_inc] + self.affine_b[1, idx])
         _, ch, _, _ = x.shape
        norm = nn.Conv2d(ch, ch, kernel_size=1, weight_init=gamma, bias_init=beta)(x ** 2)

        if self.inverse:
            norm = ops.sqrt(norm)
        else:
            norm = ops.Rsqrt(norm)

        out = x * norm

        return out