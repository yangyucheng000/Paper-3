import os
import sys
sys.path.append("..")
import numpy as np
import scipy.stats
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
import numpy as np
import numpy as numpy
import mindspore
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import ops as ops
import mindspore.numpy as nps
from mindspore.common.initializer import initializer, Zero, One, Uniform
from mindspore import Parameter,ParameterTuple




from compressai._CXX import \
    pmf_to_quantized_cdf as _pmf_to_quantized_cdf  # pylint: disable=E0611,E0401
from compressai.ops import LowerBound

import ms_adapter


class _EntropyCoder:
    """Proxy class to an actual entropy coder class.
    """
    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders
        if method not in available_entropy_coders():
            methods = ', '.join(available_entropy_coders())
            raise ValueError(f'Unknown entropy coder "{method}"'
                             f' (available: {methods})')

        if method == 'ans':
            from compressai import ans  # pylint: disable=E0611
            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == 'rangecoder':
            import range_coder  # pylint: disable=E0401
            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder
    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf, precision=16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = ms.Tensor(cdf, dtype=ms.int32)
    return cdf


class EntropyModel(nn.Cell):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """
    def __init__(self,
                 likelihood_bound=1e-9,
                 entropy_coder=None,
                 entropy_coder_precision=16):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def construct(self, *args):
        raise NotImplementedError()

    def _get_noise_cached(self, x):
        # use simple caching method to avoid creating a new tensor every call
        half = float(0.5)
        if not hasattr(self, '_noise'):
            setattr(self, '_noise', x.new(x.size()))
        self._noise.resize_(x.size())
        self._noise.uniform_(-half, half)
        return self._noise

    def _quantize(self, inputs, mode, means=None):
        # type: (Tensor, str, Optional[Tensor]) -> Tensor
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == 'noise':
           noise = self._get_noise_cached(inputs)
           inputs = inputs + noise
           return inputs

        outputs = inputs.copy()
        if means is not None:
            outputs -= means

        outputs = ms.ops.Rint()(outputs)

        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs

        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs

    @staticmethod
    def _dequantize(inputs, means=None):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.float()
        return outputs

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = ms.ops.zeros(
            (len(pmf_length), max_length + 2), ms.int32)
        for i, p in enumerate(pmf):
            prob = ms.ops.concat((p[: pmf_length[i]], tail_mass[i]), axis=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.shape[0]] = _cdf
        return cdf

    def _check_cdf_size(self):
        if  self._quantized_cdf.size== 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.shape) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.shape}")

    def _check_offsets_size(self):
        if self._offset.size == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.shape) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.shape}")

    def _check_cdf_length(self):
        if self._cdf_length.size == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.shape) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.shape}")

    
    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self._quantize(inputs, 'symbols', means)

        if len(inputs.shape) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.shape != indexes.shape:
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().asnumpy().tolist(),
                indexes[i].reshape(-1).int().asnumpy().tolist(),
                self._quantized_cdf.asnumpy().tolist(),
                self._cdf_length.reshape(-1).int().asnumpy().tolist(),
                self._offset.reshape(-1).int().asnumpy().tolist(),
            )
            strings.append(rv)
        
        return strings

    #重点 均继承于此
    def decompress(self, strings, indexes, means=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError('Invalid `strings` parameter type.')

        if not len(strings) == indexes.size(0):
            raise ValueError('Invalid strings or indexes parameters')

        if len(indexes.shape) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.shape[:-2] != indexes.shape[:-2]:
                raise ValueError('Invalid means or indexes parameters')
            if means.shape != indexes.shape and \
                    (means.shape[2] != 1 or means.shape[3] != 1):
                raise ValueError('Invalid means parameters')

        cdf = self._quantized_cdf
        # outputs = cdf.new(indexes.shape) #是个函数，对应cdf的数据
        outputs = ms.numpy.zeros(shape=indexes.shape, dtype=cdf.dtype)

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s, indexes[i].reshape(-1).int().asnumpy().tolist(), cdf.asnumpy().tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().asnumpy().tolist())
            outputs[i] = ms.Tensor(
                values, dtype=outputs.dtype
            ).reshape(outputs[i].shape)
        outputs = self.dequantize(outputs, means, dtype)
        return outputs

'''
class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`_
    for an introduction.
    """
    def __init__(self,
                 channels,
                 *args,
                 tail_mass=1e-9,
                 init_scale=10,
                 filters=(3, 3, 3, 3),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        filters = (1, ) + self.filters + (1, )
        scale = self.init_scale**(1 / (len(self.filters) + 1))
        channels = self.channels
        
        self._biases = []
        self._factors = []
        self._matrices = []
        

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            # matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            # matrix.data.fill_(init)
            # self._matrices.append(nn.Parameter(matrix))
            # change
            matrix = initializer(init, shape=(channels, filters[i + 1], filters[i]), dtype=mindspore.float32)
            self._matrices.append(mindspore.Parameter(matrix))

            # bias = torch.Tensor(channels, filters[i + 1], 1)
            # nn.init.uniform_(bias, -0.5, 0.5)
            # self._biases.append(nn.Parameter(bias))
            # change
            bias = np.random.uniform(0.5, -0.5, (channels, filters[i + 1], 1))
            bias = Tensor(bias, mindspore.float32)
            self._biases.append(mindspore.Parameter(bias))

            if i < len(self.filters):
                # factor = torch.Tensor(channels, filters[i + 1], 1)
                # nn.init.zeros_(factor)
                # self._factors.append(nn.Parameter(factor))
                # change
                zeros = ops.Zeros()
                factor = zeros((channels, filters[i + 1], 1), mindspore.float32)
                self._factors.append(mindspore.Parameter(factor))

        self.quantiles = ms.Parameter(ms.Tensor(np.zeros((channels, 1, 3), dtype=np.float32)))
        init = ms.Tensor([-self.init_scale, 0, self.init_scale], dtype=ms.float32)
        # self.quantiles = ms_adapter.repeat(init, self.quantiles.shape[0], 1, 1)
        self.quantiles.set_data(ms_adapter.repeat(init, self.quantiles.shape[0], 1, 1))
        
        target = numpy.log(2 / self.tail_mass - 1)
        #self.register_buffer("target", torch.Tensor([-target, 0, target]))
        self.target = ms.Parameter(ms.Tensor((-target, 0, target), dtype=ms.float32),  requires_grad=False, name="target")

    def _medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # pylint: disable=E0203
            return

        medians = self.quantiles[:, 0, 1]

        # minima = medians - self.quantiles[:, 0, 0]
        # minima = torch.ceil(minima).int()
        # minima = torch.clamp(minima, min=0)
        #
        # maxima = self.quantiles[:, 0, 2] - medians
        # maxima = torch.ceil(maxima).int()
        # maxima = torch.clamp(maxima, min=0)
        # change

        minima = medians - self.quantiles[:, 0, 0]
        rint = ops.Rint()
        ceil = ops.Ceil()
        minima = rint(ops.ceil(minima))
        # minima = torch.clamp(minima, min=0)  TODO BY J: no clamp in mindspore

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = rint(ceil(maxima))
        # maxima = torch.clamp(maxima, min=0)  TODO BY J: no clamp in mindspore

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max()
        # samples = torch.arange(max_length)
        # change
        samples = Tensor(mindspore.numpy.arange(max_length), mindspore.float32)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        # sign = -torch.sign(lower + upper)
        # pmf = torch.abs(
        #     torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        sign = ops.Sign()
        abs = ops.Abs()
        sigmoid = ops.Sigmoid()
        sign = -sign(lower + upper)
        pmf = abs(
            sigmoid(sign * upper) - sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        # tail_mass = torch.sigmoid(lower[:, 0, :1]) +\
        #     torch.sigmoid(-upper[:, 0, -1:])
        # change
        tail_mass = sigmoid(lower[:, 0, :1]) + \
                    sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        

    def loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = ms.numpy.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self._matrices[i]
            # change no detach in mindspore
            # if stop_gradient:
            #     matrix = matrix.detach()
            # logits = torch.matmul(F.softplus(matrix), logits) #矩阵与input相乘，每一层乘在一起
            # change
            logits = ms.ops.matmul(ops.Softplus()(matrix), logits)

            bias = self._biases[i]
            # change no detach in mindspore
            # if stop_gradient:
            #     bias = bias.detach()
            logits += bias

            if i < len(self._factors):
                factor = self._factors[i]
                # change no detach in mindspore
                # if stop_gradient:
                #     factor = factor.detach()
                # logits += torch.tanh(factor) * torch.tanh(logits)
                # change
                tanh = ops.Tanh()
                logits += tanh(factor) * tanh(logits)
        return logits

    #@torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -ms.ops.Sign()(lower + upper)
        sign = ms.ops.stop_gradient(sign)
        likelihood = ms.numpy.abs(
            ms_adapter.sigmoid(sign * upper) - ms_adapter.sigmoid(sign * lower)
        )
        return likelihood

    def construct(self, x):
        x_shape = x.shape
        perm = np.arange(len(x_shape))
        tmp = perm[0]
        perm[0] = perm[1]
        perm[1] = tmp
        # Compute inverse permutation
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x = x.permute(*perm)
        shape = x.shape
        values = x.reshape(x.shape[0], 1, -1)
        

        outputs = self._quantize(values,
                                 'noise' if self.training else 'dequantize',
                                 self._medians())

        
        likelihood = self._likelihood(outputs)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm)

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm)
        
        print("outputs:",outputs)
        print("likelihood:",likelihood)

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        # indexes = torch.arange(C).view(*view_dims)
        indexes = ms_adapter.view(ms.numpy.arange(C), *view_dims)
        indexes = indexes.int()

        #return indexes.repeat(N, 1, *size[2:])
        return ms_adapter.repeat(indexes, N, 1, *size[2:])
        
    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)
    
    def compress(self, x):
        indexes = self._build_indexes(x.shape)
        medians = ms.ops.stop_gradient(self._medians())
        spatial_dims = len(x.shape) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        #medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        medians = ms_adapter.expand(medians, x.shape[0], *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.shape[0], *size)
        #indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        indexes = self._build_indexes(output_size)
        medians = self._extend_ndims(ms.ops.stop_gradient(self._medians()), len(size))
        #medians = medians.expand(len(strings), *([-1] * (len(size) + 1))) 
        medians = ms_adapter.expand(medians, len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians)
'''

class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(channels,*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        self.params_matrix = ParameterTuple(())  # 动态添加参数
        self.params_bias = ParameterTuple(())
        self.params_factor = ParameterTuple(())
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            #matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            #matrix.data.fill_(init)
            #self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))
            matrix = ms.Tensor(np.zeros((channels, filters[i + 1], filters[i]), dtype=np.float32))
            matrix = ms_adapter.fill(matrix, float(init))
            #self.p_matrix = ms.Parameter(matrix, name=f"_matrix{i:d}", requires_grad=True)
            self.params_matrix = self.params_matrix + ParameterTuple((ms.Parameter(matrix, name=f"tuple_matrix_{i:d}", requires_grad=True),))

            #bias = torch.Tensor(channels, filters[i + 1], 1)
            #nn.init.uniform_(bias, -0.5, 0.5)
            #self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))
            bias = ms.Tensor(np.zeros((channels, filters[i + 1], 1), dtype=np.float32))
            bias = ms_adapter.uniform(bias, -0.5, 0.5)
            #self.p_bias = ms.Parameter(bias, name=f"_bias{i:d}", requires_grad=True)
            self.params_bias = self.params_bias + ParameterTuple((ms.Parameter(bias, name=f"tuple_bias_{i:d}", requires_grad=True),)) #验证过，初始化成功

            if i < len(self.filters):
                #factor = torch.Tensor(channels, filters[i + 1], 1)
                #nn.init.zeros_(factor)
                #self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))
                factor = ms.Tensor(np.zeros((channels, filters[i + 1], 1), dtype=np.float32))
                factor = ms_adapter.zeros(factor)
                #self.p_factor = ms.Parameter(factor, name=f"_factor{i:d}", requires_grad=True)
                self.params_factor = self.params_factor + ParameterTuple((ms.Parameter(factor, name=f"tuple_factor_{i:d}", requires_grad=True),))

        #self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        #init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        #self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        self.quantiles = ms.Parameter(ms.Tensor(np.zeros((channels, 1, 3), dtype=np.float32)))
        init = ms.Tensor([-self.init_scale, 0, self.init_scale], dtype=ms.float32)
        # self.quantiles = ms_adapter.repeat(init, self.quantiles.shape[0], 1, 1)
        self.quantiles.set_data(ms_adapter.repeat(init, self.quantiles.shape[0], 1, 1))
        
        target = numpy.log(2 / self.tail_mass - 1)
        #self.register_buffer("target", torch.Tensor([-target, 0, target]))
        self.target = ms.Parameter(ms.Tensor((-target, 0, target), dtype=ms.float32),  requires_grad=False, name="target")

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        #if self._offset.numel() > 0 and not force:
        if ms.ops.size(self._offset) > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        #minima = torch.ceil(minima).int()
        minima = ms.ops.ceil(minima).int()
        #minima = torch.clamp(minima, min=0)
        minima = ms_adapter.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = ms.ops.ceil(maxima).int()
        maxima = ms_adapter.clamp(maxima, min=0)
        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        #max_length = pmf_length.max().item()

        max_length =  int(pmf_length.to(ms.float32).max().item(0))
        #device = pmf_start.device
        #samples = torch.arange(max_length, device=device)
        samples = ms.numpy.arange(max_length)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        #sign = -torch.sign(lower + upper)
        sign = -ms.ops.Sign()(lower + upper)
        #pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)) ms.numpy.abs
        pmf = ms.numpy.abs(ms_adapter.sigmoid(sign * upper) - ms_adapter.sigmoid(sign * lower))
        
        pmf = pmf[:, 0, :]
        tail_mass = ms_adapter.sigmoid(lower[:, 0, :1]) + ms_adapter.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf.set_data(quantized_cdf, slice_shape=True)
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = ms.numpy.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            #matrix = getattr(self, f"_matrix{i:d}")
            matrix = self.params_matrix[i]
            # assert(matrix.name in 'matrix')

            if stop_gradient:
                matrix = ms.ops.stop_gradient(matrix)
            #logits = torch.matmul(F.softplus(matrix), logits)
            logits = ms.ops.matmul(ops.Softplus()(matrix), logits)

            #bias = getattr(self, f"_bias{i:d}")
            bias = self.params_bias[i]
            # assert(bias.name in 'bias')

            if stop_gradient:
                bias = ms.ops.stop_gradient(bias)
            logits += bias

            if i < len(self.filters):
                #factor = getattr(self, f"_factor{i:d}")
                factor = self.params_factor[i]
                # assert(factor.name in 'factor')

                if stop_gradient:
                    factor = ms.ops.stop_gradient(factor)
                logits += ms.ops.tanh(factor) * ms.ops.tanh(logits)
        return logits

    #@torch.jit.unused
    def _likelihood(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -ms.ops.Sign()(lower + upper)
        sign = ms.ops.stop_gradient(sign)
        likelihood = ms.numpy.abs(
            ms_adapter.sigmoid(sign * upper) - ms_adapter.sigmoid(sign * lower)
        )
        return likelihood

    def construct(self, x: Tensor, training: Optional[bool] = None) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        # if not torch.jit.is_scripting():
        #     # x from B x C x ... to C x B x ...
        #     perm = np.arange(len(x.shape))
        #     perm[0], perm[1] = perm[1], perm[0]
        #     # Compute inverse permutation
        #     inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        # else:
        #     raise NotImplementedError()
        #     # TorchScript in 2D for static inference
        #     # Convert to (channels, ... , batch) format
        #     # perm = (1, 2, 3, 0)
        #     # inv_perm = (3, 0, 1, 2)
        x_shape = x.shape
        perm = np.arange(len(x_shape))
        tmp = perm[0]
        perm[0] = perm[1]
        perm[1] = tmp
        # Compute inverse permutation
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]

        x = x.permute(*perm)
        shape = x.shape
        values = x.reshape(x.shape[0], 1, -1)
        # values = x.reshape(x.shape[0], 1, shape[1]*shape[2]*shape[3])

        # Add noise or quantize
        
        outputs = self._quantize(values,
                                 'noise' if self.training else 'dequantize',
                                 self._get_medians())

        # if not torch.jit.is_scripting():
        #     likelihood = self._likelihood(outputs)
        #     if self.use_likelihood_bound:
        #         likelihood = self.likelihood_lower_bound(likelihood)
        # else:
        #     raise NotImplementedError()
        #     # TorchScript not yet supported
        #     # likelihood = torch.zeros_like(outputs)
        likelihood = self._likelihood(outputs)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm)

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm)
        
        # print("outputs:",outputs)
        # print("likelihood:",likelihood)

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        # indexes = torch.arange(C).view(*view_dims)
        indexes = ms_adapter.view(ms.numpy.arange(C), *view_dims)
        indexes = indexes.int()

        #return indexes.repeat(N, 1, *size[2:])
        return ms_adapter.repeat(indexes, N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.shape)
        medians = ms.ops.stop_gradient(self._get_medians())
        spatial_dims = len(x.shape) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        #medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        medians = ms_adapter.expand(medians, x.shape[0], *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.shape[0], *size)
        #indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        indexes = self._build_indexes(output_size)
        medians = self._extend_ndims(ms.ops.stop_gradient(self._get_medians()), len(size))
        #medians = medians.expand(len(strings), *([-1] * (len(size) + 1))) 
        medians = ms_adapter.expand(medians, len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)        

##################################################################  今天改到这里  2023/02/09   #####################################################################################  
####  这段代码没有改，没有用到

class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`_.
    """
    def __init__(self,
                 scale_table=None,
                 *args,
                 scale_bound=0.11,
                 tail_mass=1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(
                f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(
                f'Invalid scale_table length "{len(scale_table)}"')


        self.scale_table = self._prepare_scale_table(scale_table) if scale_table else None

        self.scale_bound = Tensor([float(scale_bound)], mindspore.float32) if scale_bound is not None else None
        
        
        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError('Invalid parameters')

    @staticmethod
    def _prepare_scale_table(scale_table):
        return Tensor(tuple(float(s) for s in scale_table), mindspore.float32)

    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * ms.ops.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.size > 0 and not force:
            return
        self.scale_table = self._prepare_scale_table(scale_table)
        self.update()

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = ms.ops.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = ms_adapter.max(pmf_length).asnumpy().tolist()

        samples = ms.numpy.abs(
            ms.numpy.arange(max_length).int() - pmf_center[:, None]
        )
        samples_scale = ops.expand_dims(self.scale_table, 1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = ms.Tensor(np.zeros((len(pmf_length), max_length + 2), dtype=np.float32))
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf.set_data(quantized_cdf, slice_shape=True)
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

        values = ms.numpy.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def construct(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        outputs = self._quantize(inputs,
                                 'noise' if self.training else 'dequantize',
                                 means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = ms.numpy.full(scales.shape, len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes  #统计数量



class GaussianMixtureConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`_.
    """
    def __init__(self,
                 K,
                 scale_table=None,
                 mean_table=None,
                 weight_table=None,
                 *args,
                 scale_bound=0.11,
                 tail_mass=1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)

        #ywz for mixture numbers:K
        self.K = K

        if scale_table and \
                (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        # self.register_buffer(
        #     'scale_table',
        #     self._prepare_scale_table(scale_table)
        #     if scale_table else torch.Tensor())
        #
        # self.register_buffer(
        #     'scale_bound',
        #     torch.Tensor([float(scale_bound)])
        #     if scale_bound is not None else None)
        # change # no register in mindspore
        self.scale_table = self._prepare_scale_table(scale_table) if scale_table else None

        self.scale_bound = Tensor([float(scale_bound)], mindspore.float32) if scale_bound is not None else None

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError('Invalid parameters')

    @staticmethod
    def _prepare_scale_table(scale_table):
        # return torch.Tensor(tuple(float(s) for s in scale_table))
        # change
        return Tensor(tuple(float(s) for s in scale_table), mindspore.float32)

    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * ms.ops.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return
        self.scale_table = self._prepare_scale_table(scale_table)
        self.update()

    def update(self):
        # multiplier = -self._standardized_quantile(self.tail_mass / 2)
        # pmf_center = torch.ceil(self.scale_table * multiplier).int()
        # pmf_length = 2 * pmf_center + 1
        # max_length = torch.max(pmf_length).item()
        # change
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        ceil = ops.Ceil()
        pmf_center = ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = pmf_length.max()

        # samples = torch.abs(
        #     torch.arange(max_length).int() - pmf_center[:, None])
        # samples_scale = self.scale_table.unsqueeze(1)
        # samples = samples.float()
        # samples_scale = samples_scale.float()
        # upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        # lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        # pmf = upper - lower
        # change
        abs = ops.Abs()
        samples = abs(
            Tensor(mindspore.numpy.arange(max_length), mindspore.float32).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = Tensor(len(pmf_length), max_length + 2, mindspore.float32)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2


    #GMM
    def _likelihood(self, inputs, scales, means=None,weights=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor

        likelihood = None
        #获取M
        # M = inputs.size()[1] #通道数  除了inputs，另外三个的通道数均为M*K!!
        # change
        M = inputs.shape[1]  # 通道数  除了inputs，另外三个的通道数均为M*K!!

        # add_k  = 0
        for k in range(self.K):
            half = float(0.5)

            values = inputs - means[:,(M*k):((k+1)*M)]

            temp_scales = self.lower_bound_scale(scales[:,(M*k):((k+1)*M)])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

            # values = torch.abs(values)
            # change
            abs = ops.Abs()
            values = abs(values)
            upper = self._standardized_cumulative((half - values) / temp_scales)
            lower = self._standardized_cumulative((-half - values) / temp_scales)
            if likelihood==None:
                likelihood = (upper - lower)*weights[:,(M*k):((k+1)*M)] #指定对应的M个通道
            else:
                likelihood += (upper - lower)*weights[:,(M*k):((k+1)*M)] #指定对应的M个通道

            # if likelihood==None:
            #     likelihood = (upper - lower)*1/self.K #指定对应的M个通道
            # else:
            #     likelihood += (upper - lower)*1/self.K #指定对应的M个通道


        return likelihood

    #GMM
    def construct(self, inputs, scales, means=None,weights=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        outputs = self._quantize(inputs,
                                 'noise' if self.training else 'dequantize',
                                 means=None) #changed: means=None
        #GMM
        likelihood = self._likelihood(outputs, scales, means,weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(),
                                  len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes  #统计数量
 

if __name__ == '__main__':
    '''
    # test for EntropyBottleneck
    a = EntropyBottleneck(128)
    zeros = ops.Zeros()
    x = zeros((16, 128, 4, 4), mindspore.float32)
    out, likeli = a.construct(x)
    print(out.shape, likeli.shape)
    '''
    
 
    # test for GaussianCondition
    a = GaussianConditional()
    zeros = ops.Zeros()
    y = zeros((32, 192, 256, 256), mindspore.float32)
    g1 = zeros((32, 192, 256, 256), mindspore.float32)
    g2 = zeros((32, 192, 256, 256), mindspore.float32)
    output, likelihood = a.construct(y,g1,g2)
    print(output.shape, likelihood.shape)
    print("GC construct ok!")


   

    # test for GaussianMixtureConditional
    GMC = GaussianMixtureConditional(K = 5)
    zeros = ops.Zeros()
    y = zeros((32, 192, 256, 256), mindspore.float32)
    g1 = zeros((32, 960, 256, 256), mindspore.float32)
    g2 = zeros((32, 960, 256, 256), mindspore.float32)
    g3 = zeros((32, 960, 1, 1), mindspore.float32)
    y1_hat, y1_likelihoods = GMC.construct(y, g1, g2, g3)
    print(y1_hat.shape, y1_likelihoods.shape)
    print("GMC construct ok!")

