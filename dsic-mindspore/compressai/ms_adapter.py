# -*- coding: utf-8 -*-
import mindspore as ms
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.ops import constexpr

def fill(input, val):
    if not isinstance(val, (int, float, bool)):
            raise TypeError("For 'Tensor.fill', the type of the argument 'value' must be int, float or bool, "
                            "but got {}.".format(type(val)))
    output = tensor_operator_registry.get("fill")(input.dtype, input.shape, val)
    return output

def uniform(input, a, b):
    return ms.Tensor(ms.common.initializer._init_random_uniform(a, b, input.shape), dtype=input.dtype)

def zeros(input):
    return tensor_operator_registry.get("fill")(input.dtype, input.shape, 0.0)

def repeat(inputx, *sizes):
    if isinstance(sizes[0], (tuple, list)):
        output = ms.ops.tile(inputx, *sizes)
    else:
        output = ms.ops.tile(inputx, sizes)
    return output

def clamp(input, min=None, max=None, out=None):
    type = input.dtype
    if min is not None and max is not None and min > max:
        output = ms.ops.ones_like(input).astype(type)*max
    else:
        if min is not None:
            min = ms.Tensor(min, type)
        if max is not None:
            max = ms.Tensor(max, type)
        output = ms.ops.clip_by_value(input, min, max)
    return output

def sigmoid(input):
    return 1 / (ms.ops.exp(0 - input) + 1)

def view(input, *shape):
    #shape必须是（）不能是list
    output = tensor_operator_registry.get('reshape')()(input, shape)
    return output

def expand(input_ms, *size, is_under_gpu_context=True):
    @constexpr
    def size_to_ms_tensor(size):
        if isinstance(size[0], (list, tuple)):
            size = ms.Tensor(size[0])
        else:
            size = ms.Tensor(size)
        return size
    
    _size = size_to_ms_tensor(size)
    # TODO: ms.ops.expand() to support on GPU and delete 'broadcast_to' code.
    if is_under_gpu_context:
        return ms.ops.broadcast_to(input_ms, size)
    return input_ms.expand(_size)

def max(input, dim=None, keepdim=False):
    if dim is None:
        return input.max()
    #TODO
    # Until now, P.max do not support when `input` is type of `int32`, `int64``.
    if self.dtype == ms.int64 or self.dtype == ms.int32:
        raise TypeError("For 'Tensor.max', the type of `input` do not support `int64` and "
                        "`int32`, got {}.".format(dtype_name))

    indices, result = P.max(input, axis=dim, keep_dims=keepdim)
    return result, indices

# def tensor_bool_select(tensor_ms, index):
#     ms_shape_len = len(tensor_ms.shape)
#     index_shape_len = len(index.shape)
#     out_shape = [-1]
#     while index_shape_len < ms_shape_len:
#         out_shape.append(tensor_ms.shape[index_shape_len])
#         index = index.expand_dims(-1)
#         index_shape_len += 1
#     out = ms.ops.masked_select(tensor_ms, index)
#     if len(out_shape) > 1:
#         out = out.reshape(out_shape)

#     return out

def linspace(start, end, steps, dtype=None):
    if dtype is None:
        dtype = ms.float32
    start = ms.Tensor(start, dtype)
    end = ms.Tensor(end, dtype)
    output = ms.ops.linspace(start, end, steps)
    return output


def pad(input, pad, mode="constant", value=0):
    if mode == "replicate":
        mode = "edge"

    value = ms.Tensor(value, dtype=input.dtype)
    dims = len(input.shape)
    list_pad = [pad[i:i+2] for i in range(0, len(pad), 2)]
    list_pad.reverse()
    new_pad = [[0,0],] * int((dims - len(pad) /2))
    new_pad.extend(list_pad)

    @cast_tensor
    def _call_ms_api(input):
        # TODO: -> ms.ops.PadV3
        return ms.ops.operations.nn_ops.PadV3(mode=mode)(input, pad, value)

    outputs = _call_ms_api(input)
    return outputs