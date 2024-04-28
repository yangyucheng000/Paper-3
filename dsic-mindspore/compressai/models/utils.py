# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from compressai import ms_adapter


def find_named_module(cell, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """
    #return next((m for n, m in module.named_modules() if n == query), None)
    return next((c for n, c in cell.cells_and_names() if n == query), None)


def find_named_buffer(cell, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    #return next((b for n, b in module.named_buffers() if n == query), None)
    return next((b for n, b in cell.untrainable_params() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=ms.int32,
):
    new_size = state_dict[state_dict_key].shape
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.size == 0:
            registered_buf = registered_buf.resize(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        #module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
        ms_tensor = ms_adapter.fill(ms.numpy.empty(new_size, dtype=dtype), 0)
        module.buffer_name = ms.Parameter(ms_tensor, requires_grad=False, name=buffer_name)
        

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=ms.int32,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.untrainable_params()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_mode = 'pad',
        padding=kernel_size // 2,
        has_bias=True)

'''
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2dTranspose(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_mode='same',
        padding=0,
        has_bias=True
    )
'''

'''
def deconv(in_channels, out_channels, kernel_size=5, stride=2):  
    return nn.SequentialCell([
            nn.Conv2dTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=kernel_size // 2,
                has_bias=True
            ),
            nn.Pad(paddings=((0,0),(0,0),(0,1),(0,1)), mode="CONSTANT")
            
    ])
'''    
'''
def deconv1(in_channels, out_channels, kernel_size=5, stride=2):  
    return nn.Conv2dTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=kernel_size // 2,
                has_bias=True
            )
    )
'''    
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2dTranspose(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_mode='pad',
        padding=kernel_size // 2,
        has_bias=True
    )    
            
  