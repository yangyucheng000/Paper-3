import math
import mindspore
from mindspore import nn, ops, Tensor
from mindspore import numpy as np


class PositionEmbeddingSine(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.dim_t = Tensor(np.arange(self.num_pos_feats, dtype=np.float32))
        self.eps = 1e-6
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.stack = ops.Stack(axis=4)
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.reshape = ops.Reshape()
        self.cumsum = ops.CumSum()
        self.pow = ops.Pow()
        self.concat = ops.Concat(axis=3)
        self.transpose = ops.Transpose()

    def construct(self, x, mask):
        mask = self.cast(mask, mindspore.int32)
        not_mask = ops.Abs()(mask - 1)
        y_embed = self.cumsum(not_mask, 1)
        x_embed = self.cumsum(not_mask, 2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = self.pow(self.temperature, 2 * (self.dim_t // 2) / self.num_pos_feats)
        dim_t = self.cast(dim_t, x.dtype)

        pos_x = self.expand_dims(x_embed, -1)
        pos_y = self.expand_dims(y_embed, -1)
        pos_x = pos_x / dim_t
        pos_y = pos_y / dim_t

        a, b, c = pos_x.shape[:3]
        pos_x = self.stack((self.sin(pos_x[:, :, :, 0::2]), self.cos(pos_x[:, :, :, 1::2])))
        pos_x = self.reshape(pos_x, (a, b, c, -1))
        pos_y = self.stack((self.sin(pos_y[:, :, :, 0::2]), self.cos(pos_y[:, :, :, 1::2])))
        pos_y = self.reshape(pos_y, (a, b, c, -1))
        pos = self.concat((pos_y, pos_x))
        pos = self.transpose(pos, (0, 3, 1, 2))

        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    return PositionEmbeddingSine(N_steps, normalize=True)
