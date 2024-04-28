from typing import List, Optional, Tuple

import mindspore
from mindspore import ops, nn, Parameter
from mindspore.common.initializer import initializer

Tensor = mindspore.Tensor

def _pad(input, pad, mode='constant', value=0):
    # type: (Tensor, List[int], str, float) -> Tensor
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim(), 'Padding length too large'
    return ops.pad(input, pad, mode, value)


# We define this function as _pad because it takes an argument
# named pad, which clobbers the recursive reference to the pad
# function needed for __torch_function__ support
pad = _pad


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    tgt_len, bsz, embed_dim = query.shape
    assert embed_dim == embed_dim_to_check
    assert key.shape == value.shape

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if ops.equal(query, key) and ops.equal(key, value):
            # self-attention
            q, k, v = ops.dense(query, in_proj_weight, in_proj_bias).chunk(3, axis=-1)

        elif ops.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = ops.dense(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = ops.dense(key, _w, _b).chunk(2, axis=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = ops.dense(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = ops.dense(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = ops.dense(value, _w, _b)
    else:
        q_proj_weight_non_opt = _unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == query.shape[-1]

        k_proj_weight_non_opt = _unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == key.shape[-1]

        v_proj_weight_non_opt = _unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == value.shape[-1]

        if in_proj_bias is not None:
            q = ops.dense(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = ops.dense(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = ops.dense(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = ops.dense(query, q_proj_weight_non_opt, in_proj_bias)
            k = ops.dense(key, k_proj_weight_non_opt, in_proj_bias)
            v = ops.dense(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.ndim == 3:

            if list(attn_mask.shape) != [bsz * num_heads, query.shape[0], key.shape[0]]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.ndim))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = ops.cat([k, bias_k.tile((1, bsz, 1))])
            v = ops.cat([v, bias_v.tile((1, bsz, 1))])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.view(tgt_len, bsz * num_heads, head_dim).swapaxes(0, 1)
    if k is not None:
        k = k.view(-1, bsz * num_heads, head_dim).swapaxes(0, 1)
    if v is not None:
        v = v.view(-1, bsz * num_heads, head_dim).swapaxes(0, 1)

    if static_k is not None:
        assert static_k.shape[0] == bsz * num_heads
        assert static_k.shape[2] == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.shape[0] == bsz * num_heads
        assert static_v.shape[2] == head_dim
        v = static_v

    src_len = k.shape[1]

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len

    if add_zero_attn:
        src_len += 1
        k = ops.cat([k, ops.zeros((k.shape[0], 1) + k.shape[2:], dtype=k.dtype)], axis=1)
        v = ops.cat([v, ops.zeros((v.shape[0], 1) + v.shape[2:], dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = ops.bmm(q, k.swapaxes(1, 2))
    assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == mindspore.bool_:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = ops.softmax(attn_output_weights, axis=-1)
    attn_output_weights = ops.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = ops.bmm(attn_output_weights, v)
    assert list(attn_output.shape) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.swapaxes(0, 1).view(tgt_len, bsz, embed_dim)
    attn_output = ops.dense(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Cell):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(ops.zeros((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(ops.zeros((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(ops.zeros((embed_dim, self.vdim)))
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(ops.zeros((3 * embed_dim, embed_dim)))
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = Parameter(ops.zeros((3 * embed_dim)))
        else:
            self.in_proj_bias = None
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(ops.zeros((1, 1, embed_dim)))
            self.bias_v = Parameter(ops.zeros((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(initializer('xavier_uniform', self.in_proj_weight.shape, mindspore.float32))
        else:
            self.q_proj_weight = Parameter(initializer('xavier_uniform', self.q_proj_weight, mindspore.float32))
            self.k_proj_weight = Parameter(initializer('xavier_uniform', self.k_proj_weight, mindspore.float32))
            self.v_proj_weight = Parameter(initializer('xavier_uniform', self.v_proj_weight, mindspore.float32))

        if self.in_proj_bias is not None:
            self.in_proj_bias = Parameter(initializer('zeros', self.in_proj_bias.shape, mindspore.float32))
            self.out_proj.bias = Parameter(initializer('zeros', self.out_proj.bias.shape, mindspore.float32))

        if self.bias_k is not None:
            self.bias_k = Parameter(initializer('xavier_normal', self.bias_k.shape, mindspore.float32))

        if self.bias_v is not None:
            self.bias_v = Parameter(initializer('xavier_normal', self.bias_v.shape, mindspore.float32))

    def construct(self, query, key, value, key_padding_mask=None,
                  need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
