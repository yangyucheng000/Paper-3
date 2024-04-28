import copy
from typing import Optional, List

import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from .Shumted_MHA import MultiheadAttention as Shumted_MultiheadAttention
from util.box_ops import box_cxcywh_to_xyxy
import random


class CDN(nn.Cell):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, shuffle=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm((d_model,)) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        decoder_norm = nn.LayerNorm((d_model,))

        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        interaction_decoder_norm = nn.LayerNorm((d_model,))

        self.interaction_decoder = TransformerDecoderLocalMTGTMH(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.shuffle = shuffle

    def _reset_parameters(self):
        for p in self.get_parameters():
            if p.ndim > 1:
                p = Parameter(initializer('xavier_uniform', p.shape, p.dtype))

    def construct(self, src, mask, query_embed, pos_embed, tgt_padding_mask=None, attn_mask=None, gt_len=None, orig_target_sizes=None, query_box=None):
        bs, c, h, w = src.shape
        src = ops.flatten(src, start_dim=2).permute(2, 0, 1)
        pos_embed = ops.flatten(pos_embed, start_dim=2).permute(2, 0, 1)

        if attn_mask is not None:
            # todo query have been converted to (2, max_len+100, 256)
            query_embed = query_embed.permute(1, 0, 2)
        else:
            query_embed = query_embed.unsqueeze(1).tile((1, bs, 1))
        mask_ = mask
        mask = ops.flatten(mask, start_dim=1)

        tgt = ops.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # todo add attn mask and padding mask
        hopd_out = self.decoder(tgt, memory, tgt_mask=attn_mask, tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        ###
        hopd_out = hopd_out.swapaxes(1, 2)

        interaction_query_embed = hopd_out[-1]

        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)

        interaction_tgt = ops.zeros_like(interaction_query_embed)
        # todo add attn mask and padding mask
        if self.training:
            interaction_decoder_out = self.interaction_decoder.construct_AT(interaction_tgt, memory, tgt_mask=attn_mask, tgt_key_padding_mask=tgt_padding_mask,
                                 memory_key_padding_mask=mask, pos=pos_embed, query_pos=interaction_query_embed, mask=mask_, gt_len=gt_len,
                                orig_target_sizes=orig_target_sizes, query_box=query_box, LA=True, shuffle=self.shuffle)
        else:
            interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, tgt_mask=attn_mask,
                                                               tgt_key_padding_mask=tgt_padding_mask,
                                                               memory_key_padding_mask=mask, pos=pos_embed,
                                                               query_pos=interaction_query_embed)
        ######
        interaction_decoder_out = interaction_decoder_out.swapaxes(1, 2)

        return hopd_out, interaction_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Cell):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Cell):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return ops.stack(intermediate)

        return output


class TransformerDecoderLocalMTGTMH(nn.Cell):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, nhead=8):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.hum_bbox_embed = None
        self.obj_bbox_embed = None
        self.hoi_class_fc = None
        self.obj_class_fc = None
        self.matcher = None
        self.obj_logit_scale = None
        self.obj_visual_projection = None
        self.nhead = nhead
        self.nROI_head = 2
        self.global_head = nhead-self.nROI_head*2
        self.scale_object = 1.5
        self.scale_verb = 1.5
        self.start_layer = 1
        self.indx = [0, 1, 2]

    def get_mask_from_query_gt_union(self, outputs_sub_coord, outputs_obj_coord, img_GT, mask, num_queries_max, scale=2):

        outputs_sub_coord = outputs_sub_coord.swapaxes(0, 1)
        outputs_obj_coord = outputs_obj_coord.swapaxes(0, 1)
        bboxes = []

        for human_box, obj_box in zip(outputs_sub_coord, outputs_obj_coord):
            human_box = box_cxcywh_to_xyxy(human_box)
            obj_box = box_cxcywh_to_xyxy(obj_box)

            assert human_box.shape[0] == obj_box.shape[0]
            lt = ops.minimum(human_box[:, :2], obj_box[:, :2])
            rb = ops.maximum(human_box[:, 2:], obj_box[:, 2:])
            u = ops.cat((lt, rb), axis=1)
            bboxes.append(u)

        orig_target_sizes = ops.stack([t["orig_size"] for t in img_GT], axis=0)
        img_h, img_w = orig_target_sizes.unbind(1)

        scale_fcts = ops.stack([img_w // 32, img_h // 32, img_w // 32, img_h // 32], axis=1)

        gt_coord_xyxy = [(bbox * scale_fct.view(1, 4)).long() for bbox, scale_fct in zip(bboxes, scale_fcts)]

        mask_object = []

        l = 100
        assert len(gt_coord_xyxy) == mask.shape[0]
        for i in range(mask.shape[0]):  # bz

            bbox = gt_coord_xyxy[i]
            bbox = bbox.view(-1, 4)
            num_bbox = bbox.shape[0]

            for j in range(num_queries_max):
                if j >= num_bbox:
                    cur_mask_object = ops.zeros((mask.shape[1], mask.shape[2]), dtype=mindspore.bool_)
                else:
                    w1, h1, w2, h2 = bbox[j]

                    if w2 - w1 < l:
                        w1 = (w1 // scale).long()
                        w2 = (w2 * scale).long()
                    if h2 - h1 < l:
                        h1 = (h1 // scale).long()
                        h2 = (h2 * scale).long()
                    w2 = ops.minimum(w2, Tensor(mask.shape[2] - 1).long())
                    h2 = ops.minimum(h2, Tensor(mask.shape[1] - 1).long())
                    cur_mask_object = ops.zeros((mask.shape[1], mask.shape[2]), dtype=mindspore.bool_)

                    if h1 > 0:
                        cur_mask_object = cur_mask_object.index_fill(0, ops.arange(0, h1), True)
                    if h2 + 1 < cur_mask_object.shape[0]:
                        cur_mask_object = cur_mask_object.index_fill(0, ops.arange(h2 + 1,cur_mask_object.shape[0]), True)
                    if w1 > 0:
                        cur_mask_object = cur_mask_object.index_fill(1, ops.arange(0, w1), True)
                    if w2 + 1 < cur_mask_object.shape[1]:
                        cur_mask_object = cur_mask_object.index_fill(1, ops.arange(w2 + 1, cur_mask_object.shape[1]),True)

                mask_object.append(cur_mask_object)

        mask_object = ops.stack(mask_object, 0)  # (bz * num_query, H/32, W/32)
        mask_object = mask_object.view(mask.shape[0], num_queries_max, mask_object.shape[-2], mask_object.shape[-1])

        return mask_object

    def get_mask_from_query_global(self, outputs_sub_coord, outputs_obj_coord, img_GT, mask, num_queries_max, scale=2):

        mask_object = ops.zeros((mask.shape[0], num_queries_max, mask.shape[1], mask.shape[2]), dtype=mindspore.bool_)

        return mask_object


    def get_mask_from_query_gt_shift(self, outputs_coord, orig_target_sizes, mask, num_queries_max, union=False, scale=2):
        img_h, img_w = orig_target_sizes.unbind(1)

        scale_fcts = ops.stack([img_w // 32, img_h // 32, img_w // 32, img_h // 32], axis=1)
        if scale_fcts.shape[0]!=outputs_coord.shape[0]:
            outputs_coord = outputs_coord.transpose(0, 1)

        bboxes = box_cxcywh_to_xyxy(outputs_coord)
        gt_coord_xyxy = bboxes * scale_fcts[:, None, :]

        mask_object = []
        l = 100

        for i in range(mask.shape[0]): # bz

            bbox = gt_coord_xyxy[i]
            bbox = bbox.view(-1, 4)
            num_bbox = bbox.shape[0]

            for j in range(num_queries_max):
                if j >= num_bbox:
                    cur_mask_object = ops.zeros((mask.shape[1], mask.shape[2]), dtype=mindspore.bool_)
                else:
                    w1, h1, w2, h2 = bbox[j]

                    if w2 - w1 < l:
                        w1 = (w1 // scale).long()
                        w2 = (w2 * scale).long()
                    if h2-h1<l:
                        h1 = (h1 // scale).long()
                        h2 = (h2 * scale).long()
                    w2 = ops.minimum(w2, Tensor(mask.shape[2]-1).long())
                    h2 = ops.minimum(h2, Tensor(mask.shape[1]-1).long())
                    cur_mask_object = ops.zeros((mask.shape[1], mask.shape[2]), dtype=mindspore.bool_)

                    if h1 > 0:
                        cur_mask_object = cur_mask_object.index_fill(0, ops.arange(0, h1), True)
                    if h2 + 1 < cur_mask_object.shape[0]:
                        cur_mask_object = cur_mask_object.index_fill(0, ops.arange(h2 + 1, cur_mask_object.shape[0]), True)
                    if w1 > 0:
                        cur_mask_object = cur_mask_object.index_fill(1, ops.arange(0, w1), True)
                    if w2 + 1 < cur_mask_object.shape[1]:
                        cur_mask_object = cur_mask_object.index_fill(1, ops.arange(w2 + 1, cur_mask_object.shape[1]), True)

                mask_object.append(cur_mask_object)

        mask_object = ops.stack(mask_object, 0) # (bz * num_query, H/32, W/32)
        mask_object = mask_object.view(mask.shape[0], num_queries_max, mask_object.shape[-2], mask_object.shape[-1])

        return mask_object

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return ops.stack(intermediate)

        return output

    def construct_AT(self, tgt, memory,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None,
                   query_pos: Optional[Tensor] = None,
                   mask=None,
                   gt_len=None, orig_target_sizes=None, query_box=None, LA=False, shuffle=False):

        output = tgt

        intermediate = []

        # AT
        for i, layer in enumerate(self.layers):

            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos

            if i >= self.start_layer:

                if gt_len == 0:
                    memory_mask = None
                else:
                    outputs_sub_coord = query_box[0]
                    outputs_obj_coord = query_box[1]

                    assert memory_mask is None

                    mask_object = self.get_mask_from_query_gt_shift(outputs_obj_coord, orig_target_sizes, mask, gt_len,
                                                                    scale=self.scale_verb)

                    mask_object = ops.flatten(mask_object, start_dim=2, end_dim=3).unsqueeze(1).broadcast_to((-1, self.nROI_head, -1, -1))

                    mask_human = self.get_mask_from_query_gt_shift(outputs_sub_coord, orig_target_sizes, mask, gt_len,
                                                                   scale=self.scale_verb)

                    mask_human = ops.flatten(mask_human, start_dim=2, end_dim=3).unsqueeze(1).broadcast_to((-1, self.nROI_head, -1, -1))

                    memory_mask_global = self.get_mask_from_query_global(outputs_sub_coord, outputs_obj_coord, orig_target_sizes,
                                                                         mask, gt_len, scale=self.scale_verb)

                    memory_mask_global = ops.flatten(memory_mask_global, start_dim=2, end_dim=3).unsqueeze(1).broadcast_to((-1, self.global_head, -1, -1))
                    mask_list = [memory_mask_global, mask_human, mask_object]

                    if shuffle:
                        random.shuffle(self.indx)
                    memory_mask_gt = ops.cat((mask_list[self.indx[0]], mask_list[self.indx[1]], mask_list[self.indx[2]]), 1)

                    memory_mask_gt = ops.flatten(memory_mask_gt, start_dim=0, end_dim=1)
                    memory_mask = ops.zeros((memory_mask_gt.shape[0], this_query_pos.shape[0], memory_mask_gt.shape[-1]), dtype=mindspore.bool_)
                    memory_mask[:, :memory_mask_gt.shape[1], :] = memory_mask_gt

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            memory_mask = None

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return ops.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def construct_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def construct_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def construct(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.construct_pre(src, src_mask, src_key_padding_mask, pos)
        return self.construct_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Cell):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Shumted_MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.norm3 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def construct_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def construct_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def construct(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.construct_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.construct_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(Cell, N):
    return nn.CellList([copy.deepcopy(Cell) for i in range(N)])


def build_cdn(args):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        shuffle=args.shuffle
    )

def _get_activation_fn(activation):
    if activation == "relu":
        return ops.relu
    if activation == "gelu":
        return ops.gelu
    if activation == "glu":
        return ops.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
