import mindspore
from mindspore import nn, ops, Tensor

from util.box_ops import box_cxcywh_to_xyxy

from models.backbone import build_backbone
from models.Shunted_CA.cdn_doq import build_cdn
from models.Shunted_CA.doq_components import prepare_for_hqm, doq_post_process


class CDNHOI(nn.Cell):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False,
                 args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Dense(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Dense(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1, has_bias=True, pad_mode='pad')
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction
        self.transformer.interaction_decoder.nROI_Head = args.nROI_Head
        self.transformer.interaction_decoder.start_layer = args.start_layer
        self.transformer.interaction_decoder.scale_verb = args.scale_verb

        if self.use_matching:
            self.matching_embed = nn.Dense(hidden_dim, 2)

        # todo add fc to encoder sp
        self.bbox_enc = MLP(12, hidden_dim, hidden_dim, 2)
        #######

    def construct(self, x, m, doq_args=None, hard=True, gt_lr_cross=False):

        src, mask, pos = self.backbone(x, m)
        assert mask is not None

        if doq_args is not None:
            # todo prepare for doq
            input_query_with_gt_position_query, attn_mask, tgt_padding_mask, mask_dict, query_box, orig_target_sizes, gt_len = \
                prepare_for_hqm(doq_args, src.shape[0], self.bbox_enc, self.query_embed.embedding_table, hard, gt_lr_cross)
            ####

            hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, input_query_with_gt_position_query, pos[-1], None,
                                                                 attn_mask, gt_len, orig_target_sizes, query_box)[:2]

            outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
            outputs_obj_class = self.obj_class_embed(hopd_out)
            if self.use_matching:
                outputs_matching = self.matching_embed(hopd_out)

            outputs_verb_class = self.verb_class_embed(interaction_decoder_out)

            # todo
            outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord = \
                doq_post_process(outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, mask_dict)
            #####

            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
            if self.use_matching:
                out['pred_matching_logits'] = outputs_matching[-1]

            if self.aux_loss:
                if self.use_matching:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            outputs_matching)
                else:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord)

            return out, mask_dict
        else:

            hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.embedding_table,
                                                                 pos[-1])[:2]

            outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
            outputs_obj_class = self.obj_class_embed(hopd_out)
            if self.use_matching:
                outputs_matching = self.matching_embed(hopd_out)

            outputs_verb_class = self.verb_class_embed(interaction_decoder_out)

            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
            if self.use_matching:
                out['pred_matching_logits'] = outputs_matching[-1]

            if self.aux_loss:
                if self.use_matching:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            outputs_matching)
                else:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord)

            return out

    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord,
                      outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, \
                     'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in
                    zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1], \
                        outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1], \
                        outputs_matching[-min_dec_layers_num: -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in
                    zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1], \
                        outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1])]


class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList(nn.Dense(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = ops.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PostProcessHOI(object):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    def __call__(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = ops.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = ops.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = ops.stack([img_w, img_h, img_w, img_h], axis=1)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob = obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[
                index]
            sl = ops.full_like(ol, self.subject_category_id)
            l = ops.cat((sl, ol))
            b = ops.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = ops.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    backbone = build_backbone(args)

    cdn = build_cdn(args)

    model = CDNHOI(
        backbone,
        cdn,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, postprocessors
