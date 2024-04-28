import copy

import mindspore
from mindspore import ops, Tensor

from util.box_ops import box_cxcywh_to_xyxy, box_iou, box_area


def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = ops.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = ops.minimum(boxes1[:, 2:], boxes2[:, 2:])

    wh = ops.clamp(rb - lt, min=0)
    inter = wh[:, 0] * wh[:, 1]

    iou = inter / (area1 + area2 - inter)
    return iou


def encoder_gt_postion(targets, batch_size):
    enc_targets = copy.deepcopy(targets)
    gt_positions = []
    for i in range(batch_size):
        sub_boxes = enc_targets[i]['sub_boxes']  # 4
        obj_boxes = enc_targets[i]['obj_boxes']  # 4

        sub_diff = ops.zeros_like(sub_boxes)
        sub_diff[:, :2] = sub_boxes[:, 2:] / 2
        sub_diff[:, 2:] = sub_boxes[:, 2:]
        sub_boxes += ops.mul((ops.rand_like(sub_boxes) * 2 - 1.0), sub_diff) * 0.2
        sub_boxes = sub_boxes.clamp(min=0.0, max=1.0)

        obj_diff = ops.zeros_like(obj_boxes)
        obj_diff[:, :2] = obj_boxes[:, 2:] / 2
        obj_diff[:, 2:] = obj_boxes[:, 2:]
        obj_boxes += ops.mul((ops.rand_like(obj_boxes) * 2 - 1.0), obj_diff) * 0.2
        obj_boxes = obj_boxes.clamp(min=0.0, max=1.0)

        # note that the position and area is normalized by img's orig size(w, h)
        # because the boxes is normalized after data transformation
        sub_obj_position = sub_boxes[:, 0:2] - obj_boxes[:, 0:2]  # 2
        sub_area = (sub_boxes[:, 2] * sub_boxes[:, 3]).unsqueeze(1)  # 1
        obj_area = (obj_boxes[:, 2] * obj_boxes[:, 3]).unsqueeze(1)  # 1

        gt_position = ops.cat([sub_boxes, obj_boxes, sub_obj_position, sub_area, obj_area], axis=1)

        if len(gt_position) == 0:
            gt_position = ops.zeros((0, 12), dtype=mindspore.float32)
        else:
            gt_position = Tensor(gt_position, dtype=mindspore.float32)

        gt_positions.append(gt_position)
    return gt_positions


def encoder_gt_postion_iou(targets, batch_size):

    enc_targets = copy.deepcopy(targets)
    gt_positions = []
    for i in range(batch_size):
        sub_boxes = enc_targets[i]['sub_boxes']  # 4
        obj_boxes = enc_targets[i]['obj_boxes']  # 4

        thres_low = 0.6
        thres_high = 1
        shift_r = 0.7

        count = 0
        count_max = 40
        augmented1 = True

        while augmented1 and count < count_max:

            sub_diff = ops.zeros_like(sub_boxes)
            sub_diff[:, :2] = sub_boxes[:, 2:] / 2
            sub_diff[:, 2:] = sub_boxes[:, 2:]
            sub_boxes_o = sub_boxes + ops.mul((ops.rand_like(sub_boxes) * 2 - 1.0), sub_diff) * 0.2
            sub_boxes_o = sub_boxes_o.clamp(min=0.0, max=1.0)

            iou = box_iou(box_cxcywh_to_xyxy(sub_boxes_o), box_cxcywh_to_xyxy(sub_boxes))
            iou_rate = ops.bitwise_and(ops.cast((iou > thres_low), mindspore.int32),
                                       ops.cast((iou < thres_high), mindspore.int32))

            # print(f"iou_rate: {iou_rate}, count: {count}")
            if (iou_rate.float().sum() / len(sub_boxes)) >= shift_r or count == count_max:
                augmented1 = False
                sub_boxes[iou_rate] = sub_boxes_o[iou_rate]
            count += 1

        count = 0
        count_max = 40
        augmented1 = True

        while augmented1 and count < count_max:
            obj_diff = ops.zeros_like(obj_boxes)
            obj_diff[:, :2] = obj_boxes[:, 2:] / 2
            obj_diff[:, 2:] = obj_boxes[:, 2:]
            obj_boxes_o = obj_boxes + ops.mul((ops.rand_like(obj_boxes) * 2 - 1.0), obj_diff) * 0.2
            obj_boxes_o = obj_boxes_o.clamp(min=0.0, max=1.0)

            iou = box_iou(box_cxcywh_to_xyxy(obj_boxes_o), box_cxcywh_to_xyxy(obj_boxes))
            iou_rate = ops.bitwise_and(ops.cast((iou > thres_low), mindspore.int32),
                                       ops.cast((iou < thres_high), mindspore.int32))
            # iou_rate = (iou > thres_low) & (iou < thres_high)

            if (iou_rate.float().sum() / len(obj_boxes)) >= shift_r or count == count_max:
                augmented1 = False
                obj_boxes[iou_rate] = obj_boxes_o[iou_rate]
            count += 1

        # note that the position and area is normalized by img's orig size(w, h)
        # because the boxes is normalized after data transformation
        sub_obj_position = sub_boxes[:, 0:2] - obj_boxes[:, 0:2]  # 2
        sub_area = (sub_boxes[:, 2] * sub_boxes[:, 3]).unsqueeze(1)  # 1
        obj_area = (obj_boxes[:, 2] * obj_boxes[:, 3]).unsqueeze(1)  # 1

        gt_position = ops.cat([sub_boxes, obj_boxes, sub_obj_position, sub_area, obj_area], axis=1)

        if len(gt_position) == 0:
            gt_position = ops.zeros((0, 12), dtype=mindspore.float32)
        else:
            gt_position = Tensor(gt_position, dtype=mindspore.float32)

        gt_positions.append(gt_position)

    return gt_positions


def prepare_for_hqm(doq_args, batch_size, bbox_enc, query_embedding_weight, high_q=True, gt_lr_cross=False):

    targets = doq_args

    if high_q:
        gt_positions = encoder_gt_postion_iou(targets, batch_size)
    else:
        gt_positions = encoder_gt_postion(targets, batch_size)
    gt_positions_padding_size = int(max([len(gt_positions[v]) for v in range(batch_size)]))
    gt_positions_arrays = ops.zeros((gt_positions_padding_size, batch_size, 12), dtype=mindspore.float32)
    gt_positions_mask = ops.zeros((batch_size, gt_positions_padding_size), dtype=mindspore.bool_)
    gt_body_box = ops.zeros([gt_positions_padding_size, batch_size, 4]).astype(gt_positions_arrays.dtype)
    for i in range(batch_size):
        gt_position = gt_positions[i]
        if len(gt_position) > 0:
            gt_positions_arrays[0:len(gt_position), i, :] = gt_position
            gt_positions_mask[i, len(gt_position):] = True

    query_embedding_len = query_embedding_weight.shape[0]
    query_embedding_mask = ops.zeros(batch_size, query_embedding_len, dtype=mindspore.bool_)
    gt_positions_mask = ops.cat([gt_positions_mask, query_embedding_mask], axis=1)

    gt_positions_arrays = gt_positions_arrays
    human_box = gt_positions_arrays[:, :, :4]
    obj_box = gt_positions_arrays[:, :, 4:8]
    query_box = [human_box, obj_box]
    input_gt_position_embedding = ops.tanh(bbox_enc(gt_positions_arrays)).permute(1, 0, 2)
    query_embedding_weight = query_embedding_weight.tile((batch_size, 1, 1))
    input_query_with_gt_position_query = ops.cat([input_gt_position_embedding, query_embedding_weight], axis=1)

    query_size = input_query_with_gt_position_query.shape[1]
    attn_mask = ops.ones(query_size, query_size, dtype=mindspore.bool_) * float('-inf')
    attn_mask[0:gt_positions_padding_size, 0:gt_positions_padding_size] = 0
    attn_mask[gt_positions_padding_size:, gt_positions_padding_size:] = 0
    if gt_lr_cross:
        attn_mask[0:gt_positions_padding_size, gt_positions_padding_size:] = 0
    mask_dict = {
        'padding_size': gt_positions_padding_size,
        'targets': targets
    }
    orig_target_sizes = ops.stack([t["orig_size"] for t in doq_args], axis=0)
    gt_len = mask_dict['padding_size']
    return input_query_with_gt_position_query, attn_mask, gt_positions_mask, mask_dict, query_box, orig_target_sizes, gt_len


def doq_post_process(outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, mask_dict):
    """
    post process of doq after output from the transformer
    put the doq part in the mask_dict
    """
    if mask_dict and mask_dict['padding_size'] > -1:
        outputs_known_obj_class = outputs_obj_class[:, :, :mask_dict['padding_size'], :]
        outputs_known_verb_class = outputs_verb_class[:, :, :mask_dict['padding_size'], :]
        outputs_known_sub_coord = outputs_sub_coord[:, :, :mask_dict['padding_size'], :]
        outputs_known_obj_coord = outputs_obj_coord[:, :, :mask_dict['padding_size'], :]

        outputs_obj_class = outputs_obj_class[:, :, mask_dict['padding_size']:, :]
        outputs_verb_class = outputs_verb_class[:, :, mask_dict['padding_size']:, :]
        outputs_sub_coord = outputs_sub_coord[:, :, mask_dict['padding_size']:, :]
        outputs_obj_coord = outputs_obj_coord[:, :, mask_dict['padding_size']:, :]

        mask_dict['output_known_gt'] = {'gt_obj_logits': outputs_known_obj_class[-1], 'gt_verb_logits': outputs_known_verb_class[-1], 'gt_sub_boxes': outputs_known_sub_coord[-1], 'gt_obj_boxes': outputs_known_obj_coord[-1]}
        aux_output = [{'gt_obj_logits': a, 'gt_verb_logits': b, 'gt_sub_boxes': c, 'gt_obj_boxes': d} for a, b, c, d in zip(outputs_known_obj_class[:-1], outputs_known_verb_class[:-1], outputs_known_sub_coord[:-1], outputs_known_obj_coord[:-1])]

        mask_dict['output_known_gt'].update({'aux_output': aux_output})
    return outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord
