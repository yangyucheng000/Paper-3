import mindspore
from mindspore import ops, Tensor
tensor0 = Tensor(0, dtype=mindspore.float32)
tensor100 = Tensor(100, dtype=mindspore.float32)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_m(boxes1, boxes2, mode='max'):
    """
    boxes1: [n, 2]
    boxes2: [m, 2]
    return: [n, m, 2]
    """
    n, _ = boxes1.shape
    m, _ = boxes2.shape

    boxes1 = ops.Tile()(boxes1, (1, m))
    boxes1 = ops.Reshape()(boxes1, (n*m, 2))
    boxes2 = ops.Tile()(boxes2, (n, 1))
    if mode == 'max':
        outputs = ops.Maximum()(boxes1, boxes2)
    else:
        outputs = ops.Minimum()(boxes1, boxes2)
    outputs = ops.Reshape()(outputs, (n, m, 2))
    return outputs


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = ops.Unstack(axis=-1)(x)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return ops.Stack(axis=-1)(b)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = ops.Unstack(axis=-1)(x)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return ops.Stack(axis=-1)(b)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = box_m(boxes1[:, :2], boxes2[:, :2], mode='max')  # [N,M,2]
    rb = box_m(boxes1[:, 2:], boxes2[:, 2:], mode='min')  # [N,M,2]

    wh = rb - lt
    wh = ops.clip_by_value(wh, tensor0, tensor100)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = ops.ExpandDims()(area1, 1) + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)

    lt = box_m(boxes1[:, :2], boxes2[:, :2], mode='min')
    rb = box_m(boxes1[:, 2:], boxes2[:, 2:], mode='max')

    wh = rb - lt
    wh = ops.clip_by_value(wh, tensor0, tensor100)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return ops.zeros((0, 4))

    h, w = masks.shape[-2:]

    y = ops.arange(0, h, dtype=mindspore.float32)
    x = ops.arange(0, w, dtype=mindspore.float32)
    y, x = ops.meshgrid(y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten().max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten().min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten().min(-1)[0]

    return ops.stack([x_min, y_min, x_max, y_max], 1)
