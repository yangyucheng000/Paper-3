import mindspore
from mindspore import nn, ops
from models.resnet import resnet50
from models.position_encoding import build_position_encoding


class Joiner(nn.Cell):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        self.position_embedding = position_embedding
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=0)

    def construct(self, x, mask):
        x = self.backbone(x)
        mask = ops.ResizeNearestNeighbor(size=x.shape[-2:])(self.expand_dims(mask, 0))
        mask = self.squeeze(mask)
        mask = self.cast(mask, mindspore.bool_)
        pos_embed = self.cast(self.position_embedding(x, mask), x.dtype)
        return x, mask, pos_embed


def build_backbone(args):
    backbone = resnet50(pretrained=args.pretrained)
    position_embedding = build_position_encoding(args)
    model = Joiner(backbone, position_embedding)
    return model
