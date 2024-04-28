import copy
#import torchvision
import mindspore.nn as nn
#import torch.nn.functional as F
import mindspore.ops as ops
from compressai.models.priors import MeanScaleHyperprior
from models.slim_ops import AdaConv2d, AdaConvTranspose2d, SlimGDNPlus
from models.mlp_mixer import mlp_backbone, mlp_mixer_backbone

conv = lambda c_in, c_out, kernel_size=5, stride=2, in_shape_static=False, out_shape_static=False, M_mapping=None : AdaConv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, M_mapping=M_mapping, in_shape_static=in_shape_static, out_shape_static=out_shape_static)
deconv = lambda c_in, c_out, kernel_size=5, stride=2, in_shape_static=False, M_mapping=None : AdaConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, M_mapping=M_mapping, output_padding=stride-1, in_shape_static=in_shape_static)


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)


class AdaMeanScaleHyperprior(MeanScaleHyperprior):
    def __init__(self, N_list, M_list, **kwargs):
        super().__init__(N_list[-1], M_list[-1], **kwargs)
        self.N_list = N_list
        self.M_list = sorted(list(set(M_list)))
        self.M_mapping = M_mapping = [self.M_list.index(x) for x in M_list]
        M_list = self.M_list
        self.g_a = nn.SequentialCell([
            conv(3, N_list),  # 3 -> N
            SlimGDNPlus(N_list),
            conv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list),
            conv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list),
            conv(N_list, M_list, out_shape_static=True, M_mapping=M_mapping),  # N -> M
        ])

        self.g_s = nn.SequentialCell([
            deconv(M_list, N_list, in_shape_static=True, M_mapping=M_mapping),  # M -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, 3),  # N -> 3
        ])
        
        self.h_a = nn.SequentialCell([
            conv(M_list, N_list, stride=1, kernel_size=3, in_shape_static=True, M_mapping=M_mapping), # M -> N (fake M[-1])
            nn.LeakyReLU(),
            conv(N_list, N_list), # N -> N
            nn.LeakyReLU(),
            conv(N_list, N_list, out_shape_static=True), # N -> N (fake N[-1])
        ])

        self.h_s = nn.SequentialCell([
            deconv(N_list, M_list, in_shape_static=True, M_mapping=M_mapping), # N -> M (fake N[-1])
            nn.LeakyReLU(),
            deconv(M_list, [k * 3 // 2 for k in M_list], M_mapping=M_mapping), # M -> M
            nn.LeakyReLU(),
            conv([k * 3 // 2 for k in M_list], [k * 2 for k in M_list], stride=1, kernel_size=3, out_shape_static=True, M_mapping=M_mapping), # M -> M (fake M[-1])
        ])
        
    def set_running_N(self, N):
        idx = self.g_a[0].out_channels_list.index(N)
        #for n, m in self.named_modules():
        for n, m in self.cells_and_names():
            if hasattr(m, "out_channels_list") and len(m.out_channels_list) > 1:
                # print(n)
                set_exist_attr(m, "channel_choice", idx)
    
    def set_running_width(self, N):
        self.set_running_N(N)

class RoutingAgent(nn.Cell):
    def __init__(self, pred_delta=True, use_mbv3=None, use_mlp=None, downsample=False, legacy=False):
        super().__init__()
        self.use_mbv3 = use_mbv3
        self.use_mlp = use_mlp
        self.downsample = downsample
        # 修改待定 ------------------------------------------------------
        if self.use_mbv3 is not None:
            print("Using MBV3")
            if self.use_mbv3 == "large":
                self.backbone = torchvision.models.mobilenet_v3_large()
            else:
                self.backbone = torchvision.models.mobilenet_v3_small()
            self.cost_head = copy.deepcopy(self.backbone.classifier)
            self.route_head = copy.deepcopy(self.backbone.classifier)
            self.cost_head[-1] = nn.Dense(self.cost_head[-1].in_features, 4 if pred_delta else 5)
            self.route_head[-1] = nn.Dense(self.route_head[-1].in_features, 5)
        elif self.use_mlp is not None:
            print("Using MLP agent")
            depth = 8  # formerly 8 or 12
            dim = 768
            # self.backbone = mlp_backbone(image_size=256, channels=3, depth=depth,
            #                              patch_size=16, dim=dim, expansion_factor=0.5)
            self.backbone = mlp_mixer_backbone(image_size=256, channels=3, depth=depth,
                                               patch_size=16, dim=dim)
            self.cost_head = nn.Dense(dim, 4 if pred_delta else 5)  # for regression
            self.route_head = nn.Dense(dim, 5)  # for classification (decision-making)
        else:
            print("Using in-house agent")
            self.conv1 = nn.Conv2d(3, 32, 3, 2)  # 63, 3.4M
            self.conv2 = nn.Conv2d(32, 64, 3, 2)  # 31, 17.7M
            self.conv3 = nn.Conv2d(64, 128, 3, 2)  # 15, 16.6M
            self.conv4 = nn.Conv2d(128, 256, 3, 2)  # 7, 14.5M
            self.pool = nn.AdaptiveAvgPool2d(1) if not legacy else nn.AvgPool2d(7)  # for ckpt compatability
            self.fc = nn.Dense(256, 512)
            self.gelu = nn.GELU()
            self.cost_head = nn.Dense(512, 4 if pred_delta else 5)  # for regression
            self.route_head = nn.Dense(512, 5)  # for classification (decision-making)

    def forward(self, x):
        if self.downsample:
            #x = F.interpolate(x, size=(128, 128))
            x = ops.interpolate(x, size=(128, 128))
        if self.use_mbv3 is not None:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = x.flatten(1)
        elif self.use_mlp is not None:
            x = self.backbone(x)
        else:
            #         print(x.shape)
            x = self.conv1(x)
            #         print(x.shape)
            x = self.gelu(x)
            x = self.conv2(x)
            #         print(x.shape)
            x = self.gelu(x)
            x = self.conv3(x)
            #         print(x.shape)
            x = self.gelu(x)
            x = self.conv4(x)
            #         print(x.shape)
            x = self.pool(x)
            x = self.fc(x.flatten(1))
            x = self.gelu(x)

        return self.cost_head(x), self.route_head(x)
