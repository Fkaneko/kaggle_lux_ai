from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# from timm.models.resnet import downsample_avg
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.modeling.model_arch.layers import BasicConv2d


# GeM from
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.2/cirtorch/layers/pooling.py#L36
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def downsample_avg(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
):
    """
    copy from timm resnet
    """
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = (
            # AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            nn.AvgPool2d
        )
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[
            pool,
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
        ]
    )


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad: bool = True):
        super().__init__()
        self.p = nn.Parameter(data=torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class LuxResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        out_channels: int = 5,
        num_layers: int = 12,
        filters: int = 32,
        filter_size: Tuple[int, int] = (3, 3),
        zero_init_residual: bool = True,
        remove_head: bool = False,
        use_point_conv: bool = False,
    ):
        """
        from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
        taking (20, 32, 32) input and return (5,) with unit position mask
        """
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, filters, filter_size, True)
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    inplanes=filters,
                    planes=filters,
                    stride=1,
                    downsample=None,
                )
                for _ in range(num_layers // 2)
            ]
        )
        self.remove_head = remove_head

        if remove_head:
            self.stride_blocks = nn.ModuleList(
                [
                    BasicBlock(
                        inplanes=filters * 2 ** i,
                        planes=filters * 2 ** (1 + i),
                        stride=2,
                        downsample=downsample_avg(
                            filters * 2 ** i, filters * 2 ** (1 + i), 3, stride=2
                        ),
                    )
                    for i in range(2)
                ]
            )

            self.use_point_conv = use_point_conv
            if use_point_conv:
                self.point_conv = BasicConv2d(filters * 4, filters * 2, (1, 1), True)
            else:
                self.pooled_head = nn.Linear(filters * 4, filters * 2, bias=False)

            self.pool = nn.AdaptiveAvgPool2d(1)

        else:
            self.head_p = nn.Linear(filters, out_channels, bias=False)
        # from https://github.com/pytorch/vision/blob/a00c905d8ec906e5f1e9496c8b77dbc7fd1b9235/torchvision/models/resnet.py#L206-L221
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(
        self,
        x: torch.Tensor,
        aux_inputs: Optional[torch.Tensor] = None,
    ):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = block(h)
        if self.remove_head:
            downsampled_feat = h.clone()
            for block in self.stride_blocks:
                downsampled_feat = block(downsampled_feat)
            if self.use_point_conv:
                downsampled_feat = self.point_conv(downsampled_feat)
                pooled = self.pool(downsampled_feat).squeeze()
            else:
                pooled = self.pool(downsampled_feat).squeeze()
                pooled = self.pooled_head(pooled)
            return {"feature": h, "final_hidden": downsampled_feat, "pooled": pooled}
        else:
            # mask with target unit with * x[:, :] and then take Global Sum Pooling
            h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
            p = self.head_p(h_head)
            return {"outputs": p, "aux_out": None}


class LuxNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        out_channels: int = 5,
        num_layers: int = 12,
        filters: int = 32,
        filter_size: Tuple[int, int] = (3, 3),
    ):
        """
        from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
        taking (20, 32, 32) input and return (5,) with unit position mask
        """
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, filters, filter_size, True)
        self.blocks = nn.ModuleList(
            [
                BasicConv2d(filters, filters, filter_size, True)
                for _ in range(num_layers)
            ]
        )
        self.head_p = nn.Linear(filters, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        aux_inputs: Optional[torch.Tensor] = None,
    ):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        # mask with target unit with * x[:, :] and then take Global Sum Pooling
        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return {"outputs": p, "aux_out": None}
