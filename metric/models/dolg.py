import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .resnetrs import Resnet101RSGeM


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class MultiAtrousModule(nn.Module):

    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 1024) -> None:
        super(MultiAtrousModule, self).__init__()

        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, 1, bias=False), nn.BatchNorm2d(out_channels // 2),
                                     nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels // 2, rate))

        modules.append(ASPPPooling(in_channels, out_channels // 2))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(nn.Conv2d(len(self.convs) * out_channels // 2, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class AttentionModule(nn.Module):
    '''
    https://github.com/tensorflow/models/blob/master/research/delf/delf/python/training/model/delf_model.py
    https://github.com/feymanpriv/DELG/blob/master/model/delg_model.py
    '''

    def __init__(self, in_channels: int, out_channels: int = 1024) -> None:
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = x
        x0 = F.normalize(x0, p=2, dim=1)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.softplus(x)
        return x * x0, x


class FusionModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = 512, eps: float = 1e-7) -> None:
        super(FusionModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)
        self.eps = eps

    def forward(self, global_feature: torch.Tensor, local_feature: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            global_feature (N,C)
            local_feature (N,C,W,H)
        """
        if local_feature is None:
            x = self.fc(global_feature)
            return x
        fl, fg = local_feature, global_feature
        fl = self.pool(fl)
        fl = fl.flatten(start_dim=1)
        fl_dot = (fl * fg).sum(dim=1, keepdim=True)
        fg_norm = torch.norm(fg, dim=1, keepdim=True)
        fl_proj = fl_dot / (fg_norm + self.eps) * fg
        fl_orth = fl - fl_proj
        x = torch.cat([global_feature, fl_orth], dim=1)
        x = self.fc(x)
        return x


class Dolg(nn.Module):

    def __init__(self, embedding_size: int, pretrained: bool = True, cs: int = 1024, global_only: bool = False) -> None:
        super(Dolg, self).__init__()
        self.global_model = Resnet101RSGeM(embedding_size=cs, pretrained=pretrained)

        # Atrous rates:
        # 512x512: 6, 12, 18
        # 256x256: 3, 6, 9
        self.local_model = nn.Sequential(MultiAtrousModule(in_channels=1024, atrous_rates=[3, 6, 9], out_channels=cs),
                                         AttentionModule(in_channels=cs, out_channels=cs))
        self.global_only = global_only
        if self.global_only:
            self.fusion = FusionModule(cs, embedding_size)
        else:
            self.fusion = FusionModule(2 * cs, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feature, feamap = self.global_model(x)
        if self.global_only:
            x = self.fusion(global_feature)
        else:
            local_feature, _ = self.local_model(feamap)
            x = self.fusion(global_feature, local_feature)
        x = F.normalize(x, p=2, dim=1)
        return x


class DolgInfer(Dolg):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feature, feamap = self.global_model(x)
        if self.global_only:
            x = self.fusion(global_feature)
        else:
            local_feature, local_attn = self.local_model(feamap)
            x = self.fusion(global_feature, local_feature)
        x = F.normalize(x, p=2, dim=1)
        return x, local_attn
