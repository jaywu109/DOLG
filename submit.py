import os
import csv
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import faiss
import math
from itertools import repeat
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

# configuration
MODEL_PATH = 'saved_models/dolg_256_global/highest_recall.ckpt'
CROP_SIZE = 256
GLOBAL_ONLY = True

# Other
BATCH_SIZE = 64
DATA_PATH = '/home/data/kaggle/google_landmark_2021'
VISUALIZE = False


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), 1).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class RawDataset(Dataset):

    def __init__(self, root, rgb=True):
        self.root = root
        self.rgb = rgb
        self.image_path_list = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = sorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (256, 256))
            else:
                img = Image.new('L', (256, 256))

        return (img, self.image_path_list[index])


class ResizeCollate(object):

    def __init__(self, short_size=256, crop_size=227, augment=False):
        self.short_size = short_size
        self.crop_size = crop_size
        self.augment = augment

    def _train_preprocessing(self, img):
        transforms = T.Compose([
            T.Resize(self.short_size),
            T.RandomResizedCrop(self.crop_size),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(img)

    def _valid_preprocessing(self, img):
        transforms = T.Compose([
            T.Resize((self.short_size, self.short_size)),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(img)

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.augment:
            image_tensors = [self._train_preprocessing(x) for x in images]
        else:
            image_tensors = [self._valid_preprocessing(x) for x in images]
        image_tensors = torch.stack(image_tensors)
        labels = torch.as_tensor(labels)
        return image_tensors, labels


class ResizeCollateDemo(ResizeCollate):

    def __init__(self, short_size=256, crop_size=227):
        self.short_size = short_size
        self.crop_size = crop_size

    def __call__(self, batch):
        images, path = zip(*batch)

        image_tensors = [self._valid_preprocessing(x) for x in images]
        image_tensors = torch.stack(image_tensors)
        return image_tensors, path


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 reduction: int = 4,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 reduction: int = 4,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = tuple(repeat(kernel_size, 2))
        stride = tuple(repeat(stride, 2))
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class ResNet(nn.Module):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 reduction: int = 4,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.stem_width = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.reduction = reduction
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(nn.Conv2d(3, self.stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                                   norm_layer(self.stem_width), nn.ReLU(inplace=True),
                                   nn.Conv2d(self.stem_width, self.stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                                   norm_layer(self.stem_width), nn.ReLU(inplace=True),
                                   nn.Conv2d(self.stem_width, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                     norm_layer(self.inplanes), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                AvgPool2dSame(2, stride=stride, ceil_mode=True, count_include_pad=False),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, self.reduction, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      reduction=self.reduction,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x = self.layer4(x3)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, x3

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


class Resnet152GeM(nn.Module):

    def __init__(self, embedding_size, pretrained=True):
        super(Resnet152GeM, self).__init__()

        self.model = resnet152(pretrained)
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.avgpool = GeM(p=3)

    def forward(self, x):
        x, x3 = self.model(x)
        x = x.flatten(start_dim=1)
        x = self.model.embedding(x)
        return x, x3


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
        self.global_model = Resnet152GeM(embedding_size=cs, pretrained=pretrained)
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


class BaselineModelInfer(pl.LightningModule):

    def __init__(self, embedding_dim=512, cs=1024, global_only=False):
        super(BaselineModelInfer, self).__init__()
        self.embedding_dim = embedding_dim
        self.cs = cs
        self.global_only = global_only
        self.model = Dolg(self.embedding_dim, pretrained=False, cs=self.cs, global_only=self.global_only)

    def forward(self, x):
        return self.model(x)


@dataclass
class RetrievalResult:
    test_id: str
    chosen_ids: List[str]


def get_file_names(path):
    file_names = glob(os.path.join(path, *['*'] * 3, '*.jpg'))
    return file_names


def convert_to_image_ids(fnames):
    image_ids = []
    for fname in fnames:
        image_id = os.path.splitext(os.path.basename(fname))[0]
        image_ids.append(image_id)
    return image_ids


def get_image_ids(path):
    file_names = get_file_names(path)
    image_ids = convert_to_image_ids(file_names)
    return image_ids


def single_model_inference(test_path, index_path, model_path, crop_size, output_path=True):

    model = BaselineModelInfer.load_from_checkpoint(model_path, strict=False, embedding_dim=512, cs=1024, global_only=GLOBAL_ONLY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    dataset = RawDataset(index_path)
    # inference all index images
    print('Inferencing all index images...')
    data_loader = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=False,
                             collate_fn=ResizeCollateDemo(crop_size=crop_size, short_size=crop_size))

    index_features = []
    if output_path:
        index_imgpaths = []
    with torch.no_grad():
        for img_tensor, img_path in tqdm(data_loader, total=len(data_loader)):
            feature = model(img_tensor.to(device)).cpu().numpy()
            index_features.append(feature)
            if output_path:
                index_imgpaths.extend(img_path)
    index_features = np.concatenate(index_features, axis=0)
    if output_path:
        index_imgpaths = np.array(index_imgpaths)
    print('Inferencing all index images...Done')
    print('Index image feature shape: {}'.format(index_features.shape))

    # Inference on test image
    print('Inferencing test image...')
    dataset = RawDataset(test_path)
    data_loader = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             pin_memory=True,
                             shuffle=False,
                             collate_fn=ResizeCollateDemo(crop_size=crop_size, short_size=crop_size))

    test_features = []
    if output_path:
        test_imgpaths = []
    with torch.no_grad():
        for img_tensor, img_path in tqdm(data_loader, total=len(data_loader)):
            feature = model(img_tensor.to(device)).cpu().numpy()
            test_features.append(feature)
            if output_path:
                test_imgpaths.extend(img_path)
    test_features = np.concatenate(test_features, axis=0)
    print('Inferencing all test images...Done')
    print('Test image feature shape: {}'.format(test_features.shape))
    if output_path:
        return index_features, index_imgpaths, test_features, test_imgpaths
    else:
        return index_features, None, test_features, None


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    n_dim = feat_index.shape[1]

    cpu_index = faiss.IndexFlatIP(n_dim)
    cpu_index.add(feat_index)

    dists, topk_idx = cpu_index.search(x=feat_test, k=topk)
    cpu_index.reset()
    return dists, topk_idx


def generate_inference_result(test_path, index_path, visualize=False):
    # DOLG
    index_features, index_imgpaths, test_features, test_imgpaths = single_model_inference(test_path,
                                                                                          index_path,
                                                                                          MODEL_PATH,
                                                                                          CROP_SIZE,
                                                                                          output_path=True)

    torch.cuda.empty_cache()

    _, topk_idx = search_with_faiss_cpu(test_features, index_features, topk=100)

    for row, path in zip(topk_idx, test_imgpaths):
        nearest_paths = index_imgpaths[row]
        test_id = os.path.basename(path).split('.')[0]
        chosen_ids = [os.path.basename(x).split('.')[0] for x in nearest_paths]
        if visualize:
            plot_figure(test_id, chosen_ids, test_path, index_path)
        yield RetrievalResult(test_id=test_id, chosen_ids=chosen_ids)


def write_submission(output_fname, results: List[RetrievalResult]):
    with open(output_fname, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=('id', 'images'))
        writer.writeheader()
        for result in results:
            writer.writerow({'id': result.test_id, 'images': ' '.join(result.chosen_ids)})


def plot_figure(test_id, chosen_ids, test_path, index_path, visual_dir='visual'):
    test_image = os.path.join(test_path, test_id[0], test_id[1], test_id[2], test_id + '.jpg')
    chosen_images = [os.path.join(index_path, x[0], x[1], x[2], x + '.jpg') for x in chosen_ids]

    outdir = os.path.join(visual_dir, os.path.basename(test_image).split('.')[0])
    indexdir = os.path.join(outdir, 'index')
    os.makedirs(indexdir, exist_ok=True)

    shutil.copy(test_image, os.path.join(outdir, os.path.basename(test_image)))
    for i, image in enumerate(chosen_images):
        imname = '{:03d}_{}'.format(i, os.path.basename(image))
        shutil.copy(image, os.path.join(indexdir, imname))


def main():
    index_path = os.path.join(DATA_PATH, 'index')
    test_path = os.path.join(DATA_PATH, 'test')

    results = generate_inference_result(test_path, index_path, visualize=VISUALIZE)
    write_submission(output_fname='submission.csv', results=results)


if __name__ == '__main__':
    main()
