from collections import OrderedDict
from torchvision.models import resnet18
import torch.nn.functional as F
from torch import nn
import torch


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.01, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class CNNHyper(nn.Module):
    def __init__(self, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [nn.Linear(embedding_dim, hidden_dim)]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseHyper(nn.Module):
    def __init__(self, embedding_dim, out_dim=10, in_features=1280, target_hidden_dim=500, hidden_dim=100, n_hidden=1):
        super().__init__()

        self.in_features = in_features
        self.target_hidden_dim = target_hidden_dim
        self.out_dim = out_dim

        self.noise = GaussianNoise()
        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.l1_weights = nn.Linear(hidden_dim, in_features * target_hidden_dim)
        self.l1_bias = nn.Linear(hidden_dim, target_hidden_dim)
        self.l2_weights = nn.Linear(hidden_dim, target_hidden_dim * out_dim)
        self.l2_bias = nn.Linear(hidden_dim, out_dim)

    def forward(self, idx):
        emd = self.embeddings(idx)
        # emd = self.noise(emd)
        return self.forward_after_embedding(emd)

    def forward_after_embedding(self, emd):
        if self.normalize:
            emd = F.normalize(emd)
        features = self.mlp(emd)

        weights = OrderedDict({
            "fc1.weight": self.l1_weights(features).view(self.target_hidden_dim, self.in_features),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(self.out_dim, self.target_hidden_dim),
            "fc2.bias": self.l2_bias(features).view(-1),
        })
        return weights

    def get_embedding(self, idx):
        emd = self.embeddings(idx)
        if self.normalize:
            emd = F.normalize(emd)
        return emd


class DenseTarget(nn.Module):
    def __init__(self, in_features=1280, hidden_dim=500, dropout=0.2, out_dim=1203):
        super(DenseTarget, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



from torch import Tensor
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
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
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
                  zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                  replace_stride_with_dilation: Optional[List[bool]] = None,
                  norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
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

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
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
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


class HNForResnet(nn.Module):
    def __init__(self, task_embedding_size, image_embedding_size, hn_hidden_dim=120, out_dim=2):
        super().__init__()
        self.out_dim = out_dim
        self.hn_hidden_dim = hn_hidden_dim

        self.image_embedding_size = image_embedding_size

        self.fc1 = nn.Linear(task_embedding_size, hn_hidden_dim)
        self.fc2 = nn.Linear(hn_hidden_dim, hn_hidden_dim)

        self.zsc_weights = nn.Linear(hn_hidden_dim, self.out_dim * self.image_embedding_size)
        self.zsc_bias = nn.Linear(hn_hidden_dim, self.out_dim)


    def forward(self, x):
        x = torch.cat(x, dim=1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        hnet_features = F.relu(self.fc2(x))

        weights = OrderedDict({
            "linear_combiner.weight": self.zsc_weights(hnet_features).view(self.out_dim, self.image_embedding_size),
            "linear_combiner.bias": self.zsc_bias(hnet_features).view(-1),
        })
        return weights


class ZSCombiner(nn.Module):
    def __init__(self, input_dim=50, out_dim=2):
        super().__init__()
        self.linear_combiner = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.linear_combiner(x)
        return x


class ZSCombiner_2l(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=120, out_dim=2):
        super().__init__()
        self.hidden_combiner = nn.Linear(input_dim, hidden_dim)
        self.linear_combiner = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.hidden_combiner(x))
        x = self.linear_combiner(x)
        return x


class AOClevrNet_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(7056*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EVLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.siam = nn.Linear(input_size, output_size)
        self.context = nn.Linear(input_size, output_size)

    def forward(self, embedding_list):
        # embedding_list = [z.unsqueeze(0) for z in embedding_list]
        c = torch.cat(embedding_list, dim=0).sum(dim=0) / len(embedding_list)
        c = self.context(c)
        out = [self.siam(z) + c for z in embedding_list]
        return out


class INVLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.siam = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(input_size, hidden_size)
        self.common = nn.Linear(hidden_size, output_size)

    def forward(self, embedding_list):
        # embedding_list = [z.unsqueeze(0) for z in embedding_list]
        c = torch.cat(embedding_list, dim=0).sum(dim=0) / len(embedding_list)
        c = self.context(c)
        h = torch.cat([self.siam(z) + c for z in embedding_list], dim=0).sum(dim=0)
        out = self.common(F.relu(h))
        return out


class EVHN(nn.Module):
    def __init__(self, task_embedding_size, image_embedding_size, hn_hidden_dim=120):
        super().__init__()
        self.Hs1 = EVLayer(task_embedding_size, hn_hidden_dim)
        self.Hs2 = EVLayer(hn_hidden_dim, hn_hidden_dim)
        self.H1_weights = EVLayer(hn_hidden_dim, image_embedding_size)
        self.H1_bias = EVLayer(hn_hidden_dim, 1)

    def forward(self, embedding_list):
        hnet_features_list = [F.relu(z) for z in self.Hs1(embedding_list)]
        hnet_features_list = [F.relu(z) for z in self.Hs2(hnet_features_list)]

        W1_w = self.H1_weights(hnet_features_list)
        W1_b = self.H1_bias(hnet_features_list)

        weights = OrderedDict({
            "linear_combiner.weight": torch.cat(W1_w),
            "linear_combiner.bias": torch.cat(W1_b).view(-1),
        })
        return weights


class IVEVHN(nn.Module):
    def __init__(self, task_embedding_size, image_embedding_size, hn_hidden_dim=120):
        super().__init__()
        self.hn_hidden_dim = hn_hidden_dim
        self.image_embedding_size = image_embedding_size
        self.Hs1 = EVLayer(task_embedding_size, hn_hidden_dim)
        self.Hs2 = EVLayer(hn_hidden_dim, hn_hidden_dim)

        self.Hinv_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, image_embedding_size*hn_hidden_dim)
        self.Hinv_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, hn_hidden_dim)

        self.H1_weights = EVLayer(hn_hidden_dim, hn_hidden_dim)
        self.H1_bias = EVLayer(hn_hidden_dim, 1)



    def forward(self, embedding_list):
        hnet_features_list = [F.relu(z) for z in self.Hs1(embedding_list)]
        hnet_features_list = [F.relu(z) for z in self.Hs2(hnet_features_list)]

        Winv_w = self.Hinv_weights(hnet_features_list)
        Winv_b = self.Hinv_bias(hnet_features_list)
        W1_w = self.H1_weights(hnet_features_list)
        W1_b = self.H1_bias(hnet_features_list)

        weights = OrderedDict({
            "hidden_combiner.weight": Winv_w.view(self.hn_hidden_dim, self.image_embedding_size),
            "hidden_combiner.bias": Winv_b.view(-1),
            "linear_combiner.weight": torch.cat(W1_w),
            "linear_combiner.bias": torch.cat(W1_b).view(-1),
        })
        return weights


class WWHN(nn.Module):
    def __init__(self, task_embedding_size, image_embedding_size, hn_hidden_dim=120,
                 target_hidden_dim=120, target_out_dim=2):
        super().__init__()
        self.target_out_dim = target_out_dim
        self.hn_hidden_dim = hn_hidden_dim

        self.target_hidden_dim = target_hidden_dim
        self.image_embedding_size = image_embedding_size

        self.fc1 = nn.Linear(task_embedding_size, self.hn_hidden_dim)
        self.fc2 = nn.Linear(self.hn_hidden_dim, self.hn_hidden_dim)

        self.hc_weights = nn.Linear(self.hn_hidden_dim, self.target_hidden_dim * self.image_embedding_size)
        self.hc_bias = nn.Linear(self.hn_hidden_dim, self.target_hidden_dim)

        self.zsc_weights = nn.Linear(self.hn_hidden_dim, self.target_out_dim * self.target_hidden_dim)
        self.zsc_bias = nn.Linear(self.hn_hidden_dim, self.target_out_dim)


    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc1(x))
        hnet_features = F.relu(self.fc2(x))

        weights = OrderedDict({
            "hidden_combiner.weight": self.hc_weights(hnet_features).view(self.target_hidden_dim, self.image_embedding_size),
            "hidden_combiner.bias": self.hc_bias(hnet_features).view(-1),

            "linear_combiner.weight": self.zsc_weights(hnet_features).view(self.target_out_dim, self.target_hidden_dim),
            "linear_combiner.bias": self.zsc_bias(hnet_features).view(-1),
        })
        return weights


class AOClevrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(7056*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 24)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AOHN_full_target(nn.Module):
    def __init__(self, task_embedding_size=384, hn_hidden_dim=120, in_channels=3, n_kernels=16):
        super().__init__()
        self.hn_hidden_dim = hn_hidden_dim
        self.in_channels = in_channels
        self.n_kernels = n_kernels

        # HN backbone
        self.Hs1 = EVLayer(task_embedding_size, hn_hidden_dim)
        self.Hs2 = EVLayer(hn_hidden_dim, hn_hidden_dim)

        # HN heads
        self.c1_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, self.n_kernels)
        self.c2_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, 2 * self.n_kernels)
        self.l1_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, 120 * 32 * 21 * 21)
        self.l1_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, 120)
        self.l2_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, 84 * 120)
        self.l2_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, 84)

        self.l3_weights = EVLayer(hn_hidden_dim, 84)
        self.l3_bias = EVLayer(hn_hidden_dim, 1)

    def forward(self, embedding_list):
        hnet_features_list = [F.relu(z) for z in self.Hs1(embedding_list)]
        hnet_features_list = [F.relu(z) for z in self.Hs2(hnet_features_list)]

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(hnet_features_list).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(hnet_features_list).view(-1),
            "conv2.weight": self.c2_weights(hnet_features_list).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(hnet_features_list).view(-1),
            "fc1.weight": self.l1_weights(hnet_features_list).view(120, 32 * 21 * 21),
            "fc1.bias": self.l1_bias(hnet_features_list).view(-1),
            "fc2.weight": self.l2_weights(hnet_features_list).view(84, 120),
            "fc2.bias": self.l2_bias(hnet_features_list).view(-1),

            "fc3.weight": torch.cat(self.l3_weights(hnet_features_list)),
            "fc3.bias": torch.cat(self.l3_bias(hnet_features_list)).view(-1),
        })
        return weights


class AOIVEVHN(nn.Module):
    def __init__(self, task_embedding_size=384, image_embedding_size=512, hn_hidden_dim=120):
        super().__init__()
        self.hn_hidden_dim = hn_hidden_dim
        self.image_embedding_size = image_embedding_size

        # HN backbone
        self.Hs1 = EVLayer(task_embedding_size, hn_hidden_dim)
        self.Hs2 = EVLayer(hn_hidden_dim, hn_hidden_dim)

        # HN heads
        self.W1_weights = INVLayer(hn_hidden_dim, hn_hidden_dim, image_embedding_size * hn_hidden_dim)
        self.W1_bias = INVLayer(hn_hidden_dim, hn_hidden_dim, hn_hidden_dim)

        self.W2_weights = EVLayer(hn_hidden_dim, hn_hidden_dim)
        self.W2_bias = EVLayer(hn_hidden_dim, 1)

    def forward(self, embedding_list):
        hnet_features_list = [F.relu(z) for z in self.Hs1(embedding_list)]
        hnet_features_list = [F.relu(z) for z in self.Hs2(hnet_features_list)]

        weights = OrderedDict({
            "hidden_combiner.weight": self.W1_weights(hnet_features_list).view(self.hn_hidden_dim, self.image_embedding_size),
            "hidden_combiner.bias": self.W1_bias(hnet_features_list).view(-1),

            "linear_combiner.weight": torch.cat(self.W2_weights(hnet_features_list)),
            "linear_combiner.bias": torch.cat(self.W2_bias(hnet_features_list)).view(-1),
        })
        return weights


class AOEVHN(nn.Module):
    def __init__(self, task_embedding_size=384, image_embedding_size=512, hn_hidden_dim=120):
        super().__init__()
        self.hn_hidden_dim = hn_hidden_dim
        self.image_embedding_size = image_embedding_size

        # HN backbone
        self.Hs1 = EVLayer(task_embedding_size, hn_hidden_dim)
        self.Hs2 = EVLayer(hn_hidden_dim, hn_hidden_dim)

        # HN heads
        self.W2_weights = EVLayer(hn_hidden_dim, image_embedding_size)
        self.W2_bias = EVLayer(hn_hidden_dim, 1)

    def forward(self, embedding_list):
        hnet_features_list = [F.relu(z) for z in self.Hs1(embedding_list)]
        hnet_features_list = [F.relu(z) for z in self.Hs2(hnet_features_list)]

        weights = OrderedDict({
            "linear_combiner.weight": torch.cat(self.W2_weights(hnet_features_list)),
            "linear_combiner.bias": torch.cat(self.W2_bias(hnet_features_list)).view(-1),
        })
        return weights