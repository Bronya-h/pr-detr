import copy
from dataclasses import dataclass
from typing import Dict, List
from git import Optional

import torch
from torch import nn

from .conv import ConvNormLayer, SparseConvNormLayer


@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None

    def __init__(self, channels=None, height=None, width=None, stride=None) -> None:
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride


class ChannelMapper(nn.Module):
    """Channel Mapper for reduce/increase channels of backbone features. Modified
    from `mmdet <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/channel_mapper.py>`_.

    This is used to reduce/increase the channels of backbone features.

    Args:
        input_shape (Dict[str, ShapeSpec]): A dict which contains the backbone features meta infomation,
            e.g. ``input_shape = {"res5": ShapeSpec(channels=2048)}``.
        in_features (List[str]): A list contains the keys which maps the features output from the backbone,
            e.g. ``in_features = ["res"]``.
        out_channels (int): Number of output channels for each scale.
        kernel_size (int, optional): Size of the convolving kernel for each scale.
            Default: 3.
        stride (int, optional): Stride of convolution for each scale. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output of each scale.
            Default: True.
        groups (int, optional): Number of blocked connections from input channels to
            output channels for each scale. Default: 1.
        dilation (int, optional): Spacing between kernel elements for each scale.
            Default: 1.
        norm_layer (nn.Module, optional): The norm layer used for each scale. Default: None.
        activation (nn.Module, optional): The activation layer used for each scale. Default: None.
        num_outs (int, optional): Number of output feature maps. There will be ``extra_convs`` when
            ``num_outs`` is larger than the length of ``in_features``. Default: None.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from detrex.modeling import ChannelMapper
        >>> from detectron2.modeling import ShapeSpec
        >>> input_features = {
        ... "p0": torch.randn(1, 128, 128, 128),
        ... "p1": torch.randn(1, 256, 64, 64),
        ... "p2": torch.randn(1, 512, 32, 32),
        ... "p3": torch.randn(1, 1024, 16, 16),
        ... }
        >>> input_shapes = {
        ... "p0": ShapeSpec(channels=128),
        ... "p1": ShapeSpec(channels=256),
        ... "p2": ShapeSpec(channels=512),
        ... "p3": ShapeSpec(channels=1024),
        ... }
        >>> in_features = ["p0", "p1", "p2", "p3"]
        >>> neck = ChannelMapper(
        ... input_shapes=input_shapes,
        ... in_features=in_features,
        ... out_channels=256,
        ... norm_layer=nn.GroupNorm(num_groups=32, num_channels=256)
        >>> outputs = neck(input_features)
        >>> for i in range(len(outputs)):
        ... print(f"output[{i}].shape = {outputs[i].shape}")
        output[0].shape = torch.Size([1, 256, 128, 128])
        output[1].shape = torch.Size([1, 256, 64, 64])
        output[2].shape = torch.Size([1, 256, 32, 32])
        output[3].shape = torch.Size([1, 256, 16, 16])
    """

    def __init__(
        self,
        input_shapes: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        norm_layer='bn',
        activation=None,
        num_outs: int = None,
        **kwargs,
    ):
        super(ChannelMapper, self).__init__()
        self.input_shapes = input_shapes
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.groups = groups
        self.dilation = dilation
        self.norm_layer = norm_layer
        self.activation = activation
        self.num_outs = num_outs

        self.extra_convs = None

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        if num_outs is None:
            num_outs = len(input_shapes)

        self.convs = nn.ModuleList()
        for in_channel in in_channels_per_feature:
            self.convs.append(
                ConvNormLayer(ch_in=in_channel, ch_out=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias, norm_layer=norm_layer, act=activation))

        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(ConvNormLayer(ch_in=in_channel, ch_out=out_channels, kernel_size=3, stride=2, padding=1, bias=bias, norm_layer=norm_layer, act=activation))

    def forward(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (Dict[str, torch.Tensor] | List[torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        if isinstance(inputs, list):
            inputs = dict(zip(self.in_features, inputs))
        # Use 1x1 convolution to map channels to hidden_dim, and then apply batch normalization to ensure consistent scale of all tokens in the feature map
        outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                # If num_levels > len(feat_channels), it means the top-level feature map of the backbone network is not deep enough, so use convolution downsampling to generate new higher-level feature maps.
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return outs
    
    def prune_input_channel(self, in_ch_mask_list=None):
        for i in range(len(in_ch_mask_list)):
            if in_ch_mask_list[i].dtype != torch.bool:
                in_ch_mask_list[i] = in_ch_mask_list[i].to(torch.bool)
        
        for i, in_ch_mask in enumerate(in_ch_mask_list):
            self.convs[i].prune_input_channel(in_ch_mask)

        if len(in_ch_mask_list) == len(self.convs) and self.extra_convs:
            in_ch_mask_last = in_ch_mask_list[-1]
            self.extra_convs[0].conv.weight = nn.Parameter(self.extra_convs[0].conv.weight[:, in_ch_mask_last, ...])
            if self.extra_convs[0].conv.weight.grad is not None:
                self.extra_convs[0].conv.weight.grad = nn.Parameter(self.extra_convs[0].conv.weight[:, in_ch_mask_last, ...])


    def get_params_count(self):
        total_params = 0
        for conv_layer in self.convs:
            sub_total_params, _ = conv_layer.get_params_count()
            total_params += sub_total_params

        for econv_layer in self.extra_convs:
            sub_total_params, _ = econv_layer.get_params_count()
            total_params += sub_total_params

        return total_params, total_params

    def get_flops(self, feats_shapes):
        total_flops = 0
        for i, feat_shape in enumerate(feats_shapes):
            sub_flops, _ = self.convs[i].get_flops(feat_shape)
            total_flops += sub_flops

        last_feat_shape = feats_shapes[-1]
        for j in range(len(self.extra_convs)):
            sub_flops, _ = self.extra_convs[j].get_flops(last_feat_shape)
            total_flops += sub_flops
            if self.extra_convs[j].stride > 1:
                st = self.extra_convs[j].stride
                # last_feat_shape = [x // st for x in last_feat_shape]
                last_feat_shape[-3] = self.extra_convs[j].out_channels
                last_feat_shape[-2] = last_feat_shape[-2] // st
                last_feat_shape[-1] = last_feat_shape[-1] // st

        return total_flops, total_flops

class SparseChannelMapper(nn.Module):
    """Channel Mapper for reduce/increase channels of backbone features.

    This is used to reduce/increase the channels of backbone features.

    Args:
        input_shape (Dict[str, ShapeSpec]): A dict which contains the backbone features meta infomation,
            e.g. ``input_shape = {"res5": ShapeSpec(channels=2048)}``.
        in_features (List[str]): A list contains the keys which maps the features output from the backbone,
            e.g. ``in_features = ["res"]``.
        out_channels (int): Number of output channels for each scale.
        kernel_size (int, optional): Size of the convolving kernel for each scale.
            Default: 3.
        stride (int, optional): Stride of convolution for each scale. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output of each scale.
            Default: True.
        groups (int, optional): Number of blocked connections from input channels to
            output channels for each scale. Default: 1.
        dilation (int, optional): Spacing between kernel elements for each scale.
            Default: 1.
        norm_layer (nn.Module, optional): The norm layer used for each scale. Default: None.
        activation (nn.Module, optional): The activation layer used for each scale. Default: None.
        num_outs (int, optional): Number of output feature maps. There will be ``extra_convs`` when
            ``num_outs`` is larger than the length of ``in_features``. Default: None.
    """

    def __init__(
        self,
        input_shapes: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        norm_layer='bn',
        activation=None,
        num_outs: int = None,
        **kwargs,
    ):
        super(SparseChannelMapper, self).__init__()
        self.input_shapes = input_shapes
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.groups = groups
        self.dilation = dilation
        self.norm_layer = norm_layer
        self.activation = activation
        self.num_outs = num_outs
        self.extra_convs = None

        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        if num_outs is None:
            num_outs = len(input_shapes)

        # 通过第一个ConvNormLayer的mask来控制所有ConvNormLayer的稀疏剪枝
        self.convs = nn.ModuleList()
        for idx, in_channel in enumerate(in_channels_per_feature):
            if idx == 0:
                newconv = SparseConvNormLayer(ch_in=in_channel,
                                        ch_out=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - 1) // 2,
                                        bias=bias,
                                        norm_layer=norm_layer,
                                        act=activation)
            else:
                newconv = ConvNormLayer(ch_in=in_channel,
                                        ch_out=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - 1) // 2,
                                        bias=bias,
                                        norm_layer=norm_layer,
                                        act=activation)
            self.convs.append(newconv)

        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels

                self.extra_convs.append(ConvNormLayer(ch_in=in_channel, ch_out=out_channels, kernel_size=3, stride=2, padding=1, bias=bias, norm_layer=norm_layer, act=activation))

    def forward(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (Dict[str, torch.Tensor] | List[torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        if isinstance(inputs, list):
            inputs = dict(zip(self.in_features, inputs))

        # 1x1conv 通道映射hidden_dim，再经过batchnorm保持特征图所有token量纲一致
        if not self.convs[0].is_pruned:
            z  = self.convs[0].searched_zeta if self.convs[0].is_searched else self.convs[0].get_zeta()
            outs = []
            for i in range(len(inputs)):
                if i == 0:
                    out0 = self.convs[i](inputs[self.in_features[i]])
                    outs.append(out0)
                else:
                    # out_x = self.convs[i].conv(inputs[self.in_features[i]])
                    # out_x = out_x * z
                    # out_x = self.convs[i].norm(out_x)
                    # out_x = self.convs[i].act(out_x)
                    out_x = self.convs[i](inputs[self.in_features[i]], z)
                    outs.append(out_x)

            if self.extra_convs:
                for i in range(len(self.extra_convs)):
                    # 如果num_levels > len(feat_channels), 说明主干网络的顶层特征图还不够深, 使用conv下采样产生新的更高层特征图
                    if i == 0:
                        # outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                        # out_x = self.extra_convs[i].conv(inputs[self.in_features[-1]])
                        # out_x = out_x * z
                        # out_x = self.extra_convs[i].norm(out_x)
                        # out_x = self.extra_convs[i].act(out_x)
                        out_x = self.extra_convs[i](inputs[self.in_features[-1]], z)
                        outs.append(out_x)
                    else:
                        # outs.append(self.extra_convs[i](outs[-1]))
                        # out_x = self.extra_convs[i].conv(outs[-1])
                        # out_x = out_x * z
                        # out_x = self.extra_convs[i].norm(out_x)
                        # out_x = self.extra_convs[i].act(out_x)
                        out_x = self.extra_convs[i](outs[-1], z)
                        outs.append(out_x)
        else:
            outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
            if self.extra_convs:
                for i in range(len(self.extra_convs)):
                    # 如果num_levels > len(feat_channels), 说明主干网络的顶层特征图还不够深, 使用conv下采样产生新的更高层特征图
                    if i == 0:
                        outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))

        return outs

    def prune_input_channel(self, in_ch_mask_list=None):
        """ embedding剪枝 """
        for i, in_ch_mask in enumerate(in_ch_mask_list):
            self.convs[i].prune_input_channel(in_ch_mask)

        if len(in_ch_mask_list) == len(self.convs) and self.extra_convs:
            in_ch_mask_last = in_ch_mask_list[-1]
            self.extra_convs[0].conv.weight = nn.Parameter(self.extra_convs[0].conv.weight[:, in_ch_mask_last, ...])
            if self.extra_convs[0].conv.weight.grad is not None:
                self.extra_convs[0].conv.weight.grad = nn.Parameter(self.extra_convs[0].conv.weight[:, in_ch_mask_last, ...])

    def prune_output_channel(self, even_number=False):
        to_device = self.convs[0].conv.weight.device
        out_ch_mask = torch.squeeze(self.convs[0].searched_zeta)
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        for i in range(len(self.convs)):
            if i == 0:
                self.convs[i].prune_output_channel()
            else:
                self.convs[i].prune_output_channel(out_ch_mask)

        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i != 0:
                    self.extra_convs[i].prune_input_channel(out_ch_mask)
                
                self.extra_convs[i].prune_output_channel(out_ch_mask)

    def get_output_channel_mask(self):
        zeta = self.convs[0].searched_zeta
        zeta = torch.squeeze(zeta)
        return zeta

    def get_params_count(self):
        total_params = 0
        for conv_layer in self.convs:
            sub_total_params, _ = conv_layer.get_params_count()
            total_params += sub_total_params

        for econv_layer in self.extra_convs:
            sub_total_params, _ = econv_layer.get_params_count()
            total_params += sub_total_params

        return total_params, total_params

    def get_flops(self, feats_shapes):
        total_flops = 0
        for i, feat_shape in enumerate(feats_shapes):
            sub_flops, _ = self.convs[i].get_flops(feat_shape)
            total_flops += sub_flops

        last_feat_shape = feats_shapes[-1]
        for j in range(len(self.extra_convs)):
            sub_flops, _ = self.extra_convs[j].get_flops(last_feat_shape)
            total_flops += sub_flops
            if self.extra_convs[j].stride > 1:
                st = self.extra_convs[j].stride
                # last_feat_shape = [x // st for x in last_feat_shape]
                last_feat_shape[-3] = self.extra_convs[j].out_channels
                last_feat_shape[-2] = last_feat_shape[-2] // st
                last_feat_shape[-1] = last_feat_shape[-1] // st

        return total_flops, total_flops
