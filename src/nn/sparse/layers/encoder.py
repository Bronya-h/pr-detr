import copy

import torch
from torch import nn
import torch.nn.functional as F

from .channel_map import ChannelMapper, ShapeSpec, SparseChannelMapper
from .utils import pad_tensors_to_same_size, get_activation
from .attention import (
    MultiHeadAttentionViT,
    SparseMultiHeadAttentionViT,
    SrMultiHeadAttentionViT,
    SrSparseMultiHeadAttentionViT,
    FocusedSrSparseMultiHeadAttentionViT,
)
from .mlp import MLP, SparseMLP
from .conv import ConvNormLayer, SparseConvNormLayer, prune_normalization_layer

from src.core import register


class RepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, norm_layer='bn', act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, norm_layer='bn', act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_params_count(self):
        total_params = 0
        conv1_pms, _ = self.conv1.get_params_count()
        conv2_pms, _ = self.conv2.get_params_count()
        total_params += conv1_pms + conv2_pms

        return total_params, total_params

    def get_flops(self, x_shape):
        """SparseConvNormLayer剪枝前后的FLOPs

        Arguments:
            h_out -- 输出特征图的height.
            w_out -- 输出特征图的width.

        Returns:
            total_flops -- 没剪枝前的FLOPs
            activate_flops -- 剪枝后的FLOPs
        """
        h_out = x_shape[-2]
        w_out = x_shape[-1]
        total_flops = 0
        # conv layer
        conv1_flops, _ = self.conv1.get_flops(x_shape)
        conv2_flops, _ = self.conv2.get_flops(x_shape)
        total_flops += conv1_flops + conv2_flops
        # residual
        total_flops += h_out * w_out * x_shape[-3]  # w * h * c, batch=1
        # activation
        if not isinstance(self.act, nn.Identity):
            total_flops += h_out * w_out * x_shape[-3]  # w * h * c, batch=1

        return total_flops, total_flops


class SparseRepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = SparseConvNormLayer(ch_in, ch_out, 3, 1, padding=1, norm_layer='bn', act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, norm_layer='bn', act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        x1 = self.conv1(x)
        # x2 = self.conv2(x)
        x2 = self.conv2.conv(x)
        if not self.conv1.is_pruned:  # only use conv1 soft mask.
            z1 = self.conv1.searched_zeta if self.conv1.is_searched else self.conv1.zeta
            x2 = x2 * z1

        x2 = self.conv2.norm(x2)
        x2 = self.conv2.act(x2)

        y = x1 + x2
        y = self.act(y)
        return y

    def prune_input_channel(self, in_ch_mask):
        """ embedding剪枝 """
        self.conv1.prune_input_channel(in_ch_mask)
        self.conv2.conv.weight = nn.Parameter(self.conv2.conv.weight[:, in_ch_mask, ...])
        if self.conv2.conv.weight.grad is not None:
            self.conv2.conv.weight.grad = nn.Parameter(self.conv2.conv.weight[:, in_ch_mask, ...])

        pruned_input_channels = torch.sum(in_ch_mask).item()
        self.conv2.conv.in_channels = pruned_input_channels

    def prune_output_channel(self):
        # prune conv2
        out_ch_mask = torch.squeeze(self.conv1.searched_zeta)
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        pruned_output_channels = torch.sum(out_ch_mask).item()

        self.conv1.prune_output_channel()

        to_device = self.conv2.conv.weight.device
        # conv weight
        self.conv2.conv.weight = nn.Parameter(self.conv2.conv.weight[out_ch_mask, ...]).to(to_device)
        if self.conv2.conv.weight.grad is not None:
            self.conv2.conv.weight.grad = nn.Parameter(self.conv2.conv.weight.grad[out_ch_mask, ...]).to(to_device)
        # conv.bias
        if self.conv2.conv.bias is not None:
            self.conv2.conv.bias = nn.Parameter(self.conv2.conv.bias[out_ch_mask]).to(to_device)
            if self.conv2.conv.bias.grad is not None:
                self.conv2.conv.bias.grad = nn.Parameter(self.conv2.conv.bias.grad[out_ch_mask]).to(to_device)
    
        new_weight = nn.Parameter(self.conv2.norm.weight[out_ch_mask]).to(to_device)
        new_bias = nn.Parameter(self.conv2.norm.bias[out_ch_mask]).to(to_device)
        if isinstance(self.conv2.norm, nn.BatchNorm2d):
            new_norm = nn.BatchNorm2d(pruned_output_channels, device=to_device)
        elif isinstance(self.conv2.norm, nn.LayerNorm):
            new_norm = nn.LayerNorm(pruned_output_channels, device=to_device)
        elif isinstance(self.conv2.norm, nn.GroupNorm):
            new_norm = nn.GroupNorm(self.conv2.norm.num_groups, pruned_output_channels, device=to_device)
        new_norm.weight = new_weight
        new_norm.bias = new_bias
        self.conv2.norm = new_norm

        self.conv2.conv.out_channels = pruned_output_channels
        # self.conv2.norm.num_features = pruned_output_channels

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
    ):
        """Cross Stage Partial network structure. Before inputting into the bottleneck_block, the input is divided into two parts. One part is computed through the block, while the other part is directly concatenated through a shortcut.

        Arguments:
            in_channels -- The number of input channels for CSP
            out_channels -- The number of output channels for CSP

        Keyword Arguments:
            num_blocks -- The number of RepVggBlock bottleneck structures (default: {3})
            expansion -- The expansion rate of the channel count for the intermediate hidden layer (default: {1.0})
            bias -- Whether to enable bias for the internal input/output ConvNormLayer (default: {None})
            act -- The activation function name for the internal input/output ConvNormLayer (default: {"silu"})
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.bias = bias
        hidden_channels = int(out_channels * expansion)
        self.hidden_channels = hidden_channels

        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

    def get_params_count(self):
        total_params = 0
        # conv_norm_act
        conv1_pms, _ = self.conv1.get_params_count()
        conv2_pms, _ = self.conv2.get_params_count()
        total_params += conv1_pms + conv2_pms
        if not isinstance(conv3_pms, nn.Identity):
            conv3_pms, _ = self.conv3.get_params_count()
            total_params += conv3_pms
        # RepVggBlock
        for i in range(self.num_blocks):
            repvggblock_pms, _ = self.bottlenecks[i].get_params_count()
            total_params += repvggblock_pms

        return total_params, total_params

    def get_flops(self, x_shape):
        total_flops = 0
        c_in = x_shape[-3]
        h_in = x_shape[-2]
        w_in = x_shape[-1]
        total_flops = 0
        # conv layer
        conv1_flops, _ = self.conv1.get_flops([1, c_in, h_in, w_in])
        conv2_flops, _ = self.conv2.get_flops([1, c_in, h_in, w_in])
        total_flops += conv1_flops + conv2_flops
        if not isinstance(self.conv3, nn.Identity):
            conv3_flops, _ = self.conv3.get_flops([1, self.hidden_channels, h_in, w_in])
            total_flops += conv3_flops

        # RepVggBlock
        for i in range(self.num_blocks):
            repvggblock_flops, _ = self.bottlenecks[i].get_flops([1, self.hidden_channels, h_in, w_in])
            total_flops += repvggblock_flops

        # residual
        total_flops += h_in * w_in * self.hidden_channels  # w * h * c, batch=1

        return total_flops, total_flops


class SparseCSPRepLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
        search_out_ch=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.bias = bias
        hidden_channels = int(out_channels * expansion)
        self.hidden_channels = hidden_channels
        self.search_out_ch = search_out_ch

        self.conv1 = SparseConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[SparseRepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            if self.search_out_ch:
                self.conv3 = SparseConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
            else:
                self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)

        x2 = self.conv2.conv(x)

        # if not self.conv1.is_pruned:  # only use conv1 soft mask.
        if not self.bottlenecks[-1].conv1.is_pruned:  # only use conv1 soft mask.
            z = self.bottlenecks[-1].conv1.searched_zeta if self.bottlenecks[-1].conv1.is_searched else self.bottlenecks[-1].conv1.zeta
            x2 = x2 * z

        x2 = self.conv2.norm(x2)
        x_2 = self.conv2.act(x2)
        # x_2 = self.conv2(x)
        x_3 = self.conv3(x_1 + x_2)

        return x_3

    def prune_input_channel(self, in_ch_mask):
        """ embedding剪枝 """
        in_ch_mask = torch.squeeze(in_ch_mask)
        if in_ch_mask.dtype != torch.bool:
            in_ch_mask = in_ch_mask.to(torch.bool)

        pruned_input_channel = torch.sum(in_ch_mask).item()

        self.conv1.prune_input_channel(in_ch_mask)
        to_device = self.conv2.conv.weight.device
        self.conv2.conv.weight = nn.Parameter(self.conv2.conv.weight[:, in_ch_mask, ...]).to(to_device)
        if self.conv2.conv.weight.grad is not None:
            self.conv2.conv.weight.grad = nn.Parameter(self.conv2.conv.weight[:, in_ch_mask, ...]).to(to_device)

        self.conv2.conv.in_channels = pruned_input_channel

        self.in_channels = pruned_input_channel

    def prune_output_channel(self):
        zeta_con1 = self.conv1.searched_zeta
        zeta_con1 = torch.squeeze(zeta_con1)
        if zeta_con1.dtype != torch.bool:
            zeta_con1 = zeta_con1.to(torch.bool)

        zeta_last = zeta_con1
        self.conv1.prune_output_channel()
        for i in range(len(self.bottlenecks)):
            self.bottlenecks[i].prune_input_channel(zeta_last)

            zeta_last = self.bottlenecks[i].conv1.searched_zeta
            zeta_last = torch.squeeze(zeta_last)
            if zeta_last.dtype != torch.bool:
                zeta_last = zeta_last.to(torch.bool)

            self.bottlenecks[i].prune_output_channel()

        # prune conv2
        to_device = self.conv2.conv.weight.device
        pruned_output_channel = torch.sum(zeta_last).item()

        # conv weight
        self.conv2.conv.weight = nn.Parameter(self.conv2.conv.weight[zeta_last, ...]).to(to_device)
        if self.conv2.conv.weight.grad is not None:
            self.conv2.conv.weight.grad = nn.Parameter(self.conv2.conv.weight.grad[zeta_last, ...]).to(to_device)
        # conv.bias
        if self.conv2.conv.bias is not None:
            self.conv2.conv.bias = nn.Parameter(self.conv2.conv.bias[zeta_last]).to(to_device)
            if self.conv2.conv.bias.grad is not None:
                self.conv2.conv.bias.grad = nn.Parameter(self.conv2.conv.bias.grad[zeta_last]).to(to_device)
        # conv2.norm
        new_weight = nn.Parameter(self.conv2.norm.weight[zeta_last]).to(to_device)
        new_bias = nn.Parameter(self.conv2.norm.bias[zeta_last]).to(to_device)
        if isinstance(self.conv2.norm, nn.BatchNorm2d):
            new_norm = nn.BatchNorm2d(pruned_output_channel, device=to_device)
        elif isinstance(self.conv2.norm, nn.LayerNorm):
            new_norm = nn.LayerNorm(pruned_output_channel, device=to_device)
        elif isinstance(self.conv2.norm, nn.GroupNorm):
            new_norm = nn.GroupNorm(self.conv2.norm.num_groups, pruned_output_channel, device=to_device)
        new_norm.weight = new_weight
        new_norm.bias = new_bias
        self.conv2.norm = new_norm

        self.conv2.conv.out_channels = pruned_output_channel
        # self.conv2.norm.num_features = pruned_output_channel

        if not isinstance(self.conv3, nn.Identity):
            # prune conv3 input channels
            to_device = self.conv3.conv.weight.device
            self.conv3.conv.weight = nn.Parameter(self.conv3.conv.weight[:, zeta_last, ...]).to(to_device)
            if self.conv3.conv.weight.grad is not None:
                self.conv3.conv.weight.grad = nn.Parameter(self.conv3.conv.weight[:, zeta_last, ...]).to(to_device)

            self.conv3.conv.in_channels = pruned_output_channel

            if self.search_out_ch:
                conv3_zeta = torch.squeeze(self.conv3.searched_zeta)
                pruned_output_channel = torch.sum(conv3_zeta).item()
                self.conv3.prune_output_channel()
                self.out_channels = pruned_output_channel
        else:
            self.out_channels = pruned_output_channel


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout_prob=0.1,
        activation_str="relu",
        normalize_before=False,
        sr_ratio: int = 1,
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation_str = activation_str
        self.normalize_before = normalize_before
        if sr_ratio > 0:
            self.self_attn = SrMultiHeadAttentionViT(
                d_model,
                head_num=nhead,
                dropout_prob=dropout_prob,
                shortcut=True,
                linear_sr_atten=linear_sr_atten,
                sr_ratio=sr_ratio,
                linear_sr_minval=linear_sr_minval,
            )
        else:
            self.self_attn = MultiHeadAttentionViT(d_model, head_num=nhead, dropout_prob=dropout_prob, shortcut=True)

        self.mlp = MLP(d_model, dim_feedforward, d_model, num_layers=2, act='relu', act_final_layer=False, use_dropout=True, dropout_prob=dropout_prob, dropout_final_layer=True)
        self.norm1 = nn.LayerNorm(d_model) if self.normalize_before else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if self.normalize_before else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_prob)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        spatial_shapes: list[list[int] | tuple[int]] = [],
        **kwargs,
    ) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            src, _ = self.self_attn(q, k, src, src_mask, spatial_shapes=spatial_shapes)
        else:
            src = self.self_attn(q, k, src, src_mask, spatial_shapes=spatial_shapes)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.mlp(src)
        src = residual + src
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class SparseTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout_prob: float = 0.1,
        activation_str: str = "relu",
        normalize_before: bool = False,
        mha_search_type: str = 'embed',
        sr_ratio: int = 0,
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation_str = activation_str
        self.normalize_before = normalize_before
        if sr_ratio > 1:
            if focus_atten:
                self.self_attn = FocusedSrSparseMultiHeadAttentionViT(
                    d_model,
                    nhead,
                    dropout_prob=dropout_prob,
                    search_type=mha_search_type,
                    linear_sr_atten=linear_sr_atten,
                    sr_ratio=sr_ratio,
                    linear_sr_minval=linear_sr_minval,
                    focusing_factor=focusing_factor,
                    dwc_kernel=dwc_kernel,
                    kernel_fun=kernel_fun,
                )
            else:
                self.self_attn = SrSparseMultiHeadAttentionViT(
                    d_model,
                    nhead,
                    dropout_prob=dropout_prob,
                    search_type=mha_search_type,
                    linear_sr_atten=linear_sr_atten,
                    sr_ratio=sr_ratio,
                    linear_sr_minval=linear_sr_minval,
                )
        else:
            self.self_attn = SparseMultiHeadAttentionViT(d_model, nhead, dropout_prob=dropout_prob, search_type=mha_search_type)
        # self.mlp = MLP(d_model, dim_feedforward, d_model, num_layers=2, act='relu', act_final_layer=False, use_dropout=True, dropout_final_layer=True, dropout_prob=dropout_prob)
        self.mlp = SparseMLP(d_model, dim_feedforward, d_model, num_layers=2, act='relu', act_final_layer=False, use_dropout=True, dropout_final_layer=True, dropout_prob=dropout_prob)
        self.norm1 = nn.LayerNorm(d_model) if self.normalize_before else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if self.normalize_before else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if self.normalize_before else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.activation = get_activation(activation_str)
        self.is_searched = False

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shapes=None, **kwargs):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)

        src = self.self_attn(q, k, src, src_mask, spatial_shapes=spatial_shapes)
        if isinstance(src, tuple):
            src, sr_spatial_shapes = src

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm2(src)

        residual = src
        if self.normalize_before:
            src = self.norm3(src)

        src = self.mlp(src)
        src = residual + src
        if not self.normalize_before:
            src = self.norm3(src)

        return src

    def compress(self):
        self.self_attn.compress()
        self.mlp.compress()

    def decompres(self):
        self.self_attn.decompress()
        self.mlp.decompress()

    def prune_input_channel(self, in_mask):
        if in_mask.dtype != torch.bool:
            in_mask = in_mask.to(torch.bool)

        self.self_attn.prune_input_channel(in_mask)
        self.self_attn.prune_output_proj_output_channel(in_mask)

        self.mlp.prune_input_channel(in_mask)
        self.mlp.prune_lastlayer_output_channel(in_mask)
        if self.normalize_before:
            self.norm1 = prune_normalization_layer(self.norm1, in_mask)
            self.norm2 = prune_normalization_layer(self.norm2, in_mask)
            self.norm3 = prune_normalization_layer(self.norm3, in_mask)

    def prune_output_channel(self):
        self.self_attn.prune_output_channel()
        self.mlp.prune_output_channel()

    def get_params_count(self, ):
        total_params = 0
        activate_params = 0

        self_attn_total_params, self_attn_activate_params = self.self_attn.get_params_count()
        total_params += self_attn_total_params
        activate_params += self_attn_activate_params

        mlp_total_params, mlp_activate_params = self.mlp.get_params_count()
        total_params += mlp_total_params
        activate_params += mlp_activate_params

        if self.normalize_before:
            # three norm layer.
            total_params += 3 * self.d_model
            self_attn_zeta = self.self_attn.searched_zeta if self.self_attn.is_searched else self.self_attn.zeta
            self_attn_zeta_num = self_attn_zeta.sum().data
            activate_params += 2 * self.d_model
            activate_params += self_attn_zeta_num

        return total_params, activate_params

    def get_flops(self, x_shape):
        B, N, C = x_shape
        total_flops = 0
        activate_flops = 0

        self_atten_total_flops, self_atten_activate_flops = self.self_attn.get_flops(N)
        total_flops += self_atten_total_flops
        activate_flops += self_atten_activate_flops

        mlp_total_flops, mlp_activate_flops = self.mlp.get_flops(x_shape)
        total_flops += mlp_total_flops
        activate_flops += mlp_activate_flops
        # 残差连接
        self_attn_zeta = self.self_attn.searched_zeta if self.self_attn.is_searched else self.self_attn.zeta
        self_attn_zeta_num = self_attn_zeta.sum().data
        total_flops += B * N * C * 2  # 两次残差连接
        activate_flops += B * N * self_attn_zeta_num * 2  # 两次残差连接
        # Layer Norm
        if self.normalize_before:
            # norm1
            total_flops += N * 7 * self.d_model + self.d_model * 2
            activate_flops += N * 7 * self.d_model + self.d_model * 2
            # norm2
            total_flops += N * 7 * self.d_model + self.d_model * 2
            activate_flops += N * 7 * self_attn_zeta_num + self_attn_zeta_num * 2
            # norm3
            total_flops += N * 7 * self.d_model + self.d_model * 2
            activate_flops += N * 7 * self_attn_zeta_num + self_attn_zeta_num * 2
            total_flops += N * 7 * self.d_model + self.d_model * 2
            activate_flops += N * 7 * self_attn_zeta_num + self_attn_zeta_num * 2

        return total_flops, activate_flops

    @staticmethod
    def from_encoder_layer(encoder_layer: TransformerEncoderLayer, mha_search_type='embed'):
        encoder_module = SparseTransformerEncoderLayer(encoder_layer.d_model, encoder_layer.nhead, encoder_layer.dim_feedforward, encoder_layer.dropout_prob, encoder_layer.activation_str,
                                                       encoder_layer.normalize_before, mha_search_type)
        return encoder_module


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout_prob=0.1,
        activation_str="relu",
        normalize_before=False,
        num_layers=1,
        norm=None,
        sr_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout_prob,
                activation_str,
                normalize_before,
                sr_ratio=sr_ratio_list[i if i < len(sr_ratio_list) else -1] if sr_ratio_list else 0,
                linear_sr_atten=linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
            ) for i in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shapes=None) -> torch.Tensor:
        output = src
        for idx, layer in enumerate(self.layers):
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, spatial_shapes=spatial_shapes[idx:idx + 1])

        if self.norm is not None:
            output = self.norm(output)

        return output


class SparseTransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout_prob=0.1,
        activation_str="relu",
        normalize_before=False,
        num_layers=1,
        norm=None,
        mha_search_type='embed',
        sr_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob

        self.num_layers = num_layers
        self.norm = norm
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout_prob,
                activation_str,
                normalize_before,
                mha_search_type,
                sr_ratio=sr_ratio_list[i if i < len(sr_ratio_list) else -1] if sr_ratio_list else 0,
                linear_sr_atten=linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
                focus_atten=focus_atten,
                focusing_factor=focusing_factor,
                dwc_kernel=dwc_kernel,
                kernel_fun=kernel_fun,
            ) for i in range(self.num_layers)
        ])

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shapes=None) -> torch.Tensor:
        output = src
        for idx, layer in enumerate(self.layers):
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, spatial_shapes=spatial_shapes[idx:idx + 1])

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_params_count(self, ):
        total_params, activate_params = 0, 0
        for layer in self.layers:
            total, activate = layer.get_params_count()
            total_params += total
            activate_params += activate

        if self.norm is not None:
            total_params += self.d_model
            last_layer_zeta = self.layers[-1].self_attn.searched_zeta if self.layers[-1].self_attn.is_searched else self.layers[-1].self_attn.zeta
            last_layer_zeta_num = last_layer_zeta.sum().data
            activate_params += self.d_model * self.nhead * last_layer_zeta_num

        return total_params, activate_params

    def get_flops(self, x_shape):
        B, N, C = x_shape
        total_flops, activate_flops = 0, 0
        for layer in self.layers:
            total, activate = layer.get_flops(x_shape)
            total_flops += total
            activate_flops += activate

        # norm layer.
        if self.norm is not None:
            total_flops += N * 7 * C + C * 2
            last_layer_zeta = self.layers[-1].self_attn.searched_zeta if self.layers[-1].self_attn.is_searched else self.layers[-1].self_attn.zeta
            last_layer_zeta_num = last_layer_zeta.sum().data
            ac = self.nhead * last_layer_zeta_num
            activate_flops += N * 7 * ac + ac * 2

        return total_flops, activate_flops

    def prune_input_channel(self, in_mask):
        self.layers[0].prune_input_channel(in_mask)

    def prune_output_channel(self):
        for i in range(len(self.layers)):
            self.layers[i].prune_output_channel()


class FpnBlock(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        act='silu',
        depth_mult=1.0,
        expansion=1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.act = act
        self.depth_mult = depth_mult
        self.expansion = expansion

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # 1x1 的卷积映射，此处不改变通道数量，只是为了产生一个backbone到fpn特征空间的映射
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            # CSP结构完成上采样操作，将顶层特征图的通道数降低一倍
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

    def forward(self, ms_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        # broadcasting and fusion
        # 执行自顶向下的融合
        inner_outs = [ms_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 从最顶层的主干网络通道数开始遍历
            feat_height = inner_outs[0]  # 高层特征
            feat_low = ms_feats[idx - 1]  # 当前层的低一层的特征
            # lateral_convs的0个卷积层对应c5, 1对应c4, 2对应c3
            feat_height = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_height)
            inner_outs[0] = feat_height
            upsample_feat = F.interpolate(feat_height, scale_factor=2.0, mode='nearest')  # nearest进行上采样
            upsample_feat, feat_low = pad_tensors_to_same_size(upsample_feat, feat_low)  # 主干网络来的特征图的h和w可能有奇数，与upsample后的大小不一致
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))  # 拼接上采样的特征和低一层的特征
            inner_outs.insert(0, inner_out)  # 越新的特征放越前面, 即插入0位置

        return inner_outs


class SparseFpnBlock(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        act='silu',
        depth_mult=1.0,
        expansion=1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.act = act
        self.depth_mult = depth_mult
        self.expansion = expansion

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # 1x1 的卷积映射，此处不改变通道数量，只是为了产生一个backbone到fpn特征空间的映射
            self.lateral_convs.append(SparseConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            # CSP结构完成上采样操作，将顶层特征图的通道数降低一倍
            self.fpn_blocks.append(SparseCSPRepLayer(
                hidden_dim * 2,
                hidden_dim,
                round(3 * depth_mult),
                act=act,
                expansion=expansion,
                search_out_ch=False,
            ))

        self.output_channel_masks = []

    def forward(self, ms_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        # broadcasting and fusion
        # 执行自顶向下的融合
        inner_outs = [ms_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 从最顶层的主干网络通道数开始遍历
            feat_height = inner_outs[0]  # 高层特征
            feat_low = ms_feats[idx - 1]  # 当前层的低一层的特征
            layer_idx = len(self.in_channels) - 1 - idx
            # lateral_convs的0个卷积层对应c5, 1对应c4, 2对应c3
            feat_height = self.lateral_convs[layer_idx](feat_height)
            inner_outs[0] = feat_height
            upsample_feat = F.interpolate(feat_height, scale_factor=2.0, mode='nearest')  # nearest进行上采样
            upsample_feat, feat_low = pad_tensors_to_same_size(upsample_feat, feat_low)  # 主干网络来的特征图的h和w可能有奇数，与upsample后的大小不一致
            concat_feat = torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[layer_idx](concat_feat)  # 拼接上采样的特征和低一层的特征
            inner_outs.insert(0, inner_out)  # 越新的特征放越前面, 即插入0位置

        return inner_outs

    def get_zeta(self):
        return self.zeta

    def prune_input_channel(self, in_ch_mask):
        self.lateral_convs[len(self.in_channels) - 1].prune_input_channel(in_ch_mask)
        # 如果主干网络没有剪枝，拼接过来的feat_low是原始的通道数量
        pass

    def prune_output_channel(self, feats_channel_masks=[]):
        for i in range(len(feats_channel_masks)):
            self.in_channels[i] = torch.sum(feats_channel_masks[i]).item()
            if feats_channel_masks[i].dtype != torch.bool:
                feats_channel_masks[i] = feats_channel_masks[i].to(torch.bool)

        to_device = self.lateral_convs[0].conv.weight.device
        last_fpn_out_mask = feats_channel_masks[-1] if feats_channel_masks else torch.ones([self.lateral_convs[0].ch_in]).to(to_device)

        for idx in range(len(self.in_channels) - 1, 0, -1):
            layer_idx = len(self.in_channels) - 1 - idx

            in_ch_mask = torch.squeeze(self.lateral_convs[layer_idx].searched_zeta)
            low_feat_channel_mask = feats_channel_masks[idx - 1] if feats_channel_masks else torch.ones_like(in_ch_mask)
            in_ch_mask_cat = torch.cat([in_ch_mask, low_feat_channel_mask], dim=0)

            self.lateral_convs[layer_idx].prune_input_channel(last_fpn_out_mask)

            self.lateral_convs[layer_idx].prune_output_channel()
            self.fpn_blocks[layer_idx].prune_input_channel(in_ch_mask_cat)

            if isinstance(self.fpn_blocks[layer_idx].conv3, nn.Identity):
                last_fpn_out_mask = self.fpn_blocks[layer_idx].bottlenecks[-1].conv1.searched_zeta
                last_fpn_out_mask = torch.squeeze(last_fpn_out_mask)
            else:
                last_fpn_out_mask = torch.squeeze(self.fpn_blocks[layer_idx].conv3.searched_zeta) if self.fpn_blocks[layer_idx].search_out_ch else torch.ones(
                    [self.fpn_blocks[layer_idx].out_channels]).to(to_device)

            self.output_channel_masks.insert(0, last_fpn_out_mask)

            self.fpn_blocks[layer_idx].prune_output_channel()

    def get_output_channel_masks(self):

        to_device = self.lateral_convs[0].conv.weight.device

        feats_channel_masks = []
        # ch_mask = torch.squeeze(self.lateral_convs[0].searched_zeta)
        # feats_channel_masks.append(ch_mask)

        for idx in range(len(self.in_channels) - 1, 0, -1):
            layer_idx = len(self.in_channels) - 1 - idx
            ch_mask = torch.squeeze(self.lateral_convs[layer_idx].searched_zeta)
            feats_channel_masks.insert(0, ch_mask)

        if isinstance(self.fpn_blocks[layer_idx].conv3, nn.Identity):
            ch_mask = self.fpn_blocks[layer_idx].bottlenecks[-1].conv1.searched_zeta
            ch_mask = torch.squeeze(ch_mask)
        else:
            ch_mask = torch.squeeze(self.fpn_blocks[layer_idx].conv3.searched_zeta) if self.fpn_blocks[layer_idx].search_out_ch else torch.ones([self.fpn_blocks[layer_idx].out_channels
                                                                                                                                                 ]).to(to_device)

        feats_channel_masks.insert(0, ch_mask)
        # feats_channel_masks.append(ch_mask)

        return feats_channel_masks
        # return self.output_channel_masks


class PAnetBlock(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        act='silu',
        depth_mult=1.0,
        expansion=1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.act = act
        self.depth_mult = depth_mult
        self.expansion = expansion

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            # stride 为 2 的卷积层完成特征图像素的下采样
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            # CSPRep完成通道数量降低一倍
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

    def forward(self, ms_feats):
        outs = [ms_feats[0]]  # fpn后的最低层特征
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]  # 越高级的层放越后面
            feat_height = ms_feats[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # 较低层使用卷积下采样
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))  # 使用CSPRep层融合低层下采样特征和当前层特征
            outs.append(out)

        return outs


class SparsePAnetBlock(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        act='silu',
        depth_mult=1.0,
        expansion=1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.act = act
        self.depth_mult = depth_mult
        self.expansion = expansion

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            # stride 为 2 的卷积层完成特征图像素的下采样
            self.downsample_convs.append(SparseConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            # CSPRep完成通道数量降低一倍
            self.pan_blocks.append(SparseCSPRepLayer(
                hidden_dim * 2,
                hidden_dim,
                round(3 * depth_mult),
                act=act,
                expansion=expansion,
                search_out_ch=False,
            ))

    def forward(self, ms_feats):
        outs = [ms_feats[0]]  # fpn后的最低层特征
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]  # 越高级的层放越后面
            feat_height = ms_feats[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # 较低层使用卷积下采样
            concat_feat = torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](concat_feat)  # 使用CSPRep层融合低层下采样特征和当前层特征
            outs.append(out)

        return outs

    def prune_input_channel(self, feats_channel_masks):
        # for i in range(len(feats_channel_masks)):
        #     if feats_channel_masks[i].dtype != torch.bool:
        #         feats_channel_masks[i] = feats_channel_masks[i].to(torch.bool)

        # last_inmask = feats_channel_masks[0]
        # for idx in range(len(self.in_channels) - 1):
        #     downconv_mask = torch.squeeze(self.downsample_convs[0].searched_zeta)
        #     if downconv_mask.dtype != torch.bool:
        #         downconv_mask = downconv_mask.to(torch.bool)

        #     self.downsample_convs[0].prune_input_channel(last_inmask)
        #     concat_mask = torch.concat([downconv_mask, feats_channel_masks[idx + 1]], dim=0)
        #     self.pan_blocks[idx].prune_input_channel(concat_mask)
        pass

    def prune_output_channel(self, feats_channel_masks=[]):
        # for idx in range(len(self.in_channels) - 1):
        #     in_ch_mask = torch.squeeze(self.downsample_convs[idx].searched_zeta)
        #     in_ch_mask = torch.tile(in_ch_mask, (2, ))
        #     self.downsample_convs[idx].prune_output_channel()
        #     self.pan_blocks[idx].prune_input_channel(in_ch_mask)
        #     self.pan_blocks[idx].prune_output_channel()

        for i in range(len(feats_channel_masks)):
            self.in_channels[i] = torch.sum(feats_channel_masks[i]).item()
            if feats_channel_masks[i].dtype != torch.bool:
                feats_channel_masks[i] = feats_channel_masks[i].to(torch.bool)

        to_device = self.downsample_convs[0].conv.weight.device
        last_inmask = feats_channel_masks[0] if feats_channel_masks else torch.ones([self.downsample_convs[0].ch_in]).to(to_device)
        for idx in range(len(self.in_channels) - 1):
            downconv_mask = torch.squeeze(self.downsample_convs[idx].searched_zeta)
            if downconv_mask.dtype != torch.bool:
                downconv_mask = downconv_mask.to(torch.bool)

            self.downsample_convs[idx].prune_input_channel(last_inmask)
            self.downsample_convs[idx].prune_output_channel()

            concat_mask = torch.concat([downconv_mask, feats_channel_masks[idx + 1]], dim=0)
            self.pan_blocks[idx].prune_input_channel(concat_mask)

            if isinstance(self.pan_blocks[idx].conv3, nn.Identity):
                last_inmask = self.pan_blocks[idx].bottlenecks[-1].conv1.searched_zeta
                last_inmask = torch.squeeze(last_inmask)
            else:
                last_inmask = torch.squeeze(self.pan_blocks[idx].conv3.searched_zeta) if self.pan_blocks[idx].search_out_ch else torch.ones([self.pan_blocks[idx].out_channels]).to(to_device)
            self.pan_blocks[idx].prune_output_channel()

    def get_output_channel_masks(self, feats_channel_masks=[]):
        to_device = self.downsample_convs[0].conv.weight.device
        out_feats_channel_masks = [feats_channel_masks[0]]

        for idx in range(len(self.in_channels) - 1):
            layer_idx = idx
            if isinstance(self.pan_blocks[layer_idx].conv3, nn.Identity):
                ch_mask = self.pan_blocks[layer_idx].bottlenecks[-1].conv1.searched_zeta
                ch_mask = torch.squeeze(ch_mask)
            else:
                ch_mask = torch.squeeze(self.pan_blocks[layer_idx].conv3.searched_zeta) if self.pan_blocks[layer_idx].search_out_ch else torch.ones([self.pan_blocks[layer_idx].out_channels
                                                                                                                                                     ]).to(to_device)
            out_feats_channel_masks.append(ch_mask)

        return out_feats_channel_masks


@register
class OriginRTDETREncoder(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=None,
        in_channel_names=["c3", "c4", "c5"],
        spatial_reduce_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
    ):
        """对应论文中Efficent Hybrid Encoder模块，对一个特征图使用AIFI encoder执行尺度内的融合后，执行自顶向下和自底向上的多尺度特征融合。

        Args:
            in_channels (list, optional): 来自主干网络的各个尺度特征图的通道数. Defaults to [512, 1024, 2048].
            feat_strides (list, optional): 输入的各个尺度特征图的下采样倍数. Defaults to [8, 16, 32].
            hidden_dim (int, optional): encoder的注意力机制隐层维度大小. Defaults to 256.
            nhead (int, optional): encoder多头注意力的head数量. Defaults to 8.
            dim_feedforward (int, optional): encoder前馈网络的隐层维度. Defaults to 1024.
            dropout (float, optional): encoder结构的dropout层的丢弃概率. Defaults to 0.0.
            enc_act (str, optional): encoder结构的前馈网络后的激活函数. Defaults to 'gelu'.
            use_encoder_idx (list, optional): 来自主干网络feats特征图列表的用于做RT-DETR论文中AIFI操作的序号. 例如use_encoder_idx=[2], 使用['res3', 'res4', 'res5']中的res5特征做AIFI. Defaults to [2].
            num_encoder_layers (int, optional): encoder网络深度. Defaults to 1.
            pe_temperature (int, optional): position embedding的温度. Defaults to 10000.
            expansion (float, optional): CSPRep层的中间隐层的通道数量对输入特征通道数的比例, 0~1. Defaults to 1.0.
            depth_mult (float, optional): CSPRep层的中间隐层数量对3的倍数, 乘积后round到最接近的整数, 0~1. Defaults to 1.0.
            act (str, optional): FPN自顶向下和PAN自底向上的用于特征融合的卷积层的激活函数. Defaults to 'silu'.
            eval_spatial_size (int, optional): 推理阶段的图片大小. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.eval_feats_size = [[self.eval_spatial_size[0] // x, self.eval_spatial_size[1] // x] for x in self.feat_strides]
        self.eval_feats_area = [ls[0] * ls[1] for ls in self.eval_feats_size]

        # channel projection
        # 将主干网络的所有尺度的特征图的通道数量通过linear映射到统一的通道数量
        input_proj_shapes = {in_channel_names[i]: ShapeSpec(channels=in_channels[i]) for i in range(len(in_channels))}
        self.input_proj = ChannelMapper(input_shapes=input_proj_shapes, in_features=in_channel_names, out_channels=hidden_dim, kernel_size=1, bias=False, norm_layer='bn')

        # encoder transformer
        self.encoder = nn.ModuleList([  # 用 TransformerEncoder 构建一个完整的encoder, 不使用norm
            TransformerEncoder(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                enc_act,
                num_layers=num_encoder_layers,
                sr_ratio_list=spatial_reduce_ratio_list,
                linear_sr_atten=linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
            ) for i in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.fpn_blocks = FpnBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)

        # bottom-up pan
        self.panet_blocks = PAnetBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)

        self._reset_parameters()  # 默认以eval图片大小，设置每一个特征图的像素点的pos_emb，

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                feat_w, feat_h = self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride
                pos_embed = self.build_2d_sincos_position_embedding(feat_w, feat_h, self.hidden_dim, self.pe_temperature)
                # setattr(self, f'pos_embed{idx}', pos_embed)
                setattr(self, f'pos_embed_h{feat_h}_w{feat_w}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward_encoder(self, feats):
        """encoder 部分的前向推理

        Arguments:
            feats -- 多尺度特征图

        Returns:
            feats: encoder forward feature list.
        """
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):  # 绑定encoder序号
                feat_h, feat_w = feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = feats[enc_ind].flatten(2).permute(0, 2, 1)
                # if self.training or (self.eval_spatial_size is None) or (h * w != self.eval_feats_area[enc_ind]):
                pos_embed_name = f'pos_embed_h{feat_h}_w{feat_w}'
                if getattr(self, pos_embed_name, None) is None:
                    # 根据实际的特征图大小设置pos_embed
                    pos_embed = self.build_2d_sincos_position_embedding(feat_w, feat_h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                    setattr(self, pos_embed_name, pos_embed)
                else:
                    # infer阶段使用默认的pos_embed
                    pos_embed = getattr(self, pos_embed_name, None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shapes=[(feat_h, feat_w)])  # 每个encoder根据其要使用的特征层的序号做forward
                # flatten [B, HxW, hidden_dim] to [B, hidden_dim, H, W]
                feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, feat_h, feat_w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        return feats

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 通道映射，所有尺度映射到同一个数量
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats = self.input_proj(feats)

        # encoder
        encoder_feats = self.forward_encoder(proj_feats)

        # broadcasting and fusion
        # 执行自顶向下的融合
        inner_outs = self.fpn_blocks(encoder_feats)

        # bottom-up panet
        # 自底向上融合
        outs = self.panet_blocks(inner_outs)

        return outs


@register
class SparseRTDETREncoder(nn.Module):

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=None,
        in_channel_names=["c3", "c4", "c5"],
        mha_search_type='embed',
        sr_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
        input_proj_prune: bool = True,
        **kwargs,
    ):
        """对应论文中Efficent Hybrid Encoder模块，对一个特征图使用AIFI encoder执行尺度内的融合后，执行自顶向下和自底向上的多尺度特征融合。

        Args:
            in_channels (list, optional): 来自主干网络的各个尺度特征图的通道数. Defaults to [512, 1024, 2048].
            feat_strides (list, optional): 输入的各个尺度特征图的下采样倍数. Defaults to [8, 16, 32].
            hidden_dim (int, optional): encoder的注意力机制隐层维度大小. Defaults to 256.
            nhead (int, optional): encoder多头注意力的head数量. Defaults to 8.
            dim_feedforward (int, optional): encoder前馈网络的隐层维度. Defaults to 1024.
            dropout (float, optional): encoder结构的dropout层的丢弃概率. Defaults to 0.0.
            enc_act (str, optional): encoder结构的前馈网络后的激活函数. Defaults to 'gelu'.
            use_encoder_idx (list, optional): 来自主干网络feats特征图列表的用于做RT-DETR论文中AIFI操作的序号. 例如use_encoder_idx=[2], 使用['res3', 'res4', 'res5']中的res5特征做AIFI. Defaults to [2].
            num_encoder_layers (int, optional): encoder网络深度. Defaults to 1.
            pe_temperature (int, optional): position embedding的温度. Defaults to 10000.
            expansion (float, optional): CSPRep层的中间隐层的通道数量对输入特征通道数的比例, 0~1. Defaults to 1.0.
            depth_mult (float, optional): CSPRep层的中间隐层数量对3的倍数, 乘积后round到最接近的整数, 0~1. Defaults to 1.0.
            act (str, optional): FPN自顶向下和PAN自底向上的用于特征融合的卷积层的激活函数. Defaults to 'silu'.
            eval_spatial_size (int, optional): 推理阶段的图片大小. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.eval_feats_size = [[self.eval_spatial_size[0] // x, self.eval_spatial_size[1] // x] for x in self.feat_strides]
        self.eval_feats_area = [ls[0] * ls[1] for ls in self.eval_feats_size]

        # channel projection
        # 将主干网络的所有尺度的特征图的通道数量通过linear映射到统一的通道数量
        input_proj_shapes = {in_channel_names[i]: ShapeSpec(channels=in_channels[i]) for i in range(len(in_channels))}
        if not input_proj_prune:
            self.input_proj = ChannelMapper(input_shapes=input_proj_shapes, in_features=in_channel_names, out_channels=hidden_dim, kernel_size=1, bias=False, norm_layer='bn')
        else:
            self.input_proj = SparseChannelMapper(input_shapes=input_proj_shapes, in_features=in_channel_names, out_channels=hidden_dim, kernel_size=1, bias=False, norm_layer='bn')

        # encoder transformer
        self.encoder = nn.ModuleList([  # 用 TransformerEncoder 构建一个完整的encoder, 不使用norm
            SparseTransformerEncoder(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout,
                enc_act,
                num_layers=num_encoder_layers,
                mha_search_type=mha_search_type,
                sr_ratio_list=sr_ratio_list,
                linear_sr_atten=linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
                focus_atten=focus_atten,
                focusing_factor=focusing_factor,
                dwc_kernel=dwc_kernel,
                kernel_fun=kernel_fun,
            ) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        # self.fpn_blocks = FpnBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)
        self.fpn_blocks = SparseFpnBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)

        # bottom-up pan
        # self.panet_blocks = PAnetBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)
        self.panet_blocks = SparsePAnetBlock(self.in_channels, self.hidden_dim, act=act, depth_mult=depth_mult, expansion=expansion)

        self._reset_parameters()  # 默认以eval图片大小，设置每一个特征图的像素点的pos_emb，

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                feat_w, feat_h = self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride
                pos_embed = self.build_2d_sincos_position_embedding(feat_w, feat_h, self.hidden_dim, self.pe_temperature)
                # setattr(self, f'pos_embed{idx}', pos_embed)
                setattr(self, f'pos_embed_h{feat_h}_w{feat_w}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward_encoder(self, feats):
        """encoder 部分的前向推理

        Arguments:
            feats -- 多尺度特征图

        Returns:
            feats: encoder forward feature list.
        """
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):  # 绑定encoder序号
                feat_h, feat_w = feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = feats[enc_ind].flatten(2).permute(0, 2, 1)
                # if self.training or (self.eval_spatial_size is None) or (h * w != self.eval_feats_area[enc_ind]):
                pos_embed_name = f'pos_embed_h{feat_h}_w{feat_w}'
                if getattr(self, pos_embed_name, None) is None or getattr(self, pos_embed_name).shape[-1] != src_flatten.shape[-1]:
                    # 根据实际的特征图大小设置pos_embed
                    pos_embed = self.build_2d_sincos_position_embedding(feat_w, feat_h, src_flatten.shape[-1], self.pe_temperature).to(src_flatten.device)
                    setattr(self, pos_embed_name, pos_embed)
                else:
                    # infer阶段使用默认的pos_embed
                    pos_embed = getattr(self, pos_embed_name, None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shapes=[(feat_h, feat_w)])  # 每个encoder根据其要使用的特征层的序号做forward
                # flatten [B, HxW, hidden_dim] to [B, hidden_dim, H, W]
                feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, feat_h, feat_w).contiguous()

        return feats

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 通道映射，所有尺度映射到同一个数量
        proj_feats = self.input_proj(feats)

        # encoder
        encoder_feats = self.forward_encoder(proj_feats)

        # broadcasting and fusion
        # 执行自顶向下的融合
        inner_outs = self.fpn_blocks(encoder_feats)

        # bottom-up panet
        # 自底向上融合
        outs = self.panet_blocks(inner_outs)

        return outs

    def get_params_count(self, ):
        total_params, activate_params = 0, 0
        for layer in self.layers:
            total, activate = layer.get_params_count()
            total_params += total
            activate_params += activate

        if self.norm is not None:
            total_params += self.d_model
            last_layer_zeta = self.layers[-1].self_attn.searched_zeta if self.layers[-1].self_attn.is_searched else self.layers[-1].self_attn.zeta
            last_layer_zeta_num = last_layer_zeta.sum().data
            activate_params += self.d_model * self.nhead * last_layer_zeta_num

        return total_params, activate_params

    def get_flops(self, x_shape):
        B, N, C = x_shape
        total_flops, activate_flops = 0, 0
        for layer in self.layers:
            total, activate = layer.get_flops(x_shape)
            total_flops += total
            activate_flops += activate

        # norm layer.
        if self.norm is not None:
            total_flops += N * 7 * C + C * 2
            last_layer_zeta = self.layers[-1].self_attn.searched_zeta if self.layers[-1].self_attn.is_searched else self.layers[-1].self_attn.zeta
            last_layer_zeta_num = last_layer_zeta.sum().data
            ac = self.nhead * last_layer_zeta_num
            activate_flops += N * 7 * ac + ac * 2

        return total_flops, activate_flops

    def prune_input_channel(self, in_masks):
        self.input_proj.prune_input_channel(in_masks)

    def prune_output_channel(self, *args, **kwargs):
        input_proj_out_mask = None
        if isinstance(self.input_proj, SparseChannelMapper):
            input_proj_out_mask = self.input_proj.get_output_channel_mask()
            self.input_proj.prune_output_channel()

        for i in range(len(self.encoder)):
            if input_proj_out_mask is not None:
                self.encoder[i].prune_input_channel(input_proj_out_mask)

            if isinstance(self.encoder[i], SparseTransformerEncoder):
                self.encoder[i].prune_output_channel()

        fpn_in_ch_masks = []
        if isinstance(self.fpn_blocks, SparseFpnBlock):
            if input_proj_out_mask is not None:
                fpn_in_ch_masks = [input_proj_out_mask] * len(self.in_channels)
                # self.fpn_blocks.prune_input_channel(input_proj_out_mask)

            fpn_outs_feats_ch_masks = self.fpn_blocks.get_output_channel_masks()
            self.fpn_blocks.prune_output_channel(fpn_in_ch_masks)
            self.fpn_outs_feats_ch_masks = fpn_outs_feats_ch_masks

        if isinstance(self.panet_blocks, SparsePAnetBlock):
            pan_outs_feats_ch_masks = self.panet_blocks.get_output_channel_masks(fpn_outs_feats_ch_masks)
            self.panet_blocks.prune_output_channel(fpn_outs_feats_ch_masks)
            self.pan_outs_feats_ch_masks = pan_outs_feats_ch_masks

    def get_output_channel_mask(self):
        return self.pan_outs_feats_ch_masks
