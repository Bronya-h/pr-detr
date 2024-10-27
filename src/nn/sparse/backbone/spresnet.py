'''by lyuwenyu
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from ..layers import ConvNormLayer, SparseConvNormLayer, FrozenBatchNorm2d, get_activation

__all__ = ['SparsePResNet']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}

donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)), ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        self.act_str = act
        self.variant_str = variant
        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)), ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = SparseConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        if self.shortcut:
            self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        else:
            self.branch2b = SparseConvNormLayer(ch_out, ch_out, 3, 1, act=None)

        self.act = nn.Identity() if act is None else get_activation(act)

        self.is_pruned = False

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        if self.is_pruned:
            short_x = x
            if not self.shortcut:
                short_x = self.short(short_x)
        else:
            z = self.branch2b.searched_zeta if self.branch2b.is_searched else self.branch2b.get_zeta()
            short_x = x
            if not self.shortcut:
                if self.variant_str == 'd' and self.stride == 2:
                    short_x = self.short.pool(short_x)
                    short_x = self.short.conv.conv(short_x)
                    short_x = short_x * z
                    short_x = self.short.conv.norm(short_x)
                    short_x = self.short.conv.act(short_x)
                else:
                    short_x = self.short.conv(short_x)
                    short_x = short_x * z
                    short_x = self.short.norm(short_x)
                    short_x = self.short.act(short_x)

        out = out + short_x
        out = self.act(out)

        return out

    def prune_input_channel(self, in_ch_mask):
        pruned_input_channel = torch.sum(in_ch_mask).item()
        pruned_input_channel = int(pruned_input_channel)

        self.branch2a.prune_input_channel(in_ch_mask)

        if not self.short:
            if self.variant_str == 'd' and self.stride == 2:
                self.short.conv.prune_input_channel(in_ch_mask)
            else:
                self.short.prune_input_channel(in_ch_mask)

        self.ch_in = pruned_input_channel

    def prune_output_channel(self):
        zeta_bh2a = torch.squeeze(self.branch2a.searched_zeta)
        self.branch2a.prune_output_channel()

        self.branch2b.prune_input_channel(zeta_bh2a)
        if not self.shortcut:
            zeta_bh2b = torch.squeeze(self.branch2b.searched_zeta)
            self.branch2b.prune_output_channel()

            if self.variant_str == 'd' and self.stride == 2:
                self.short.conv.prune_output_channel(zeta_bh2b)
            else:
                self.short.prune_output_channel(zeta_bh2b)

            pruned_output_channel = torch.sum(zeta_bh2b).item()
            pruned_output_channel = int(pruned_output_channel)
            self.ch_out = pruned_output_channel

        self.is_pruned = True

    def get_output_channel_mask(self):
        to_device = self.branch2a.conv.weight.device
        if self.shortcut:
            z = torch.ones([self.branch2b.ch_out], device=to_device)
        else:
            z = self.branch2b.searched_zeta
            z = torch.squeeze(z).to(torch.bool)
        return z


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)), ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

    def prune_input_channel(self, in_ch_mask):
        pruned_input_channel = torch.sum(in_ch_mask).item()
        pruned_input_channel = int(pruned_input_channel)

        self.branch2a.prune_input_channel(in_ch_mask)
        self.banch2a.ch_in = pruned_input_channel

        if not self.short:
            if self.variant_str == 'd' and self.stride == 2:
                self.short.conv.prune_input_channel(in_ch_mask)
                self.short.conv.ch_in = pruned_input_channel
            else:
                self.short.prune_input_channel(in_ch_mask)
                self.short.ch_in = pruned_input_channel

    def prune_output_channel(self):
        zeta_bh2a = torch.squeeze(self.branch2a.searched_zeta)
        self.branch2a.prune_output_channel()

        self.branch2b.prune_input_channel(zeta_bh2a)
        zeta_bh2b = torch.squeeze(self.branch2b.searched_zeta)
        self.branch2b.prune_output_channel()

        self.branch2c.prune_input_channel(zeta_bh2b)
        zeta_bh2c = torch.squeeze(self.branch2c.searched_zeta)
        self.branch2c.prune_output_channel()

        pruned_output_channel = torch.sum(zeta_bh2c).item()
        pruned_output_channel = int(pruned_output_channel)
        self.ch_out = pruned_output_channel


class SparseBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        self.shortcut = shortcut
        self.act_str = act
        self.variant_str = variant

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = SparseConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = SparseConvNormLayer(width, width, 3, stride2, act=act)
        if self.shortcut:
            self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        else:
            self.branch2c = SparseConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)), ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

        self.is_pruned = False

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.is_pruned:
            short_x = x
            if not self.shortcut:
                short_x = self.short(short_x)
        else:
            short_x = x
            if not self.shortcut:
                z = self.branch2c.searched_zeta if self.branch2c.is_searched else self.branch2c.get_zeta()
                if self.variant_str == 'd' and self.stride == 2:
                    short_x = self.short.pool(short_x)
                    short_x = self.short.conv.conv(short_x)
                    short_x = short_x * z
                    short_x = self.short.conv.norm(short_x)
                    short_x = self.short.conv.act(short_x)
                else:
                    short_x = self.short.conv(short_x)
                    short_x = short_x * z
                    short_x = self.short.norm(short_x)
                    short_x = self.short.act(short_x)

        out = out + short_x
        out = self.act(out)

        return out

    def prune_input_channel(self, in_ch_mask):
        pruned_input_channel = torch.sum(in_ch_mask).item()
        pruned_input_channel = int(pruned_input_channel)

        self.branch2a.prune_input_channel(in_ch_mask)
        self.branch2a.ch_in = pruned_input_channel

        if not self.shortcut:
            if self.variant_str == 'd' and self.stride == 2:
                self.short.conv.prune_input_channel(in_ch_mask)
                self.short.conv.ch_in = pruned_input_channel
            else:
                self.short.prune_input_channel(in_ch_mask)
                self.short.ch_in = pruned_input_channel

        self.ch_in = pruned_input_channel

    def prune_output_channel(self, last_bottleneck_mask=None):
        """prune output channel dimension of this class object.

        Keyword Arguments:
            last_bottleneck_mask -- last bottleneck output mask. (default: {None})
        """
        zeta_bh2a = torch.squeeze(self.branch2a.searched_zeta)
        self.branch2a.prune_output_channel()

        self.branch2b.prune_input_channel(zeta_bh2a)
        zeta_bh2b = torch.squeeze(self.branch2b.searched_zeta)
        self.branch2b.prune_output_channel()

        self.branch2c.prune_input_channel(zeta_bh2b)

        if last_bottleneck_mask is None:
            zeta_bh2c = torch.squeeze(self.branch2c.searched_zeta)
            self.branch2c.prune_output_channel()
            
            if not self.shortcut:
                if self.variant_str == 'd' and self.stride == 2:
                    self.short.conv.prune_output_channel(zeta_bh2c)
                else:
                    self.short.prune_output_channel(zeta_bh2c)

            pruned_output_channel = torch.sum(zeta_bh2c).item()
            pruned_output_channel = int(pruned_output_channel)
            self.ch_out = pruned_output_channel
            
        else:
            self.branch2c.prune_output_channel(last_bottleneck_mask)
            if not self.shortcut:
                if self.variant_str == 'd' and self.stride == 2:
                    self.short.conv.prune_output_channel(last_bottleneck_mask)
                else:
                    self.short.prune_output_channel(last_bottleneck_mask)

        self.is_pruned = True

    def get_output_channel_mask(self):
        to_device = self.branch2a.conv.weight.device
        if self.shortcut:
            # z = torch.ones([self.branch2c.ch_out], device=to_device)
            z = None
        else:
            z = self.branch2c.searched_zeta
            z = torch.squeeze(z).to(torch.bool)
        
        return z


class Blocks(nn.Module):

    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(ch_in, ch_out, stride=2 if i == 0 and stage_num != 2 else 1, shortcut=False if i == 0 else True, variant=variant, act=act))

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class SparseBlocks(nn.Module):

    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(block(ch_in, ch_out, stride=2 if i == 0 and stage_num != 2 else 1, shortcut=False if i == 0 else True, variant=variant, act=act))

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for idx, block in enumerate(self.blocks):
            out = block(out)
        return out

    def prune_input_channel(self, in_ch_mask):
        self.blocks[0].prune_input_channel(in_ch_mask)

    def prune_output_channel(self):
        to_device = self.blocks[0].branch2a.conv.weight.device
        last_out_mask = None
        for i in range(len(self.blocks)):
            if i != 0:
                self.blocks[i].prune_input_channel(last_out_mask)
            
            if self.blocks[i].shortcut:
                if last_out_mask is None:
                    last_out_mask = torch.ones([self.blocks[max(0, i - 1)].ch_out], device=to_device)
                self.blocks[i].prune_output_channel(last_out_mask)
            else:
                last_out_mask = self.blocks[i].get_output_channel_mask()
                self.blocks[i].prune_output_channel()
        
        self.last_out_mask = last_out_mask

    def get_output_channel_mask(self):
        """需要先执行prune_output_channel才能获得last_out_mask。"""
        return self.last_out_mask


@register
class SparsePResNet(nn.Module):

    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3], act='relu', freeze_at=-1, freeze_norm=True, pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],  # [3, 32, 3, 2]
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"], # [32, 32, 3, 1]
                [ch_in // 2, ch_in, 3, 1, "conv1_3"], # [32, 64, 3, 1]
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        # TODO: prune conv1
        self.conv1 = nn.Sequential(OrderedDict([(_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def]))

        ch_out_list = [64, 128, 256, 512]
        block = SparseBottleNeck if depth >= 50 else SparseBasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(SparseBlocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant))
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state, strict=False)
            print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    def prune_input_channel(self, in_ch_mask):
        pass

    def prune_output_channel(self):
        # TODO: prune conv1
        last_out_mask = None
        for i in range(len(self.res_layers)):
            if i != 0:
                self.res_layers[i].prune_input_channel(last_out_mask)

            self.res_layers[i].prune_output_channel()
            last_out_mask = self.res_layers[i].get_output_channel_mask()

    def get_output_channel_mask(self):
        """获得每个特征图的通道剪枝masks.

        Returns:
            out_feats_channel_masks [lis]
        """
        out_feats_channel_masks = []
        sorted_reture_idx = sorted(self.return_idx)
        for idx in sorted_reture_idx:
            out_ch_masks = self.res_layers[idx].get_output_channel_mask()
            out_feats_channel_masks.append(out_ch_masks)

        return out_feats_channel_masks


@register
class OriginPResNet(nn.Module):

    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3], act='relu', freeze_at=-1, freeze_norm=True, pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([(_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant))
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
