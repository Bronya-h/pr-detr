from functools import partial
from cv2 import norm
import torch
from torch import nn

from .utils import get_activation, multiply_list_elements


class ConvNormLayer(nn.Module):

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, norm_layer='bn', act=None, group_norm_num=None, groups:int=1, dimension: int = 2):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.norm_layer = norm_layer
        self.act = act
        self.group_norm_num = group_norm_num
        if dimension == 1:
            self.conv = nn.Conv1d(ch_in, ch_out, kernel_size, stride, padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias, groups=groups)
        elif dimension == 2:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias, groups=groups)
        elif dimension == 3:
            self.conv = nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias, groups=groups)
        
        if norm_layer == 'bn':
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm_layer == 'layer':
            self.norm = nn.LayerNorm(ch_out)
        elif norm_layer == 'group':
            self.norm = nn.GroupNorm(8 if self.group_norm_num is None else self.group_norm_num, ch_out)
        elif norm_layer == 'instance':
            self.norm = nn.InstanceNorm2d(ch_out)
        elif norm_layer is None or norm_layer == 'none':
            self.norm = nn.Identity()

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor, mask=None):
        x = self.conv(x)
        if mask is not None:
            x = mask * x
        if isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        x = self.act(x)
        return x

    def prune_input_channel(self, in_ch_mask):
        prune_conv_layer(self.conv, in_ch_mask, direction='input')

        pruned_input_channel = torch.sum(in_ch_mask).item()
        self.ch_in = pruned_input_channel

    def prune_output_channel(self, out_ch_mask):
        prune_conv_layer(self.conv, out_ch_mask, direction='output')
        self.norm = prune_normalization_layer(self.norm, out_ch_mask)
        pruned_output_channel = torch.sum(out_ch_mask).item()
        self.ch_out = pruned_output_channel

    def get_params_count(self):
        total_params = 0
        conv_in, conv_out, conv_k1, conv_k2 = self.conv.weight.shape
        bias_len = len(self.conv.bias) if self.conv.bias is not None else 0
        total_params += conv_in * conv_out * conv_k1 * conv_k2 + bias_len

        return total_params, total_params

    def get_flops(self, x_shape):
        """FLOPs of SparseConvNormLayer before and after pruning

        Arguments:
            h_out -- The height of the output feature map.
            w_out -- The width of the output feature map.

        Returns:
            total_flops -- FLOPs before pruning
            activate_flops -- FLOPs after pruning
        """
        h_out = x_shape[-2]
        w_out = x_shape[-1]
        total_flops = 0
        ch_in, ch_out, ksz1, ksz2 = self.conv.weight.shape
        # conv layer
        total_flops = 2 * ksz1 * ksz2 * ch_in * ch_out * h_out * w_out
        # norm layer
        total_flops += h_out * w_out * 7 * ch_out + ch_out * 2
        # activation
        total_flops += h_out * w_out * ch_out

        return total_flops, total_flops


class SparseConvNormLayer(ConvNormLayer):

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, norm_layer='bn', act=None, group_norm_num=None, groups:int=1, dimension: int = 2):
        super().__init__(ch_in, ch_out, kernel_size, stride, padding, bias, norm_layer, act, group_norm_num, groups, dimension)
        self.is_searched = False
        self.is_pruned = False
        self.num_gates = ch_out
        self.zeta = nn.Parameter(torch.ones(1, self.num_gates, 1, 1))  # 只剪枝输出通道的数量
        self.searched_zeta = torch.ones_like(self.zeta)

        self.loss_weight = 1.0

    def forward(self, x):
        x = self.conv(x)
        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.get_zeta()
            x = x * z

        x = self.norm(x)
        x = self.act(x)
        return x

    def get_zeta(self):
        return self.zeta

    def compress(self, threshold, min_gates_ratio=0.1):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta() >= threshold).float()
        activate_gates = torch.sum(self.searched_zeta).item()
        if activate_gates % 2 != 0:
            sotred_indices = torch.sort(self.get_zeta(), descending=True).indices
            
        # If the sum of searched_zeta is detected as an odd number, regenerate searched_zeta to make the sum an even number
        if (activate_gates / self.num_gates) < min_gates_ratio: # Prevent the number of neurons in the output layer from being 0
            thres_rank = min(self.num_gates - 1, max(0, int((1 - min_gates_ratio) * self.num_gates)))
            act_thres = torch.sort(torch.squeeze(self.get_zeta())).values[thres_rank]
            self.searched_zeta = (self.get_zeta() >= act_thres).float()

        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def prune_input_channel(self, in_ch_mask):
        """ embedding prune """
        prune_conv_layer(self.conv, in_ch_mask, direction='input')
        pruned_input_channel = torch.sum(in_ch_mask).item()
        self.ch_in = pruned_input_channel

    def prune_output_channel(self):
        out_ch_mask = torch.squeeze(self.searched_zeta)
        prune_conv_layer(self.conv, out_ch_mask, direction='output')
        self.norm = prune_normalization_layer(self.norm, out_ch_mask)
        pruned_output_channel = torch.sum(out_ch_mask).item()
        self.ch_out = pruned_output_channel

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None

    def get_params_count(self):
        ch_in, ch_out, ksz1, ksz2 = self.conv.weight.shape
        active_ch_out = self.searched_zeta.sum().data
        # conv layer
        total_params = ch_in * ch_out * ksz1 * ksz2
        total_params = total_params + ch_out if self.conv.bias is True else total_params
        active_params = ch_in * active_ch_out * ksz1 * ksz2
        active_params = active_params + active_ch_out if self.conv.bias is True else total_params
        # norm layer
        total_params += ch_out
        active_params += active_ch_out
        return total_params, active_params

    def get_flops(self, x_shape):
        h_out = x_shape[-2]
        w_out = x_shape[-1]
        ch_in, ch_out, ksz1, ksz2 = self.conv.weight.shape
        active_ch_out = self.searched_zeta.sum().data
        # conv layer
        total_flops = 2 * ksz1 * ksz2 * ch_in * ch_out * h_out * w_out
        activate_flops = 2 * ksz1 * ksz2 * ch_in * active_ch_out * h_out * w_out
        # norm layer
        if self.norm_layer == 'bn':
            total_flops += h_out * w_out * 7 * ch_out + ch_out * 2
            activate_flops += h_out * w_out * 7 * active_ch_out + active_ch_out * 2
        elif self.norm_layer == "layer":
            total_flops += h_out * w_out * 7 * ch_out + ch_out * 2
            activate_flops += h_out * w_out * 7 * active_ch_out + active_ch_out * 2
        elif self.norm_layer == "group":
            total_flops += h_out * w_out * 7 * ch_out + ch_out * 2
            activate_flops += h_out * w_out * 7 * active_ch_out + active_ch_out * 2
        else:
            total_flops += h_out * w_out * 7 * ch_out + ch_out * 2
            activate_flops += h_out * w_out * 7 * active_ch_out + active_ch_out * 2
        # activation
        total_flops += h_out * w_out * ch_out
        activate_flops += h_out * w_out * active_ch_out

        return total_flops, activate_flops

    @staticmethod
    def from_convnorm(convnorm_module: ConvNormLayer):
        convnorm_module = SparseConvNormLayer(convnorm_module.ch_in, convnorm_module.ch_out, convnorm_module.kernel_size, convnorm_module.stride, convnorm_module.padding, convnorm_module.bias,
                                              convnorm_module.norm_layer, convnorm_module.act, convnorm_module.group_norm_num)
        return convnorm_module


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features, eps=1e-5, device=None, dtype=None):
        super(FrozenBatchNorm2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        n = num_features
        self.register_buffer("weight", torch.ones(n, **factory_kwargs))
        self.register_buffer("bias", torch.zeros(n, **factory_kwargs))
        self.register_buffer("running_mean", torch.zeros(n, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(n, **factory_kwargs))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return ("{num_features}, eps={eps}".format(**self.__dict__))


def prune_normalization_layer(norm_layer: nn.Module, out_ch_mask):
    out_ch_mask = torch.squeeze(out_ch_mask)
    if out_ch_mask.dtype != torch.bool:
        out_ch_mask = out_ch_mask.to(torch.bool)

    to_device = norm_layer.weight.device
    pruned_output_channel = torch.sum(out_ch_mask).item()
    
    belong_class = type(norm_layer)
    if not isinstance(norm_layer, nn.GroupNorm):
        new_norm =  belong_class(pruned_output_channel, device=to_device)
    else:
        new_norm =  belong_class(norm_layer.num_groups, pruned_output_channel, device=to_device, eps=norm_layer.eps, affine=norm_layer.affine)
    
    if isinstance(norm_layer.weight, nn.Parameter):
        new_norm.weight = nn.Parameter(norm_layer.weight[out_ch_mask]).to(to_device)
        new_norm.bias = nn.Parameter(norm_layer.bias[out_ch_mask]).to(to_device)
    else:
        new_norm.weight = norm_layer.weight[out_ch_mask].to(to_device)
        new_norm.bias = norm_layer.bias[out_ch_mask].to(to_device)
    
    if hasattr(norm_layer, 'running_mean'):
        new_norm.running_mean = norm_layer.running_mean[out_ch_mask].to(to_device)
        new_norm.running_var = norm_layer.running_var[out_ch_mask].to(to_device)
    
    return new_norm


def prune_conv_layer(conv_layer: nn.Conv2d, ch_mask: torch.Tensor, direction='output'):
    ch_mask = torch.squeeze(ch_mask)
    if ch_mask.dtype != torch.bool:
        ch_mask = ch_mask.to(torch.bool)

    to_device = conv_layer.weight.device
    pruned_channel_num = torch.sum(ch_mask).item()
    
    if direction == 'output':
        # conv weight
        conv_layer.weight = nn.Parameter(conv_layer.weight[ch_mask, ...]).to(to_device)
        # conv.bias
        if conv_layer.bias is not None:
            conv_layer.bias = nn.Parameter(conv_layer.bias[ch_mask]).to(to_device)
        conv_layer.out_channels = pruned_channel_num
    elif direction == 'input':
        # conv weight
        conv_layer.weight = nn.Parameter(conv_layer.weight[:, ch_mask, ...]).to(to_device)
        conv_layer.in_channels = pruned_channel_num
    else:
        raise ValueError("'direction' is not correct.")
