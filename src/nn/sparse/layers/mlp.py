import copy
import torch
import torch.nn as nn

from .conv import prune_normalization_layer
from .utils import get_activation, multiply_list_elements


def prune_linear_layer(layer: nn.Linear, mask, direction='output'):
    mask = torch.squeeze(mask)
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    to_device = layer.weight.device
    pruned_channel = torch.sum(mask).item()
    pruned_channel = int(pruned_channel)
    
    if direction == 'output':
        layer.weight = nn.Parameter(layer.weight[mask, ...]).to(to_device)
        # conv.bias
        if layer.bias is not None:
            layer.bias = nn.Parameter(layer.bias[mask]).to(to_device)
        layer.out_features = pruned_channel
    elif direction == 'input':
        layer.weight = nn.Parameter(layer.weight[:, mask, ...]).to(to_device)
        layer.in_features = pruned_channel
    else:
        raise ValueError("'direction' is not correct.")


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu', act_final_layer=False, bias=True, use_dropout=False, dropout_prob=0.0, dropout_final_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act_final_layer = act_final_layer
        self.bias = bias
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.dropout_final_layer = dropout_final_layer

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k, bias=True) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout_layer = nn.Dropout(dropout_prob) if self.use_dropout else nn.Identity()
        # self.dropout_layers = nn.ModuleList(nn.Dropout(self.dropout_prob) if self.use_dropout else nn.Identity() for i in range(self.num_layers))
        if isinstance(act, str):
            self.act = get_activation(act)
        elif isinstance(act, nn.Module):
            self.act = copy.copy(act)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.act(x)
                x = self.dropout_layer(x)
            else:
                x = self.act(x) if self.act_final_layer else x
                x = self.dropout_layer(x) if self.dropout_final_layer else x

        return x


class SparseMLP(MLP):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu', act_final_layer=False, bias=True, use_dropout=False, dropout_prob=0.0, dropout_final_layer=False):
        """A MLP Module using learnable mask to prune. 

        Arguments:
            input_dim -- _description_
            hidden_dim -- _description_
            output_dim -- _description_
            num_layers -- _description_

        Keyword Arguments:
            act -- _description_ (default: {'relu'})
            act_final_layer -- _description_ (default: {False})
            bias -- _description_ (default: {True})
            use_dropout -- _description_ (default: {False})
            dropout_prob -- _description_ (default: {0.0})
            dropout_final_layer -- _description_ (default: {False})
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, act, act_final_layer, bias, use_dropout, dropout_prob, dropout_final_layer)
        self.is_searched = False
        self.is_pruned = False
        self.num_gates = self.hidden_dim
        self.zeta = nn.Parameter(torch.ones(1, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)

        self.loss_weight = 1.0

    def forward(self, x):
        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.get_zeta()
        # z = torch.sigmoid(z) if not self.is_searched else z
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.act(x)
                if not self.is_pruned:
                    x = z * x
                x = self.dropout_layer(x)
            else:
                x = self.act(x) if self.act_final_layer else x
                x = self.dropout_layer(x) if self.dropout_final_layer else x

        return x

    def get_zeta(self):
        return self.zeta

    def compress(self, threshold, min_gates_ratio=0.1):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta() >= threshold).float()

        activate_gates = torch.sum(self.searched_zeta)
        if activate_gates / self.num_gates < min_gates_ratio:  # 触发保底机制，防止输出层的神经元个数为0
            thres_rank = min(self.num_gates - 1, max(0, int((1 - min_gates_ratio) * self.num_gates)))
            act_thres = torch.sort(torch.squeeze(self.get_zeta())).values[thres_rank]
            self.searched_zeta = (self.get_zeta() >= act_thres).float()

        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def prune_input_channel(self, in_ch_mask):
        if in_ch_mask.dtype != torch.bool:
            in_ch_mask = in_ch_mask.to(torch.bool)
        
        self.layers[0].weight = nn.Parameter(self.layers[0].weight[:, in_ch_mask])
        if self.layers[0].weight.grad is not None:
            self.layers[0].weight.grad = nn.Parameter(self.layers[0].weight[:, in_ch_mask])

        pruned_input_channel = torch.sum(in_ch_mask).item()
        pruned_input_channel = int(pruned_input_channel)
        self.layers[0].in_features = pruned_input_channel

    def prune_output_channel(self):
        out_ch_mask = torch.squeeze(self.searched_zeta)
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        pruned_output_channel = torch.sum(out_ch_mask).item()

        to_device = self.layers[0].weight.device
        """ embedding剪枝 """
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                # linear weight output_features
                self.layers[i].weight = nn.Parameter(self.layers[i].weight[out_ch_mask, :]).to(to_device)
                if self.layers[i].weight.grad is not None:
                    self.layers[i].weight.grad = nn.Parameter(self.layers[i].weight[out_ch_mask, :]).to(to_device)
                # linear bias output_features
                if self.layers[i].bias is not None:
                    self.layers[i].bias = nn.Parameter(self.layers[i].bias[out_ch_mask]).to(to_device)
                    if self.layers[i].bias.grad is not None:
                        self.layers[i].bias.grad = nn.Parameter(self.layers[i].bias.grad[out_ch_mask]).to(to_device)

                self.layers[i].out_features = pruned_output_channel

            if i != 0:
                # linear weight input_features
                self.layers[i].weight = nn.Parameter(self.layers[i].weight[:, out_ch_mask]).to(to_device)
                if self.layers[i].weight.grad is not None:
                    self.layers[i].weight.grad = nn.Parameter(self.layers[i].weight[:, out_ch_mask]).to(to_device)

                self.layers[i].in_features = pruned_output_channel

        self.hidden_dim = torch.sum(out_ch_mask).item()
        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None

    def prune_lastlayer_output_channel(self, out_ch_mask):
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)
            
        prune_output_ch_num = torch.sum(out_ch_mask).item()
        prune_output_ch_num = int(prune_output_ch_num)
        
        to_device = self.layers[-1].weight.device
        self.layers[-1].weight = nn.Parameter(self.layers[-1].weight[out_ch_mask, :]).to(to_device)
        if self.layers[-1].bias is not None:
            self.layers[-1].bias = nn.Parameter(self.layers[-1].bias[out_ch_mask]).to(to_device)
        
        self.layers[-1].out_features = prune_output_ch_num
        
    def get_params_count(self):
        dim_in = self.input_dim
        dim_hidden = self.hidden_dim
        dim_out = self.output_dim
        active_dim_hiddn = self.searched_zeta.sum().data
        total_params = 0
        active_params = 0
        for i in range(self.num_layers):
            if i == 0:
                # input layer
                total_params += dim_in * dim_hidden + dim_hidden
                active_params += dim_in * active_dim_hiddn + active_dim_hiddn
            elif i == self.num_layers - 1:
                # output layer
                total_params += dim_hidden * dim_out + dim_out
                active_params += active_dim_hiddn * dim_out + dim_out
            else:
                # medium layer
                total_params += dim_hidden * dim_hidden + dim_hidden
                active_params += active_dim_hiddn * active_dim_hiddn + active_dim_hiddn

        return total_params, active_params

    def get_flops(self, x_shape):
        c_in = x_shape[-1]
        vec_num = multiply_list_elements(x_shape[:-1]) if len(x_shape) > 1 else 1  # project vector number.

        dim_in = self.input_dim
        dim_hidden = self.hidden_dim
        dim_out = self.output_dim
        active_dim_hiddn = self.searched_zeta.sum().data

        total_flops = 0
        active_flops = 0
        for i in range(self.num_layers):
            if i == 0:
                # input layer
                total_flops += 2 * dim_in * dim_hidden
                active_flops += 2 * dim_in * active_dim_hiddn
            elif i == self.num_layers - 1:
                # output layer
                total_flops += 2 * dim_hidden * dim_out
                active_flops += 2 * active_dim_hiddn * dim_out
            else:
                # medium layer
                total_flops += 2 * dim_hidden * dim_hidden
                active_flops += 2 * active_dim_hiddn * active_dim_hiddn

        total_flops *= vec_num
        active_flops *= vec_num
        return total_flops, active_flops

    @staticmethod
    def from_mlp(mlp_module: MLP):
        mlp_module = SparseMLP(mlp_module.input_dim, mlp_module.hidden_dim, mlp_module.output_dim, mlp_module.num_layers, mlp_module.act, mlp_module.act_final_layer, mlp_module.bias,
                               mlp_module.use_dropout, mlp_module.dropout_prob, mlp_module.dropout_final_layer)
        return mlp_module


class LinearNormAct(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, norm_layer='bn', act=None, group_norm_num=None, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.norm_layer = norm_layer
        self.act = act
        self.group_norm_num = group_norm_num
        
        self.linear = nn.Linear(in_features, out_features, bias, device=device, dtype=dtype)
        
        if norm_layer is None or norm_layer == 'bn':
            self.norm_layer = norm_layer
            self.norm = nn.BatchNorm2d(out_features)
        elif norm_layer == 'layer':
            self.norm = nn.LayerNorm(out_features)
        elif norm_layer == 'group':
            self.norm = nn.GroupNorm(8 if self.group_norm_num is None else self.group_norm_num, out_features)
        
        self.act = nn.Identity() if act is None else get_activation(act)
        
    def forward(self, x, mask=None):
        x = self.linear(x)
        if mask is not None:
            x = mask * x
        
        x = self.norm(x)
        x = self.act(x)
        return x
        
    def prune_input_channel(self, in_ch_mask):
        prune_linear_layer(self.linear, in_ch_mask, direction='input')
        pruned_input_channel = int(torch.sum(in_ch_mask).item())
        self.in_features = pruned_input_channel

    def prune_output_channel(self, out_ch_mask):
        out_ch_mask = torch.squeeze(out_ch_mask)
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        pruned_output_channel = torch.sum(out_ch_mask).item()
        pruned_output_channel = int(pruned_output_channel)
        
        prune_linear_layer(self.linear, out_ch_mask, direction='output')
        self.norm = prune_normalization_layer(self.norm, out_ch_mask)

        self.out_features = pruned_output_channel