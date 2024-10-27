import math
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

from .mlp import prune_linear_layer
from .conv import prune_normalization_layer, prune_conv_layer, ConvNormLayer


class MultiHeadAttentionViT(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(
        self,
        d_model,
        head_num,
        dropout_prob=0.1,
        d_k=None,
        d_v=None,
        identity_map_reordering=False,
        can_be_stateful=False,
        shortcut=True,
        **kwargs,
    ):
        """Multiple head attention from ViT repository.

        Arguments:
            d_model -- embed dim
            d_k -- Dimensionality of queries and keys.
            d_v -- Dimensionality of values.
            head_num -- Number of heads.

        Keyword Arguments:
            dropout -- _description_ (default: {.1})
            identity_map_reordering -- _description_ (default: {False})
            can_be_stateful -- _description_ (default: {False})
            shortcut -- _description_ (default: {True})
            attention_module -- _description_ (default: {None})
            attention_module_kwargs -- _description_ (default: {None})
        """
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        assert self.d_model % self.head_num == 0, f"d_model-{self.d_model} can not be exact divide by head_num-{self.head_num}"
        self.dropout_prob = dropout_prob
        self.d_k = d_k if d_k is not None else self.d_model // self.head_num
        self.d_v = d_v if d_v is not None else self.d_model // self.head_num
        self.identity_map_reordering = identity_map_reordering
        self.shortcut = shortcut

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, **kwargs):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            b_s, nq = queries.shape[:2]
            nk = keys.shape[1]
            q = self.fc_q(q_norm).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(k_norm).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(v_norm).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))

            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            b_s, nq = queries.shape[:2]
            nk = keys.shape[1]
            q = self.fc_q(queries).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)
        return out


class SparseMultiHeadAttentionViT(MultiHeadAttentionViT):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(
        self,
        d_model,
        head_num,
        dropout_prob=0.1,
        d_k=None,
        d_v=None,
        identity_map_reordering=False,
        can_be_stateful=False,
        shortcut=True,
        search_type='embed',
    ):
        """Multiple head attention from ViT repository.

        Arguments:
            d_model -- embed dim
            d_k -- Dimensionality of queries and keys.
            d_v -- Dimensionality of values.
            head_num -- Number of heads.

        Keyword Arguments:
            dropout -- _description_ (default: {.1})
            identity_map_reordering -- _description_ (default: {False})
            can_be_stateful -- _description_ (default: {False})
            shortcut -- _description_ (default: {True})
            attention_module -- _description_ (default: {None})
            attention_module_kwargs -- _description_ (default: {None})
        """
        super(SparseMultiHeadAttentionViT, self).__init__(d_model, head_num, dropout_prob, d_k, d_v, identity_map_reordering, can_be_stateful, shortcut)
        self.is_searched = False
        self.is_pruned = False
        self.num_gates = self.d_model // self.head_num
        self.search_type = search_type

        if self.search_type == 'head':
            self.zeta = nn.Parameter(torch.ones(1, self.head_num, 1, 1))  # [B, numhead, N, embed_dim]
        elif self.search_type == 'embed':
            self.zeta = nn.Parameter(torch.ones(1, 1, 1, self.num_gates))  # 在embed_dim维度剪枝
        elif self.search_type == 'uniform':
            self.zeta = nn.Parameter(torch.ones(1, self.head_num, 1, self.num_gates))  # 在head和embed_dim两个维度上搜索
        else:
            raise ValueError("search_type is not correct, must be 'head', 'embed' or 'uniform'.")

        self.searched_zeta = torch.ones_like(self.zeta)

        self.is_pruned = False

        self.loss_weight = 1.0

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, **kwargs):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.zeta
        # z = torch.sigmoid(z) if not self.is_searched else z
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            b_s, nq = queries.shape[:2]
            nk = keys.shape[1]

            q = self.fc_q(q_norm).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            # k = self.fc_k(k_norm).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            k = self.fc_k(k_norm).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, d_k, nk)
            v = self.fc_v(v_norm).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            k = torch.transpose(k, -1, -2)
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))

            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            b_s, nq = queries.shape[:2]
            nk = keys.shape[1]
            q = self.fc_q(queries).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            # k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = self.fc_v(values).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            k = torch.transpose(k, -1, -2)
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)

        return out

    def get_zeta(self):
        return self.zeta

    def compress(self, threshold_attn, min_gates_ratio=0.1):
        self.is_searched = True
        self.searched_zeta = (self.zeta >= threshold_attn).float()

        activate_gates = torch.sum(self.searched_zeta)
        if (activate_gates / self.num_gates) < min_gates_ratio:  # 触发保底机制，防止输出层的神经元个数为0
            thres_rank = min(self.num_gates - 1, max(0, int((1 - min_gates_ratio) * self.num_gates)))
            act_thres = torch.sort(torch.squeeze(self.get_zeta())).values[thres_rank]
            self.searched_zeta = (self.get_zeta() >= act_thres).float()

        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def prune_input_channel(self, in_ch_mask):
        """ embedding剪枝 """
        if in_ch_mask != torch.bool:
            in_ch_mask = in_ch_mask.to(torch.bool)

        pruned_input_channel = torch.sum(in_ch_mask).item()

        prune_linear_layer(self.fc_q, mask=in_ch_mask, direction='input')
        prune_linear_layer(self.fc_k, mask=in_ch_mask, direction='input')
        prune_linear_layer(self.fc_v, mask=in_ch_mask, direction='input')

    def prune_output_channel(self):
        if self.search_type == 'head':
            pruning_dim = 1
            out_head_mask = torch.squeeze(self.searched_zeta)
        elif self.search_type == "embed":
            pruning_dim = 3
            out_ch_mask = torch.squeeze(self.searched_zeta)
        elif self.search_type == "uniform":
            pruning_dim = [1, 3]
            # out_head_mask = self.searched_zeta[]
            raise RuntimeError("uniform search is not complement.")

        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        if self.search_type == "embed":
            out_ch_mask = torch.tile(out_ch_mask, (self.head_num, ))

        pruned_output_channel = torch.sum(out_ch_mask).item()

        to_device = self.fc_q.weight.device

        if self.search_type == "embed":
            """ embedding剪枝 """
            prune_linear_layer(self.fc_q, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_k, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_v, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_o, mask=out_ch_mask, direction='input')

            self.d_k = self.d_v = pruned_output_channel // self.head_num

        elif self.search_type == "head":
            self.head_num = torch.sum(torch.squeeze(self.searched_zeta)).item()
            if self.head_num % 2 == 1:
                self.head_num = max(1, self.head_num - 1)
            self.num_gates = self.d_model // self.head_num
            self.d_k = self.d_model // self.head_num
            self.d_v = self.d_model // self.head_num
        else:
            raise ValueError("uniform prune has not complement yet.")

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None

    def prune_output_proj_output_channel(self, out_ch_mask):
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        prune_output_ch_num = torch.sum(out_ch_mask).item()
        prune_output_ch_num = int(prune_output_ch_num)

        to_device = self.fc_o.weight.device
        self.fc_o.weight = nn.Parameter(self.fc_o.weight[out_ch_mask, :]).to(to_device)
        if self.fc_o.bias is not None:
            self.fc_o.bias = nn.Parameter(self.fc_o.bias[out_ch_mask]).to(to_device)

        self.layer_norm = prune_normalization_layer(self.layer_norm, out_ch_mask)

        self.fc_o.out_features = prune_output_ch_num

    def get_params_count(self):
        dim = self.d_model
        active = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1:
            active *= self.num_gates
        elif self.zeta.shape[2] == 1:
            active *= self.head_num
        total_params = dim * dim * 3 + dim * 3  # qkv weights and bias
        total_params += dim * dim + dim  # proj weights and bias
        active_params = dim * active * 3 + active * 3
        active_params += active * dim + dim
        return total_params, active_params

    def get_flops(self, seqnum):
        H = self.head_num
        N = seqnum
        d = self.num_gates
        sd = self.searched_zeta.sum().data if self.is_searched else self.zeta.sum().data
        if self.zeta.shape[-1] == 1:  # Head Elimination
            sd *= self.num_gates
        elif self.zeta.shape[1] == 1:  # Uniform Search
            sd *= self.head_num  # TODO

        total_flops, active_flops = 0, 0

        total_flops += 3 * N * 2 * (H * d) * (H * d)  #linear: qkv
        active_flops += 3 * N * 2 * (H * sd) * (H * sd)  #linear: qkv

        total_flops += 2 * H * d * N * N + N * N  #q@k * scale
        active_flops += 2 * H * sd * N * N + N * N  #q@k * scale

        total_flops += 5 * H * N * N  #softmax: exp, sum(exp), div, max, x-max
        active_flops += 5 * H * N * N  #softmax: exp, sum(exp), div, max, x-max

        total_flops += H * 2 * N * N * d  #attn@v
        active_flops += H * 2 * N * N * sd  #attn@v

        total_flops += N * 2 * (H * d) * (H * d)  #linear: proj
        active_flops += N * 2 * (H * sd) * (H * sd)  #linear: proj

        return total_flops, active_flops


class SrMultiHeadAttentionViT(MultiHeadAttentionViT):
    def __init__(
        self,
        d_model,
        head_num,
        dropout_prob=0.1,
        d_k=None,
        d_v=None,
        identity_map_reordering=False,
        can_be_stateful=False,
        shortcut=True,
        search_type='embed',
        linear_sr_atten: bool = False,
        sr_ratio: int = 1,
        linear_sr_minval: int = 1,
        **kwargs,
    ):
        """Multiple head attention from ViT repository.
        modify to support spatial reduction attention.

        Arguments:
            d_model -- embed dim
            d_k -- Dimensionality of queries and keys.
            d_v -- Dimensionality of values.
            head_num -- Number of heads.
            self_atten -- whether this object is use to self attention

        Keyword Arguments:
            dropout -- _description_ (default: {.1})
            identity_map_reordering -- _description_ (default: {False})
            can_be_stateful -- _description_ (default: {False})
            shortcut -- _description_ (default: {True})
            attention_module -- _description_ (default: {None})
            attention_module_kwargs -- _description_ (default: {None})
        """
        super().__init__(
            d_model,
            head_num,
            dropout_prob,
            d_k,
            d_v,
            identity_map_reordering,
            can_be_stateful,
            shortcut,
            **kwargs,
        )
        self.linear_sr_atten = linear_sr_atten
        self.sr_ratio = sr_ratio
        self.linear_sr_minval = linear_sr_minval
        if sr_ratio > 1:
            if not linear_sr_atten:
                self.sr_layer = ConvNormLayer(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio, norm_layer='layer')
            else:
                self.sr_layer = nn.Sequential(
                    # nn.AdaptiveAvgPool2d(output_size=7),
                    ConvNormLayer(d_model, d_model, kernel_size=1, stride=1, norm_layer='layer'),
                    nn.GELU(),
                )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        attention_weights: torch.Tensor = None,
        spatial_shapes: list = None,
        **kwargs,
    ):
        """
        计算多头自注意力机制的前向传播。
        
        Args:
            queries (torch.Tensor): 查询张量，形状为 [batch_size, num_queries, dim_query]。
            keys (torch.Tensor): 键张量，形状为 [batch_size, num_keys, dim_key]。
            values (torch.Tensor): 值张量，形状为 [batch_size, num_values, dim_value]。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置上的注意力。默认为 None。
            attention_weights (torch.Tensor, optional): 注意力权重张量，用于加权注意力得分。默认为 None。
            **kwargs: 其他关键字参数。
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_queries, dim_output]。
        
        """
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        b_s, nq, dim_query = queries.shape
        nk, dim_key = keys.shape[1:]

        split_shape = [h * w for h, w in spatial_shapes]  # [5184, 1296, 324]
        sr_spatial_shapes = []

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            if self.sr_ratio > 1:
                k_norm_list = k_norm.split(split_shape, dim=1)
                v_norm_list = v_norm.split(split_shape, dim=1)

                sr_k_norm_list = []
                sr_v_norm_list = []
                nk = 0
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    k_norm = k_norm_list[i]
                    v_norm = v_norm_list[i]

                    k_norm = k_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    v_norm = v_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if not self.linear_sr_atten:
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        k_norm = F.adaptive_avg_pool2d(k_norm, (sr_h, sr_w))
                        v_norm = F.adaptive_avg_pool2d(v_norm, (sr_h, sr_w))

                    k_norm = self.sr_layer(k_norm)
                    v_norm = self.sr_layer(v_norm)
                    sr_h, sr_w = k_norm.shape[-2:]
                    sr_spatial_shapes.append((sr_h, sr_w))
                    k_norm = k_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    v_norm = v_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_k_norm_list.append(k_norm)
                    sr_v_norm_list.append(v_norm)

                k_norm = torch.cat(sr_k_norm_list, dim=1)
                v_norm = torch.cat(sr_v_norm_list, dim=1)

            nk = k_norm.shape[1]
            q = self.fc_q(q_norm).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(k_norm).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = self.fc_v(v_norm).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            k = torch.transpose(k, -1, -2)
            # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            att = torch.matmul(q, k) / np.sqrt(self.head_num)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            if self.sr_ratio > 1:  # only do spatial reduction when sr_ratio > 1.
                key_split_list = keys.split(split_shape, dim=1)
                value_split_list = values.split(split_shape, dim=1)

                sr_key_list = []
                sr_value_list = []
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    key = key_split_list[i]
                    value = value_split_list[i]
                    key = key.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    value = value.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if self.linear_sr_atten:  # when use linear sr attention, adaptive_avg_pool2d to do the downsample.
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        key = F.adaptive_avg_pool2d(key, (sr_h, sr_w))
                        value = F.adaptive_avg_pool2d(value, (sr_h, sr_w))

                    key = self.sr_layer(key)
                    value = self.sr_layer(value)
                    sr_h, sr_w = key.shape[-2:]
                    sr_spatial_shapes.append((sr_h, sr_w))
                    key = key.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    value = value.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_key_list.append(key)
                    sr_value_list.append(value)

                keys = torch.cat(sr_key_list, dim=1)
                values = torch.cat(sr_value_list, dim=1)

            nk = keys.shape[1]  # update the number of keys tokens if do spatial reduction.
            q = self.fc_q(queries).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = self.fc_v(values).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
            
            k = k.transpose(-1, -2)
            # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            att = torch.matmul(q, k) / np.sqrt(self.head_num)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)

        return out


class SrSparseMultiHeadAttentionViT(SparseMultiHeadAttentionViT):
    """Multiscale Spatial Reduction Multiple Head Attention.
    多尺度空间下采样多头注意力机制。"""

    def __init__(
        self,
        d_model,
        head_num,
        dropout_prob=0.1,
        d_k=None,
        d_v=None,
        identity_map_reordering=False,
        can_be_stateful=False,
        shortcut=True,
        search_type='embed',
        linear_sr_atten: bool = False,
        sr_ratio: int = 1,
        linear_sr_minval: int = 1,
        **kwargs,
    ):
        """Multiple head attention from ViT repository.
        modify to support spatial reduction attention.

        Arguments:
            d_model -- embed dim
            d_k -- Dimensionality of queries and keys.
            d_v -- Dimensionality of values.
            head_num -- Number of heads.
            self_atten -- whether this object is use to self attention

        Keyword Arguments:
            dropout -- _description_ (default: {.1})
            identity_map_reordering -- _description_ (default: {False})
            can_be_stateful -- _description_ (default: {False})
            shortcut -- _description_ (default: {True})
            attention_module -- _description_ (default: {None})
            attention_module_kwargs -- _description_ (default: {None})
        """
        super().__init__(
            d_model,
            head_num,
            dropout_prob,
            d_k,
            d_v,
            identity_map_reordering,
            can_be_stateful,
            shortcut,
            search_type,
            **kwargs,
        )
        self.linear_sr_atten = linear_sr_atten
        self.sr_ratio = sr_ratio
        self.linear_sr_minval = linear_sr_minval
        if sr_ratio > 1:
            if not linear_sr_atten:
                self.sr_layer = ConvNormLayer(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio, norm_layer='layer')
            else:
                self.sr_layer = nn.Sequential(
                    # nn.AdaptiveAvgPool2d(output_size=7),
                    ConvNormLayer(d_model, d_model, kernel_size=1, stride=1, norm_layer='layer'),
                    nn.GELU(),
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        attention_weights: torch.Tensor = None,
        spatial_shapes: list = None,
        **kwargs,
    ):
        """
        计算多头自注意力机制的前向传播。
        
        Args:
            queries (torch.Tensor): 查询张量，形状为 [batch_size, num_queries, dim_query]。
            keys (torch.Tensor): 键张量，形状为 [batch_size, num_keys, dim_key]。
            values (torch.Tensor): 值张量，形状为 [batch_size, num_values, dim_value]。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置上的注意力。默认为 None。
            attention_weights (torch.Tensor, optional): 注意力权重张量，用于加权注意力得分。默认为 None。
            **kwargs: 其他关键字参数。
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_queries, dim_output]。
        
        """
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        b_s, nq, dim_query = queries.shape
        nk, dim_key = keys.shape[1:]

        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.zeta

        split_shape = [h * w for h, w in spatial_shapes]  # [5184, 1296, 324]
        sr_spatial_shapes = []

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            if self.sr_ratio > 1:
                k_norm_list = k_norm.split(split_shape, dim=1)
                v_norm_list = v_norm.split(split_shape, dim=1)

                sr_k_norm_list = []
                sr_v_norm_list = []
                nk = 0
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    k_norm = k_norm_list[i]
                    v_norm = v_norm_list[i]

                    k_norm = k_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    v_norm = v_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if not self.linear_sr_atten:
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        k_norm = F.adaptive_avg_pool2d(k_norm, (sr_h, sr_w))
                        v_norm = F.adaptive_avg_pool2d(v_norm, (sr_h, sr_w))

                    k_norm = self.sr_conv(k_norm)
                    v_norm = self.sr_conv(v_norm)
                    sr_h, sr_w = k_norm.shape[-2:]
                    sr_spatial_shapes.append((sr_h, sr_w))
                    k_norm = k_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    v_norm = v_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_k_norm_list.append(k_norm)
                    sr_v_norm_list.append(v_norm)

                k_norm = torch.cat(sr_k_norm_list, dim=1)
                v_norm = torch.cat(sr_v_norm_list, dim=1)

            nk = k_norm.shape[1]
            q = self.fc_q(q_norm).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(k_norm).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = self.fc_v(v_norm).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            k = torch.transpose(k, -1, -2)
            # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            att = torch.matmul(q, k) / np.sqrt(self.head_num)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))

            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            if self.sr_ratio > 1:  # only do spatial reduction when sr_ratio > 1.
                key_split_list = keys.split(split_shape, dim=1)
                value_split_list = values.split(split_shape, dim=1)

                sr_key_list = []
                sr_value_list = []
                nk = 0
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    key = key_split_list[i]
                    value = value_split_list[i]
                    key = key.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    value = value.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if self.linear_sr_atten:  # when use linear sr attention, adaptive_avg_pool2d to do the downsample.
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        key = F.adaptive_avg_pool2d(key, (sr_h, sr_w))
                        value = F.adaptive_avg_pool2d(value, (sr_h, sr_w))

                    key = self.sr_layer(key)
                    value = self.sr_layer(value)
                    sr_h, sr_w = key.shape[-2:]
                    nk += sr_h * sr_w
                    sr_spatial_shapes.append((sr_h, sr_w))
                    key = key.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    value = value.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_key_list.append(key)
                    sr_value_list.append(value)

                keys = torch.cat(sr_key_list, dim=1)
                values = torch.cat(sr_value_list, dim=1)

            nk = keys.shape[1]  # update the number of keys tokens if do spatial reduction.
            q = self.fc_q(queries).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = self.fc_v(values).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            k = torch.transpose(k, -1, -2)
            # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            att = torch.matmul(q, k) / np.sqrt(self.head_num)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                att = att * attention_weights
                att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask.bool(), -1e9)

            att = torch.softmax(att, -1)
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)

        return out


class FocusedSrSparseMultiHeadAttentionViT(SrSparseMultiHeadAttentionViT):

    def __init__(self,
                 d_model,
                 head_num,
                 dropout_prob=0.1,
                 d_k=None,
                 d_v=None,
                 identity_map_reordering=False,
                 can_be_stateful=False,
                 shortcut=True,
                 search_type='embed',
                 linear_sr_atten: bool = False,
                 sr_ratio: int = 1,
                 linear_sr_minval: int = 1,
                 focusing_factor: int = 3,
                 dwc_kernel: int = 5,
                 kernel_fun: str = 'relu',
                 **kwargs):
        super().__init__(d_model, head_num, dropout_prob, d_k, d_v, identity_map_reordering, can_be_stateful, shortcut, search_type, linear_sr_atten, sr_ratio, linear_sr_minval, **kwargs)

        self.focusing_factor = focusing_factor
        # self.dwc = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=dwc_kernel, groups=self.d_v, padding=dwc_kernel // 2)
        self.dwc = ConvNormLayer(
            ch_in=self.d_v,
            ch_out=self.d_v,
            kernel_size=dwc_kernel,
            stride=1,
            groups=self.d_v,
            padding=dwc_kernel // 2,
            norm_layer=None,
            dimension=1,  # nn.Conv1d
        )
        self.focus_scale = nn.Parameter(torch.zeros(size=(1, 1, d_model)))
        if kernel_fun == 'relu':
            self.kernel_fun = nn.ReLU()
        elif kernel_fun == 'sigmoid':
            self.kernel_fun = nn.Sigmoid()
        elif kernel_fun == 'tanh':
            self.kernel_fun = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        attention_weights: torch.Tensor = None,
        spatial_shapes: list = None,
        **kwargs,
    ):
        """
        计算多头自注意力机制的前向传播。
        
        Args:
            queries (torch.Tensor): 查询张量，形状为 [batch_size, num_queries, dim_query]。
            keys (torch.Tensor): 键张量，形状为 [batch_size, num_keys, dim_key]。
            values (torch.Tensor): 值张量，形状为 [batch_size, num_values, dim_value]。
            attention_mask (torch.Tensor, optional): 注意力掩码张量，用于屏蔽某些位置上的注意力。默认为 None。
            attention_weights (torch.Tensor, optional): 注意力权重张量，用于加权注意力得分。默认为 None。
            **kwargs: 其他关键字参数。
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_queries, dim_output]。
        
        """
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        b_s, nq, dim_query = queries.shape
        nk, dim_key = keys.shape[1:]

        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.zeta

        split_shape = [h * w for h, w in spatial_shapes]  # [5184, 1296, 324]
        sr_spatial_shapes = []

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)

            if self.sr_ratio > 1:
                k_norm_list = k_norm.split(split_shape, dim=1)
                v_norm_list = v_norm.split(split_shape, dim=1)

                sr_k_norm_list = []
                sr_v_norm_list = []
                nk = 0
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    k_norm = k_norm_list[i]
                    v_norm = v_norm_list[i]

                    k_norm = k_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    v_norm = v_norm.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if not self.linear_sr_atten:
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        k_norm = F.adaptive_avg_pool2d(k_norm, (sr_h, sr_w))
                        v_norm = F.adaptive_avg_pool2d(v_norm, (sr_h, sr_w))

                    k_norm = self.sr_conv(k_norm)
                    v_norm = self.sr_conv(v_norm)
                    sr_h, sr_w = k_norm.shape[-2:]
                    sr_spatial_shapes.append((sr_h, sr_w))
                    k_norm = k_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    v_norm = v_norm.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_k_norm_list.append(k_norm)
                    sr_v_norm_list.append(v_norm)

                k_norm = torch.cat(sr_k_norm_list, dim=1)
                v_norm = torch.cat(sr_v_norm_list, dim=1)

            nk = k_norm.shape[1]
            q = self.fc_q(q_norm)
            k = self.fc_k(k_norm)
            v = self.fc_v(v_norm)

            focus_scale = self.softplus(self.focus_scale)
            q = self.kernel_fun(q) + 1e-6
            k = self.kernel_fun(k) + 1e-6
            q = q / focus_scale
            k = k / focus_scale
            q_norm = torch.norm(q, dim=-1, keepdim=True)  # norm throuout the d_model dimention.
            k_norm = torch.norm(k, dim=-1, keepdim=True)
            q = q**self.focusing_factor
            k = k**self.focusing_factor
            q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
            k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

            q = q.view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = k.view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = v.view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            kv = (k.transpose(-2, -1) * (nk**-0.5)) @ (v * (nk**-0.5))
            out = q @ kv / att  # [bs, head, n_q, dim_q]
            out = out.permute(0, 2, 1, 3).reshape(b_s, nq, self.head_num * self.d_v)  #

            if self.sr_ratio > 1:
                v = F.interpolate(v.permute(0, 2, 1, 3), size=out.shape[1], mode='linear')  # [bs, nk, h, d_v]
                v = v.permute(0, 2, 1, 3)  # [bs, h, nk, d_v]

            nv = v.shape[2]
            v = v.reshape(b_s * self.head_num, nv, self.d_v).permute(0, 2, 1)  # [bs*h, d_v, nv]
            dwc_v = self.dwc(v)
            dwc_v = dwc_v.reshape(b_s, self.head_num * self.d_v, -1)  # [bs, head*dim_v, nv]
            dwc_v = dwc_v.permute(0, 2, 1)  # [bs, nv, head*dim_v]

            out = out + dwc_v

            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            if self.sr_ratio > 1:  # only do spatial reduction when sr_ratio > 1.
                key_split_list = keys.split(split_shape, dim=1)
                value_split_list = values.split(split_shape, dim=1)

                sr_key_list = []
                sr_value_list = []
                nk = 0
                for i in range(len(spatial_shapes)):
                    ori_h, ori_w = spatial_shapes[i]
                    sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio

                    key = key_split_list[i]
                    value = value_split_list[i]
                    key = key.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    value = value.permute(0, 2, 1).reshape(b_s, dim_key, ori_h, ori_w)
                    if self.linear_sr_atten:  # when use linear sr attention, adaptive_avg_pool2d to do the downsample.
                        sr_h = max(sr_h, self.linear_sr_minval)
                        sr_w = max(sr_w, self.linear_sr_minval)
                        key = F.adaptive_avg_pool2d(key, (sr_h, sr_w))
                        value = F.adaptive_avg_pool2d(value, (sr_h, sr_w))

                    key = self.sr_layer(key)
                    value = self.sr_layer(value)
                    sr_h, sr_w = key.shape[-2:]
                    nk += sr_h * sr_w
                    sr_spatial_shapes.append((sr_h, sr_w))
                    key = key.reshape(b_s, dim_key, -1).permute(0, 2, 1)
                    value = value.reshape(b_s, dim_key, -1).permute(0, 2, 1)

                    sr_key_list.append(key)
                    sr_value_list.append(value)

                keys = torch.cat(sr_key_list, dim=1)
                values = torch.cat(sr_value_list, dim=1)

            nk = keys.shape[1]  # update the number of keys tokens if do spatial reduction.
            # q = self.fc_q(queries).view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            # k = self.fc_k(keys).view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            # v = self.fc_v(values).view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
            q = self.fc_q(queries)  # [bs, nq, dim_query]
            k = self.fc_k(keys)  # [bs, nk, dim_key]
            v = self.fc_v(values)  # [bs, nk, dim_key]

            focus_scale = self.softplus(self.focus_scale)
            q = self.kernel_fun(q) + 1e-6
            k = self.kernel_fun(k) + 1e-6
            q = q / focus_scale
            k = k / focus_scale
            q_norm = torch.norm(q, dim=-1, keepdim=True)  # norm throuout the d_model dimention.
            k_norm = torch.norm(k, dim=-1, keepdim=True)
            q = q**self.focusing_factor
            k = k**self.focusing_factor
            q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
            k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

            q = q.view(b_s, nq, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = k.view(b_s, nk, self.head_num, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
            v = v.view(b_s, nk, self.head_num, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

            if not self.is_pruned:
                # soft mask prune
                q = q * z
                k = k * z
                v = v * z

            att = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6
            # if attention_weights is not None:
            #     att = att * attention_weights
            #     att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
            # if attention_mask is not None:
            #     att = att.masked_fill(attention_mask.bool(), -1e9)

            kv = (k.transpose(-2, -1) * (nk**-0.5)) @ (v * (nk**-0.5))

            out = q @ kv / att  # [bs, head, n_q, dim_q]
            out = out.permute(0, 2, 1, 3).reshape(b_s, nq, self.head_num * self.d_v)  #

            if self.sr_ratio > 1:
                v = F.interpolate(v.permute(0, 2, 1, 3), size=out.shape[1], mode='linear')  # [bs, nk, h, d_v]
                v = v.permute(0, 2, 1, 3)  # [bs, h, nk, d_v]

            nv = v.shape[2]
            v = v.reshape(b_s * self.head_num, nv, self.d_v).permute(0, 2, 1)  # [bs*h, d_v, nv]
            dwc_v = self.dwc(v)
            dwc_v = dwc_v.reshape(b_s, self.head_num * self.d_v, -1)  # [bs, head*dim_v, nv]
            dwc_v = dwc_v.permute(0, 2, 1)  # [bs, nv, head*dim_v]

            out = out + dwc_v

            out = self.fc_o(out)  # (b_s, nq, d_model)
            out = self.dropout(out)

            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)

        return out

    def prune_input_channel(self, in_ch_mask):
        """ embedding剪枝 """
        if in_ch_mask != torch.bool:
            in_ch_mask = in_ch_mask.to(torch.bool)

        pruned_input_channel = torch.sum(in_ch_mask).item()

        prune_linear_layer(self.fc_q, mask=in_ch_mask, direction='input')
        prune_linear_layer(self.fc_k, mask=in_ch_mask, direction='input')
        prune_linear_layer(self.fc_v, mask=in_ch_mask, direction='input')

    def prune_output_channel(self):
        if self.search_type == 'head':
            pruning_dim = 1
            out_head_mask = torch.squeeze(self.searched_zeta)
        elif self.search_type == "embed":
            pruning_dim = 3
            out_ch_mask = torch.squeeze(self.searched_zeta)
        elif self.search_type == "uniform":
            pruning_dim = [1, 3]
            # out_head_mask = self.searched_zeta[]
            raise RuntimeError("uniform search is not complement.")

        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        if self.search_type == "embed":
            dwc_ch_mask = out_ch_mask
            out_ch_mask = torch.tile(out_ch_mask, (self.head_num, ))

        pruned_output_channel = torch.sum(out_ch_mask).item()

        to_device = self.fc_q.weight.device

        if self.search_type == "embed":
            """ embedding剪枝 """
            prune_linear_layer(self.fc_q, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_k, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_v, mask=out_ch_mask, direction='output')
            prune_linear_layer(self.fc_o, mask=out_ch_mask, direction='input')

            self.d_k = self.d_v = pruned_output_channel // self.head_num

        elif self.search_type == "head":
            self.head_num = torch.sum(torch.squeeze(self.searched_zeta)).item()
            if self.head_num % 2 == 1:
                self.head_num = max(1, self.head_num - 1)
            self.num_gates = self.d_model // self.head_num
            self.d_k = self.d_model // self.head_num
            self.d_v = self.d_model // self.head_num
        else:
            raise ValueError("uniform prune has not complement yet.")

        # prune depthwise conv layer
        prune_conv_layer(self.dwc, ch_mask=dwc_ch_mask, direction='input')
        prune_conv_layer(self.dwc, ch_mask=dwc_ch_mask, direction='output')

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None


class MSDeformableAttention(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        head_num=8,
        num_levels=4,
        num_points=4,
    ):
        """多头可变形注意力机制。

        Args:
            embed_dim (int, optional): MLP中间隐层数量. Defaults to 256.
            head_num (int, optional): 多头注意力头的数量. Defaults to 8.
            num_levels (int, optional): 输入的多尺度特征图的数量. Defaults to 4.
            num_points (int, optional): 参考点坐标描述点的轴数量，如(x,y,w,h)是4个轴. Defaults to 4.
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num  # 论文符号M
        self.num_levels = num_levels  # 论文符号L
        self.num_points = num_points  # 论文符号K
        self.total_points = head_num * num_levels * num_points

        self.head_dim = embed_dim // head_num
        assert self.head_dim * head_num == self.embed_dim, "embed_dim must be divisible by head_num"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,  # 前两个通道，2MK。多尺度是2MLK
        )
        self.attention_weights = nn.Linear(embed_dim, self.total_points)  # 后一个通道, MK。多尺度是MLK
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = self.deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.head_num, dtype=torch.float32) * (2.0 * math.pi / self.head_num)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.head_num, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]  # 8, 492
        Len_v = value.shape[1]  # value [8, 10164, 256]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.head_num, self.head_dim)  # value为每个head分配输入数据, [8, 10164, 8, 32]
        # 采样偏置 [bs, lenq, M, L, K, 2]--[8, 492, 8, 3, 4, 2]
        sampling_offsets = self.sampling_offsets(query).reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points, 2)
        # 获得各个参考点的注意力加权 成 [bs, lenq, M, L*K]--[8, 492, 8, 12]
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.head_num, self.num_levels * self.num_points)
        # 多尺度上进行各个参考点的softmax，所有参考点概率和为1。[bs, lenq, M, L, K] -- [8, 492, 8, 3, 4]
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:  # [8, 492, 1, 2]
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # [8, 492, 1, 4]
            # [8, 492, 1, 1, 1, 2] 在(w,h)两个维度上乘sampling_offsets，对w/h进行伸缩。
            sampling_locations = (reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1]))

        # output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)  # [8, 492, 256]

        output = self.output_proj(output)  # 线性层映射，使output参数得到一个聚合。

        return output

    def deformable_attention_core_func(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        """
        可变形注意力机制的核心计算函数。

        Args:
            value (Tensor): [bs, value_length, n_head, c]
            value_spatial_shapes (Tensor|List): [n_levels, 2]
            value_level_start_index (Tensor|List): [n_levels]
            sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
            attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, _, n_head, c = value.shape  # [8, 6804, 8, 32]
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape  # [8, 492, 8, 3, 4, 2]

        split_shape = [h * w for h, w in value_spatial_shapes]  # [5184, 1296, 324]
        # 每个尺度的value用split分离出来
        value_list = value.split(split_shape, dim=1)  # [[8, 5184, 8, 32], [8, 1296, 8, 32], [8, 324, 8, 32]]
        # [0, 1]范围的samp_loc放缩偏移到[-1, 1]
        sampling_grids = 2 * sampling_locations - 1  # [8, 492, 8, 3, 4, 2]
        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            # [N_, H_*W_, M_, D_] -> [N_, H_*W_, M_*D_] -> [N_, M_*D_, H_*W_] -> [N_*M_, D_, H_, W_]
            value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)  # [64, 32, 80, 80]
            # TODO: spatial reduce.
            # [N_, Lq_, M_, P_, 2] -> [N_, M_, Lq_, P_, 2] -> [N_*M_, Lq_, P_, 2]
            sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)  # [64, 500, 4, 2]
            # N_*M_, D_, Lq_, P_
            # 用双线性插值，用sampling_grid预测的偏移点的对周围4个点进行不同权重的上采样，形状从[N_*M_, D_, H_, W_]变为[N_*M_, D_, Lq_, P_]
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)  # [64, 32, 500, 4]
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, Len_q, n_levels * n_points)
        """此处是DeformableAttention论文中的Aggregated过程。"""
        # 聚合多尺度后的samp_value与atten_weight: (N_*M_, D_, Lq_, L_*P_) * (N_*M_, 1, Lq_, L_*P_)
        # 在最后一维进行sum, [N_*M_, D_, Lq_], reshape后[N_, M_*D_, Lq_]
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)
        # permute成[N_, Lq_, M_*D_], 其中M_*D_与value的维度相同。
        return output.permute(0, 2, 1)


class SrMSDeformableAttention(MSDeformableAttention):
    """Spatial Reduction Multiple Scale Deformable Attention."""
    def __init__(
        self,
        embed_dim: int = 256,
        head_num: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        linear_sr_atten: bool = False,
        sr_ratio: list[int] = 2,
        linear_sr_minval: int = 1,
    ) -> None:
        super().__init__(embed_dim, head_num, num_levels, num_points)
        self.linear_atten = linear_sr_atten
        self.sr_ratio = sr_ratio
        self.linear_sr_minval = linear_sr_minval
        if sr_ratio > 1:
            if not linear_sr_atten:
                self.sr_layer = ConvNormLayer(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio, norm_layer='layer')
            else:
                self.sr_layer = nn.Sequential(
                    # nn.AdaptiveAvgPool2d(output_size=7),
                    ConvNormLayer(embed_dim, embed_dim, kernel_size=1, stride=1, norm_layer='layer'),
                    nn.GELU(),
                )
        else:
            self.sr_layer = nn.Identity()
            
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list[tuple[int] | list[int]],
        value_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q, dim_q = query.shape  # 8, 492
        Len_v, dim_v = value.shape[1:]  # value [8, 10164, 256]

        if self.sr_ratio > 1:  # do spatial reduction operation
            split_shape = [h * w for h, w in value_spatial_shapes]  # [5184, 1296, 324]
            # 每个尺度的value用split分离出来
            value_list = value.split(split_shape, dim=1)  # [[8, 5184, 256], [8, 1296, 256], [8, 324, 256]]
            sr_spatial_shapes = []
            sr_value_list = []
            for i in range(len(split_shape)):
                ori_h, ori_w = value_spatial_shapes[i]
                sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio
                sr_value = value_list[i]
                sr_value = sr_value.permute(0, 2, 1).reshape(bs, dim_v, ori_h, ori_w)
                if self.linear_atten:
                    sr_h = max(sr_h, self.linear_sr_minval)
                    sr_w = max(sr_w, self.linear_sr_minval)
                    sr_value = F.adaptive_avg_pool2d(sr_value, (sr_h, sr_w))
                sr_value: torch.Tensor = self.sr_layer(sr_value)
                sr_h, sr_w = sr_value.shape[-2:]  # update the sr_h and sr_w to actually spatial shape of sr_value Tensor.
                sr_spatial_shapes.append((sr_h, sr_w))
                sr_value = sr_value.reshape(bs, dim_v, -1).permute(0, 2, 1)
                sr_value_list.append(sr_value)

            sr_values = torch.cat(sr_value_list, dim=1)
            value = sr_values
            value_spatial_shapes = sr_spatial_shapes
            Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask

        value = value.reshape(bs, Len_v, self.head_num, self.head_dim)  # value为每个head分配输入数据, [8, 10164, 8, 32]

        # 采样偏置 [bs, lenq, M, L, K, 2]--[8, 492, 8, 3, 4, 2]
        sampling_offsets = self.sampling_offsets(query)  # [8, 492, 8*3*4*2]
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points, 2)
        # 获得各个参考点的注意力加权 成 [bs, lenq, M, L*K]--[8, 492, 8, 12]
        attention_weights = self.attention_weights(query)  # [8, 492, 8*12]
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels * self.num_points)
        # 多尺度上进行各个参考点的softmax，所有参考点概率和为1。[bs, lenq, M, L, K]--[8, 492, 8, 3, 4]
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:  # [8, 492, 1, 2]
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # [8, 492, 1, 4]
            # [8, 492, 1, 1, 1, 2] 在(w,h)两个维度上乘sampling_offsets，对w/h进行伸缩。
            sampling_locations = (reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1]))

        output = self.deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)  # [8, 492, 256]

        output = self.output_proj(output)  # 线性层映射，使output参数得到一个聚合。

        return output


class SparseMSDeformableAttention(MSDeformableAttention):

    def __init__(
        self,
        embed_dim=256,
        head_num=8,
        num_levels=4,
        num_points=4,
        search_type='embed',
    ) -> None:
        """可稀疏掩码学习的多头可变形注意力机制。

        Args:
            embed_dim (int, optional): MLP中间隐层数量. Defaults to 256.
            head_num (int, optional): 多头注意力头的数量. Defaults to 8.
            num_levels (int, optional): 输入的多尺度特征图的数量. Defaults to 4.
            num_points (int, optional): 参考点坐标描述点的轴数量，如(x,y,w,h)是4个轴. Defaults to 4.
        """
        super(SparseMSDeformableAttention, self).__init__(embed_dim, head_num, num_levels, num_points)
        self.is_searched = False
        self.is_pruned = False
        self.num_gates = self.embed_dim // self.head_num
        self.search_type = search_type

        # only apply to value_proj / output_proj
        if self.search_type == 'head':
            self.zeta = nn.Parameter(torch.ones(1, 1, self.head_num, 1))  # [B, numhead, N, embed_dim]
        elif self.search_type == 'embed':
            self.zeta = nn.Parameter(torch.ones(1, 1, 1, self.num_gates))  # 在embed_dim维度剪枝
        elif self.search_type == 'uniform':
            self.zeta = nn.Parameter(torch.ones(1, 1, self.head_num, self.num_gates))  # 在head和embed_dim两个维度上搜索
        else:
            raise ValueError("search_type is not correct, must be 'head', 'embed' or 'uniform'.")

        self.searched_zeta = torch.ones_like(self.zeta)

        self.is_pruned = False

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]  # 8, 492
        Len_v = value.shape[1]  # value [8, 10164, 256]

        # TODO:spatial reduce
        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask

        value = value.reshape(bs, Len_v, self.head_num, self.head_dim)  # value为每个head分配输入数据, [8, 10164, 8, 32]
        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.zeta
            value = value * z

        # 采样偏置 [bs, lenq, M, L, K, 2]--[8, 492, 8, 3, 4, 2]
        sampling_offsets = self.sampling_offsets(query)  # [8, 492, 8*3*4*2]
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points, 2)
        # 获得各个参考点的注意力加权 成 [bs, lenq, M, L*K]--[8, 492, 8, 12]
        attention_weights = self.attention_weights(query)  # [8, 492, 8*12]
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels * self.num_points)
        # 多尺度上进行各个参考点的softmax，所有参考点概率和为1。[bs, lenq, M, L, K]--[8, 492, 8, 3, 4]
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:  # [8, 492, 1, 2]
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # [8, 492, 1, 4]
            # [8, 492, 1, 1, 1, 2] 在(w,h)两个维度上乘sampling_offsets，对w/h进行伸缩。
            sampling_locations = (reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1]))

        output = self.deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)  # [8, 492, 256]

        output = self.output_proj(output)  # 线性层映射，使output参数得到一个聚合。

        return output

    def get_zeta(self):
        return self.zeta

    def compress(self, threshold_attn, min_gates_ratio=0.1):
        self.is_searched = True
        self.searched_zeta = (self.zeta >= threshold_attn).float()

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
        """剪枝输入通道

        Arguments:
            in_ch_mask -- 输入通道的剪枝的Hard Mask.
        """
        if in_ch_mask.dtype != torch.bool:
            in_ch_mask = in_ch_mask.to(torch.bool)

        self.value_proj.weight = nn.Parameter(self.value_proj.weight[:, in_ch_mask])
        if self.value_proj.weight.grad is not None:
            self.value_proj.weight.grad = nn.Parameter(self.value_proj.weight[:, in_ch_mask])

        self.sampling_offsets.weight = nn.Parameter(self.sampling_offsets.weight[:, in_ch_mask])
        if self.sampling_offsets.weight.grad is not None:
            self.sampling_offsets.weight.grad = nn.Parameter(self.sampling_offsets.weight[:, in_ch_mask])

        self.attention_weights.weight = nn.Parameter(self.attention_weights.weight[:, in_ch_mask])
        if self.attention_weights.weight.grad is not None:
            self.attention_weights.weight.grad = nn.Parameter(self.attention_weights.weight[:, in_ch_mask])

    def prune_output_channel(self):
        """剪枝隐层的输出通道.
        """
        if self.search_type == "embed":
            out_ch_mask = torch.squeeze(self.searched_zeta)
            if out_ch_mask.dtype != torch.bool:
                out_ch_mask = out_ch_mask.to(torch.bool)

            self.head_dim = torch.sum(out_ch_mask).item()

            out_ch_mask = torch.tile(out_ch_mask, (self.head_num, ))  # tile block to multi head.

            to_device = self.value_proj.weight.device

            self.value_proj.weight = nn.Parameter(self.value_proj.weight[out_ch_mask, :]).to(to_device)
            if self.value_proj.weight.grad is not None:
                self.value_proj.weight.grad = nn.Parameter(self.value_proj.weight[out_ch_mask, :]).to(to_device)
            if self.value_proj.bias is not None:
                self.value_proj.bias = nn.Parameter(self.value_proj.bias[out_ch_mask]).to(to_device)
                if self.value_proj.bias.grad is not None:
                    self.value_proj.bias.grad = nn.Parameter(self.value_proj.bias.grad[out_ch_mask]).to(to_device)

            self.output_proj.weight = nn.Parameter(self.output_proj.weight[:, out_ch_mask]).to(to_device)
            if self.output_proj.weight.grad is not None:
                self.output_proj.weight.grad = nn.Parameter(self.output_proj.weight[:, out_ch_mask]).to(to_device)

        elif self.search_type == "head":
            self.head_num = torch.sum(torch.squeeze(self.searched_zeta)).item()
            if self.head_num % 2 == 1:
                self.head_num = max(1, self.head_num - 1)
            self.num_gates = self.embed_dim // self.head_num
            self.head_dim = self.embed_dim // self.head_num
        else:
            raise ValueError("uniform prune has not complement yet.")

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None

    def prune_output_proj_output_channel(self, out_ch_mask):
        if out_ch_mask.dtype != torch.bool:
            out_ch_mask = out_ch_mask.to(torch.bool)

        prune_linear_layer(self.output_proj, out_ch_mask, direction='output')

        prune_output_ch_num = torch.sum(out_ch_mask).item()
        prune_output_ch_num = int(prune_output_ch_num)

        self.embed_dim = prune_output_ch_num

    def get_params_count(self):
        dim = self.d_model
        active = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1:
            active *= self.num_gates
        elif self.zeta.shape[2] == 1:
            active *= self.head_num
        total_params = dim * dim * 3 + dim * 3  # qkv weights and bias
        total_params += dim * dim + dim  # proj weights and bias
        active_params = dim * active * 3 + active * 3
        active_params += active * dim + dim
        return total_params, active_params

    def get_flops(self, seqnum):
        H = self.head_num
        N = seqnum
        d = self.num_gates
        sd = self.searched_zeta.sum().data if self.is_searched else self.zeta.sum().data
        if self.zeta.shape[-1] == 1:  # Head Elimination
            sd *= self.num_gates
        elif self.zeta.shape[1] == 1:  # Uniform Search
            sd *= self.head_num  # TODO

        total_flops, active_flops = 0, 0

        total_flops += 3 * N * 2 * (H * d) * (H * d)  #linear: qkv
        active_flops += 3 * N * 2 * (H * sd) * (H * sd)  #linear: qkv

        total_flops += 2 * H * d * N * N + N * N  #q@k * scale
        active_flops += 2 * H * sd * N * N + N * N  #q@k * scale

        total_flops += 5 * H * N * N  #softmax: exp, sum(exp), div, max, x-max
        active_flops += 5 * H * N * N  #softmax: exp, sum(exp), div, max, x-max

        total_flops += H * 2 * N * N * d  #attn@v
        active_flops += H * 2 * N * N * sd  #attn@v

        total_flops += N * 2 * (H * d) * (H * d)  #linear: proj
        active_flops += N * 2 * (H * sd) * (H * sd)  #linear: proj

        return total_flops, active_flops


class SrSparseMSDeformableAttention(SparseMSDeformableAttention):
    """Spatial Reduction Multiple Scale Deformable Attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        head_num: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        search_type: str = 'embed',
        linear_sr_atten: bool = False,
        sr_ratio: list[int] = 2,
        linear_sr_minval: int = 1,
    ) -> None:

        super().__init__(embed_dim, head_num, num_levels, num_points, search_type)
        self.linear_atten = linear_sr_atten
        self.sr_ratio = sr_ratio
        self.linear_sr_minval = linear_sr_minval
        if sr_ratio > 1:
            if not linear_sr_atten:
                self.sr_layer = ConvNormLayer(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio, norm_layer='layer')
            else:
                self.sr_layer = nn.Sequential(
                    # nn.AdaptiveAvgPool2d(output_size=7),
                    ConvNormLayer(embed_dim, embed_dim, kernel_size=1, stride=1, norm_layer='layer'),
                    nn.GELU(),
                )
        else:
            self.sr_layer = nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list[tuple[int] | list[int]],
        value_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q, dim_q = query.shape  # 8, 492
        Len_v, dim_v = value.shape[1:]  # value [8, 10164, 256]

        if self.sr_ratio > 1:  # do spatial reduction operation
            split_shape = [h * w for h, w in value_spatial_shapes]  # [5184, 1296, 324]
            # 每个尺度的value用split分离出来
            value_list = value.split(split_shape, dim=1)  # [[8, 5184, 256], [8, 1296, 256], [8, 324, 256]]
            sr_spatial_shapes = []
            sr_value_list = []
            for i in range(len(split_shape)):
                ori_h, ori_w = value_spatial_shapes[i]
                sr_h, sr_w = ori_h // self.sr_ratio, ori_w // self.sr_ratio
                sr_value = value_list[i]
                sr_value = sr_value.permute(0, 2, 1).reshape(bs, dim_v, ori_h, ori_w)
                if self.linear_atten:
                    sr_h = max(sr_h, self.linear_sr_minval)
                    sr_w = max(sr_w, self.linear_sr_minval)
                    sr_value = F.adaptive_avg_pool2d(sr_value, (sr_h, sr_w))
                sr_value: torch.Tensor = self.sr_layer(sr_value)
                sr_h, sr_w = sr_value.shape[-2:]  # update the sr_h and sr_w to actually spatial shape of sr_value Tensor.
                sr_spatial_shapes.append((sr_h, sr_w))
                sr_value = sr_value.reshape(bs, dim_v, -1).permute(0, 2, 1)
                sr_value_list.append(sr_value)

            sr_values = torch.cat(sr_value_list, dim=1)
            value = sr_values
            value_spatial_shapes = sr_spatial_shapes
            Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask

        value = value.reshape(bs, Len_v, self.head_num, self.head_dim)  # value为每个head分配输入数据, [8, 10164, 8, 32]
        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.zeta
            value = value * z

        # 采样偏置 [bs, lenq, M, L, K, 2]--[8, 492, 8, 3, 4, 2]
        sampling_offsets = self.sampling_offsets(query)  # [8, 492, 8*3*4*2]
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points, 2)
        # 获得各个参考点的注意力加权 成 [bs, lenq, M, L*K]--[8, 492, 8, 12]
        attention_weights = self.attention_weights(query)  # [8, 492, 8*12]
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels * self.num_points)
        # 多尺度上进行各个参考点的softmax，所有参考点概率和为1。[bs, lenq, M, L, K]--[8, 492, 8, 3, 4]
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(bs, Len_q, self.head_num, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:  # [8, 492, 1, 2]
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:  # [8, 492, 1, 4]
            # [8, 492, 1, 1, 1, 2] 在(w,h)两个维度上乘sampling_offsets，对w/h进行伸缩。
            sampling_locations = (reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1]))

        output = self.deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)  # [8, 492, 256]

        output = self.output_proj(output)  # 线性层映射，使output参数得到一个聚合。

        return output


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


