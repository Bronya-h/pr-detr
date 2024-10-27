import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.core import register

from .mlp import MLP, SparseMLP, LinearNormAct
from .attention import (
    MSDeformableAttention,
    MultiHeadAttentionViT,
    SparseMultiHeadAttentionViT,
    SparseMSDeformableAttention,
    SrMSDeformableAttention,
    SrMultiHeadAttentionViT,
    SrSparseMultiHeadAttentionViT,
    SrSparseMSDeformableAttention,
    FocusedSrSparseMultiHeadAttentionViT,
)
from .channel_map import ChannelMapper, ShapeSpec, SparseChannelMapper
from .conv import ConvNormLayer, prune_normalization_layer
from .utils import bias_init_with_prob, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, inverse_sigmoid
from .couple import SparseCouple
from ..fts import (
    ForegroundTokenScorePredictor,
    MultiCategoryScorePredictor,
    ForeGroundScoreSeleter,
)
from ..token_merge import bipartite_soft_matching, merge_wavg, ToMe


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.,
        activation="relu",
        n_levels=4,
        n_points=4,
        sr_ratio: int = 2,
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.n_levels = n_levels
        self.n_points = n_points
        assert self.d_model % n_head == 0, f"d_model-{d_model} can not be exact division by n_head-{n_head}"
        self.self_attn = MultiHeadAttentionViT(d_model, head_num=n_head, dropout_prob=dropout_prob, shortcut=True)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        if sr_ratio > 1:
            self.cross_attn = SrMSDeformableAttention(d_model, n_head, n_levels, n_points, linear_sr_atten=linear_sr_atten, sr_ratio=sr_ratio, linear_sr_minval=linear_sr_minval)
        else:
            self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = MLP(d_model, dim_feedforward, d_model, num_layers=2, act=activation, act_final_layer=False, use_dropout=True, dropout_prob=dropout_prob, dropout_final_layer=True)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, memory_level_start_index, attn_mask=None, memory_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        else:
            tgt2 = self.self_attn(q, k, tgt, attn_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class SparseTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.,
        activation="relu",
        n_levels=4,
        n_points=4,
        mha_search_type='embed',
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.n_levels = n_levels
        self.n_points = n_points
        self.mha_search_type = mha_search_type
        assert self.d_model % n_head == 0, f"d_model-{d_model} can not be exact division by n_head-{n_head}"
        # self attention
        self.self_attn = SparseMultiHeadAttentionViT(d_model, n_head, dropout_prob=dropout_prob, search_type=mha_search_type)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = SparseMSDeformableAttention(d_model, n_head, n_levels, n_points, mha_search_type)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = SparseMLP(d_model, dim_feedforward, d_model, num_layers=2, act=activation, act_final_layer=False, use_dropout=True, dropout_prob=dropout_prob, dropout_final_layer=True)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, memory_level_start_index, attn_mask=None, memory_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        else:
            tgt2 = self.self_attn(q, k, tgt, attn_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt

    def compress(self):
        self.self_attn.compress()
        self.cross_attn.compress()
        self.ffn.compress()

    def decompres(self):
        self.self_attn.decompress()
        self.cross_attn.decompress()
        self.ffn.decompress()

    def prune_input_channel(self, in_mask):
        if in_mask.dtype != torch.bool:
            in_mask = in_mask.to(torch.bool)

        self.self_attn.prune_input_channel(in_mask)
        self.self_attn.prune_output_proj_output_channel(in_mask)
        self.cross_attn.prune_input_channel(in_mask)
        self.cross_attn.prune_output_proj_output_channel(in_mask)

        self.ffn.prune_input_channel(in_mask)
        self.ffn.prune_lastlayer_output_channel(in_mask)

        self.norm1 = prune_normalization_layer(self.norm1, in_mask)
        self.norm2 = prune_normalization_layer(self.norm2, in_mask)
        self.norm3 = prune_normalization_layer(self.norm3, in_mask)

    def prune_output_channel(self):
        self.self_attn.prune_output_channel()
        self.cross_attn.prune_output_channel()
        self.ffn.prune_output_channel()

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
        self_attn_zeta = self.self_attn.searched_zeta if self.self_attn.is_searched else self.self_attn.zeta
        self_attn_zeta_num = self_attn_zeta.sum().data
        total_flops += B * N * C * 2  
        activate_flops += B * N * self_attn_zeta_num * 2 
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


class SrSparseTransformerDecoderLayer(SparseTransformerDecoderLayer):
    """Spatial reduction Sparse Transformer Decoder Layer."""

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        mha_search_type='embed',
        sr_ratio: int = 2,
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout_prob=dropout_prob,
            activation=activation,
            n_levels=n_levels,
            n_points=n_points,
            mha_search_type=mha_search_type,
            **kwargs,
        )
        self.sr_ratio = sr_ratio  # how many tokens should be remove in the one decoder layer.
        if focus_atten == True:
            self.self_attn = FocusedSrSparseMultiHeadAttentionViT(
                d_model=d_model,
                head_num=n_head,
                dropout_prob=dropout_prob,
                search_type=mha_search_type,
                sr_ratio=0,
                focusing_factor=focusing_factor,
                dwc_kernel=dwc_kernel,
                kernel_fun=kernel_fun,
            )
        # query 之间的自注意力计算不需要空间下采样

        self.cross_attn = SrSparseMSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            mha_search_type,
            linear_sr_atten=linear_sr_atten,
            sr_ratio=sr_ratio,
            linear_sr_minval=linear_sr_minval,
        )

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, memory_level_start_index, attn_mask=None, memory_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask, spatial_shapes=memory_spatial_shapes)
        else:
            tgt2 = self.self_attn(q, k, tgt, attn_mask, spatial_shapes=memory_spatial_shapes)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class TomeSparseTransformerDecoderLayer(SparseTransformerDecoderLayer):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        mha_search_type='embed',
        tome_ratio=0.5,
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout_prob=dropout_prob,
            activation=activation,
            n_levels=n_levels,
            n_points=n_points,
            mha_search_type=mha_search_type,
        )
        self.tome_ratio = tome_ratio  # how many tokens should be remove in the one decoder layer.
        self.tome_block = ToMe(tome_ratio)

    def forward(
        self,
        tgt,
        reference_points,
        ref_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
        dn_target_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if isinstance(self.self_attn, nn.MultiheadAttention):
            tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        else:
            tgt2 = self.self_attn(q, k, tgt, attn_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.tome_ratio > 0:
            tgt, reference_points, attn_mask, ref_points, dn_target_mask = self.tome_block(tgt, reference_points, attn_mask, ref_points, dn_target_mask)

        # ffn
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt, reference_points, attn_mask, ref_points, dn_target_mask


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        num_layers=3,
        eval_idx=-1,
        sr_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.n_levels = n_levels
        self.n_points = n_points
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=self.d_model,
                n_head=self.n_head,
                dim_feedforward=self.dim_feedforward,
                dropout_prob=self.dropout_prob,
                activation=self.activation,
                n_levels=self.n_levels,
                n_points=self.n_points,
                sr_ratio=sr_ratio_list[i] if i < len(sr_ratio_list) else 0,
                linear_sr_atten=linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
            ) for i in range(num_layers)
        ])

    def forward(self, tgt, ref_points_unact, memory, memory_spatial_shapes, memory_level_start_index, bbox_head, score_head, query_pos_head, attn_mask=None, memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        ref_points = None
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, memory_level_start_index, attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class SparseTransformerDecoder(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        num_layers=3,
        eval_idx=-1,
        mha_search_type: str = 'embed',
        # couple_layer_head = False,
        sr_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.n_levels = n_levels
        self.n_points = n_points
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        # self.couple_layer_head = couple_layer_head
        if len(sr_ratio_list) > 0:
            self.layers = nn.ModuleList([
                SrSparseTransformerDecoderLayer(
                    d_model=self.d_model,
                    n_head=self.n_head,
                    dim_feedforward=self.dim_feedforward,
                    dropout_prob=self.dropout_prob,
                    activation=self.activation,
                    n_levels=self.n_levels,
                    n_points=self.n_points,
                    mha_search_type=mha_search_type,
                    sr_ratio=sr_ratio_list[i] if i < len(sr_ratio_list) else 1,
                    linear_sr_atten=linear_sr_atten,
                    linear_sr_minval=linear_sr_minval,
                    focus_atten=focus_atten,
                    focusing_factor=focusing_factor,
                    dwc_kernel=dwc_kernel,
                    kernel_fun=kernel_fun,
                ) for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                SparseTransformerDecoderLayer(
                    d_model=self.d_model,
                    n_head=self.n_head,
                    dim_feedforward=self.dim_feedforward,
                    dropout_prob=self.dropout_prob,
                    activation=self.activation,
                    n_levels=self.n_levels,
                    n_points=self.n_points,
                    mha_search_type=mha_search_type,
                ) for _ in range(num_layers)
            ])
    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
        couple_layers=None,
        **kwargs,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        ref_points = None
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, memory_level_start_index, attn_mask, memory_mask, query_pos_embed)

            output = couple_layers[i](output) if couple_layers is not None else output

            bbox_head_out = bbox_head[i](output)
            score_head_out = score_head[i](output)

            inter_ref_bbox = F.sigmoid(bbox_head_out + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head_out)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head_out + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head_out)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), None

    def prune_input_channel(self, in_mask):
        for i in range(len(self.layers)):
            self.layers[i].prune_input_channel(in_mask)

    def prune_output_channel(self):
        for i in range(len(self.layers)):
            self.layers[i].prune_output_channel()


class TpSparseTransformerDecoder(SparseTransformerDecoder):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout_prob=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        num_layers=3,
        eval_idx=-1,
        mha_search_type: str = 'embed',
        cascade_end: float = 0.4,
        token_remain_ratio: float = 0.5,
        num_classes: int = 80,
        token_prune_method: str = 'tome',
        **kwargs,
    ):
        super().__init__(d_model, n_head, dim_feedforward, dropout_prob, activation, n_levels, n_points, num_layers, eval_idx, mha_search_type, **kwargs)
        self.token_prune_method = token_prune_method
        if token_prune_method == 'fg':
            self.fg_token_selector = ForeGroundScoreSeleter(d_model, num_layers, n_levels, cascade_end, token_remain_ratio)
        else:
            self.layers = nn.ModuleList([
                TomeSparseTransformerDecoderLayer(
                    d_model=self.d_model,
                    n_head=self.n_head,
                    dim_feedforward=self.dim_feedforward,
                    dropout_prob=self.dropout_prob,
                    activation=self.activation,
                    n_levels=self.n_levels,
                    n_points=self.n_points,
                    mha_search_type=mha_search_type,
                    tome_ratio=1 - token_remain_ratio,
                ) for _ in range(num_layers)
            ])
        # self.fg_token_selector = ForeGroundScoreSeleter(d_model, 1, n_levels, cascade_end, token_remain_ratio)
        # self.mcateg_score_predictor = MultiCategoryScorePredictor(d_model, num_classes)

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
        couple_layers=None,
        dn_target_mask=None,
        **kwargs,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dn_target_mask_list = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        ref_points = None
        for i, layer in enumerate(self.layers):
            # ref_points_input = ref_points_detach.unsqueeze(2)
            ref_points_input: torch.Tensor = ref_points_detach
            query_pos_embed = query_pos_head(ref_points_detach)

            if self.token_prune_method == 'tome':
                output, ref_points_input, attn_mask, ref_points, dn_target_mask = layer(
                    output,
                    ref_points_input.unsqueeze(2),
                    ref_points,
                    memory,
                    memory_spatial_shapes,
                    memory_level_start_index,
                    attn_mask,
                    memory_mask,
                    query_pos_embed,
                    dn_target_mask=dn_target_mask,
                )
                ref_points_input = ref_points_input.squeeze(2)
            else:
                output = layer(output, ref_points_input.unsqueeze(2), memory, memory_spatial_shapes, memory_level_start_index, attn_mask, memory_mask, query_pos_embed, 1)

            output = couple_layers[i](output) if couple_layers is not None else output  # struct prune.

            bbox_head_out = bbox_head[i](output)
            score_head_out = score_head[i](output)

            inter_ref_bbox = F.sigmoid(bbox_head_out + inverse_sigmoid(ref_points_input))

            if self.training:
                dec_out_logits.append(score_head_out)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head_out + inverse_sigmoid(ref_points.squeeze(dim=-2))))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head_out)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            # 当前的inter_ref_bbox是sigmoid之后的
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

            dn_target_mask_list.append(dn_target_mask)

        return dec_out_bboxes, dec_out_logits, dn_target_mask_list


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """Preprocess the contrastive denoising training pairs.

    Args:
        targets (dict): Information about classes and coordinates in the image.
        num_classes (int): Number of classes in the dataset.
        num_queries (int): Number of queries in the decoder.
        class_embed (int): Embedding matrix for converting discrete classes to vectors.
        num_denoising (int, optional): Number of contrastive denoising training pairs. Defaults to 100.
        label_noise_ratio (float, optional): Noise intensity for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise intensity for bbox coordinates. Defaults to 1.0.

    Returns:
        input_query_class (torch.Tensor): Noisy and vectorized query class representations, with num_group tuples of 2.
        input_query_bbox (torch.Tensor): Noisy query bbox coordinate matrix, with num_group tuples of 2.
        attn_mask (torch.Tensor): Attention mask focusing on queries and ignoring denoising areas.
        dn_meta (dict): Denoising training metadata, including dn_positive_idx, dn_num_group, dn_num_split (num_denoising, num_queries).
    """
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]  # Each sample has several targets
    device = targets[0]['labels'].device

    max_gt_num = max(num_gts)  # There are several possible targets gt in the current picture.
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num  # Possible denoising groups for each target
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], 4, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1

    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])  
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    # 负 gtmask 的生成
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1  
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask 
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts]) 
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox) 
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0) 
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {"dn_positive_idx": dn_positive_idx, "dn_num_group": num_group, "dn_num_split": [num_denoising, num_queries]}

    return input_query_class, input_query_bbox, attn_mask, dn_meta


@register
class OriginRTDETRDecoder(nn.Module):
    __share__ = ['num_classes']

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        position_embed_type='sine',
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout_prob=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        num_coordinates=4,
        in_channel_names=["p3", "p4", "p5"],
        spatial_reduce_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        **kwargs,
    ):

        super(OriginRTDETRDecoder, self).__init__()
        assert position_embed_type in ['sine', 'learned'], f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.num_coordinates = num_coordinates
        self.in_channel_names = in_channel_names
        self.dropout_prob = dropout_prob

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.decoder = TransformerDecoder(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout_prob,
            activation,
            num_levels,
            num_decoder_points,
            num_decoder_layers,
            eval_idx,
            sr_ratio_list=spatial_reduce_ratio_list,
            linear_sr_atten=linear_sr_atten,
            linear_sr_minval=linear_sr_minval,
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(self.num_classes + 1, hidden_dim, padding_idx=self.num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(self.num_coordinates, 2 * hidden_dim, hidden_dim, num_layers=2)  # TODO: convert to SparseMLP

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim, ))  # TODO: convert to SparseMLP 1 layer.
        self.enc_score_head = nn.Linear(hidden_dim, self.num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, self.num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3) for _ in range(num_decoder_layers)])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        input_proj_shapespec = {k: ShapeSpec(channels=chn) for k, chn in zip(self.in_channel_names, feat_channels)}
        self.input_proj = ChannelMapper(
            input_shapes=input_proj_shapespec,
            in_features=self.in_channel_names,
            out_channels=self.hidden_dim,
            kernel_size=1,
            bias=False,
            num_outs=self.num_levels,
            norm_layer='bn',
            activation=None,
        )

    def _get_encoder_input(self, feats):
        """多尺度特征投影成一个维度的通道数量，展平多尺度特征，记录每个尺度的开始位置。

        Args:
            feats (list[torch.Tensor]): 主干网络的多尺度特征列表。

        Returns:
            tuple(feat_flatten, spatial_shapes, level_start_index): 展平特征、每个特征图的长和宽、每个尺度在展平矩阵中的开始位置
        """
        # get projection features
        proj_feats = self.input_proj(feats)

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        # 将所有的多尺度特征flatten成一维向量，并用level_start_index记录每一个尺度特征图的第一个特征点在flatten向量中的位置
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])  # 这是每个尺度特征图在一维展平向量中的末尾位置

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)  # 在1维concat, 形成了[b, h*w*scale_num, c]
        level_start_index.pop()  # 丢弃最后一个末尾位置就变成了开头位置，因为第一个元素是0
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        """生成每个多尺度特征图的anchor

        Args:
            spatial_shapes (list, optional): 多尺度特征图的长宽. Defaults to None.
            grid_size (float, optional): 每个grid的大小. Defaults to 0.05.
            dtype (torch.dtype, optional): torch的数据类型. Defaults to torch.float32.
            device (str, optional): 生成的anchor张量的设备位置. Defaults to 'cpu'.

        Returns:
            anchors (torch.Tensor): 所有尺度特征图的anchor.
            valid_mask (torch.Tensor): 有效的anchor的掩码, 对应于anchors.
        """
        if spatial_shapes is None:
            # 为空就使用默认的eval的图片大小生成每个尺度的特征图的大小
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 每个尺度生成anchor网格
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)  # [1, 2]
            # 0维拓展成[grid_num, 2], 加 0.5 移动到网格中心点，除以valid_WH归一化到0~1范围
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)  # 每个grid的宽高，2**lvl代表特征图下采样后，特征图变少
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, self.num_coordinates))  # 最后一维拼接[x,y,w,h], 形成维度[1, h*w, self.num_coordinates]

        anchors = torch.concat(anchors, 1).to(device)  # 所有尺度的anchor拼接成[1, h*w*scale, self.num_coordinates]
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  # 筛选坐标值在[0.01, 0.99]之间的anchor
        anchors = torch.log(anchors / (1 - anchors))  # anchor 的非线性变换
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)  # 提取出mask有效的anchors

        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        """从encoder输出中处理出能用于decoder输入的query。

        Args:
            memory (torch.Tensor): encoder的输出。
            spatial_shapes (torch.Tensor): 多尺度特征图的长宽形状
            denoising_class (torch.Tensor, optional): 去噪训练的类别张量. Defaults to None.
            denoising_bbox_unact (torch.Tensor, optional): 去噪训练的bbox坐标张量. Defaults to None.

        Returns:
            target (torch.Tensor): 从encoder的所有多尺度flatten后的token集合中，抽取出来的top num_queries个token向量表示.
            reference_points_unact (torch.Tensor): top num_queries个token的经过线性层预测的bbox坐标与anchor grid相加后结果.
            enc_topk_bboxes (torch.Tensor): reference_points_unact经过sigmoid激活函数计算后的结果。
            enc_topk_logits (torch.Tensor):  top num_queries个token的经过线性层预测的class score logit.
        """
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        # 只计算anchor有效的多尺度特征图的token
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export

        output_memory = self.enc_output(memory)  # 经过一层linear和layernorm, 把所有尺度在token维度做归一化

        enc_outputs_class = self.enc_score_head(output_memory)  # 一个linear获得num_classes维度的一个logit输出
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # 3层MLP, 预测每个token的4个位置信息(cx,cy,w,h)

        # 在class score维度上取最大值得到每个token的最大类别置信度，再计算出前num_queries个token的indices
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # 从预测出的bbox的坐标张量中，获取预测分数topk个box坐标作为reference_points
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)  # 经过sigmoid, 限制参考点分数在0~1
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        # 获取topk个类别score的logit
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            # 获取encoder输出没有经过线性层预测的分数topk的token表征向量
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)  # 将去噪训练的类别向量与target向量拼接在一起

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        """RTDETR的对比去噪训练和decoder前向传播部分

        Args:
            feats (list): 主干网络提取的多尺度特征图。
            targets (dict|list, optional): 一个batch的标注目标类别和bbox. Defaults to None.

        Returns:
            out (dict): 包含out["pred_logits"]: Tensor, out["pred_boxes"]: Tensor,
                        out["aux_outputs"]: list, out["dn_meta"]: dict.
        """
        # input projection and embedding
        # 特征通道投影，展平多尺度成一维向量，记录每个尺度位置
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              level_start_index,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])  # 根据decoder输出的logits/bboxes计算辅助损失
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))  # 根据encoder输出logit/boxes计算辅助损失

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


@register
class SparseRTDETRDecoder(nn.Module):
    __share__ = ['num_classes']

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        position_embed_type='sine',
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout_prob=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        num_coordinates=4,
        in_channel_names=["p3", "p4", "p5"],
        mha_search_type: str = 'embed',
        token_prune: bool = False,
        token_prune_method: str = 'tome',
        cascade_end: int = 0.4,
        token_remain_ratio: float = 0.5,
        spatial_reduce_ratio_list: list[int] = [],
        linear_sr_atten: bool = False,
        linear_sr_minval: int = 1,
        focus_atten: bool = False,
        focusing_factor: int = 3,
        dwc_kernel: int = 5,
        kernel_fun: str = 'relu',
        **kwargs,
    ):

        super(SparseRTDETRDecoder, self).__init__()
        assert position_embed_type in ['sine', 'learned'], f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.num_coordinates = num_coordinates
        self.in_channel_names = in_channel_names
        self.token_prune = token_prune
        self.token_prune_method = token_prune_method
        self.cascade_end = cascade_end
        self.token_remain_ratio = token_remain_ratio

        # backbone feature projection
        input_proj_shapespec = {k: ShapeSpec(channels=chn) for k, chn in zip(self.in_channel_names, feat_channels)}
        # TODO: SparseChannelMapper
        self.input_proj = ChannelMapper(
            input_shapes=input_proj_shapespec,
            in_features=self.in_channel_names,
            out_channels=self.hidden_dim,
            kernel_size=1,
            bias=False,
            num_outs=self.num_levels,
            norm_layer='bn',
            activation=None,
        )

        # Transformer module

        # decoder_class = SparseTransformerDecoder if not token_prune else TpSparseTransformerDecoder
        if not token_prune:
            self.decoder = SparseTransformerDecoder(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout_prob,
                activation,
                num_levels,
                num_decoder_points,
                num_decoder_layers,
                eval_idx,
                mha_search_type,
                spatial_reduce_ratio_list,
                linear_sr_atten,
                linear_sr_minval=linear_sr_minval,
                focus_atten=focus_atten,
                focusing_factor=focusing_factor,
                dwc_kernel=dwc_kernel,
                kernel_fun=kernel_fun,
            )
        else:
            self.decoder = TpSparseTransformerDecoder(
                hidden_dim,
                nhead,
                dim_feedforward,
                dropout_prob,
                activation,
                num_levels,
                num_decoder_points,
                num_decoder_layers,
                eval_idx,
                mha_search_type,
                cascade_end,
                token_remain_ratio=token_remain_ratio,
                num_classes=self.num_classes,
                token_prune_method=token_prune_method,
            )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(self.num_classes + 1, hidden_dim, padding_idx=self.num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        # self.query_pos_head = MLP(self.num_coordinates, 2 * hidden_dim, hidden_dim, num_layers=2)  # TODO: convert to SparseMLP
        self.query_pos_head = SparseMLP(self.num_coordinates, 2 * hidden_dim, hidden_dim, num_layers=2)  # TODO: convert to SparseMLP

        # encoder head
        # self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim, ))  # TODO: convert to SparseMLP 1 layer.
        self.enc_output = LinearNormAct(hidden_dim, hidden_dim, norm_layer='layer')
        self.enc_score_head = nn.Linear(hidden_dim, self.num_classes)
        # self.enc_bbox_head = MLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3)
        self.enc_bbox_head = SparseMLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, self.num_classes) for _ in range(num_decoder_layers)])
        # self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([SparseMLP(hidden_dim, hidden_dim, self.num_coordinates, num_layers=3) for _ in range(num_decoder_layers)])

        self.couple_enc = SparseCouple([self.enc_output], [self.enc_score_head, self.enc_bbox_head],
                                       zeta_shapes=[1, 1, self.enc_output.linear.out_features],
                                       num_gates=self.enc_output.linear.out_features)
        # self.couple_dec_list = [SparseCouple([self.decoder.layers[i]], [self.dec_score_head[i], self.dec_bbox_head[i]], zeta_shapes=[], num_gates=100) for i in range(len(self.decoder.layers))]

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output.linear.weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, feats):
        """多尺度特征投影成一个维度的通道数量，展平多尺度特征，记录每个尺度的开始位置。

        Args:
            feats (list[torch.Tensor]): 主干网络的多尺度特征列表。

        Returns:
            tuple(feat_flatten, spatial_shapes, level_start_index): 展平特征、每个特征图的长和宽、每个尺度在展平矩阵中的开始位置
        """
        # get projection features
        proj_feats = self.input_proj(feats)

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        # 将所有的多尺度特征flatten成一维向量，并用level_start_index记录每一个尺度特征图的第一个特征点在flatten向量中的位置
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])  # 这是每个尺度特征图在一维展平向量中的末尾位置

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)  # 在1维concat, 形成了[b, h*w*scale_num, c]
        level_start_index.pop()  # 丢弃最后一个末尾位置就变成了开头位置，因为第一个元素是0
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        """生成每个多尺度特征图的anchor

        Args:
            spatial_shapes (list, optional): 多尺度特征图的长宽. Defaults to None.
            grid_size (float, optional): 每个grid的大小. Defaults to 0.05.
            dtype (torch.dtype, optional): torch的数据类型. Defaults to torch.float32.
            device (str, optional): 生成的anchor张量的设备位置. Defaults to 'cpu'.

        Returns:
            anchors (torch.Tensor): 所有尺度特征图的anchor.
            valid_mask (torch.Tensor): 有效的anchor的掩码, 对应于anchors.
        """
        if spatial_shapes is None:
            # 为空就使用默认的eval的图片大小生成每个尺度的特征图的大小
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 每个尺度生成anchor网格
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)  # [1, 2]
            # 0维拓展成[grid_num, 2], 加 0.5 移动到网格中心点，除以valid_WH归一化到0~1范围
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)  # 每个grid的宽高，2**lvl代表特征图下采样后，特征图变少
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, self.num_coordinates))  # 最后一维拼接[x,y,w,h], 形成维度[1, h*w, self.num_coordinates]

        anchors = torch.concat(anchors, 1).to(device)  # 所有尺度的anchor拼接成[1, h*w*scale, self.num_coordinates]
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  # 筛选坐标值在[0.01, 0.99]之间的anchor
        anchors = torch.log(anchors / (1 - anchors))  # anchor 的非线性变换
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)  # 提取出mask有效的anchors

        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes):
        """从encoder输出中处理出能用于decoder输入的query。

        Args:
            memory (torch.Tensor): encoder的输出。
            spatial_shapes (torch.Tensor): 多尺度特征图的长宽形状。
            denoising_class (torch.Tensor, optional): 去噪训练的类别张量. Defaults to None.
            denoising_bbox_unact (torch.Tensor, optional): 去噪训练的bbox坐标张量. Defaults to None.

        Returns:
            target (torch.Tensor): 从encoder的所有多尺度flatten后的token集合中，抽取出来的top num_queries个token向量表示.
            reference_points_unact (torch.Tensor): top num_queries个token的经过线性层预测的bbox坐标与anchor grid相加后结果.
            enc_topk_bboxes (torch.Tensor): reference_points_unact经过sigmoid激活函数计算后的结果。
            enc_topk_logits (torch.Tensor):  top num_queries个token的经过线性层预测的class score logit.
        """
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        # 只计算anchor有效的多尺度特征图的token
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export
        # Couple [SparseChannelMap,]  [enc_output]'
        output_memory = self.enc_output(memory)  # 经过一层linear和layernorm, 把所有尺度在token维度做归一化
        # SparseCouple 2 [self.enc_output,] [self.enc_score_head, self.enc_bbox_head]
        output_memory = self.couple_enc(output_memory)
        enc_outputs_class = self.enc_score_head(output_memory)  # 一个linear获得num_classes维度的一个logit输出
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # 3层MLP, 预测每个token的4个位置信息(cx,cy,w,h)

        # 在class score维度上取最大值得到每个token的最大类别置信度，再计算出前num_queries个token的indices
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # 从预测出的bbox的坐标张量中，获取预测分数topk个box坐标作为reference_points
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)  # 经过sigmoid, 限制参考点分数在0~1

        # 获取topk个类别score的logit
        enc_topk_logits = enc_outputs_class.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            # 获取encoder输出没有经过线性层预测的分数topk的token表征向量
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets=None):
        """RTDETR的对比去噪训练和decoder前向传播部分

        Args:
            feats (list): 主干网络提取的多尺度特征图。
            targets (dict|list, optional): 一个batch的标注目标类别和bbox. Defaults to None.

        Returns:
            out (dict): 包含out["pred_logits"]: Tensor, out["pred_boxes"]: Tensor,
                        out["aux_outputs"]: list, out["dn_meta"]: dict.
        """
        # input projection and embedding
        # 特征通道投影，展平多尺度成一维向量，记录每个尺度位置
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale
                )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes)

        if denoising_bbox_unact is not None:
            init_ref_points_unact = torch.concat([denoising_bbox_unact, init_ref_points_unact], 1)
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)  # 将去噪训练的类别向量与target向量拼接在一起

        if dn_meta is not None and self.token_prune and self.token_prune_method == 'tome':
            # 对比去噪训练的token的位置的掩码生成
            dn_target_mask = torch.zeros(dn_meta['dn_num_split'][0] + dn_meta['dn_num_split'][1], device=target.device, dtype=torch.bool)
            dn_target_mask[:dn_meta['dn_num_split'][0]] = True
        else:
            dn_target_mask = None

        # decoder
        out_bboxes, out_logits, dn_target_mask_list = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            dn_target_mask=dn_target_mask,
        )

        if self.training and dn_meta is not None:
            if self.token_prune and self.token_prune_method == 'tome':
                # 根据dn_target_mask滤除掉用于对比去噪训练的token
                dn_out_bboxes_list = []
                dn_out_logits_list = []
                out_bboxes_list = []
                out_logits_list = []
                for i in range(len(out_bboxes)):
                    dn_bboxes = out_bboxes[i][:, dn_target_mask_list[i], ...]
                    bboxes = out_bboxes[i][:, ~dn_target_mask_list[i], ...]
                    dn_logits = out_logits[i][:, dn_target_mask_list[i], ...]
                    logits = out_logits[i][:, ~dn_target_mask_list[i], ...]
                    dn_out_bboxes_list.append(dn_bboxes)
                    dn_out_logits_list.append(dn_logits)
                    out_bboxes_list.append(bboxes)
                    out_logits_list.append(logits)

                dn_out_logits = dn_out_logits_list
                dn_out_bboxes = dn_out_bboxes_list
                out_logits = out_logits_list
                out_bboxes = out_bboxes_list
            else:
                dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])  # 根据decoder输出的logits/bboxes计算辅助损失
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))  # 根据encoder输出logit/boxes计算辅助损失

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]

    def prune_input_channel(self, in_masks=None):
        self.input_proj.prune_input_channel(in_masks)

    def prune_output_channel(self):
        # input_proj_out_mask = None
        # if isinstance(self.input_proj, SparseChannelMapper):
        #     input_proj_out_mask = self.input_proj.get_output_channel_mask()
        #     self.input_proj.prune_output_channel()

        # if input_proj_out_mask is not None:
        #     self.decoder.prune_input_channel(input_proj_out_mask)
        self.decoder.prune_output_channel()
        self.query_pos_head.prune_output_channel()
        self.enc_bbox_head.prune_output_channel()
        for i in range(len(self.dec_bbox_head)):
            self.dec_bbox_head[i].prune_output_channel()

        self.couple_enc.prune_last_layer_channel()
        self.couple_enc.prune_next_layer_channel()
