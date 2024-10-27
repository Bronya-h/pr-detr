from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from ..layers.attention import SparseMultiHeadAttentionViT, SparseMSDeformableAttention
from ..layers.conv import SparseConvNormLayer
from ..layers.mlp import SparseMLP
from ..layers.encoder import SparseRTDETREncoder
from ..layers.decoder import SparseRTDETRDecoder
from ..backbone.spresnet import SparsePResNet

@register
class SparseRTDETR(nn.Module):
    __inject__ = [
        'backbone',
        'encoder',
        'decoder',
    ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        # self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        self.searchable_modules = self.get_searchable_modules()
        print("----------------Searchable Modules-------------------------")
        print(*list(self.searchable_modules.keys()), sep='\n')
        print("-----------------------------------------------------------")
        # self.device = next(self.parameters()).device
        self.device = None
        self.is_pruned = False

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def get_searchable_modules(self):
        # searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        searchable_modules = {n: m for n, m in self.named_modules(prefix='SparseRTDETR') if hasattr(m, 'zeta')}
        return searchable_modules

    def calculate_search_threshold(self, budget_attn, budget_mlp, budget_conv):
        """macro calculate search threshold for basic network block.
        scope of statistics is all the same searched Modules.

        Arguments:
            budget_attn -- budget for attention block.
            budget_mlp -- budget for MLP block.
            budget_conv -- budget for convolution block.

        Returns:
            threshold_attn, threshold_mlp, threshold_conv [float]: threshold for basic block.
        """
        zetas_attn, zetas_mlp, zetas_conv = self.give_zetas()
        zetas_attn = sorted(zetas_attn)
        zetas_mlp = sorted(zetas_mlp)
        zetas_conv = sorted(zetas_conv)
        threshold_attn = -float('inf') if len(zetas_attn) == 0 else zetas_attn[int((1.0 - budget_attn) * len(zetas_attn))]
        threshold_mlp = -float('inf') if len(zetas_mlp) == 0 else zetas_mlp[int((1.0 - budget_mlp) * len(zetas_mlp))]
        threshold_conv = -float('inf') if len(zetas_conv) == 0 else zetas_conv[int((1.0 - budget_conv) * len(zetas_conv))]
        return threshold_attn, threshold_mlp, threshold_conv
    
    def calculate_search_threshold_micro(self, zeta, budget=0.9):
        """micro calculate search threshold for basic network block.
        scope of statistics is one searched Modules that have zeta.

        Arguments:
            zeta -- soft mask of one searched Module.

        Keyword Arguments:
            budget -- budget for remain structure number of this block. (default: {0.9})

        Returns:
            thres [float]: threshold for this block.
        """
        zeta = torch.squeeze(zeta)
        zeta = torch.sort(zeta).values
        n = zeta.numel()
        tidx = min(n, max(0, round((1 - budget) * n)))
        thres = zeta[tidx].item()
        return thres
        

    def n_remaining(self, m):
        # if hasattr(m, 'num_heads'):
        # if isinstance(m, SparseMultiHeadAttentionViT):
        #     return (m.searched_zeta if m.is_searched else m.zeta).sum()
        return (m.searched_zeta if m.is_searched else m.get_zeta()).sum()

    def get_remaining(self):
        """return the fraction of active zeta"""
        n_rem_attn, n_total_attn = 0, 1e-9
        n_rem_mlp, n_total_mlp = 0, 1e-9
        n_rem_conv, n_total_conv = 0, 1e-9

        for l_block_name, l_block in self.searchable_modules.items():
            # if isinstance(l_block, SparseMultiHeadAttentionViT):
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                if l_block.search_type == 'head':
                    n_rem_attn += self.n_remaining(l_block)
                    n_total_attn += l_block.head_num
                elif l_block.search_type == "embed":
                    n_rem_attn += self.n_remaining(l_block) * l_block.head_num
                    n_total_attn += l_block.num_gates * l_block.head_num
                elif l_block.search_type == "uniform":
                    n_rem_attn += self.n_remaining(l_block)
                    n_total_attn += l_block.num_gates * l_block.head_num

            elif isinstance(l_block, SparseMLP):
                n_rem_mlp += self.n_remaining(l_block)
                n_total_mlp += l_block.num_gates
            elif isinstance(l_block, SparseConvNormLayer):
                n_rem_conv += self.n_remaining(l_block)
                n_total_conv += l_block.num_gates

        return n_rem_attn / n_total_attn, n_rem_mlp / n_total_mlp, n_rem_conv / n_total_conv

    def get_sparsity_loss(self):
        """获取稀疏掩码SoftMask的所有加和，送入L1正则项中进行稀疏掩码学习。

        Returns:
            loss_attn, loss_mlp, loss_conv (torch.Tensor): atten/mlp/conv层的稀疏加权和。 
        """
        if self.device is None:
            self.device = next(self.parameters()).device

        loss_attn = torch.FloatTensor([]).to(self.device)
        loss_mlp = torch.FloatTensor([]).to(self.device)
        loss_conv = torch.FloatTensor([]).to(self.device)

        for l_block_name, l_block in self.searchable_modules.items():
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                zeta_attn = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn, torch.abs(zeta_attn.view(-1))])
                # loss_attn = torch.cat([loss_attn, torch.sigmoid(zeta_attn.view(-1))])
            elif isinstance(l_block, SparseMLP):
                zeta_mlp = l_block.get_zeta()
                loss_mlp = torch.cat([loss_mlp, torch.abs(zeta_mlp.view(-1))])
                # loss_mlp = torch.cat([loss_mlp, torch.sigmoid(zeta_mlp.view(-1))])
            elif isinstance(l_block, SparseConvNormLayer):
                zeta_conv = l_block.get_zeta()
                loss_conv = torch.cat([loss_conv, torch.abs(zeta_conv.view(-1))])
                # loss_conv = torch.cat([loss_conv, torch.sigmoid(zeta_conv.view(-1))])

        loss_attn  = torch.sum(loss_attn).to(self.device) if len(loss_attn) > 0 else torch.tensor(0).to(self.device)
        loss_mlp  = torch.sum(loss_mlp).to(self.device) if len(loss_mlp) > 0 else torch.tensor(0).to(self.device)
        loss_conv  = torch.sum(loss_conv).to(self.device) if len(loss_conv) > 0 else torch.tensor(0).to(self.device)

        return loss_attn, loss_mlp, loss_conv

    def give_zetas(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_conv = []
        for l_block_name, l_block in self.searchable_modules.items():
            zetas = l_block.get_zeta()
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                zetas_attn.append(zetas.cpu().detach().reshape(-1).numpy().tolist())
            elif isinstance(l_block, SparseMLP):
                zetas_mlp.append(zetas.cpu().detach().reshape(-1).numpy().tolist())
            elif isinstance(l_block, SparseConvNormLayer):
                zetas_conv.append(zetas.cpu().detach().reshape(-1).numpy().tolist())

        new_zetas_attn = []
        new_zetas_mlp = []
        new_zetas_conv = []
        _ = [new_zetas_attn.extend(z) for z in zetas_attn]
        _ = [new_zetas_mlp.extend(z) for z in zetas_mlp]
        _ = [new_zetas_conv.extend(z) for z in zetas_conv]

        return new_zetas_attn, new_zetas_mlp, new_zetas_conv

    def compress_macro(self, budget_attn=0.8, budget_mlp=0.8, budget_conv=0.8, min_gates_ratio=0.1):
        """compress the network to make zeta exactly 1 and 0."""
        if len(self.searchable_modules) < 1:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]

        thresh_attn, thresh_mlp, thresh_conv = self.calculate_search_threshold(budget_attn, budget_mlp, budget_conv)

        for l_block_name, l_block in self.searchable_modules.items():
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                l_block.compress(thresh_attn, min_gates_ratio)
            elif isinstance(l_block, SparseMLP):
                l_block.compress(thresh_mlp, min_gates_ratio)
            elif isinstance(l_block, SparseConvNormLayer):
                l_block.compress(thresh_conv, min_gates_ratio)

        return thresh_attn, thresh_mlp, thresh_conv
    
    def compress_micro(self, budget_attn=0.8, budget_mlp=0.8, budget_conv=0.8, min_gates_ratio=0.1, *args, **kwargs):
        if len(self.searchable_modules) < 1:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
            
        for l_block_name, l_block in self.searchable_modules.items():
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                thres = self.calculate_search_threshold_micro(l_block.get_zeta(), budget_attn)
                l_block.compress(thres, min_gates_ratio)
            elif isinstance(l_block, SparseMLP):
                thres = self.calculate_search_threshold_micro(l_block.get_zeta(), budget_mlp)
                l_block.compress(thres, min_gates_ratio)
            elif isinstance(l_block, SparseConvNormLayer):
                thres = self.calculate_search_threshold_micro(l_block.get_zeta(), budget_conv)
                l_block.compress(thres, min_gates_ratio)

    def flatten_group_zetas(self, zeta_list: list):
        zeta_list = [torch.squeeze(zeta) for zeta in zeta_list]
        flat_zetas = torch.concat(zeta_list, dim=0)
        return flat_zetas
    
    def get_block_type(self, l_block):
        if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
            return 'atten'
        elif isinstance(l_block, SparseMLP):
            return 'mlp'
        elif isinstance(l_block, SparseConvNormLayer):
            return 'conv'
        else:
            return 'other'

    def compress_group(self, budget_backbone=0.7, budget_encoder=0.7, budget_decoder=0.7, min_gates_ratio=0.1, *args, **kwargs):
        backbone_group = {'conv': [], 'atten': [], 'mlp': [], 'other': []}
        encoder_group = {'conv': [], 'atten': [], 'mlp': [], 'other': []}
        decoder_group = {'conv': [], 'atten': [], 'mlp': [], 'other': []}
        
        # get group
        for l_block_name, l_block in self.searchable_modules.items():
            spstr = l_block_name.split('.')
            if spstr[1] == 'backbone':
                backbone_group[self.get_block_type(l_block)].append(l_block)
            elif spstr[1] == 'encoder':
                encoder_group[self.get_block_type(l_block)].append(l_block)
            elif spstr[1] == 'decoder':
                decoder_group[self.get_block_type(l_block)].append(l_block)
        
        # compress block for each different block type in each group.
        for zip_group, zip_budget in zip([backbone_group, encoder_group, decoder_group], [budget_backbone, budget_encoder, budget_decoder]):
            for block_type, block_list in zip_group.items():
                if len(block_list) == 0:
                    continue
                zeta_list = [block.get_zeta() for block in block_list]
                flat_zeta = self.flatten_group_zetas(zeta_list)
                thres = self.calculate_search_threshold_micro(flat_zeta, zip_budget)
                _ = [l_block.compress(thres, min_gates_ratio) for l_block in block_list]

    def correct_require_grad(self, attn_w, mlp_w, conv_w):
        conditions = {
            SparseMultiHeadAttentionViT: attn_w,
            SparseMSDeformableAttention: attn_w,
            SparseMLP: mlp_w,
            SparseConvNormLayer: conv_w
        }
        for l_block_name, l_block in self.searchable_modules.items():
            if type(l_block) in conditions and conditions[type(l_block)] <= 0:
                l_block.zeta.requires_grad = False

    def decompress(self):
        for l_block_name, l_block in self.searchable_modules.items():
            l_block.decompress()

    def get_channels(self):
        active_channels_attn = []
        active_channels_mlp = []
        active_channels_conv = []
        for l_block_name, l_block in self.searchable_modules.items():
            if isinstance(l_block, (SparseMultiHeadAttentionViT, SparseMSDeformableAttention)):
                active_channels_attn.append(l_block.searched_zeta.numpy())
            elif isinstance(l_block, SparseMLP):
                active_channels_mlp.append(l_block.searched_zeta.numpy())
            elif isinstance(l_block, SparseConvNormLayer):
                active_channels_conv.append(l_block.searched_zeta.numpy())

        return np.squeeze(np.array(active_channels_attn)), np.squeeze(np.array(active_channels_mlp)), np.squeeze(np.array(active_channels_conv))

    def get_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        searched_params = total_params
        for l_block_name, l_block in self.searchable_modules.items():
            searched_params -= l_block.get_params_count()[0]
            searched_params += l_block.get_params_count()[1]
        return total_params, searched_params.item()

    def prune(self):
        backbone_out_ch_masks = []
        if isinstance(self.backbone, (SparsePResNet, )):
            self.backbone.prune_output_channel()
            backbone_out_ch_masks = self.backbone.get_output_channel_mask()

        pan_out_ch_masks = []
        if isinstance(self.encoder, (SparseRTDETREncoder, )):
            self.encoder.prune_input_channel(backbone_out_ch_masks)
            self.encoder.prune_output_channel()
            pan_out_ch_masks = self.encoder.get_output_channel_mask()
        
        if isinstance(self.decoder,(SparseRTDETRDecoder, )):
            self.decoder.prune_input_channel(pan_out_ch_masks)
            self.decoder.prune_output_channel()

        self.is_pruned = True


@register
class OriginRTDETR(nn.Module):
    __inject__ = [
        'backbone',
        'encoder',
        'decoder',
    ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
