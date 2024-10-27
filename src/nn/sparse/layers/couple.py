import copy
import torch
import torch.nn as nn

from .mlp import LinearNormAct, SparseMLP, prune_linear_layer
from .conv import prune_normalization_layer, SparseConvNormLayer
from .attention import SparseMultiHeadAttentionViT, SparseMSDeformableAttention
from .utils import get_activation, multiply_list_elements


class SparseCouple(nn.Module):

    def __init__(self, last_layers: list[nn.Module], next_layers: list[nn.Module], zeta_shapes: list[int] = None, num_gates: int = None) -> None:
        super().__init__()
        self.last_layers = last_layers
        self.next_layers = next_layers

        self.num_gates = num_gates
        self.zeta_shapes = zeta_shapes
        if self.num_gates is None:
            self.num_gates = max(self.zeta_shapes)
        
        self.zeta = nn.Parameter(torch.ones(self.zeta_shapes))
        self.searched_zeta = torch.ones_like(self.zeta)

        self.is_searched = False
        self.is_pruned = False

    def forward(self, x):
        if not self.is_pruned:
            z = self.searched_zeta if self.is_searched else self.get_zeta()
            x = x * z
        
        return x

    def get_zeta(self):
        return self.zeta

    def compress(self, threshold, min_gates_ratio=0.1):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta() >= threshold).float()

        activate_gates = torch.sum(self.searched_zeta)
        if activate_gates / self.num_gates < min_gates_ratio:
            thres_rank = min(self.num_gates - 1, max(0, int((1 - min_gates_ratio) * self.num_gates)))
            act_thres = torch.sort(torch.squeeze(self.get_zeta())).values[thres_rank]
            self.searched_zeta = (self.get_zeta() >= act_thres).float()

        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def prune_last_layer_channel(self):
        """ last_layers prune """
        zeta = self.searched_zeta
        ch_mask = torch.squeeze(zeta)
        for i in range(len(self.last_layers)):
            if isinstance(self.last_layers[i], nn.Linear):
                prune_linear_layer(self.last_layers[i], ch_mask, direction='output')
            elif isinstance(self.last_layers[i], LinearNormAct):
                self.last_layers[i].prune_output_channel(ch_mask)
            else:
                raise ValueError("prune last layer for '{}' Module is not completed in this function.".format(type(self.last_layers[i])))

    def prune_next_layer_channel(self):
        zeta = self.searched_zeta
        mask = torch.squeeze(zeta)
        for i in range(len(self.next_layers)):
            if isinstance(self.next_layers[i], nn.Linear):
                prune_linear_layer(self.next_layers[i], mask, direction='input')
            elif isinstance(self.next_layers[i], LinearNormAct):
                self.next_layers[i].prune_input_channel(mask)
            elif isinstance(self.next_layers[i], SparseMLP):
                self.next_layers[i].prune_input_channel(mask)
            elif isinstance(self.next_layers[i], SparseConvNormLayer):
                self.next_layers[i].prune_input_channel(mask)
            elif isinstance(self.next_layers[i], SparseMultiHeadAttentionViT):
                self.next_layers[i].prune_input_channel(mask)
            else:
                raise ValueError("prune last layer for '{}' Module is not completed in this function.".format(type(self.next_layers[i])))

        self.is_pruned = True

        self.zeta = None
        self.searched_zeta = None
