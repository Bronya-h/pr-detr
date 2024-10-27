# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch
from torch import nn


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


class ToMe(nn.Module):

    def __init__(self, token_remove_ratio: float = 0.5):
        super().__init__()
        self.token_remove_ratio = token_remove_ratio  # only affect half of the number of tokens, because it is .

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        atten_mask: torch.Tensor,
        ref_points: torch.Tensor,
        dn_target_mask: torch.Tensor,
        **kwargs,
    ):
        bs, num_tokens, hdim = target.shape
        rnum = int(num_tokens // 2 * self.token_remove_ratio)
        # 1. Get the correlation scores
        unm_idx, src_idx, dst_idx = self.get_corr_scores(target, rnum)
        # 2. Merge the target
        target = self.merge_target(target, unm_idx, src_idx, dst_idx, rnum)
        # 3. Update reference points
        reference_points = self.gather_but_not_merged(reference_points.squeeze(dim=-2), unm_idx, src_idx, dst_idx, rnum)
        reference_points = reference_points.unsqueeze(dim=-2)
        # 4. Update attention mask
        atten_mask = atten_mask.unsqueeze(dim=0).expand(bs, -1, -1)
        atten_mask = self.gather_but_not_merged(atten_mask, unm_idx, src_idx, dst_idx, rnum)
        atten_mask = atten_mask.transpose(-1, -2)
        atten_mask = self.gather_but_not_merged(atten_mask, unm_idx, src_idx, dst_idx, rnum)
        atten_mask = atten_mask[0].squeeze(dim=0)
        # 5. Update ref_points
        if ref_points is not None:
            ref_points = self.gather_but_not_merged(ref_points, unm_idx, src_idx, dst_idx, rnum)
            ref_points = ref_points.unsqueeze(dim=-2)

        # 6. Update dn_target_mask
        if dn_target_mask is not None:
            dn_target_mask = dn_target_mask.expand(bs, num_tokens)[..., None]
            dn_target_mask = self.gather_but_not_merged(dn_target_mask, unm_idx, src_idx, dst_idx, rnum)
            dn_target_mask = dn_target_mask[0].squeeze(-1)

        return target, reference_points, atten_mask, ref_points, dn_target_mask

        return target, reference_points, atten_mask, ref_points

    def get_corr_scores(self, target: torch.Tensor, rnum: int, **kwargs):
        """
        将目标张量隔元素划分成两个集合，计算两个节点之间的相似度，并返回未合并节点和合并节点的索引。
        
        Args:
            target (torch.Tensor): 目标张量，其形状应为 (B, N, C)，其中 B 表示批次大小，N 表示节点数量，C 表示特征维度。
            rnum (int): 需要合并的节点数量。
            **kwargs: 其他可选参数，但在此函数中不使用。
        
        Returns:
            tuple: 包含三个元素的元组，分别为：
            - unm_idx (torch.Tensor): 未合并节点的索引，其形状为 (B, (N - rnum), 1)。
            - src_idx (torch.Tensor): 合并节点的索引，其形状为 (B, rnum, 1)。
            - dst_idx (torch.Tensor): 与合并节点相似的其他节点的索引，其形状为 (B, rnum, 1)。
        """
        with torch.no_grad():
            target = target / target.norm(dim=-1, keepdim=True)  # 先进行归一化处理
            a, b = target[..., ::2, :], target[..., 1::2, :]  # 间隔元素获取两个集合
            scores = a @ b.transpose(-1, -2)  # 矩阵乘法获取相似度矩阵

            node_max, node_idx = scores.max(dim=-1)  # 每个节点获取相似度最大的其他节点的相似度值和下标
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # 对节点列表进行相似度值降序排序

            unm_idx = edge_idx[..., rnum:, :]  # Unmerged Tokens 相似度靠后的节点不进行合并
            src_idx = edge_idx[..., :rnum, :]  # Merged Tokens 相似度靠前的节点进行合并
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # 获得跟合并节点相似的其他节点的下标

        return unm_idx, src_idx, dst_idx

    def merge_target(self, target: torch.Tensor, unm_idx: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, rnum: int, mode="mean", **kwargs):
        src, dst = target[..., ::2, :], target[..., 1::2, :]  # 与计算相关性举证相同的隔元素划分两个集合
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - rnum, c))  # 抽取未合并节点值
        src = src.gather(dim=-2, index=src_idx.expand(n, rnum, c))  # 抽取要合并的节点值
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, rnum, c), src, reduce=mode)  # 将要合并的节点值与相似的其他节点进行合并
        return torch.cat([unm, dst], dim=1)

    def gather_but_not_merged(self, target: torch.Tensor, unm_idx: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, rnum: int, **kwargs) -> torch.Tensor:
        """
        按照merge_target相同的gather顺序取出unm和dst, 但是不对dst进行scatter_reduce操作。
        Args:
            target (torch.Tensor): 目标张量，其形状应为 (B, N, C)，其中 B 表示批次大小，N 表示节点数量，C 表示特征维度。
            unm_idx (torch.Tensor): 未合并节点的索引，其形状为 (B, (N - rnum), 1)。
        Returns:
            out (torch.Tensor): 未合并节点与合并节点的组合张量，其形状为 (B, N, C)。
        """
        src, dst = target[..., ::2, :], target[..., 1::2, :]  # 与计算相关性举证相同的隔元素划分两个集合
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - rnum, c))  # 抽取未合并节点值
        return torch.cat([unm, dst], dim=1)

    def unmerge_target(self, target: torch.Tensor, unm_idx: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor, rnum: int) -> torch.Tensor:
        """merge 操作的逆操作，将合并的节点拆分回未合并节点。

        Arguments:
            unm_idx (torch.Tensor): 未合并节点的索引，其形状为 (B, (N - rnum), 1)。
            src_idx (torch.Tensor): 合并节点的索引，其形状为 (B, rnum, 1)。
            dst_idx (torch.Tensor): 与合并节点相似的其他节点的索引，其形状为 (B, rnum, 1)。

        Returns:
            out (torch.Tensor): 未合并节点与合并节点的组合张量，其形状为 (B, N, C)。
        """
        unm_len = unm_idx.shape[1]
        unm, dst = target[..., :unm_len, :], target[..., unm_len:, :]  # 将目标张量逆concat操作
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, rnum, c))  # 抽取已经合并的节点值

        out = torch.zeros(n, target.shape[1], c, device=target.device, dtype=target.dtype)  # 创建一个和未操作前同样大小的张量

        out[..., 1::2, :] = dst  # 第1个元素起的隔元素值置为已经合并的节点值
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)  # 不合并节点的scatter逆操作
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, rnum, c), src=src)  # 抽取的合并节点值的逆操作

        return out


def kth_bipartite_soft_matching(metric: torch.Tensor, k: int) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, :(k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
