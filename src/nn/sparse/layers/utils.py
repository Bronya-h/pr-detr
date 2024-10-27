from functools import reduce
import math
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.boxes import box_area

def get_activation(act: str, inplace: bool = False):
    '''get activation
    '''
    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    """
    将边界框坐标从中心点和宽高格式 (cx, cy, w, h) 转换为左上角和右下角格式 (x1, y1, x2, y2)。
    
    Args:
        x (torch.Tensor): 形状为 (N, ..., 4) 的张量，表示边界框的中心点坐标和宽高。
            其中，N 表示边界框的数量，... 表示可以有多余的维度。
    
    Returns:
        torch.Tensor: 形状为 (N, ..., 4) 的张量，表示边界框的左上角和右下角坐标。
    
    """
    return torch.stack([(x[..., 0] - 0.5 * x[..., 2]),
                        (x[..., 1] - 0.5 * x[..., 3]),
                        (x[..., 0] + 0.5 * x[..., 2]),
                        (x[..., 1] + 0.5 * x[..., 3])], dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def pad_tensors_to_same_size(*args, padmode='constant', value=0):
    """将多个张量pad到一样的大小。

    Args:
        args (list[torch.Tensor]): 输入张量列表.
        padmode (str): pad的模式，constant或者same.
        value (any): pad的值.

    Returns:
        padded_tensors (list[torch.Tensor]): pad后的张量列表.
    """
    max_height = max([x.size(2) for x in args])
    max_width = max([x.size(3) for x in args])

    padded_tensors = []
    for x in args:
        pad_height = max_height - x.size(2)
        pad_width = max_width - x.size(3)
        padded_tensor = F.pad(x, (0, pad_width, 0, pad_height), mode=padmode, value=value)
        padded_tensors.append(padded_tensor)

    return padded_tensors


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def multiply_list_elements(lst):
    return reduce(operator.mul, lst)
