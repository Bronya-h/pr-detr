
import math
import sys
import time
from typing import Iterable

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from ..nn.sparse.criterion.distill_loss import SearchingDistillationLoss


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    max_epoches: int = 0,
                    writers: list = [],
                    **kwargs):
    model.train()
    criterion.train()
    if max_epoches is None or max_epoches <= 0:
        max_epoches = epoch + 1

    metric_logger = MetricLogger(delimiter="  ", max_epoches=max_epoches)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{:>2d}/{:>2d}]'.format(epoch, max_epoches)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, cur_epoch=epoch, max_epoches=max_epoches):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets) if not isinstance(criterion, SearchingDistillationLoss) else criterion(outputs, targets, model)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets) if not isinstance(criterion, SearchingDistillationLoss) else criterion(outputs, targets, model)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        for writer in writers:
            writer.write()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             postprocessors,
             data_loader: Iterable,
             base_ds,
             device: torch.device,
             output_dir: str = None,
             cur_epoch: int = 0,
             max_epoches: int = 1,
             **kwargs):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", max_epoches=max_epoches)
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test: [{}]'.format(cur_epoch)

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )
    start_time = None
    val_seconds = []
    for samples, targets in metric_logger.log_every(data_loader, len(data_loader) // 12, header, cur_epoch, max_epoches=cur_epoch + 1):
        start_time = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)


        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        end_time = time.time()
        val_seconds.append(end_time - start_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()


    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    stats['FPS'] = sum(val_seconds) / len(val_seconds)

    return stats, coco_evaluator


def bind_bbox_metric(coco_eval_stats: list):
    """将CocoEvaluator放回的stats列表添加指标名称，放回字典。

    Args:
        coco_eval_stats (list): COCO评估结果

    Returns:
        bbox_metric (dict): coco评估结果加上key的字典。
    """
    bbox_metric_names = ['AP', 'AP-50', 'AP-75', 'AP-small', 'AP-medium', 'AP-large', 'AR-1', 'AR-10', 'AR-100', 'AR-small', 'AR-medium', 'AR-large']
    bbox_metric = zip(bbox_metric_names, coco_eval_stats)
    bbox_metric = dict(bbox_metric)
    return bbox_metric
