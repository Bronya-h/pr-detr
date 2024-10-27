'''
by lyuwenyu
'''
import datetime
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
import tqdm
from detectron2.utils.analysis import (FlopCountAnalysis,
                                       activation_count_operators,
                                       parameter_count_table)
from fvcore.nn import flop_count_table  # can also try flop_count_str
from matplotlib import pyplot as plt
from thop import profile

import src.torch_pruning as tp
from src.core import BaseConfig
from src.data import get_coco_api_from_dataset
from src.misc import dist
from src.misc.events import EventStorage, TensorboardXWriter
from src.misc.logger import MetricLogger, SmoothedValue

from ..core.yaml_utils import create as yaml_create
from ..torch_pruning.pruner.function import MSDeformableAttentionPruner
from ..zoo.rtdetr.rtdetr_decoder import MSDeformableAttention
from .det_engine import bind_bbox_metric, evaluate, train_one_epoch
from .solver import BaseSolver


class DetSolver(BaseSolver):

    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        module = self.ema.module if self.ema else self.model
        state['model'] = dist.de_parallel(module).state_dict()
        state['date'] = datetime.datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state

    def fit(self, do_eval: bool = True):
        print("Start training")
        self.train()

        args = self.cfg

        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writers = [TensorboardXWriter(log_dir=tensorboard_dir)]

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params(Million): ', n_parameters / 1e6)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {
            'epoch': -1,
        }
        best_coco_map = 0.0

        start_time = time.time()
        with EventStorage(0) as self.storage:
            for epoch in range(self.last_epoch + 1, args.epoches):
                if dist.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(self.model,
                                              self.criterion,
                                              self.train_dataloader,
                                              self.optimizer,
                                              self.device,
                                              epoch,
                                              args.clip_max_norm,
                                              print_freq=args.log_step,
                                              ema=self.ema,
                                              scaler=self.scaler,
                                              max_epoches=args.epoches,
                                              writers=self.writers)

                self.lr_scheduler.step()

                module = self.ema.module if self.ema else self.model
                if do_eval:
                    test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir)

                test_stats.pop("FPS")

                bbox_metric_dict = bind_bbox_metric(test_stats['coco_eval_bbox'])
                self.storage.put_scalars(cur_iter=(epoch + 1) * len(self.train_dataloader), **bbox_metric_dict, smoothing_hint=False)

                # TODO
                for k in test_stats.keys():
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]
                print('best_stat: ', best_stat)
                cur_coco_map = bbox_metric_dict['AP']

                if self.output_dir:
                    if cur_coco_map > best_coco_map:
                        checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                        if (epoch + 1) % args.checkpoint_step == 0:
                            checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                        model_files = glob.glob(root_dir=self.output_dir, pathname="checkpoint*.pth")
                        model_files = [os.path.join(self.output_dir, f) for f in model_files]
                        model_mtime = [os.path.getmtime(f) for f in model_files]
                        bind_model_files = list(zip(model_mtime, model_files))
                        bind_model_files.sort(key=lambda x: x[0], reverse=False)
                        model_files = [x[1] for x in bind_model_files]
                        reserve_num = getattr(self.cfg, 'max_checkpoint_num', 5)
                        delete_model_files = []
                        n = len(model_files)
                        if n > reserve_num:
                            delete_model_files = model_files[:n - reserve_num]

                        for del_f in delete_model_files:
                            os.remove(del_f)

                        best_coco_map = cur_coco_map

                    for checkpoint_path in checkpoint_paths:
                        dist.save_on_master(self.state_dict(epoch), checkpoint_path)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

                if self.output_dir and dist.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

                for writer in self.writers:
                    writer.write()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        n_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print('number of params(Million): ', n_parameters / 1e6)

        with EventStorage(0) as self.storage:
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir)
            bbox_metric_dict = bind_bbox_metric(test_stats['coco_eval_bbox'])
            print(" ".join(["{:^8}" for k in bbox_metric_dict.keys()]))
            print(" ".join(["{:.6f}".format(v) for v in bbox_metric_dict.values()]))
            print("FPS: {:.3f}".format(1.0 / test_stats["FPS"]))

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        do_parameter(self.model)
        do_flop(self.val_dataloader, self.model, to_device=self.device)
        do_structure(self.model)

    def flops_params(self, h=640, w=640, c=3):
        self.model.eval()
        cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image = torch.randn(size=[1, c, h, w], device=cur_device, dtype=torch.float32)
        flops, params = profile(self.model, inputs=(image, ))
        print("FLOPs: {:.3f}G".format(flops / 1e9))
        print("PARAMS: {:.3f}M".format(params / 1e6))

    def flops_params_partial(self, eval_model=None, h=640, w=640, c=3):
        if eval_model is None:
            eval_model = self.model

        cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image = torch.randn(size=[1, c, h, w], device=cur_device, dtype=torch.float32)

        backbone_flops, backbone_params = profile(eval_model.backbone, inputs=(image, ))
        out = eval_model.backbone(image)

        encoder_flops, encoder_params = profile(eval_model.encoder, inputs=(out, ))
        out = eval_model.encoder(out)
        decoder_flops, decoder_params = profile(eval_model.decoder, inputs=(out, ))

        flops = backbone_flops + encoder_flops + decoder_flops
        params = backbone_params + encoder_params + decoder_params

        backbone_flops_prop = backbone_flops / flops
        backbone_params_prop = backbone_params / params

        encoder_flops_prop = encoder_flops / flops
        encoder_params_prop = encoder_params / params

        decoder_flops_prop = decoder_flops / flops
        decoder_params_prop = decoder_params / params

        print("total_FLOPs: {:.3f}G".format(flops / 1e9))
        print("total_PARAMS: {:.3f}M".format(params / 1e6))

        print("backbone_FLOPs: {:.3f}G, {:.3f}".format(backbone_flops / 1e9, backbone_flops_prop))
        print("backbone_PARAMS: {:.3f}M, {:.3f}".format(backbone_params / 1e6, backbone_params_prop))

        print("encoder_FLOPs: {:.3f}G, {:.3f}".format(encoder_flops / 1e9, encoder_flops_prop))
        print("encoder_PARAMS: {:.3f}M, {:.3f}".format(encoder_params / 1e6, encoder_params_prop))

        print("decoder_FLOPs: {:.3f}G, {:.3f}".format(decoder_flops / 1e9, decoder_flops_prop))
        print("decoder_PARAMS: {:.3f}M, {:.3f}".format(decoder_params / 1e6, decoder_params_prop))

        metrics = {}
        metrics['total_flops'] = flops
        metrics['total_params'] = params

        metrics['backbone_flops'] = backbone_flops
        metrics['backbone_params'] = backbone_params

        metrics['encoder_flops'] = encoder_flops
        metrics['encoder_params'] = encoder_params

        metrics['decoder_flops'] = decoder_flops
        metrics['decoder_params'] = decoder_params

        return metrics


class SparseDetSolver(DetSolver):

    def mask_select(self):
        thres_scope = self.cfg.yaml_cfg.get("thres_scope", None)
        if thres_scope is None or thres_scope == 'macro':
            print("INFO: macro searched scope.")
            thresh_attn, thresh_mlp, thresh_conv = self.model.compress_macro(self.cfg.yaml_cfg["budget_atten"], self.cfg.yaml_cfg["budget_mlp"], self.cfg.yaml_cfg["budget_conv"],
                                                                             self.cfg.yaml_cfg["min_gates_ratio"])
            print("threshold attention: {}".format(thresh_attn))
            print("threshold MLP: {}".format(thresh_mlp))
            print("threshold convolution: {}".format(thresh_conv))
        elif thres_scope == 'micro':
            print("INFO: micro searched scope.")
            self.model.compress_micro(self.cfg.yaml_cfg["budget_atten"], self.cfg.yaml_cfg["budget_mlp"], self.cfg.yaml_cfg["budget_conv"], self.cfg.yaml_cfg["min_gates_ratio"])
        elif thres_scope == 'group':
            self.model.compress_group(self.cfg.yaml_cfg["budget_backbone"], self.cfg.yaml_cfg["budget_encoder"], self.cfg.yaml_cfg["budget_decoder"], self.cfg.yaml_cfg["min_gates_ratio"])
        else:
            raise ValueError("'thres_scope' is not a correct string value.")

        remain_atten, remain_mlp, remain_conv = self.model.get_remaining()
        print("remain attention: {}".format(remain_atten))
        print("remain mlp: {}".format(remain_mlp))
        print("remain convolution: {}".format(remain_conv))

        if True and sys.gettrace() is None:
            masked_checkpoint_path = os.path.join(self.output_dir, "masked_checkpoint.pth")
            dist.save_on_master(self.state_dict(self.cfg.epoches), masked_checkpoint_path)

    def prune(self):
        """根据HardMask剪枝模型，考虑各个模块间的输入输出的耦合关系。

        Returns:
            prune_model 剪枝后的模型。
        """
        self.model.eval()
        self.model.prune()

        if True and sys.gettrace() is None:
            pruned_checkpoint_path = os.path.join(self.output_dir, "pruned_checkpoint.pth")
            dist.save_on_master(self.state_dict(self.cfg.epoches), pruned_checkpoint_path)

    def finetuning(self, pruned_model=None, do_eval=True, ft_epoches=5):
        """微调剪枝后的模型

        Arguments:
            pruned_model -- 剪枝后的模型

        Returns:
            微调的结果
        """
        if pruned_model is not None:
            self.model = pruned_model
            self.model.train()

        if self.ema is not None:
            self.ema = yaml_create('ema', model=self.model)

        args = self.cfg

        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writers = [TensorboardXWriter(log_dir=tensorboard_dir)]

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params(Million): ', n_parameters / 1e6)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {
            'epoch': -1,
        }
        best_coco_map = 0.0

        start_time = time.time()
        with EventStorage(0) as self.storage:
            for epoch in range(ft_epoches):
                if dist.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(self.model,
                                              self.criterion,
                                              self.train_dataloader,
                                              self.optimizer,
                                              self.device,
                                              epoch,
                                              args.clip_max_norm,
                                              print_freq=args.log_step,
                                              ema=self.ema,
                                              scaler=self.scaler,
                                              max_epoches=ft_epoches,
                                              writers=self.writers)

                self.lr_scheduler.step()
                module = self.ema.module if self.ema else self.model
                if do_eval:
                    test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir)

                test_stats.pop("FPS")

                bbox_metric_dict = bind_bbox_metric(test_stats['coco_eval_bbox'])
                self.storage.put_scalars(cur_iter=(epoch + 1) * len(self.train_dataloader), **bbox_metric_dict, smoothing_hint=False)

                for k in test_stats.keys():
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]

                print('best_stat: ', best_stat)
                cur_coco_map = bbox_metric_dict['AP']

                if self.output_dir:
                    if cur_coco_map > best_coco_map:
                        checkpoint_paths = [os.path.join(self.output_dir, 'pruned_finetune.pth')]
                        if (epoch + 1) % args.checkpoint_step == 0:
                            checkpoint_paths.append(os.path.join(self.output_dir, f'pruned_finetune{epoch:04}.pth'))
                        model_files = glob.glob(root_dir=self.output_dir, pathname="pruned_finetune*.pth")
                        model_files = [os.path.join(self.output_dir, f) for f in model_files]
                        model_mtime = [os.path.getmtime(f) for f in model_files]
                        bind_model_files = list(zip(model_mtime, model_files))
                        bind_model_files.sort(key=lambda x: x[0], reverse=False)
                        model_files = [x[1] for x in bind_model_files]
                        reserve_num = getattr(self.cfg, 'max_finetune_num', 5)
                        delete_model_files = []
                        n = len(model_files)
                        if n > reserve_num:
                            delete_model_files = model_files[:n - reserve_num]

                        for del_f in delete_model_files:
                            os.remove(del_f)

                        best_coco_map = cur_coco_map

                    for checkpoint_path in checkpoint_paths:
                        dist.save_on_master(self.state_dict(epoch), checkpoint_path)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

                if self.output_dir and dist.is_main_process():
                    with open(os.path.join(self.output_dir, "pruned_finetune_log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    # for evaluation logs
                    if coco_evaluator is not None:
                        coco_eval_dir = os.path.join(self.output_dir, 'pruned_finetune_eval')
                        os.makedirs(coco_eval_dir, exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['pruned_finetune_latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval, os.path.join(coco_eval_dir, name))

                for writer in self.writers:
                    writer.write()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Finetuning time {}'.format(total_time_str))

        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        with EventStorage(0) as self.storage:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir)
            bbox_metric_dict = bind_bbox_metric(test_stats['coco_eval_bbox'])
            print(",".join(list(bbox_metric_dict.keys())))
            print(" ".join(["{:.6f}".format(x) for x in list(bbox_metric_dict.values())]))
            print("FPS: {:.6f}".format(1.0 / test_stats["FPS"]))

    def prune_pipeline(self):
        self.train()
        print("------Mask Select------")
        self.mask_select()
        src_metrics = self.flops_params_partial(self.model, h=self.cfg.eval_shape[0], w=self.cfg.eval_shape[1], c=self.cfg.eval_shape[2])

        print("------Pruned Model------")
        self.prune()
        prune_metrics = self.flops_params_partial(self.model, h=self.cfg.eval_shape[0], w=self.cfg.eval_shape[1], c=self.cfg.eval_shape[2])
        print("********compress ratio********")
        for k in src_metrics.keys():
            reduce_ratio = (src_metrics[k] - prune_metrics[k]) / src_metrics[k]
            print('reduce_ratio {} - {:.3f}'.format(k, reduce_ratio))

        print("******************************")

        print("------Fine Tuning Model------")
        self.finetuning(ft_epoches=self.cfg.yaml_cfg['finetune_epoches'])

    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        module = self.ema.module if self.ema else self.model
        module.train()  # 保存train模式的模型
        state['model'] = dist.de_parallel(module).state_dict()
        state['date'] = datetime.datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state

    def load_state_dict(self, state, strict=True):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'], strict=strict)
            else:
                try:
                    self.model.load_state_dict(state['model'], strict=strict)
                except:
                    self.model.load_state_dict(state['model'], strict=False)

            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'], strict=strict)
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    def resume(self, path, prune_state=False, **kwargs):
        '''load resume
        '''
        state = torch.load(path, map_location='cpu')
        if not prune_state:
            self.load_state_dict(state, strict=False)
        else:
            self.load_state_dict_for_pruned_model(state, strict=False)
            visit_layer_name = set()
            for n, m in self.model.named_modules():
                if n not in visit_layer_name and hasattr(m, "zeta"):
                    m.is_pruned = True
                    visit_layer_name.add(n)
            if self.ema is not None:
                visit_layer_name = set()
                for n, m in self.ema.module.named_modules():
                    if n not in visit_layer_name and hasattr(m, "zeta"):
                        m.is_pruned = True
                        visit_layer_name.add(n)
                
        
    def load_state_dict_for_pruned_model(self, state, strict=False):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                replace_parameters_with_different_shapes(self.model.module, state['model'])
            else:
                replace_parameters_with_different_shapes(self.model, state['model'])
            
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            replace_parameters_with_different_shapes(self.ema.module, state['ema']['module'])
            self.updates = state['ema']['updates']
            self.warmups = state['ema']['warmups']
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

class PruneDetSolver(DetSolver):
    """A pruning after train Detection solver.
    Pruning a model after a full train, and then fine tuning model."""

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)

        self.full_train_before_prune = False  # whether to full train a model before pruning.
        """pruning config"""
        self.pruning_imp = tp.importance.MagnitudeImportance(p=1)
        self.pruning_target_rate = 0.5  # The proportion of structural parameters pruned after pruning_iter_steps iterations
        self.max_map_drop = 0.2

        self.ignored_layer_inputs = []
        self.ignored_layer_inputs_names = []
        self.ignored_layer_outputs = []
        self.ignored_layer_outputs_names = []

        self.unwrapped_parameters = []
        self.customized_pruners = {MSDeformableAttention: MSDeformableAttentionPruner()}

        self.pruning_iter_steps = 16  # progressive pruning
        # self.example_inputs = torch.randn(size=[1, 3, 640, 640]).to(self.device)
        self.example_inputs = None
        """ fine-tuning config """
        self.ft_batch_size = 8
        self.ft_epoches = 1
        self.ft_max_iters = None
        """state reserve variants."""
        self.macs_list = []  # 计算量FLOPs的变化列表
        self.nparams_list = []  #
        self.ft_map_list = []
        self.pruned_map_list = []

    def fit(self):
        """execute pruning model.
        """
        '''full train one model.'''
        self.train()
        if self.full_train_before_prune:
            super().fit()  # full train a detection model before pruning.
        '''get first pruning estimate.'''
        if self.example_inputs is None:
            self.example_inputs = torch.randn(size=[1, 3, 640, 640]).to(self.device)
        self.base_macs, self.base_nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        # if sys.gettrace():
        #     test_stats = {'coco_eval_bbox': [0.0] * 12}
        # else:
        #     test_stats, _ = self.val()

        test_stats = {'coco_eval_bbox': [0.0] * 12}
        init_map = test_stats['coco_eval_bbox'][0]
        self.macs_list.append(self.base_macs)
        self.nparams_list.append(100)
        self.ft_map_list.append(init_map)
        self.pruned_map_list.append(init_map)
        print(f"Before Pruning: MACs={self.base_macs / 1e9: .5f} G, #Params={self.base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")
        '''get pruning parameters adaptive to model.'''
        module = self.ema.module if self.ema else self.model
        for cur_layer_name, cur_layer in module.named_modules():
            # TODO: analyse which layer should be ignored in RT-DETR.
            if isinstance(cur_layer, torch.nn.Linear):
                if (cur_layer.out_features == module.decoder.num_classes or \
                    cur_layer.out_features == module.decoder.num_coordinates):
                    self.ignored_layer_outputs.append(cur_layer)
                    self.ignored_layer_outputs_names.append(cur_layer_name)

                if (cur_layer.in_features == module.decoder.num_classes or \
                    cur_layer.in_features == module.decoder.num_coordinates):
                    self.ignored_layer_inputs.append(cur_layer)
                    self.ignored_layer_inputs_names.append(cur_layer_name)

            # elif 'cross_attn' in cur_layer_name:
            #     # TODO: custom Pruner for DeformableTransformer
            #     self.ignored_layers_names.append(cur_layer_name)
            #     self.ignored_layers.append(cur_layer)

        print("ignored_layers: ")
        print("ignored_layer_inputs:")
        print(*[">>> " + self.ignored_layer_inputs_names[i] + "---" + str(self.ignored_layer_inputs[i]) for i in range(len(self.ignored_layer_inputs_names))], sep='\n')
        print("ignored_layer_outputs:")
        print(*[">>> " + self.ignored_layer_outputs_names[i] + "---" + str(self.ignored_layer_outputs[i]) for i in range(len(self.ignored_layer_outputs_names))], sep='\n')

        self.pruning_ratio = 1 - math.pow((1 - self.pruning_target_rate), 1 / self.pruning_iter_steps)
        '''do pruning model.'''
        if self.ft_max_iters is None or self.ft_max_iters < 0:
            self.ft_max_iters = int(len(self.train_dataloader) * self.ft_epoches)

        for pi in range(self.pruning_iter_steps):
            self.model.train()
            self.pruner = tp.pruner.MagnitudePruner(
                self.model,
                self.example_inputs,
                importance=self.pruning_imp,
                iterative_steps=1,
                pruning_ratio=self.pruning_ratio,
                # ignored_layers=self.ignored_layers,
                unwrapped_parameters=self.unwrapped_parameters,
                global_pruning=True,
                # global_pruning=False,
                head_pruning_ratio=0.25,
                customized_pruners=self.customized_pruners,
                ignored_layer_outputs=self.ignored_layer_outputs,
                ignored_layer_inputs=self.ignored_layer_inputs)

            tp.utils.draw_dependency_graph(self.pruner.DG, save_as=os.path.join(self.output_dir, 'draw_dep_graph.png'), title=None)
            tp.utils.draw_groups(self.pruner.DG, save_as=os.path.join(self.output_dir, 'draw_groups.png'), title=None)
            tp.utils.draw_computational_graph(self.pruner.DG, save_as=os.path.join(self.output_dir, 'draw_comp_graph.png'), title=None)

            self.pruner.step()

            # pre fine-tuning validation
            test_stats, _ = self.val()
            pruned_map = test_stats['coco_eval_bbox'][0]
            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
            current_speed_up = float(self.macs_list[0]) / pruned_macs
            print(f"After pruning iter {pi + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
                  f"mAP={pruned_map}, speed up={current_speed_up}")

            # fine-tuning
            self.fine_tuning(pi, self.pruning_iter_steps, self.cfg.clip_max_norm, print_freq=self.cfg.log_step, ema=self.ema, scaler=self.scaler)
            # post fine-tuning validation
            test_stats, _ = self.val()
            current_map = test_stats['coco_eval_bbox'][0]
            print(f"Prune step - [{pi}/{self.pruning_iter_steps}] fine tuning mAP={current_map}")

            self.macs_list.append(pruned_macs)
            self.nparams_list.append(pruned_nparams / self.base_nparams * 100)
            self.pruned_map_list.append(pruned_map)
            self.ft_map_list.append(current_map)

            # remove pruner after single iteration
            del pruner

            self.save_pruning_performance_graph(self.nparams_list, self.ft_map_list, self.macs_list, self.pruned_map_list, pi)

            if init_map - current_map > self.max_map_drop:
                print("Pruning early stop")
                break

    def fine_tuning(self, prune_step, max_prune_steps=None, max_norm: float = 0, **kwargs):
        if max_prune_steps is None:
            max_prune_steps = prune_step + 1

        print("Start Fine tuning Step: [{}/{}].".format(prune_step, max_prune_steps))

        self.train()
        args = self.cfg

        if not self.writers:
            tensorboard_dir = os.path.join(args.output_dir, 'tensorboard', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writers = [TensorboardXWriter(log_dir=tensorboard_dir)]

        steps_per_epoch = len(self.train_dataloader)

        if not self.model.training:
            self.model.train()
        if not self.criterion.training:
            self.criterion.train()

        max_epoches = math.ceil(self.ft_max_iters / steps_per_epoch)
        metric_logger = MetricLogger(delimiter="  ", max_epoches=max_epoches)
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        print_freq = kwargs.get('print_freq', 10)
        ema = kwargs.get('ema', None)
        scaler = kwargs.get('scaler', None)

        start_time = time.time()

        with EventStorage(0) as self.storage:
            last_epoch = -1
            step = 0
            epoch = 0
            header = 'Epoch: [{:>2d}/{:>2d}]'.format(epoch, max_epoches)
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            for samples, targets in metric_logger.log_every(self.train_dataloader, print_freq, header, cur_epoch=epoch, max_epoches=max_epoches, cycle_dataloader=True):
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                if scaler is not None:
                    with torch.autocast(device_type=str(self.device), cache_enabled=True):
                        outputs = self.model(samples, targets)

                    with torch.autocast(device_type=str(self.device), enabled=False):
                        loss_dict = self.criterion(outputs, targets)

                    loss = sum(loss_dict.values())
                    scaler.scale(loss).backward()

                    if max_norm > 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                else:
                    outputs = self.model(samples, targets)
                    loss_dict = self.criterion(outputs, targets)

                    loss = sum(loss_dict.values())
                    self.optimizer.zero_grad()
                    loss.backward()

                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    self.optimizer.step()

                # ema
                if ema is not None:
                    ema.update(self.model)

                loss_dict_reduced = dist.reduce_dict(loss_dict)
                loss_value = sum(loss_dict_reduced.values())

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                metric_logger.update(loss=loss_value, **loss_dict_reduced)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                for writer in self.writers:
                    writer.write()

                step += 1
                if step >= self.ft_max_iters:
                    break  # exit fine tuning.

                epoch = step // steps_per_epoch
                if last_epoch < epoch:
                    if dist.is_dist_available_and_initialized():
                        self.train_dataloader.sampler.set_epoch(epoch)

                    self.lr_scheduler.step()
                    # gather the stats from all processes
                    metric_logger.synchronize_between_processes()
                    print("Averaged stats:", metric_logger)
                    header = 'Epoch: [{:>2d}/{:>2d}]'.format(epoch, max_epoches)

                    last_epoch = epoch

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Fine Tuning time {}'.format(total_time_str))

    def val(self, output_path=None):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        module = self.ema.module if self.ema else self.model
        with EventStorage(0) as self.storage:
            base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, output_path)
        return test_stats, coco_evaluator

    def save_pruning_performance_graph(self, x, y1, y2, y3, prune_step=0):
        """
        Draw performance change graph
        Parameters
        ----------
        x : List
            Parameter numbers of all pruning steps
        y1 : List
            mAPs after fine-tuning of all pruning steps
        y2 : List
            MACs of all pruning steps
        y3 : List
            mAPs after pruning (not fine-tuned) of all pruning steps

        Returns
        -------

        """
        try:
            plt.style.use("ggplot")
        except:
            pass

        x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
        y2_ratio = y2 / y2[0]

        # create the figure and the axis object
        fig, ax = plt.subplots(figsize=(8, 6))

        # plot the pruned mAP and recovered mAP
        ax.set_xlabel('Pruning Ratio')
        ax.set_ylabel('mAP')
        ax.plot(x, y1, label='recovered mAP')
        ax.scatter(x, y1)
        ax.plot(x, y3, color='tab:gray', label='pruned mAP')
        ax.scatter(x, y3, color='tab:gray')

        # create a second axis that shares the same x-axis
        ax2 = ax.twinx()

        # plot the second set of data
        ax2.set_ylabel('MACs')
        ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
        ax2.scatter(x, y2_ratio, color='tab:orange')

        # add a legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        ax.set_xlim(105, -5)
        ax.set_ylim(0, max(y1) + 0.05)
        ax2.set_ylim(0.05, 1.05)

        # calculate the highest and lowest points for each set of data
        max_y1_idx = np.argmax(y1)
        min_y1_idx = np.argmin(y1)
        max_y2_idx = np.argmax(y2)
        min_y2_idx = np.argmin(y2)
        max_y1 = y1[max_y1_idx]
        min_y1 = y1[min_y1_idx]
        max_y2 = y2_ratio[max_y2_idx]
        min_y2 = y2_ratio[min_y2_idx]

        # add text for the highest and lowest values near the points
        ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
        ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
        ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
        ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

        plt.title('Comparison of mAP and MACs with Pruning Ratio')
        save_path = os.path.join(self.output_dir, "{:>2d}_PruningPerfChange.png".format(prune_step))
        plt.savefig(save_path, dpi=600)


def do_flop(data_loader, model, to_device='cuda', num_inputs=10):
    model.eval()
    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(num_inputs), data_loader):  # noqa
        samples, targets = data
        samples = samples.to(to_device)
        targets = [{k: v.to(to_device) for k, v in t.items()} for t in targets]
        
        # data.to(model.device)
        flops = FlopCountAnalysis(model, samples)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    print("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    print("Average GFlops for each type of operators:\n" + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()]))
    print("Total GFlops: {:.3f}±{:.3f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9))


def do_activation(data_loader, model, to_device='cuda', num_inputs=10):
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(num_inputs), data_loader):  # noqa
        samples, targets = data
        samples = samples.to(to_device)
        targets = [{k: v.to(to_device) for k, v in t.items()} for t in targets]

        count = activation_count_operators(model, [{'image':samples,},])
        counts += count
        total_activations.append(sum(count.values()))

    print("(Million) Activations for Each Type of Operators:\n" + str([(k, v / idx) for k, v in counts.items()]))
    print("Total (Million) Activations: {}±{}".format(np.mean(total_activations), np.std(total_activations)))


def do_parameter(model, ):
    model.eval()

    print("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(model):
    # model = instantiate(cfg.model)
    model.eval()

    print("Model Structure:\n" + str(model))


# def replace_state_dict_params(model_state_dict, load_state_dict):
#     for load_param_name, load_param in load_state_dict.items():
#         if load_param_name in model_state_dict:
#             # 直接赋值，不进行数据复制
#             if hasattr(load_param, 'data'):
#                 model_state_dict[load_param_name].data = load_param.data
#             else:
#                 model_state_dict[load_param_name] = load_param


def replace_state_dict_params(model, load_state_dict):
    for load_param_name, load_param in load_state_dict.items():
        if load_param_name in model.state_dict():
            # 直接赋值，不进行数据复制
            if hasattr(load_param, 'data'):
                model.state_dict()[load_param_name].data = load_param.data
            else:
                model.state_dict()[load_param_name] = load_param
                
def replace_parameters(model, origin_model, load_state):
    """
    递归地替换模型中的参数，如果参数形状不同则跳过。
    
    :param model: 要更新的模型
    :param origin_model: 原始模型，用于获取参数形状
    :param load_state: 加载的状态字典，包含新的参数
    """
    for (name, param), (origin_name, origin_param) in zip(model.named_parameters(), origin_model.named_parameters()):
        if name in load_state and param.shape != load_state[name].shape:
            print(f"Shape mismatch for parameter {name}, skipping...")
            continue
        if name in load_state:
            param.data = load_state[name].data

    for child_model, origin_child_model in zip(model.children(), origin_model.children()):
        # 递归地对子模块进行参数替换
        replace_parameters(child_model, origin_child_model, load_state)
        
def replace_parameters_with_different_shapes(model, load_state):
    """
    递归地替换模型中的参数，即使参数形状不同。

    :param model: 要更新的模型
    :param load_state: 加载的状态字典，包含新的参数
    """
    # 遍历模型的所有参数
    model_state = model.state_dict()
    for name, param in model.named_parameters():
        if name in load_state:
            new_param = load_state[name]
            model_param = model_state[name]
            model_param_device = model_param.device
            new_param = new_param.to(model_param_device)
            new_param = nn.Parameter(new_param)
            recursive_set_parameter(model, name, new_param)

    # 递归处理子模块
    for child in model.children():
        replace_parameters_with_different_shapes(child, load_state)


def replace_normlayer_attribution(target_norm_layer, origin_norm_layer):
    pass


def recursive_set_parameter(father_layer, layer_name, new_param):
    """
    递归地遍历到最后一层基本网络结构的层，设置这一层的参数。
    :param father_layer: 父层
    :param layer_name: 子层的名称
    :param new_param: 新层
    """
    # 获取子层
    if '.' in layer_name:
        child_name, grandchilds = layer_name.split('.', 1)
        child_layer = getattr(father_layer, child_name)
        if child_layer is None:
            print(f"Layer {child_name} not found in model, skipping...")
        recursive_set_parameter(child_layer, grandchilds, new_param)
    else:
        if isinstance(father_layer, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            pass
            # replace_normlayer_attribution(father_layer, new_param)
        setattr(father_layer, layer_name, new_param)