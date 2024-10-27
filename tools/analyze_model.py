# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # 确保导入的是本地库

from detectron2.engine.defaults import _try_get_key
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str

from detectron2.engine import default_argument_parser
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

logger = None


def setuplog(args):
    output_dir = _try_get_key(args, "output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    global logger
    logger = setup_logger(output_dir, distributed_rank=rank)
    return cfg


def do_flop(data_loader, model, num_inputs=10):
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(num_inputs), data_loader):  # noqa
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())
    
    global logger

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )


def do_activation(data_loader, model):
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))

    global logger
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(model, ):
    model.eval()
    global logger
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(model):
    # model = instantiate(cfg.model)
    model.eval()
    
    global logger
    logger.info("Model Structure:\n" + str(model))


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
        Examples:
        To show parameters of a model:
        $ python tools/analyze_model.py --tasks parameter \\
            --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py
        Flops and activations are data-dependent, therefore inputs and model weights
        are needed to count them:
        $ python tools/analyze_model.py --num-inputs 100 --tasks flop \\
            --config-file projects/dab_detr/configs/dab_detr_r50_50ep.py \\
            train.init_checkpoint=/path/to/model.pkl
        """
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure", "all"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=10,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1
    
    if args.tasks == ["all"]:
        args.tasks = ["parameter", "flop", "activation", "structure"]

    cfg = setuplog(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
        }[task](cfg)
