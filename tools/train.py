import os
import signal
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import torch


if sys.gettrace() is not None:
    torch.autograd.set_detect_anomaly(True)

def cleanup_resources(signum, frame):
    torch.cuda.empty_cache()
    exit(0)

if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, cleanup_resources)
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, cleanup_resources)
if hasattr(signal, 'SIGQUIT'):
    signal.signal(signal.SIGQUIT, cleanup_resources)

os.environ['PYTHONUNBUFFERED'] = '1'

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def contains_exclude_words(file_path, exclude_words) -> bool:
    for word in exclude_words:
        if word in file_path:
            return True
    return False


def load_latest_model(dir_path, exclude_words=[]):
    model_files = glob.glob(root_dir=dir_path, pathname="*.pth")
    if exclude_words:
        model_files = [f for f in model_files if not contains_exclude_words(f, exclude_words)]
    model_files = [os.path.join(dir_path, f) for f in model_files]
    assert len(model_files) > 0, "directory: {} has not pytorch model weight files.".format(dir_path)
    model_mtime = [os.path.getmtime(f) for f in model_files]
    idx = 0
    latest_time = model_mtime[0]
    for i in range(1, len(model_files)):
        if model_mtime[i] > latest_time:
            idx = i
            latest_time = model_mtime[i]

    return model_files[idx]


def main(args, ) -> None:
    '''main
    '''
    dist.set_seed(args.seed)
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning, device=torch.device(args.device), prune=args.prune,)
    cfg.eval_shape = args.shape
    if (args.test_only or args.prune) and not args.resume:
        ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            cfg.resume = ckpt_path
        else:
            cfg.resume = load_latest_model(cfg.output_dir, exclude_words=['eval'])

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
        solver.flops_params(h=args.shape[0], w=args.shape[1], c=args.shape[2])
    elif args.prune:
        solver.prune_pipeline()
    else:
        solver.fit()
        solver.val()
        solver.flops_params(h=args.shape[0], w=args.shape[1], c=args.shape[2])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c',type=str,)
    parser.add_argument('--resume','-r',type=str,)
    parser.add_argument('--tuning','-t',type=str,)
    parser.add_argument(
        '--test-only',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--prune',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
    )
    parser.add_argument('--shape', type=int, nargs='+', default=[640, 640, 3], help="input image shape [h, w, c], to get FLOPs and PARAMS.")

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    main(args)
