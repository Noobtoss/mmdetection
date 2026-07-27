import argparse
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from mmdet.registry import RUNNERS
from mmengine.config import Config
from mmengine.runner import Runner


def train(config):
    if 'runner_type' not in config:
        runner = Runner.from_cfg(config)
    else:
        runner = RUNNERS.build(config)
    runner.train()


def parse_args():
    parser = argparse.ArgumentParser("mmdetection train parser")
    parser.add_argument("--run_name", type=str, help="exp name")
    parser.add_argument("--work_dir", type=str, help="work_dir")
    parser.add_argument("--model_cfg", type=str, default=None)
    parser.add_argument("--data_cfg", type=str, default=None)
    parser.add_argument("--train_cfg", type=str, default='../configs/runtime.py')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def build_config(args):
    config = Config({})
    config_files = [
            args.model_cfg,
            args.data_cfg,
            args.train_cfg,
    ]
    for config_file in config_files:
        if config_file is None:
            continue
        config.merge_from_dict(Config.fromfile(config_file).to_dict())

    config.work_dir = args.work_dir
    config.run_name = args.run_name
    config.model_name = Path(args.model_cfg).stem
    config.dataset_name = Path(args.data_cfg).stem
    for backend in config.visualizer.vis_backends:
        if backend["type"] == "WandbVisBackend":
            backend["init_kwargs"]["name"] = args.run_name

    return config


def update_config(config, opts):
    if not opts:
        return config
    updates = {}
    for key, value in zip(opts[0::2], opts[1::2]):
        try:
            value = eval(value)
        except Exception:
            pass
        updates[key] = value
    if 'seed' in updates:
        updates['randomness.seed'] = updates.pop('seed')
        # updates['randomness.deterministic'] = True
        updates['randomness.diff_rank_seed'] = True
    config.merge_from_dict(updates, allow_list_keys=True)
    return config


def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        warnings.warn("⚠️ Running with hardcoded test args")
        args = Namespace(
            run_name="unnamed_run",
            work_dir="/Users/noobtoss/code_nexus/mmdetection/runs/unnamed_run",
            model_cfg="../configs/models/faster-rcnn_r50_fpn.py",
            data_cfg="../configs/datasets/05ACCV2026Plus_local.py",
            train_cfg="../configs/runtime.py",
            opts=["seed", "184181",
                  "load_from", "../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth",
                  "visualizer.vis_backends.1.init_kwargs.project", "runs"
            ]
        )
    config = build_config(args)
    config = update_config(config, args.opts)
    train(config)


if __name__ == '__main__':
    main()
