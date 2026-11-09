import argparse
import sys
import warnings
from argparse import Namespace
from mmdet.registry import RUNNERS
from mmengine.runner import Runner

from build_cfg import build_cfg

DEFAULT_ARGS = {
    "faster-rcnn": Namespace(
        run_name="unnamed_run",
        work_dir="/Users/noobtoss/code_hub/mmdetection/runs/unnamed_run",
        data_cfg="../configs/datasets/base.py",
        model_cfg="../configs/models/faster-rcnn_r50_fpn.py",
        mods_cfg="../configs/mods.py",
        data_dir="/Users/noobtoss/code_hub/mmdetection/datasets/05ACCV2026Plus",
        train_ann="annotation_train.json",
        val_ann="annotation_test.json",
        test_ann="annotation_test.json",
        opts=["seed", "184181",
              "load_from", "../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth",
              "wandb_project", "runs-mmdetection",
              "cls_feat", None,
              ]
    ),
    "detr": Namespace(
        run_name="unnamed_run",
        work_dir="/Users/noobtoss/code_hub/mmdetection/runs/unnamed_run",
        data_cfg="../configs/datasets/base.py",
        model_cfg="../configs/models/detr_r50.py",
        mods_cfg="../configs/mods.py",
        data_dir="/Users/noobtoss/code_hub/mmdetection/datasets/05ACCV2026Plus",
        train_ann="annotation_train.json",
        val_ann="annotation_test.json",
        test_ann="annotation_test.json",
        opts=["seed", "184181",
              "load_from", "../../checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth",
              "wandb_project", "runs-mmdetection",
              ]
    ),
    "def-detr": Namespace(
        run_name="unnamed_run",
        work_dir="/Users/noobtoss/code_hub/mmdetection/runs/unnamed_run",
        data_cfg="../configs/datasets/base.py",
        model_cfg="../configs/models/deformable-detr-refine-twostage_r50.py",
        mods_cfg="../configs/mods.py",
        data_dir="/Users/noobtoss/code_hub/mmdetection/datasets/05ACCV2026Plus",
        train_ann="annotation_train.json",
        val_ann="annotation_test.json",
        test_ann="annotation_test.json",
        opts=["seed", "184181",
              "load_from", "../../checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth",
              "wandb_project", "runs-mmdetection",
              ]
    ),
    "dino": Namespace(
        run_name="unnamed_run",
        work_dir="/Users/noobtoss/code_hub/mmdetection/runs/unnamed_run",
        data_cfg="../configs/datasets/base.py",
        model_cfg="../configs/models/dino-5scale_swin-l.py",
        mods_cfg="../configs/mods.py",
        data_dir="/Users/noobtoss/code_hub/mmdetection/datasets/05ACCV2026Plus",
        train_ann="annotation_train.json",
        val_ann="annotation_test.json",
        test_ann="annotation_test.json",
        opts=["seed", "184181",
              "load_from", "../../checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
              "wandb_project", "runs-mmdetection",
              ]
    ),
}


def train(cfg):
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.train()


def parse_args():
    parser = argparse.ArgumentParser("mmdetection train parser")
    parser.add_argument("--run_name", type=str, help="run name")
    parser.add_argument("--work_dir", type=str, help="work_dir")
    parser.add_argument("--model_cfg", type=str, default=None)
    parser.add_argument("--data_cfg", type=str, default=None)
    parser.add_argument("--mods_cfg", type=str, default='../configs/mods.py')
    parser.add_argument("--data_dir", type=str, default=None, help="base for relative data and COCO annotation paths; also the COCO image directory")
    parser.add_argument("--train_ann", type=str, help="train ann json")
    parser.add_argument("--val_ann", type=str, help="val ann json")
    parser.add_argument("--test_ann", type=str, help="test ann json")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.test_ann and not args.val_ann: warnings.warn("--val_ann missing; using --test_ann.")
    if args.val_ann and not args.test_ann: warnings.warn("--test_ann missing; using --val_ann.")
    if not args.val_ann: args.val_ann = args.test_ann
    if not args.test_ann: args.test_ann = args.val_ann
    return args


def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        warnings.warn("⚠️ Running with hardcoded test args")
        args = DEFAULT_ARGS["faster-rcnn"]
        # args = DEFAULT_ARGS["detr"]
        # args = DEFAULT_ARGS["def-detr"]
        # args = DEFAULT_ARGS["dino"]

    cfg = build_cfg(args)
    train(cfg)


if __name__ == '__main__':
    main()
