from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.registry import RUNNERS

def main():
    ROOT_DIR = "/nfs/scratch/staff/schmittth/code_nexus/mmdetection"
    ROOT_DIR = "/Users/noobtoss/code_nexus/mmdetection"

    OUT_DIR = f"{ROOT_DIR}/runs"
    CONFIG = f"{ROOT_DIR}/custom/configs/faster-rcnn_init.py"
    CKPT = f"{ROOT_DIR}/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"

    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = OUT_DIR
    cfg.merge_from_dict({
        'custom.root_dir': ROOT_DIR,
        'load_from': CKPT,
    })

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()

if __name__ == '__main__':
    main()
