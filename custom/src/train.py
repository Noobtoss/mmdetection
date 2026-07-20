from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.registry import RUNNERS

def main():
    ROOT_DIR = "/nfs/scratch/staff/schmittth/code_nexus/mmdetection"
    ROOT_DIR = "/Users/noobtoss/code_nexus/mmdetection"

    OUT_DIR = f"{ROOT_DIR}/runs"
    CONFIG = f"{ROOT_DIR}/custom/configs/init_config.py"
    CKPT = f"{ROOT_DIR}/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

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
