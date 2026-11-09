import os
from pathlib import Path
from mmengine.config import Config

ALIASES = {
    "epochs": "max_epochs",
    "train_ann": ("train_ann", "ann_file_train"),
    "val_ann": ("val_ann", "ann_file_val"),
    "test_ann": ("test_ann", "ann_file_test"),
}

VALUE_MAPPING = {}


def tag(cfg):
    cfg.run_name = cfg.run_name  # Matches W&B since opts does not override run_name.
    cfg.model_name = Path(cfg.model_cfg).stem
    cfg.dataset_name = Path(cfg.data_cfg).stem
    cfg.imgsz = getattr(cfg, 'imgsz', next((step.get('scale') for step in cfg.test_dataloader.dataset.pipeline if step.get('type') == 'Resize'), None))
    cfg.epochs = cfg.max_epochs


def write_os_vars(opts, args=None):
    env_keys = {
        'cls_feat_loss', 'cls_feat', 'cls_feat_proj_head',
        'data_dir', 'ann_file_train', 'ann_file_val', 'ann_file_test',
        'max_epochs',
        'run_name', 'wandb_project',
    }
    updates = vars(args).copy() if args is not None else {}
    opts = opts or []
    updates.update(zip(opts[0::2], opts[1::2]))
    for key, value in updates.items():
        targets = ALIASES.get(key, key)
        for target in (targets,) if isinstance(targets, str) else targets:
            if target in env_keys:
                if not target.startswith('cls_feat') and value in (None, ''):
                    continue
                os.environ[f'MMDET_{target.upper()}'] = str(value)



def update_cfg(cfg, opts, args=None):
    updates = {}
    if args is not None:
        updates.update(vars(args))
    opts = opts or []
    for key, value in zip(opts[0::2], opts[1::2]):
        try:
            value = eval(value)
        except Exception:
            pass
        updates[key] = value
    mapped_updates = {}
    for key, value in updates.items():
        targets = ALIASES.get(key, key)
        if isinstance(targets, str):
            targets = (targets,)
        for target in targets:
            mapped_updates[target] = VALUE_MAPPING[target](value) if target in VALUE_MAPPING else value
    updates = mapped_updates
    if 'seed' in updates:
        updates['randomness.seed'] = updates.pop('seed')
        # updates['randomness.deterministic'] = True
        updates['randomness.diff_rank_seed'] = True
    updates = {key: value for key, value in updates.items()
               if value is not None and value != "" and key not in ('ann_file_train', 'ann_file_val', 'ann_file_test')}
    cfg.merge_from_dict(updates, allow_list_keys=True)
    return cfg


def init_cfg(args):
    cfg = Config({}, filename='config.py', cfg_text='# Merged configuration')
    cfg_files = [
        args.data_cfg,
        args.model_cfg,
        args.mods_cfg,
    ]
    for cfg_file in cfg_files:
        if cfg_file is None:
            continue
        part = Config.fromfile(cfg_file)
        if cfg_file == args.data_cfg:
            # Set the class count before the model config builds its head.
            os.environ['MMDET_NUM_CLASSES'] = str(part.num_classes)
        elif cfg_file == args.model_cfg:
            # Keep dataset overrides over the model's inherited COCO defaults.
            part.merge_from_dict(cfg.to_dict())
        cfg.merge_from_dict(part.to_dict())
    cfg.work_dir = args.work_dir
    cfg.run_name = args.run_name
    return cfg


def build_cfg(args):
    write_os_vars(args.opts, args)
    cfg = init_cfg(args)
    cfg = update_cfg(cfg, args.opts, args)
    tag(cfg)
    return cfg
