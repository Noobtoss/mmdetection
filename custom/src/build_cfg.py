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


def rebase_path(path, old_dir, new_dir):
    relative_path = Path(path).relative_to(old_dir)
    return str(Path(new_dir) / relative_path)


def build_epoch_update(config, updates):
    train_cfg = updates.get('train_cfg') or {}
    updates.setdefault('max_epochs', updates.get('train_cfg.max_epochs', train_cfg.get('max_epochs', config.train_cfg.max_epochs)))
    if updates.get('train_cfg') not in (None, ''):
        updates['train_cfg']['max_epochs'] = updates['max_epochs']
    else:
        updates['train_cfg.max_epochs'] = updates['max_epochs']
    ratio = updates['max_epochs'] / config.train_cfg.max_epochs
    if ratio == 1 or updates.get('param_scheduler') not in (None, ''):
        return updates
    schedulers = config.get('param_scheduler') or []
    for i, scheduler in enumerate([schedulers] if isinstance(schedulers, dict) else schedulers):
        if not scheduler.get('by_epoch', True):
            continue
        prefix = 'param_scheduler' if isinstance(schedulers, dict) else f'param_scheduler.{i}'
        for field in ('begin', 'end'):
            if field in scheduler:
                updates.setdefault(f'{prefix}.{field}', round(scheduler[field] * ratio))
        if 'milestones' in scheduler:
            updates.setdefault(f'{prefix}.milestones', [round(m * ratio) for m in scheduler['milestones']])
    return updates


def build_dataset_update(config, updates):
    updates = {key: value for key, value in updates.items() if value is not None and value != ""}
    updates['data_dir'] = os.path.abspath(updates.get('data_dir', config.data_dir))

    for split in ('train', 'val', 'test'):
        ann_key = f'ann_file_{split}'
        updates.setdefault(ann_key, rebase_path(config[ann_key], config.data_dir, updates['data_dir']))
        updates.setdefault(f'{split}_dataloader.dataset.data_root', updates['data_dir'])
        ann_file = os.path.abspath(os.path.join(updates['data_dir'], updates[ann_key]))
        updates[ann_key] = ann_file
        updates.setdefault(f'{split}_dataloader.dataset.ann_file', ann_file)
        if split != 'train':
            updates.setdefault(f'{split}_evaluator.ann_file', ann_file)
    return updates


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
    updates = build_epoch_update(cfg, updates)
    updates = build_dataset_update(cfg, updates)
    cfg.merge_from_dict(updates, allow_list_keys=True)
    return cfg


def init_cfg(args):
    cfg = Config({}, filename='config.py', cfg_text='# Merged configuration')
    cfg_files = [
        args.model_cfg,
        args.data_cfg,
        args.mods_cfg,
    ]
    for cfg_file in cfg_files:
        if cfg_file is None:
            continue
        cfg.merge_from_dict(Config.fromfile(cfg_file).to_dict())
    cfg.work_dir = args.work_dir
    cfg.run_name = args.run_name
    for backend in cfg.visualizer.vis_backends:
        if backend["type"] == "WandbVisBackend":
            backend["init_kwargs"]["name"] = args.run_name

    return cfg


def build_cfg(args):
    cfg = init_cfg(args)
    cfg = update_cfg(cfg, args.opts, args)
    tag(cfg)
    return cfg
