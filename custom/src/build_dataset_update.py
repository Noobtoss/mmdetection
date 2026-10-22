import os
from pathlib import Path


def rebase_path(path, old_dir, new_dir):
    relative_path = Path(path).relative_to(old_dir)
    return str(Path(new_dir) / relative_path)

def build_dataset_update(config, updates):
    annotation_aliases = {"ann_train": "ann_file_train", "ann_test": "ann_file_test", "ann_val": "ann_file_val"}
    updates = {annotation_aliases.get(key, key): value for key, value in updates.items()}

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
