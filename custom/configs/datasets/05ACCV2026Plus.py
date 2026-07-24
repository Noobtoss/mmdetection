# dataset settings
root_dir = '/Users/noobtoss/code_nexus/mmdetection'
root_dir = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection'
dataset_type = 'CocoDataset'
data_root = f'{root_dir}/datasets/05ACCV2026Plus'

ann_file_train = f'{data_root}/annotations/annotation_train.json'
ann_file_val = f'{data_root}/annotations/annotation_test.json'
ann_file_test = f'{data_root}/annotations/annotation_test.json'

backend_args = None

metainfo = dict(
    classes=(
        'Backware',
        'Bauernbrot',
        'Floesserbrot',
        'Salzstange',
        'Sonnenblumensemmel',
        'Kuerbiskernsemmel',
        'Roggensemmel',
        'Dinkelsemmel',
        'LaugenstangeSchinkenKaese',
        'Pfefferlaugenbrezel',
        'KernigeStange',
        'Schokocroissant',
        'Apfeltasche',
        'Quarktasche',
        'Mohnschnecke',
        'Nussschnecke',
        'Vanillehoernchen',
        'Osterei',
        'Osterbrezel',
        'Kirschtasche',
        'Fruechteschiffchen',
        'Anisbrezel',
        'Doppelsemmel',
        'Fruestuecksemmel',
        'Kaisersemmel',
        'Kornknacker',
        'Landbrot',
        'Laugenbrezel',
        'Laugenstange',
        'Laugenzopf',
        'Mohnsemmel',
        'Mohnstange',
        'Partybrot',
        'Sandwichbroetchen',
        'Sesamsemmel',
        'Sesamstange',
        'Vollgutsemmel'
    )
)
num_classes = len(metainfo['classes'])

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file_train,
        data_prefix=dict(img='Images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file_val,
        data_prefix=dict(img='Images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_file_val,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
