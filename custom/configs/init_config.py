_base_ = [
    './_base_/rtmdet_tiny_8xb32-300e_coco.py',
]

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

root_dir = '/Users/noobtoss/code_nexus/mmdetection'
# root_dir = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection'
data_root = f'{root_dir}/datasets/05ACCV2026Plus'
dataset_type = 'CocoDataset'

ann_file_train = f'{data_root}/annotations/annotation_train.json'
ann_file_val = f'{data_root}/annotations/annotation_test.json'
ann_file_test = f'{data_root}/annotations/annotation_test.json'

num_classes = len(metainfo['classes'])
max_epochs = 100

dataset_cfg = dict(
    metainfo=metainfo,
    data_root=data_root,
    data_prefix=dict(img='Images/'),
)

model = dict(
    bbox_head=dict(
    num_classes=num_classes
    )
)

train_cfg = dict(
    max_epochs=max_epochs
)

train_dataloader = dict(
    dataset=dict(
        **dataset_cfg,
        ann_file=ann_file_train,
    )
)

val_dataloader = dict(
    dataset=dict(
        **dataset_cfg,
        ann_file=ann_file_val,
    )
)

test_dataloader = dict(
    dataset=dict(
        **dataset_cfg,
        ann_file=ann_file_test,
    )
)

val_evaluator = dict(ann_file=ann_file_val)
test_evaluator = dict(ann_file=ann_file_test)

custom_imports = dict(
    imports=['mods'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(type='CustomLoggerHook')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='tmp',
                name='tmp'
            )
        )
    ]
)
