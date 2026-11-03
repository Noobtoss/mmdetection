import os

# root = '/Users/noobtoss/code_hub/mmdetection'
# root = '/nfs/scratch/staff/schmittth/code_hub/mmdetection}}'
# root = '/home/atuin/v147eb/v147eb15/code_hub/mmdetection'
dataset_type = 'CocoDataset'
data_dir = '{{$MMDET_DATA_DIR:}}'

ann_file_train = os.path.join(data_dir, '{{$MMDET_ANN_FILE_TRAIN:annotation_train.json}}')
ann_file_val = os.path.join(data_dir, '{{$MMDET_ANN_FILE_VAL:annotation_test.json}}')
ann_file_test = os.path.join(data_dir, '{{$MMDET_ANN_FILE_TEST:annotation_test.json}}')

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

train_dataloader = dict(
    dataset=dict(
        data_root=data_dir,
        metainfo=metainfo,
        ann_file=ann_file_train,
        data_prefix=dict(img='')
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_dir,
        metainfo=metainfo,
        ann_file=ann_file_val,
        data_prefix=dict(img='')
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_dir,
        metainfo=metainfo,
        ann_file=ann_file_test,
        data_prefix=dict(img='')
    )
)

val_evaluator = dict(
    ann_file=ann_file_val
)
test_evaluator = dict(
    ann_file=ann_file_test
)
