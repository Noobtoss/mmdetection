# dataset settings
# root_dir = '/Users/noobtoss/code_nexus/mmdetection'
# root_dir = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection'
root_dir = '/home/atuin/v147eb/v147eb15/code_nexus/mmdetection'
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

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file_train,
        data_prefix=dict(img='Images/')
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file_val,
        data_prefix=dict(img='Images/')
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file_val,
        data_prefix=dict(img='Images/')
    )
)

val_evaluator = dict(
    ann_file=ann_file_val
)
test_evaluator = dict(
    ann_file=ann_file_val
)
