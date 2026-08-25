_base_ = '../../../configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'

# num_classes = 37   # hard encoding bad
# load_from = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
load_from = '../../../checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

model = dict(
    bbox_head=dict(
        type='DINOHead',
        num_classes=37,
    )
)
