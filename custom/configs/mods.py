wandb_project = '{{$MMDET_WANDB_PROJECT:runs-mmdetection}}'
run_name = '{{$MMDET_RUN_NAME:unnamed_run}}'

custom_imports = dict(
    imports=['mods'],
    allow_failed_imports=False
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project=wandb_project,
            name=run_name
        )
    )
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
)
