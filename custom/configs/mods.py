custom_imports = dict(
    imports=['mods'],
    allow_failed_imports=False
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='runs-mmdetection',
            name='unnamed_run'
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
