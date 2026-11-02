from mmdet.apis import DetInferencer

# Print every model name mmdetection knows about + has weights for
models = DetInferencer.list_models('mmdet')
for m in models:
    print(m)
