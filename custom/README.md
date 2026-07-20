## Custom Scripts

This folder contains custom shell scripts and Python scripts used for SLURM job management and training.

## Cluster Install Step-by-Step

```shell
# setup conda env
conda create -n conda-mmdetection python=3.8 -y
conda activate conda-mmdetection

# load modules
module load cuda/cuda-11.8.0
module load gcc/gcc-10.5.0

# install torch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# install dependencies
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv==2.0.1
pip install yapf==0.40.1

# install mmdetection
git clone -b v3.1.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

## Additional Resources

- Installation Guide: https://github.com/open-mmlab/mmdetection/discussions/12328
