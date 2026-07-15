#!/usr/bin/env bash
# Creates the host environment used for Hinyeun training and remote inference.
# The simulator dependencies in INSTALL.md are not required for this workflow.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n dp3 python=3.12 -y
conda activate dp3

# cu128 is required for RTX 5090 (sm_120). Select a compatible PyTorch wheel
# for older GPUs if their driver does not support CUDA 12.8.
pip install torch==2.7.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

pip install \
    "zarr==2.18.3" "numcodecs<0.16" hydra-core==1.3.2 \
    "diffusers==0.31.0" einops termcolor wandb dill tqdm numba matplotlib \
    "av==15.1.0" pyarrow pandas ipdb

pip install -e "${repo}/3D-Diffusion-Policy"

# Optional for direct LeRobot dataset loading. The zarr converter itself uses
# pyarrow and av, but keeping this installed matches the verified environment.
pip install "lerobot[dataset]==0.6.0"

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import diffusion_policy_3d, zarr, hydra, diffusers, av, pyarrow; print('dp3 env ready')"
