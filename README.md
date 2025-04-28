# Code repo for Learned gridification for efficient point cloud processing

[![arXiv](https://img.shields.io/badge/arXiv-2307.14354-b31b1b.svg)](https://arxiv.org/pdf/2307.14354)

Installation: works for CUDA11.7 + conda and pip

Create env:
```
conda create --name gridifier python=3.10
conda activate gridifier
```

# install torch geometric

for pytorch < 2.0, use:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

for pytorch 2:
```
python -m pip install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

# Dependencies

```
python -m pip install pytorch-lightning
python -m pip install matplotlib
python -m pip install wandb
python -m pip install scikit-image
python -m pip install torchmetrics
python -m pip install timm
python -m pip install plotly
python -m pip install natsort
python -m pip install ml-collections
python -m pip install pandas
```

Train ModelNet40 or ShapeNet models by passing the appropriate config file, e.g. (see /cfg folder for available configs):

```
python main.py --cfg=cfg/modelnet_cfg.py
```
