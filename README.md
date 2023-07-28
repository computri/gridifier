# DensePointClouds

Env:

```
conda create --name envy python=3.10
conda activate envy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install pytorch-lightning
python -m pip install matplotlib
python -m pip install wandb
python -m pip install scikit-image
python -m pip install torchmetrics
python -m pip install timm
python -m pip install plotly
python -m pip install natsort
python -m pip install networkx
python -m pip install ml-collections
python -m pip install pandas
```

command:

```
python configure_baselines.py --aggr "max" --seed 42 --model "ConvNet3D" --sigma 1 --dataset "ModelNet40" --dropout 0.5 --d_hidden 128 --features "" --n_blocks
3 --blocktype "" --drop_path 0.1 --embed_pos 1 --fps_ratio 1.0 --layernorm 0 --batch_size 32 --update_net 0 --use_normals 1 --conditioning "relpos" --connectivity "knn" --n_neighbours 9 --n_subsamples 1000 --learning_rate 1e-4 --node_embedding 1 --grid_resolution 9 --augment_rotation 0 --k_backward_edges 4 --message_batchnorm 0
```

Using config files.

python main.py --cfg=path-to-config --cfg.arg1=1 --cfg.arg2=2 --....

Download IntrA dataset:
``` 
cd data
git clone git@github.com:intra3d2019/IntrA.git
``` 
Download IntrA from:
https://drive.google.com/drive/folders/1yjLdofRRqyklgwFOC0K4r7ee1LPKstPh

and copy its contents into the github IntrA directory

# Questions

1. If we use simultaneously backward and forward edges, is there no chance that some appear twice?
