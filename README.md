# [ICRA2022] AutoPlace: Robust Place Recognition with Single-Chip Automotive Radar

[![arxiv](https://img.shields.io/badge/arXiv-2109.08652-%23B31C1B?style=flat)](https://arxiv.org/abs/2109.08652)
[![GitHub](https://img.shields.io/website?label=Project%20Page&up_message=Up&url=https%3A%2F%2Fwww.csc.liv.ac.uk%2F~ramdrop%2Fautoplace.html)](https://www.csc.liv.ac.uk/~ramdrop/autoplace.html)
[![YouTube](https://img.shields.io/youtube/views/d_-ZYJhgGIk?label=YouTube&style=flat)](https://www.youtube.com/watch?v=d_-ZYJhgGIk&ab_channel=KaiwenCai)

![demo](demo.gif)

```shell
@article{cai2021autoplace,
  title={AutoPlace: Robust Place Recognition with Low-cost Single-chip Automotive Radar},
  author={Cai, Kaiwen and Wang, Bing and Lu, Chris Xiaoxuan},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3475--3481},
  year={2022},
  organization={IEEE}  
}
```

## 0. Environment Setup  ⚙️

* Ubuntu 18.04, python 3.8, A100
* PyTorch 1.8.1 + CUDA 11.1
* nuscenes-devkit 1.1.1

```
pip install --upgrade pip
pip install -r requirements.txt
```

Go to the nuscenes-devkit package directory (depends on which python you are using), and override the function `from_file_multisweep(...)` in `site-packages/nuscenes/utils/data_classes.py ` with the function provided in `autoplace/nuscenes-devkit_override.py`.

## 1. Dataset preprocessing 📥

You may need to download nuScenes dataset (radar) from [nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).

```bash
cd autoplace/preprocess
./gene_woDTR.sh
./gene_wDTR.sh
```

the generated processed dataset folder should be like:

```
dataset
├── 7n5s_xy11
│   ├── pcl_parameter.json
│   ├── img
│   ├── pcl
│   ├── rcs
│   ├── nuscenes_test.mat
│   ├── nuscenes_train.mat
│   ├── nuscenes_val.mat
│   ├── database.csv
│   ├── train.csv
│   └── test.csv
└── 7n5s_xy11_remove
    ├── ...
```

to save you time on downloading/preprocessing the nuScenes dataset, you may as well download my processed dataset from [Dropbox](https://www.dropbox.com/s/yaqn1qa48ot4s9g/dataset.zip?dl=0) and then arrange it in the above way.

## 2. AutoPlace 🚗

1. train SpatialEncoder (se)

   ```bash
   cd autoplace

   python train.py  --nEpochs=50 --output_dim=9216 --seqLen=1 --encoder_dim=256 --net=autoplace --logsPath=logs_autoplace --cGPU=0 --split=val --imgDir='dataset/7n5s_xy11/img' --structDir='dataset/7n5s_xy11'
   ```
2. train SpatialEncoder+DPR (se_dpr)

   ```bash
   cd autoplace

   python train.py  --nEpochs=50 --output_dim=9216 --seqLen=1 --encoder_dim=256 --net=autoplace --logsPath=logs_autoplace --cGPU=0 --split=val --imgDir='dataset/7n5s_xy11_removal/img' --structDir='dataset/7n5s_xy11'
   ```
3. train SpatialEncoder+TemporalEncoder (se_te)

   ```bash
   cd autoplace

   python train.py  --nEpochs=50 --output_dim=4096 --seqLen=3 --encoder_dim=256 --net=autoplace --logsPath=logs_autoplace --cGPU=0 --split=val --imgDir='dataset/7n5s_xy11/img' --structDir='dataset/7n5s_xy11'
   ```
4. train SpatialEncoder+TemporalEncoder+DPR (se_te_dpr)

   ```bash
   cd autoplace

   python train.py  --nEpochs=50 --output_dim=4096 --seqLen=3 --encoder_dim=256 --net=autoplace --logsPath=logs_autoplace --cGPU=0 --split=val --imgDir='dataset/7n5s_xy11_removal/img' --structDir='dataset/7n5s_xy11'

   ```
5. evaluate a model

   ```bash
   cd autoplace

   python train.py --mode='evaluate'  --cGPU=0  --split=test --resume=[logs_folder]
   ```
6. apply `RCSHR` on `SpatialEncoder+TemporalEncoder+DPR` model (You may need to evaluate SpatialEncoder+TemporalEncoder+DPR model first): modify the path `se_te_dpr` in `autoplace/postprocess/parse/resume_path.json` to [logs_folder], then

   ```bash
   cd autoplace/postprocess/parse 

   python parse.py  --rcshr --model=se_te_dpr
   ```
7. To generate (1) Reall@N curve, (2) PR curve, (3) F1 Score and (4) Average Precision

   ```bash
   cd autoplace/postprocess/vis

   python ablation_figure.py 
   python ablation_score.py 
   ```

## 3. [SOTA methods](https://github.com/ramdrop/AutoPlace/blob/main/SOTA.md) ⚔
