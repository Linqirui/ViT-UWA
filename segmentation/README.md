## Usage

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
conda create -n vit-uwa python==3.9
conda activate vit-uwa
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
pip install numpy==1.24.3
pip install yapf==0.40.1
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Prepare SUIM (Pascal VOC Format) according to the guidelines in [SUIM-Dataset](https://github.com/Linqirui/SUIM-Dataset).

## Pretraining Sources

| Name          | Year | Type       | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeiT          | 2021 | Supervised | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                               | [paper](https://arxiv.org/abs/2012.12877)                                                                                                                                       |
| AugReg        | 2021 | Supervised | ImageNet-22K | [repo](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | [paper](https://arxiv.org/abs/2106.10270)                                                                                                                                       |
| BEiT          | 2021 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit)                                             | [paper](https://arxiv.org/abs/2106.08254)                                                                                                                                       |
| BEiTv2        | 2022 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2)                                            | [paper](https://arxiv.org/abs/2208.06366)                                                                                                                                       |


## Results and Models



**SUIM**

|   Method    | Backbone  |                                                                                     Pretrain                                                                                     | Lr schd | Crop Size | mIoU (SS/MS)  | #Param |                                     Config                                     |                                                                                                                    Download                                                                                                                    |
| :---------: |:---------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| :-----: | :-------: |:-------------:|:------:|:------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   UperNet   | ViT-UWA-T |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  160k   |    512    | 71.41 / 71.60 |  38M   |        [config](./configs/SUIM/upernet_deit_uwa_tiny_512_160k_SUIM.py)         |                                                            [ckpt](https://pan.baidu.com/s/1XMwR04bbaP9eSP-vqXkCNA?pwd=suim)                                                             |
|   UperNet   | ViT-UWA-S |                                                 [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  160k   |    512    | 73.45 / 74.35 |  62M   |        [config](./configs/SUIM/upernet_deit_uwa_small_512_160k_SUIM.py)        |                                                            [ckpt](https://pan.baidu.com/s/1C__k-5zBnOmXK13__kXQlg?pwd=suim)                                                            |
|   UperNet   | ViT-UWA-B |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  160k   |    512    | 75.25 / 75.39 |  143M  |        [config](./configs/SUIM/upernet_deit_uwa_base_512_160k_SUIM.py)         |                                                           [ckpt](https://pan.baidu.com/s/1hPHt5vcYg6tBvuVWhFI5ag?pwd=suim)                                                            |
|   UperNet   | ViT-UWA-L | [AugReg-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    | 76.84 / 77.92 |  376M  |        [config](./configs/SUIM/upernet_augreg_uwa_large_512_160k_SUIM.py)        |                                                           [ckpt](https://pan.baidu.com/s/1QNce6q7oH2OCzDXev1M60A?pwd=suim)                                                           |
| Mask2Former | ViT-UWA-L |                    [BEiTv2-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip)                    |   80k   |    896    |  78.9 / 79.2  |  583M  | [config](./configs/SUIM/mask2former_beitv2_uwa_large_896_80k_SUIM.py) | [ckpt](https://pan.baidu.com/s/1yVfM-B7jBPp7_b1eFbrwKQ?pwd=suim)|


## Evaluation 

To evaluate ViT-UWA-B + UperNet on SUIM on a single node with 1 gpu run:
 
```shell
python test.py configs/SUIM/upernet_deit_uwa_base_512_160k_SUIM.py /path/to/checkpoint_file --eval mIoU
```

To compute the model complexity, including the number of parameters and FLOPs, run:

```shell
python get_flops.py configs/SUIM/upernet_deit_uwa_base_512_160k_SUIM.py
```

## Training

To train ViT-UWA-B + UperNet on SUIM on a single node with 1 gpu run:

```shell
python train.py configs/SUIM/upernet_deit_uwa_base_512_160k_SUIM.py
```

To train ViT-UWA-B + UperNet on SUIM on a single node with 2 gpus for run:

```shell
sh dist_train.sh configs/SUIM/upernet_deit_uwa_base_512_160k_SUIM.py 2
```