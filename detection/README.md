## Usage

Install [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Prepare UIIS according to the guidelines in [Watermask](https://github.com/LiamLian0727/WaterMask).

Prepare USIS10K according to the guidelines in [USIS-SAM](https://github.com/LiamLian0727/USIS10K).

## Pretraining Sources

| Name          | Type            | Year | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | --------------- | ---- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeiT          | Supervised      | 2021 | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                               | [paper](https://arxiv.org/abs/2012.12877)                                                                                                                                       |
| AugReg        | Supervised      | 2021 | ImageNet-22K | [repo](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | [paper](https://arxiv.org/abs/2106.10270)                                                                                                                                       |

## Results and Models

**Mask R-CNN & UIIS**

|   Method   |   Backbone    |                                                                                     Pretrain                                                                                     | Lr schd | box AP | mask AP | #Param |                                      Config                                      |                                                                                                                  Download                                                                                                                   |
| :--------: |:-------------:| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: |:------:|:-------:|:------:|:--------------------------------------------------------------------------------:| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask R-CNN |   ViT-UWA-T   |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  3x+MS  |  26.8  |  25.7   |  30M   |       [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_tiny_fpn_3x_UIIS.py)       |                                                           [ckpt](https://pan.baidu.com/s/1XNwtejea_b5gK8pmSsXU8w?pwd=uiis)                                                            |
| Mask R-CNN |   ViT-UWA-S   |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  3x+MS  |  28.1  |  26.1   |  52M   |      [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_small_fpn_3x_UIIS.py)       |                                                           [ckpt](https://pan.baidu.com/s/1i9IbDSHkenaiMOlOBE9nXw?pwd=uiis)                                                           |
| Mask R-CNN |   ViT-UWA-B   |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  3x+MS  |  29.8  |  26.8   |  129M  |       [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_base_fpn_3x_UIIS.py)       |                                                           [ckpt](https://pan.baidu.com/s/16UFY8AzBdxKJ9hM5gLmSEg?pwd=uiis)                                                            |
| Mask R-CNN |   ViT-UWA-B   |                                   [AugReg-B](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth)                                    |  3x+MS  |  30.9  |  29.0   |  129M  | [config](./configs/mask_rcnn/mask_rcnn_augreg_uwa_base_fpn_3x_UIIS.py) | [ckpt](https://pan.baidu.com/s/1qBR9mhLqWw8mNq110UqKkA?pwd=uiis)  |

**Other Detectors & UIIS**

|       Method       | Backbone  |                                                                                     Pretrain                                                                                     | Lr schd | box AP | #Param |                                 Config                                  |                                                                                                                  Download                                                                                                                   |
|:------------------:|:---------:| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: |:------:|:------:|:-----------------------------------------------------------------------:| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Cascade Mask R-CNN | ViT-UWA-S |                                                 [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  3x+MS  |  31.0  |  89M   | [config](./configs/cascade_rcnn/cascade_mask_rcnn_deit_uwa_small_fpn_3x_UIIS.py) |                                                           [ckpt](https://pan.baidu.com/s/1mYA18nMZ7WxV9ztska5_zA?pwd=uiis)                                                            |
|        ATSS        | ViT-UWA-S |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  3x+MS  |  29.3  |  40M   |    [config](./configs/atss/atss_deit_uwa_small_fpn_3x_UIIS.py)     |                                                           [ckpt](https://pan.baidu.com/s/1Jxmhclw87Z-5Zej7_kGuEg?pwd=uiis)                                                           |
|        GFL         | ViT-UWA-S |                                                 [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  3x+MS  |  29.7  |  40M   |     [config](./configs/gfl/gfl_deit_uwa_small_fpn_3x_UIIS.py)      |                                                           [ckpt](https://pan.baidu.com/s/1rjntVLzbRBaey9WSs4dOMA?pwd=uiis)                                                            |


**Mask R-CNN & USIS10K**

|   Method   |   Backbone    |                                                                                     Pretrain                                                                                     | Lr schd | box AP | mask AP | #Param |                                 Config                                 |                                                                                                                  Download                                                                                                                   |
| :--------: |:-------------:| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: |:------:|:-------:|:------:|:----------------------------------------------------------------------:| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Mask R-CNN |   ViT-UWA-T   |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  3x+MS  |  40.9  |  38.8   |  30M   |  [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_tiny_fpn_3x_USIS.py)  |                                                           [ckpt](https://pan.baidu.com/s/1SlEFAYVXQk48PdboKCEs5A?pwd=usis)                                                            |
| Mask R-CNN |   ViT-UWA-S   |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  3x+MS  |  43.0  |  39.7   |  52M   | [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_small_fpn_3x_USIS.py)  |                                                           [ckpt](https://pan.baidu.com/s/1awR07zfwJssDjxXqxfkYjQ?pwd=usis)                                                           |
| Mask R-CNN |   ViT-UWA-B   |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  3x+MS  |  44.9  |  42.0   |  129M  |  [config](./configs/mask_rcnn/mask_rcnn_deit_uwa_base_fpn_3x_USIS.py)  |                                                           [ckpt](https://pan.baidu.com/s/136xKKsPtk351gqX9UVGWfw?pwd=usis)                                                            |
| Mask R-CNN |   ViT-UWA-B   |                                   [AugReg-B](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth)                                    |  3x+MS  |  46.4  |  44.2   |  129M  | [config](./configs/mask_rcnn/mask_rcnn_augreg_uwa_base_fpn_3x_USIS.py) | [ckpt](https://pan.baidu.com/s/15G4fKQgTSLFK62Ru2fzTbg?pwd=usis)  |

## Evaluation

To evaluate ViT-UWA-S + Mask R-CNN on UIIS on a single node with 1 gpu run:

```shell
python test.py configs/mask_rcnn/mask_rcnn_deit_uwa_small_fpn_3x_UIIS.py /path/to/checkpoint_file --eval bbox segm
```

To benchmark the inference speed of ViT-UWA-S + Mask R-CNN on UIIS, run:

```shell
python benchmark.py configs/mask_rcnn/mask_rcnn_deit_uwa_small_fpn_3x_UIIS.py /path/to/checkpoint_file
```
To compute the model complexity, including the number of parameters and FLOPs, run:

```shell
python get_flops.py configs/mask_rcnn/mask_rcnn_deit_uwa_small_fpn_3x_UIIS.py
```

## Training

To train ViT-UWA-B + Mask R-CNN on UIIS on a single node with 1 gpu for 36 epochs run:

```shell
python train.py configs/mask_rcnn/mask_rcnn_deit_uwa_base_fpn_3x_UIIS.py
```

To train ViT-UWA-B + Mask R-CNN on USIS10K on a single node with 1 gpu for 36 epochs run:

```shell
python train.py configs/mask_rcnn/mask_rcnn_deit_uwa_base_fpn_3x_USIS.py
```

To train ViT-UWA-B + Mask R-CNN on USIS10K on a single node with 2 gpus for 36 epochs run:

```shell
sh dist_train.sh configs/mask_rcnn/mask_rcnn_deit_uwa_base_fpn_3x_USIS.py 2
```