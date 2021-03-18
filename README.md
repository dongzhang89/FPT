# Feature Pyramid Transformer

Implementation for paper: [Feature Pyramid Transformer](https://arxiv.org/abs/2007.09451).

## Contents

1. [Overview](#Overview)
2. [Requirements](#Requirements)
3. [Data Preparation](#data_preparation)
4. [Pretrained Model](#pretrained_model)
5. [Model Training](#model_training)
6. [Inference](#Inference)
7. [Citation](#Citation)
8. [Question](#Question)

## Overview
Feature interactions across space and scales underpin modern visual recognition systems because they introduce beneficial visual contexts. Conventionally, spatial contexts are passively hidden in the CNN's increasing receptive fields or actively encoded by non-local convolution. Yet, the non-local spatial interactions are not across scales, and thus they fail to capture the non-local contexts of objects (or parts) residing in different scales. To this end, we propose a fully active feature interaction across both space and scales, called Feature Pyramid Transformer (FPT). It transforms any feature pyramid into another feature pyramid of the same size but with richer contexts, by using three specially designed transformers in self-level, top-down, and bottom-up interaction fashion. FPT serves as a generic visual backbone with fair computational overhead. We conduct extensive experiments in both instance-level (i.e., object detection and instance segmentation) and pixel-level segmentation tasks, using various backbones and head networks, and observe consistent improvement over all the baselines and the state-of-the-art methods.

<div align="center">
<img src="demos/screenshot_20200731170229.png" width="700px"/>
<p> Overall structure of our proposed FPT. Different texture patterns indicate different feature transformers, and different color represents feature maps with different scales. "Conv" denotes a 3 × 3 convolution with the output dimension of 256. Without loss of generality, the top/bottom layer feature maps has no rendering/grounding transformer.</p>
</div>

## Requirements

- Packages
  - pytorch=0.4.0
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - dropblock
  - [pycocotools](https://github.com/cocodataset/cocoapi)
  - tensorboardX  — for logging the losses in Tensorboard
- 8 GPUs and CUDA 8.0 or higher. Some operations only have gpu implementation.

## Data Preparation

Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

- **COCO**:
  Download COCO images and annotations from [website](http://cocodataset.org/#download).

  And make sure to put the files as the following structure:
  ```
  coco
  ├── annotations
  |   ├── instances_minival2014.json
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   ├── instances_valminusminival2014.json
  │   ├── ...
  |
  └── images
      ├── train2014
      ├── train2017
      ├── val2014
      ├── val2017
      ├── ...
  ```
   Feel free to put COCO at any place you want, and then soft link the dataset under the `data/` folder:

   ```
   ln -s path/to/coco data/coco 
   ```

  Recommend to put the images on a SSD for possible better training performance

## Pretrained Model

#### ImageNet Pretrained Model from Caffe

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1)
- [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l)
- [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)

Download them and put them into the `{repo_root}/data/pretrained_model`.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data preprocessing as used in Pytorch pretrained model.**

#### ImageNet Pretrained Model from Detectron

- [R-50.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl)
- [R-101.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-101.pkl)
- [R-50-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl)
- [R-101-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl)
- [X-101-32x8d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl)
- [X-101-64x4d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl)
- [X-152-32x8d-IN5k.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl)

**NOTE**: Caffe pretrained weights have slightly better performance than the Pytorch pretrained weights.

## Model Training

### Train from scratch

Take mask-rcnn with resnet-50 backbone for example.
```
python tools/train_net_step.py --dataset coco2017 --cfg configs/e2e_fptnet_R-50_mask.yaml --use_tfboard --bs {batch_size} --nw {num_workers}
```

Use `--bs` to overwrite the default batch size to a proper value that fits into your GPUs. Simliar for `--nw`, number of data loader threads defaults to 4 in config.py.

Specify `—-use_tfboard` to log the losses on Tensorboard.

### Finetune from a checkpoint
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint}
```
or using Detectron's checkpoint file
```
python tools/train_net_step.py ... --load_detectron {path/to/the/checkpoint}
```

### Resume training with the same dataset and batch size
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint} --resume
```

When resume the training, **step count** and **optimizer state** will also be restored from the checkpoint. For SGD optimizer, optimizer state contains the momentum for each trainable parameter.

**NOTE**: `--resume` is not yet supported for `--load_detectron`

### Set config options in command line
```
  python tools/train_net_step.py ... --no_save --set {config.name1} {value1} {config.name2} {value2} ...
```
- For Example, run for debugging.
  ```
  python tools/train_net_step.py ... --no_save --set DEBUG True
  ```
  Load less annotations to accelarate training progress. Add `--no_save` to avoid saving any checkpoint or logging.

### Show command line help messages
```
python train_net_step.py --help
```

## Inference 

### Evaluate the training results
For example, on coco2017 val set
```
python tools/test_net.py --dataset coco2017 --cfg configs/e2e_fptnet_R-50_mask.yaml --load_ckpt {path/to/your/checkpoint}
```

### Results visualization
```
python tools/infer_simple.py --dataset coco --cfg configs/e2e_fptnet_R-50_mask.yaml --load_ckpt {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}
```

## My nn.DataParallel

- **Keep certain keyword inputs on cpu**
  Official DataParallel will broadcast all the input Variables to GPUs. However, many rpn related computations are done in CPU, and it's unnecessary to put those related inputs on GPUs.
- **Allow Different blob size for different GPU**
  To save gpu memory, images are padded seperately for each gpu.
- **Work with returned value of dictionary type**

## Citation

If our work is useful for your research, please consider citing:

```
@inproceedings{zhang2020feature,
  title={Feature pyramid transformer},
  author={Zhang, Dong and Zhang, Hanwang and Tang, Jinhui and Wang, Meng and Hua, Xiansheng and Sun, Qianru},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020},
}
```

## Questions

Please contact 'dongzhang@njust.edu.cn'
