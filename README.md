# Ultrasound Nerve Segmentation Challenge using Torchnet

This repository acts as a starting point for someone who wants to start with the [kaggle ultrasound-nerve-segmentation challenge](https://www.kaggle.com/c/ultrasound-nerve-segmentation) and exploit [torchnet](https://github.com/torchnet/torchnet) for the same. The challenge in itself acts as a learning experience for segmentation problems.

## Requirements

- CUDA Environment
- [Torch-7](http://torch.ch/)
- [Torchnet](https://github.com/torchnet/torchnet)
- Torch Libraries
    - [image](https://github.com/torch/image)
    - [hdf5](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)
    - [nngraph](https://github.com/torch/nngraph)
    - [optim](https://github.com/torch/optim)
    - [cudnn](https://github.com/soumith/cudnn.torch)
    - [paths](https://github.com/torch/paths)
    - [csvigo](https://github.com/clementfarabet/lua---csv)
    - [imagemagick](http://www.imagemagick.org/script/index.php)

## Dataset Generation

The [dataset](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) consists of 5635 training images and their masks, and 5508 testing images. The images are in .tif, and to be able to load them into lua, we convert then into .png. So firstly we need to setup dataset such that

- Train images are in ``` /path/to/train/data/images ```
- Train masks are in ``` /path/to/train/data/masks ```
- Test images are in ``` /path/to/test/data ```

Now, go to each folder and run the following command, it will generate .png file for each .tif file in the folder. Be patient the procedure takes time.

``` mogrify -format png *.tif ```

Now, we have all images in .png. So, to create datasets run the following command

```
th create_dataset.lua -train /path/to/train/data/images -trainOutput /path/to/train/data.h5 -test /path/to/test/data -testOutput /path/to/test/data.h5
```

This will generate the datasets in HDF5 format, such that train images and masks of patient number *N* are in paths */images_N* and */masks_N* of the train HDF5 file respectively. The test images are in */images* path of test HDF5 file generated.

## Model

We are using a Modified [U-Net](http://arxiv.org/abs/1505.04597) with [Kaiming-He](https://arxiv.org/abs/1502.01852) initialization. The structure of U-Net generated using nngraph can be found [here](./U-Net.svg)

## Training

```
th main.lua [OPTIONS]
```

### Options

| Option | Default value | Description |
| ------ | --- | ----------- |
| `-dataset` | `data/train.h5` | Path to training dataset to be used |
| `-model` | `models/unet.lua` | Path of the model to be used |
| `-trainSize` | 100 | Amount of data to be used for training, -1 if complete train data to be used |
| `-valSize` | 25 | Amount of data to be used for validation, -1 if complete validation to be used |
| `-trainBatchSize` | 64 | Size of batch size to be used for training |
| `-valBatchSize` | 32 | Size of batch size to be used for validation |
| `-savePath` | `data/saved_models` | Path where models must be saved |
| `-optimMethod` | `sgd` | Method to be used for training, can be sgd or adam |
| `-maxepoch` | 250 | Maximum epochs for which training must be done |
| `-cvparam` | 2 | Cross validation parameter |

### Train Validation Data Split

The images are given for each patient, and thus in dataset we have 47 patients with each patient having 119|120 images. To assess the real performance of our model, we divide the dataset into train and validation based on patients and use 80-20 split. Thus, now question arises which patients to use for validation and which for training.To solve this, we keep a parameter `-cvparam`, such that all patients with `patient_id%5==cvparam` are used in validation, else in training. Now out of these images, we select `-trainSize` number of images and `-valSize` number of images for training and validation respectively.

### Data Augmentation

Data augmentation plays a vital role in any segmentation problem with limited dataset. Here we do on-the-fly data augmentation using [Resnet's transformation file](https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua). The image goes through following transformations:

1. Horizontal flip with probability 0.5
2. Vertical flip with probability 0.5
3. Rotation between -5 to 5 degrees with uniform probability
4. Elastic transformations

### Constants

We resize the image to `imgWidth * imgHeight` and then pass to our model. For creating segmentation masks, we consider a pixel from the output to be a part of mask iff `prob_pixel > baseSegmentationProb`. One can define these values in [constants.lua](constants.lua) file.

## Submission file

```
th generate_submission.lua [OPTIONS]
```

### Options

| Option | Default value | Description |
| ------ | --- | ----------- |
| `-dataset` | `data/test.h5` | Path to dataset to be used |
| `-model` | `models/unet.t7` | Path of the model to be used |
| `-csv` | `submission.csv` | Path of the csv to be generated |
| `-testSize` | 5508 | Number of images to be used for generating test data, must be < 5508 |


## Exploiting Torchnet

[Torchnet](https://lvdmaaten.github.io/publications/papers/Torchnet_2016.pdf) was introduced in second half of June to enable code re-use and making writing code in Torch much more simple. It is basically a well structured implementation of the boilerplate code such as permutation for batches, training for loop and all such things, into a single library. In this project, we have exploited 4 major tools provided by torchnet

1. Datasets
2. Dataset Iterators
3. Engine
4. Meters

### Datasets

```lua
local dataset = tnt.ShuffleDataset{
    dataset = tnt.ListDataset{
        list = torch.range(1,#images):long(),
        load = function(idx)
            return { input = images[idx], target = masks[idx] }
        end,
    },
    size = size
}
```
