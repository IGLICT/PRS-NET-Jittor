# PRS-Net (Jittor): Planar Reflective Symmetry Detection Net for 3D Models

## Introduction
This repository is code release for PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models (arXiv pdf [here](https://arxiv.org/pdf/1910.06511.pdf)).

<img src='teaser.png' width='800'>

In geometry processing, symmetry is a universal type of high-level structural information of 3D models and benefits many geometry processing tasks including shape segmentation, alignment, matching, and completion. Thus it is an important problem to analyze various symmetry forms of 3D shapes. Planar reflective symmetry is the most fundamental one. Traditional methods based on spatial sampling can be time-consuming and may not be able to identify all the symmetry planes. In this paper, we present a novel learning framework to automatically discover global planar reflective symmetry of a 3D shape. Our framework trains an unsupervised 3D convolutional neural network to extract global model features and then outputs possible global symmetry parameters, where input shapes are represented using voxels. We introduce a dedicated symmetry distance loss along with a regularization loss to avoid generating duplicated symmetry planes. Our network can also identify generalized cylinders by predicting their rotation axes. We further provide a method to remove invalid and duplicated planes and axes. We demonstrate that our method is able to produce reliable and accurate results. Our neural network based method is hundreds of times faster than the state-of-the-art methods, which are based on sampling. Our method is also robust even with noisy or incomplete input surfaces.

In this repository, we provide PRS-Net model implementation (with Pytorch) as well as data preparation, training and testing scripts on ShapeNet.
## Installation

The code is tested with Ubuntu 18.04, Python 3.8, Jittor 1.2.2.58, CUDA 10.0 and cuDNN v7.5.

Install the following Python dependencies (with `pip install`):
    
    scipy (1.6.0)

For the jittor installation, please refer to [this link](https://cg.cs.tsinghua.edu.cn/jittor/download).
And for running MATLAB code, you need to install [gptoolbox](https://github.com/alecjacobson/gptoolbox).


## Data preprocessing
We use MATLAB to preprocess data. First download [ShapeNetCore.v2](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2/) into `preprocess/shapenet` and then run `preprocess/precomputeShapeData.m`.
## Training
    python train.py --dataroot ./datasets/shapenet --name exp --num_quat 3 --num_plane 3 --batchSize 32 --weight 25
## Inference
    python test.py --dataroot ./datasets/shapenet --name exp --num_quat 3 --num_plane 3

## Acknowledgments
The structure of this code is based on [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), and some MATLAB code is based on [volumetricPrimitives](https://github.com/shubhtuls/volumetricPrimitives).

## Citation

If you find our work useful in your research, please consider citing:

    @ARTICLE{9127500,
        author={L. {Gao} and L. -X. {Zhang} and H. -Y. {Meng} and Y. -H. {Ren} and Y. -K. {Lai} and L. {Kobbelt}},
        title={PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models},
        journal={IEEE Transactions on Visualization and Computer Graphics},
        year = {2020},
        volume = {},
        pages = {1-1},
        number = {},
        doi={10.1109/TVCG.2020.3003823}
    }
    
    @article{hu2020jittor,
    title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
    author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
    journal={Information Sciences},
    volume={63},
    number={222103},
    pages={1--21},
    year={2020}
    }
