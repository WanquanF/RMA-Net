# RMA-Net

This repo is the code of the paper: Recurrent Multi-view Alignment Network for Unsupervised Surface Registration (CVPR 2021).

Paper address: [https://arxiv.org/abs/2011.12104](https://arxiv.org/abs/2011.12104)

Project webpage: [https://wanquanf.github.io/RMA-Net.html](https://wanquanf.github.io/RMA-Net.html)

## Prerequisite Installation
The code has been tested with Python3.8, PyTorch 1.6 and Cuda 10.2:

    conda create --name rmanet
    conda activate rmanet
    conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
    conda install -c conda-forge igl

## Usage

### Pre-trained Models
Download the [pre-trained models](https://none) and put them in the *[YourProjectPath]/pre_trained* folder. 

### Datasets
Download the [datasets](https://none) and put them in the *[YourProjectPath]/data* folder.

### Test on the datasets
To test on our constructed datasets, run:

### How to train on your own dataset
To construct your own dataset, please pack your data into the same format as ours and train with our training script. An example is given here:



## Citation
Please cite this paper with the following bibtex:

    @inproceedings{feng2021recurrent,
        author    = {Wanquan Feng and Juyong Zhang and Hongrui Cai and Haofei Xu and Junhui Hou and Hujun Bao},
        title     = {Recurrent Multi-view Alignment Network for Unsupervised Surface Registration},
        booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2021}
    }

