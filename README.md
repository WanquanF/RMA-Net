# RMA-Net

This repo is the implementation of the paper: Recurrent Multi-view Alignment Network for Unsupervised Surface Registration (CVPR 2021). 

Paper address: [https://arxiv.org/abs/2011.12104](https://arxiv.org/abs/2011.12104)

Project webpage: [https://wanquanf.github.io/RMA-Net.html](https://wanquanf.github.io/RMA-Net.html)

![avatar](./images/teaser_version3_low.png)

## Prerequisite Installation
The code has been tested with Python 3.8, PyTorch 1.6 and Cuda 10.2:

    conda create --name rmanet
    conda activate rmanet
    conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
    conda install -c conda-forge igl

Other requirements include: [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page), [Openmesh](https://www.graphics.rwth-aachen.de/software/openmesh/) and [MeshlabServer](https://www.meshlab.net/).

Build the cuda extension:

    python build_cuda.py


## Usage

### Pre-trained Models
Download the [pre-trained models](https://wanquanf.github.io/rmanet_pretrained.html) and put the models in the *[YourProjectPath]/pre_trained* folder. 

### Run the registration
To run registration for a single sample, you can run:

    python inference.py --weight [pretrained-weight-path] --src [source-obj-path] --tgt [target-obj-path] --iteration [iteration-number] --device_id [gpu-id] --if_nonrigid [1 or 0]

The last argument *--if_nonrigid* represents if the translation between the source and target is non-rigid (1) or rigid (0). Registration results are listed in the folder named *source_deform_results*, including the deforming results of different stages. We have given a collection of samples in *[YourProjectPath]/samples*, and you can run the registration for them by:
    
    sh inference_samples.sh


### Datasets
The dataset used in our paper can be downloaded [here](https://wanquanf.github.io/rmanet_datasets.html).

Or you can also construct your the dataset that can be used in the code. To show how to construct a dataset that can be used in the code, we give a sample script that constructs a toy dataset that can construct the packed dataset.
Firstly, build the code for ACAP interpolation (you should change the include/lib path in the *[YourProjectPath]/data/sample_data/code_for_converting_seed_to_dataset/vertex2acap/CMakelists.txt*):

    cd [YourProjectPath]/data/sample_data/code_for_converting_seed_to_dataset/vertex2acap
    python build_acap.py

Then, download some [seed data](https://wanquanf.github.io/seed_data.html) into the *[YourProjectPath]/data/sample_data/seed* folder, and then convert the seed data into a packed dataset (you should change the *meshlabserver* path in *[YourProjectPath]/data/sample_data/code_for_converting_seed_to_dataset/sample_points_for_one_mesh.py*):

    cd [YourProjectPath]/data/sample_data/code_for_converting_seed_to_dataset
    python convert_seed_to_dataset.py

For simplicity, you can also directly download the constructed [packed dataset](https://wanquanf.github.io/packed_dataset.html) into *[YourProjectPath]/data/sample_data/code_for_converting_seed_to_dataset/packed_data*.


### Train with the dataset
To train with the constructed dataset:

    cd [YourProjectPath]/model
    python train_sample.py

The settings (the weights of the loss terms, the dataset, etc) of the training process can also be adjusted in the *train_sample.py*. The training results are saved in cd *[YourProjectPath]/model/results*.



## Citation
Please cite this paper with the following bibtex:

    @inproceedings{feng2021recurrent,
        author    = {Wanquan Feng and Juyong Zhang and Hongrui Cai and Haofei Xu and Junhui Hou and Hujun Bao},
        title     = {Recurrent Multi-view Alignment Network for Unsupervised Surface Registration},
        booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2021}
    }

## Acknowledgement
In this repo, we borrowed a lot from [DCP](https://github.com/WangYueFt/dcp) and [Raft](https://github.com/princeton-vl/RAFT).
