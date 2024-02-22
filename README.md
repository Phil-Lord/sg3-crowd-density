# Generative Models for the Synthesis and Manipulation of Crowded Scenes
### PyTorch implementation of StyleGAN3 with crowd density estimation

Abstract: _Crowd density and Pedestrian Level of Service (PLOS) are fundamental measures in public safety, events management, and architectural engineering. Being able to visualise specific locations with varying levels of crowd density could aid the organisation of city layouts, events spaces, and buildings, helping prevent crowd crushes and improve overall comfort levels. Visualisation of crowded scenes is a very complicated task due to their complex nature; the immense variation found between crowds makes them difficult to generalise. This task calls for a solution capable of understanding the appearance and behaviour of crowds in order to produce believable images of them. Generative models are a class of statistical models that generate new data instances similar to that on which they were trained. Generative Adversarial Networks (GANs) are a form of generative model with a deep learning based architecture that simultaneously trains a generator and discriminator in a ‘minimax two-player game’. The use of GANs for image synthesis is currently being researched and developed on a large scale, however, there is limited research specific to using these models in PLOS and crowd density manipulation. This project proposes the novel concept of adding accurate crowd density classification in crowd images to the StyleGAN3 architecture, enabling the generation of synthesised crowd images with varying densities. StyleGAN3 is a state-of-the-art GAN which borrows from style transfer methods. The project also includes structured experimentation of GAN models on crowd image data to evaluate the effectiveness of image generation practices, such as data augmentation and transfer learning._

## Introduction
This repository contains a copy of the [official PyTorch StyleGAN3 implementation](https://github.com/NVlabs/stylegan3) with the addition of crowd density estimation methods in [`dataset_tool.py`](dataset_tool.py). These methods enable the calculation of crowd density in [CrowdHuman](https://www.crowdhuman.org/ "CrowdHuman") images for the conditional training of StyleGAN3 models. Another addition is the [`graph_loss.py`](graph_loss.py) script for graphing the loss results of StyleGAN3 models throughout training.

## Requirements
The following requirements are taken from the [official PyTorch StyleGAN3 repository's README](https://github.com/NVlabs/stylegan3/blob/main/README.md):

>* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
>* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
>* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
>* CUDA toolkit 11.1 or later. (Why is a separate CUDA toolkit installation required? See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
>* GCC 7 or later (Linux) or Visual Studio (Windows) compilers. Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
>* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
>  - `conda env create -f environment.yml`
>  - `conda activate stylegan3`
>* Docker users:
>  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
>  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

>The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

>See [Troubleshooting](./docs/troubleshooting.md) for help on common installation and run-time problems.

The crowd density estimation methods in [`dataset_tool.py`](dataset_tool.py) are designed to be used with the CrowdHuman crowd image dataset. Head over to the [CrowdHuman download page](https://www.crowdhuman.org/download.html) and use the Google Drive links to download the training and/or validation zip file(s) and odgt annotation file(s).

## Generating crowd density labels
Once the repository is cloned locally, the Python environment is activated, and the CrowdHuman data is downloaded, you are ready to start generating crowd density labels. This is done using the `dataset_tool.py` with the `--source` option set to the locally stored CrowdHuman zip folder, containing both the images and the annotations file. Use the new `--density` option to select a crowd density estimation method; available methods include:

- `threshold`: Manual Thresholds with Crowd Counting.
- `normalised`: Manual Thresholds with Normalised Crowd Counting.
- `euclidean`: Euclidean Distance.
- `grid`: Grid Density Clustering with Adaptive Cell Sizing.
- `grid-metres`: Grid Density Clustering with m<sup>2</sup> Cell Sizing.
- `kde`: Kernel Density Estimation (KDE) (incomplete)

The example below shows `dataset_tool.py` being used to label a CrowdHuman dataset using the Euclidean Distance method:

```
python dataset_tool.py \
    --source="C:\path\to\crowdhuman\dataset.zip" \
    --dest="C:\path\to\output\destination.zip" \
    --resolution=256x256 --transform=center-crop --density=grid
```

## Conditional training with labelled datasets
Once the labelled dataset has been generated, its time to perform conditional StyleGAN3 training. No adjustments have been made to this part of the process, just point to a labelled CrowdHuman dataset with `--data`, select how many epochs with `--kimg` (currently 2,000), and use the other options in the example below.

```
python train.py \
        --outdir=C:\path\to\training-runs --cfg=stylegan3-r \
        --data=C:\path\to\crowdhuman_labelled.zip \
        --gpus=1 --batch=32 --batch-gpu=8 --gamma=2 \
        --kimg=2000 --snap=10 --cbase=16384 --cond=true
```

## Graphing loss
During training, the `--outdir` folder populates a file called `X.jsonl` containing various pieces of data from the run. The `graph_loss.py` script can be used to graph the loss of a run using by pointing to the `X.jsonl` file using the `folder_path` variable. This script even allows for plotting loss from multiple runs on one graph, such as when performing transfer learning. To do this, simply make sure all `X.jsonl` files are in the `folder_path` target folder and use the `lines_to_skip` variable to smooth spikes between files.
