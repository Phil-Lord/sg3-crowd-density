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


## Getting started
- Clone or download repo
- Download CrowdHuman (link)
- Switch to environment from file

## Generating crowd density labels
- Explain command
- List methods
- Command examples

## Conditional training with labelled datasets
Run command for training with labels option.

## Graphing loss
Generate loss graphs of training runs using the script.
