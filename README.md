# 50.039-CV-Project
50.039 Computer Vision Project

Sketch Image Classification with Shape-bias Computer Vision

# Background
With the growing popularity of touch-interface devices, more people have begun using simple sketches to communicate emotions and ideas. Sketches are more simplistic than actual photos as they are often abstractions of complex objects in real life. Sketches emphasise the overall shape language of objects, which makes it a suitable subject in trying to reduce texture-bias and focus more on the shape-bias. An image classifier able to detect the subject(s) of a hand-drawn image has many wide applications in today’s world, such as communication aids, novel dataset generation or even games.

Hand-drawn sketch recognition remains a difficult task, owing to the sketches' extremely abstract and symbolic features. Furthermore, with individual variance in skill, the same object may have vastly different shapes and degrees of abstraction. The subject of drawings may also break free from the realm of reality, depicting fantastical concepts such as magic, science fiction, and monsters. This poses an interesting challenge as compared to conventional image classification, which generally seeks to identify objects from our everyday environments.

# Introduction 
This project aims to detect the subject of various hand-drawn images and classify them using a deep neural network approach. 
There have been multiple studies regarding sketch image classification, detailing various methods and neural network architectures. In this project, we will focus on testing and comparing the methods outlined in [The Origins and
Prevalence of Texture Bias in Convolutional Neural Networks, Hermann et. al](https://proceedings.neurips.cc/paper/2020/hash/db5f9f42a7157abe65bb145000b5871a-Abstract.html)

The paper suggests that many CNNs tend to classify images based on texture information rather than shape, a texture-based approach as opposed to the shape-based approach that resonates to how humans identify images. This makes the effect of data augmentation on images much larger as they affect the texture and shape biases which determine how a neural network identifies features of an image. The paper proposes training models that can classify ambiguous images by shape by taking less aggressive random crops during training and applying simple, naturalistic augmentations such as distortion of colour and blurring.

In our implementation, we will use a combination of open source sketch datasets such as the ImageNet dataset, as well as our own hand-drawn images. Links to the various datasets we may use can be found below:

1. [HaohanWang/ImageNet-Sketch: ImageNet-Sketch data set for evaluating model's
ability in learning (out-of-domain) semantics at ImageNet scale (github.com)](https://github.com/HaohanWang/ImageNet-Sketch)
2. https://github.com/googlecreativelab/quickdraw-dataset
3. http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

# Dependencies

For the models:
>python==3.6 and above
>
>pytorch==
>
>opencv-python==3.4.17

For downloading the datasets:
>gsutil

# Downloading the Datasets and Directory Structure
For this project, since the datasets are large, please download the datasets into your project directory.

## Setting up the Directory Structure
For standardization, please ensure that your directory structure is as follows:

```
50.039-CV-PROJECT
    Datasets
        Cybertron
        ImageNet-Sketch
        Quick-Draw
```

Do take note that these datasets are quite large and will take a while to download.

## Downloading ImageNet-Sketch
Download the data set from this [google drive](https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view) or [kaggle](https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view)

Unzip the data and put it in the ImageNet-Sketch folder.

## Downloading Cybertron Dataset
Download the png version of the dataset [here](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip)

Unzip the file and put it in the Cybertron folder.

## Downloading the Google Quick!Draw Dataset
You will need to install gsutil to get the dataset from Google.
If using a conda environment, run `conda install -c conda-forge gsutil`

To install, use the following commands:
```
cd Datasets
cd Quick-Draw
gsutil -m cp "gs://quickdraw_dataset/full/numpy_bitmap/*.npy" .

```
Do make sure to cd into the Quick-Draw folder first otherwise you're gonna have alot of objects in the Dataset folder (not fun).

# Usage
