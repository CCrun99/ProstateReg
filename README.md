# ProstateReg
<div style="width:50px">Official Code for  Weakly Supervised Volumetric Prostate Registration for MRI-TRUS Image Driven by Signed Distance Map</div>

## Introduction
<div align="center">
  <img src="https://github.com/CCrun99/ProstateReg/blob/main/ProstateReg%20Architecture.jpg" style="width:500px">
</div>
<div style="width:50px">
  (1)	We propose a weakly-supervised volumetric MRI-TRUS registration method driven by segmentations and their corresponding SDMs capable of encoding organ segmentations into a higher dimensional space, implicitly capturing structure and contour information.
</div>
<div style="width:50px">
  (2)	We design a mixed DSC-SDM-based loss both robust to segmentation outliers, and optimal in terms of global alignment.
</div>

## Requirements
The packages and their corresponding version we used in this repository are listed in below.
- Python 3.8.5
- Pytorch 1.13.0
- SimpleITK
- Cuda 11.6
- Skimage

## Source
* `train.py`: Main script training the network.
* `predict.py`: Predict the trained model on test data in terms of dice score,hausdorff distance,mean surface distance and jacobian determinant.
* `loss.py`: Contains some losses/regularization functions.
* `SMR12p.pth`: You may use the pretrained model to replicate our results.

## Dataset
We use the dataset, please refer to [public prostate MRI-TRUS biopsy dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661) for details.

## Network
We use the base network, please refer to [monai](https://docs.monai.io/en/latest/data.html) for details.
