# Code for Generating BubbleNets Training Labels

## Overview
generate_labels is provided to enable users to quickly develop their own training labels for BubbleNets.
Previously generated labels for the entire DAVIS 2016 dataset are provided in ``./labels/DAVIS16_labels.pk``.

Please cite our paper if you find it useful for your research.
```
@inproceedings{GrCoCVPR19,
  author = {Griffin, Brent A. and Corso, Jason J.},
  booktitle={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames},
  year = {2019}
}
```

### Demo
Just run ``./train_osvos_models.py`` with TensorFlow sourced.
This will train all of the OSVOS segmentation models used in the paper.
Results will be dated and added to the ``./results`` folder.

__Generating Training Labels:__ We generate a training label y for each frame, which represents the segmentation performance associated with selecting that frame for annotation.
![alt text](https://github.com/griffbr/BubbleNets/blob/master/figure/label_generation.jpg "Generating Training Labels")
<br />

### Setup
Add new data to ``./source_data/`` folder following the examples already provided.
Each video folder in source_data will be used to generate new frame-specific performance labels for training BubbleNets.
Generated labels are saved in ``./labels/``.

Each unique video folder contains source images in the ``src`` folder and ground truth annotation masks in the ``usrAnnotate`` folder.
The mask should have the same name as its corresponding source image (e.g., 00054.png).

The image-trained process (currently OSVOS segmentation) and evaluation (currently mean Jaccard measure and contour accuracy) can both be changed in lines 59-63 in ``./gen_seg_performance_labels.py``.
Changing how labels are generated will, in effect, change the purpose of the frames that BubbleNets selects after training on those labels.

Currently, the segmentation network will train on each annotation for 500 iterations.
To change this, edit line 24 (``TRAIN_ITER = 500``) in  ``./gen_seg_performance_labels.py``.

### Execution Process
Run ``./gen_seg_performance_labels.py`` [native Python, requires TensorFlow].<br />
Model and segmentation files generated in ``./temp_results`` can be deleted to save disk space.<br />
Newly generated labels are saved in ``./labels``.

## Included External Files

F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung.<br />
A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation, Computer Vision and Pattern Recognition (CVPR), 2016.<br />
Video Object Segmentation Evaluation. <br />
https://github.com/davisvideochallenge/davis

## Use

This code is available for non-commercial research purposes only.