# BubbleNets (in progress)
Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames

Contact: Brent Griffin (griffb at umich dot edu)

## Publication
[BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames](https://arxiv.org/abs/1903.08336 "arXiv Paper")<br />
[Brent A. Griffin](https://www.griffb.com) and [Jason J. Corso](http://web.eecs.umich.edu/~jjcorso/)<br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

Please cite our paper if you find it useful for your research.
```
@inproceedings{GrCoCVPR19,
  author = {Griffin, Brent A. and Corso, Jason J.},
  booktitle={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames},
  year = {2019}
}
```

## Method

__Video Description:__ https://youtu.be/hlog5FV9RLs

[![IMAGE ALT TEXT HERE](https://img.https://youtu.be/0kNmm8SBnnU/0.jpg)](https://youtu.be/0kNmm8SBnnU)




## Setup

Download [resnet_v2_50.ckpt](https://www.dropbox.com/s/gn5uvc6foz10lab/resnet_v2_50.ckpt?dl=0) and add to ``./methods/annotate_suggest/ResNet/``.

Add new data to ``./data/rawData/`` folder following the examples already provided.
Each folder in rawData will be used to train a separate segmentation model using the corresponding annotated training data.
Remove folders from rawData if you do not need to train a new model for them.

## Execution Process

Run video_annotate_marshalv2_9.py       [native Python, has scikit dependency, requires TensorFlow]
										Uses automatic annotation frame selection with previous GrabCut-based annotation tool.

Run videoSegmentation_Marshalv2_9.py    [native Python, requires TensorFlow]
										New multi-annotation segmentation tool.


## Included External Files

S. Caelles*, K.K. Maninis*, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, and L. Van Gool
One-Shot Video Object Segmentation, Computer Vision and Pattern Recognition (CVPR), 2017.<br />
Video Object Segmentation. <br />
https://github.com/scaelles/OSVOS-TensorFlow

## Use

This code is available for non-commercial research purposes only.