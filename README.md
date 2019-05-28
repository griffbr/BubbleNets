# BubbleNets
Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames

Contact: Brent Griffin (griffb at umich dot edu)

## Publication
[BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames](http://arxiv.org/abs/1903.11779 "arXiv Paper")<br />
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

__CVPR 2019 Oral Presentation:__ https://youtu.be/XBEMuFVC2lg

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XBEMuFVC2lg/0.jpg)](https://youtu.be/XBEMuFVC2lg)

__Qualitative Comparison on DAVIS 2017 Validation Set:__ Segmentations from different annotation frame selection strategies.
![alt text](https://github.com/griffbr/BubbleNets/blob/master/figure/qual_compare.jpg "Qualitative Comparison of Frame Selection Strategies")
<br />

__BubbleNets Framework:__ Deep sorting compares and swaps adjacent frames using their predicted relative performance.
![alt text](https://github.com/griffbr/BubbleNets/blob/master/figure/bubblenets.jpg "BubbleNets Framework")
<br />

## Setup

Download [resnet_v2_50.ckpt](https://www.dropbox.com/s/gn5uvc6foz10lab/resnet_v2_50.ckpt?dl=0) and add to ``./methods/annotate_suggest/ResNet/``.

Add new data to ``./data/rawData/`` folder following the examples provided.
``scooter-black`` is an example with BubbleNets and annotation already complete, and ``soapbox`` is a completely unprocessed folder example.
Each folder in rawData will be used to train a separate segmentation model using the corresponding annotated training data.
Remove folders from rawData if you do not need to train a new model for them.

## Execution Process

__Run__ ``./bubblenets_select_frame.py`` <br />
Uses automatic BubbleNets annotation frame selection with GrabCut-based user annotation tool. BubbleNet selects are stored in a text file (e.g., ``./rawData/scooter-black/frame_selection/BN0.txt``), so using another annotation tool is also possible. <br />
[native Python, has scikit dependency, requires TensorFlow]

__Run__ ``./osvos_segment_video.py`` <br />
Runs OSVOS segmentation given user-provided annotated training frames. Trained OSVOS models are stored in ``./data/models/``. Results are timestamped and will appear in the ``./results/`` folder. <br />
[native Python, requires TensorFlow]

## Included External Files

S. Caelles*, K.K. Maninis*, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, and L. Van Gool. <br />
One-Shot Video Object Segmentation, Computer Vision and Pattern Recognition (CVPR), 2017.<br />
Video Object Segmentation. <br />
https://github.com/scaelles/OSVOS-TensorFlow

K. He, X. Zhang, S. Ren and J. Sun. <br />
Deep Residual Learning for Image Recognition, Computer Vision and Pattern Recognition (CVPR), 2016. <br />
Image preprocessing. <br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py

## Use

This code is available for non-commercial research purposes only.
