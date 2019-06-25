# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

__author__ = 'federico perazzi'
__version__ = '2.0.0'

from misc import log     # Logger
from misc import cfg     # Configuration parameters
from misc import phase   # Dataset working set (train,val,etc...)
from misc import overlay # Overlay segmentation on top of RGB image
from misc import Timer   # Timing utility class
from misc import io

from dataset import DAVISLoader,Segmentation,Annotation
from dataset import db_eval,db_eval_sequence
from dataset import print_results

