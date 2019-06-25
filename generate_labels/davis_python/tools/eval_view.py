#!/usr/bin/env python

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Print technique results.
"""

import os.path as osp
import argparse

import h5py
import yaml
import davis
import numpy as np

from davis import Timer,log,cfg,db_eval,print_results
from easydict import EasyDict as edict

def parse_args():
  """
  Parse input arguments.
  """

  parser = argparse.ArgumentParser(
    description="""Print technique results.""")

  parser.add_argument(
      '-i','--input',required=True,type=str,
      help='Path to the technique results (yaml)')

  args = parser.parse_args()

  return args

if __name__ == '__main__':

  args = parse_args()

  log.info("Loading evaluation from: {}".format(args.input))
  with open(args.input,'r') as f:
    evaluation = edict(yaml.safe_load(f))

  print_results(evaluation)
