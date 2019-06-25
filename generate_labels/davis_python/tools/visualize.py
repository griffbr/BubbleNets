#!/usr/bin/env python

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Visualize sequence annotations.
"""

import cv2
import sys
import argparse

import davis
import skimage.io as io
import prettytable

from davis import DAVISLoader, cfg, phase,log,overlay

def parse_args():
  """Parse input arguments."""

  parser = argparse.ArgumentParser(
      description="Visualize dataset annotations.")

  parser.add_argument(
      '-i','--input',default=None,type=str,
      help='Path to the technique to be visualized')

  parser.add_argument(
      '--year','-y',default=cfg.YEAR,type=str,choices=['2016','2017'])

  parser.add_argument(
      '--phase','-p',default=cfg.PHASE.name,type=str,choices=[e.name.lower()
        for e in davis.phase])

  parser.add_argument('--single-object',action='store_true')

  args = parser.parse_args()

  # Cast string to Enum
  args.phase = davis.phase[args.phase.upper()]

  return args

def imshow(im,an,color_palette):
  """ Display image using cv2 as backend."""

  ov = overlay(im,an,color_palette)
  cv2.imshow("Sequence",ov[...,[2,1,0]])

  ch = chr(cv2.waitKey())
  return ch

if __name__ == '__main__':

  args = parse_args()

  log.info('Loading DAVIS year: {} phase: {}'.format(
    args.year,args.phase))

  db = davis.dataset.DAVISLoader(args.year,
      args.phase,args.single_object)

  if args.input is None:
    # Visualize ground-truth data
    for images,annotations in db.iteritems():
      for im,an in zip(images,annotations):
        ch = imshow(im,an,annotations.color_palette)
        if  ch == 'q':
          sys.exit(0)
        elif ch == 's':
          break # skip to next sequence
