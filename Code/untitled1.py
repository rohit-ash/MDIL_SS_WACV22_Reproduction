# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 02:41:53 2023

@author: Rohit
"""
import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset


def is_label_city(filename):
    return filename.endswith("_labelTrainIds.png")


root = 'C:/Users/Rohit/Desktop/cityscapes/'
images_root = os.path.join(root, 'leftImg8bit/')
labels_root = os.path.join(root, 'gtFine/')

images_root += 'val'
labels_root += 'val'

images_root += '/'
labels_root += '/'


# [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
# self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
filenamesGt = [os.path.join(dp+'/', f) for dp, dn, fn in os.walk(
    os.path.expanduser(labels_root)) for f in fn if is_label_city(f)]

for root, dirs, fn in os.walk(os.path.expanduser(labels_root)):
    print(fn)