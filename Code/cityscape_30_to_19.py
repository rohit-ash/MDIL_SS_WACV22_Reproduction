# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 00:31:27 2023

@author: Rohit
"""
import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset


NUM_CHANNELS = 3
NUMC_city = 20
NUMC_bdd = 20
NUMC_idd = 27

mapping_20 = { 0: 19,1: 19,2: 19,3: 19, 4: 19,5: 19,6: 19,7: 0,8: 1,9: 19,10: 19, 11: 2,12: 3,13: 4,14: 19,15: 19, 16: 19, 17: 5,18: 19, 19: 6, 20: 7,21: 8, 22: 9,23: 10,24: 11, 25: 12,26: 13,27: 14,28: 15, 29: 19,30: 19,31: 16, 32: 17, 33: 18,-1: 19}

def is_label_city(filename):
    return filename.endswith("_labelIds.png")

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def load_image(file):
    return Image.open(file)

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask

root = 'C:/Users/Rohit/Desktop/cityscapes/'
images_root = os.path.join(root, 'leftImg8bit/')
labels_root = os.path.join(root, 'gtFine/')

images_root += 'val'
labels_root += 'val'

images_root += '/'
labels_root += '/'


# [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
# self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
filenamesGt = [os.path.join(dp+'/', f) for dp, dn, fn in os.walk(os.path.expanduser(labels_root)) for f in fn if is_label_city(f)]

for filenameGt in filenamesGt:
    with open(image_path_city(labels_root, filenameGt), 'rb') as f:
        label = load_image(f).convert('P')
        label = np.array(label)
        label = encode_labels(label)
        label = Image.fromarray(label)
        label.save(filenameGt)