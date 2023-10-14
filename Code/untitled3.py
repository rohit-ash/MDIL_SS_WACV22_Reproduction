# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:16:49 2023

@author: Rohit
"""

import numpy as np
from PIL import Image


img_PIL = Image.open(r'C:/Users/Rohit/Desktop/cityscapes/gtFine/val/frankfurt/frankfurt_000000_010763_gtFine_labelIds.png')

np_img = np.asarray(img_PIL)