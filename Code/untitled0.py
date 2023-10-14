# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:42:25 2023

@author: Rohit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as T

from dataset_custom import cityscapes, IDD, BDD # original label id gt annotations

from erfnet_RA_parallel import Net as Net_RAP # proposed model
from erfnet import Net as ERFNet_ind # single-task models 
from erfnet_ftp1 import Net as ERFNet_ft1 # 1st stage FT/FE (Eg. CS->BDD)
from erfnet_ftp2 import Net as ERFNet_ft2 # 2nd stage FT/FE (CS|BDD->IDD)

model_step2 = ERFNet_ft1(num_classes_old=20, num_classes_new=20)
model_step2 = torch.nn.DataParallel(model_step2).cuda()
saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/checkpoint_erfnet_ftp1_150_6_Finetune-CStoBDD-final.pth.tar')
model_step2.load_state_dict(saved_model['state_dict'])

model_step3 = ERFNet_ft2(20, 20, 27) 
model_step3 = torch.nn.DataParallel(model_step3).cuda()
saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/model_best_erfnet_ftp2_150_6_Finetune-code-CSBDDtoIDD-FT.pth.tar')
model_step3.load_state_dict(saved_model['state_dict'])

model_step3 = ERFNet_ft2(20, 27, 20) 
model_step3 = torch.nn.DataParallel(model_step3).cuda()
saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/model_best_erfnet_ftp2_150_6_FT_CS1_IDD2_BDD3.pth.tar')
model_step3.load_state_dict(saved_model['state_dict'])

model_step1 = Net_RAP([20], 1, 0) 
model_step1 = torch.nn.DataParallel(model_step1).cuda()
saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/RAP_FT_KLD/step1cs/model_best_cityscapes_erfnet_RA_parallel_150_6RAP_FT_step1.pth.tar')
model_step1.load_state_dict(saved_model['state_dict'])
