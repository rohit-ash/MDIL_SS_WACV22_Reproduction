# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 03:13:05 2023

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

from transform import Relabel, ToLabel, Colorize # modify IDD label ids if saving colour maps. otherwise its fine. 
from iouEval import iouEval, getColorEntry
#from torchsummary import summary

CUDA_LAUNCH_BLOCKING=1

#pass dataset name, get val_loader, criterion with suitable weight

#def main():
NUM_CHANNELS = 3
NUMC_city = 20
NUMC_bdd = 20
NUMC_idd = 27

#device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
image_transform = ToPILImage()
input_transform = Compose([
    Resize([512,1024],Image.BILINEAR),
    ToTensor(),
])
 
target_transform_cityscapes = Compose([
    Resize([512,1024],  Image.NEAREST),
    ToLabel(),
    Relabel(255, NUMC_city-1),   
])

target_transform_IDD = Compose([
    Resize([512,1024],  Image.NEAREST),
    ToLabel(),
    Relabel(255, NUMC_idd-1),  
])

def criterion_fn(data_name='cityscapes'): 
  weight_IDD = torch.tensor([3.235635601598852, 6.76221624390441, 9.458242359884549, 9.446818215454014, 9.947040673126763, 9.789672819856547, 9.476665808564432, 10.465565126694731, 9.59189547383129, 7.637805282159825, 8.990899026692638, 9.26222234098628, 10.265657138809514, 9.386517631614392, 8.357391489170013, 9.910382864314824, 10.389977663948363, 8.997422571963602, 10.418070541191673, 10.483262606962834, 9.511436923349441, 7.597725385711079, 6.1734896019878205, 9.787631041755187, 3.9178330193378708, 4.417448652936843, 10.313160683418731]).cuda()

  weight_BDD = torch.tensor([3.6525147483016243, 8.799815287822142, 4.781908267406055, 10.034828238618045, 9.5567865464289, 9.645099012085169, 10.315292989325766, 10.163473632969513, 4.791692009441432, \
                             9.556915153488912, 4.142994047786311, 10.246903827488143, 10.47145010979545, 6.006704177894196, 9.60620532303246, 9.964959813857726, 10.478333987902301, 10.468010534454706, \
                             10.440929141422366, 3.960822533003462]).cuda()

  weight_city = torch.tensor([2.8159904084894922, 6.9874672455551075, 3.7901719017455604, 9.94305485286704, 9.77037625072462, 9.511470001589007, 10.310780572569994, 10.025305236316246, 4.6341256102158805, \
                            9.561389195953845, 7.869695292372276, 9.518873463871952, 10.374050047877898, 6.662394711556909, 10.26054487392723, 10.28786101490449, 10.289883605859952, 10.405463349170795, \
                            10.138502340710136, 5.131658171724055]).cuda()

  weight_city[-1] = 0
  weight_BDD[-1] = 0
  weight_IDD[-1] = 0
  
  CS_datadir = 'C:/Users/Rohit/Desktop/cityscapes/'
  BDD_datadir = 'C:/Users/Rohit/Desktop/bdd100k/'
  IDD_datadir = 'C:/Users/Rohit/Desktop/IDD_Segmentation/'

  if data_name == 'cityscapes':
        dataset_val = cityscapes(CS_datadir, input_transform,
                      target_transform_cityscapes, 'val')
        weight = weight_city
  elif data_name == 'IDD':
      dataset_val = IDD(IDD_datadir, input_transform,
                       target_transform_IDD, 'val')
      weight = weight_IDD
  elif data_name == 'BDD':
         dataset_val = BDD(BDD_datadir, input_transform,
                           target_transform_cityscapes, 'val')
         weight = weight_BDD

  loader_val = DataLoader(dataset_val, num_workers = 0,
                        batch_size=1, shuffle=False)

#weight = weight.cuda()
  criterion = nn.CrossEntropyLoss(weight=weight,ignore_index = 19).to(device)

  return loader_val, criterion

def eval(model, dataset_loader, criterion, task, num_classes, data_name = 'cityscapes'):
    model.eval()
    epoch_loss_val = []
    num_cls = num_classes[task]
    print(num_cls)
    iouEvalVal = iouEval(num_cls, num_cls-1)
    print(iouEvalVal)


    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(dataset_loader):
        # inputs size: torch.Size([1, 20, 512, 1024])
            start_time = time.time()
            inputs = images.to(device)
            targets = labels.to(device)
            outputs = model(inputs,task)
            outputs = outputs.to(device)
            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data);
         
 
    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()

#     print('check val fn, loss, acc: ', iouVal) 
    
    return iou_classes, iouVal

loader_val_CS, criterion_CS = criterion_fn('cityscapes')
loader_val_BDD, criterion_BDD = criterion_fn('BDD')
loader_val_IDD, criterion_IDD = criterion_fn('IDD')

def eval_finetune(model, dataset_loader, criterion, task, num_classes):
    model.eval()
    epoch_loss_val = []
    num_cls = num_classes[task]

    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(dataset_loader):
            # inputs size: torch.Size([1, 20, 512, 1024])
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()
            if task == 0:
                outputs = model(inputs, decoder_old=True, decoder_new=False)
            elif task == 1:
                outputs = model(inputs, decoder_old=False, decoder_new=True)

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()

    print('check val fn, loss, acc: ', iouVal) 
        
    return iou_classes, iouVal

def eval_finetune3(model, dataset_loader, criterion, task, num_classes=[20, 20, 27]):
    model.eval()
    epoch_loss_val = []
    num_cls = num_classes[task]

    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(dataset_loader):
            # inputs size: torch.Size([1, 20, 512, 1024])
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()
            if task == 0:
                outputs = model(inputs, decoder_old1=True, decoder_old2=False, decoder_new=False)
            elif task == 1:
                outputs = model(inputs, decoder_old1=False, decoder_old2=True, decoder_new=False)
            elif task == 2:
                outputs = model(inputs, decoder_old1=False, decoder_old2=False, decoder_new=True)

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()

    print('check val fn, loss, acc: ', iouVal) 
        
    return iou_classes, iouVal

# Step 2: Finetune CS model on BDD
# model_step2 = ERFNet_ft1(num_classes_old=20, num_classes_new=20)
# model_step2 = torch.nn.DataParallel(model_step2).cuda()
# saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/checkpoint_erfnet_ftp1_150_6_Finetune-CStoBDD-final.pth.tar')
# model_step2.load_state_dict(saved_model['state_dict'])

# iou_classes_step2_FT1_CS, val_acc_step2_FT1_CS = eval_finetune(model_step2, loader_val_CS, criterion_CS, 0, [20, 20])
# print(val_acc_step2_FT1_CS)

# iou_classes_step2_FT1_BDD, val_acc_step2_FT1_BDD = eval_finetune(model_step2, loader_val_BDD, criterion_BDD, 1, [20, 20])
# print(val_acc_step2_FT1_BDD)

# Step 3: Finetune CS|BDD model on IDD
# model_step3 = ERFNet_ft2(20, 20, 27) 
# model_step3 = torch.nn.DataParallel(model_step3).cuda()
# saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/model_best_erfnet_ftp2_150_6_Finetune-code-CSBDDtoIDD-FT.pth.tar')
# model_step3.load_state_dict(saved_model['state_dict'])

# iou_classes_step3_CS, val_acc_step3_CS = eval_finetune3(model_step3, loader_val_CS, criterion_CS, 0, [20, 20, 27])
# print(val_acc_step3_CS)

# iou_classes_step3_BDD, val_acc_step3_BDD = eval_finetune3(model_step3, loader_val_BDD, criterion_BDD, 1, [20, 20, 27])
# print(val_acc_step3_BDD)

# iou_classes_step3_IDD, val_acc_step3_IDD = eval_finetune3(model_step3, loader_val_IDD, criterion_IDD, 2, [20, 20, 27])
# print(val_acc_step3_IDD)

# Step 3: Finetune CS|IDD model on BDD
model_step3 = ERFNet_ft2(20, 27, 20) 
model_step3 = torch.nn.DataParallel(model_step3).cuda()
saved_model = torch.load('C:/Users/Rohit/Desktop/MDIL-SS-Reproduction/Checkpoints/FineTune/model_best_erfnet_ftp2_150_6_FT_CS1_IDD2_BDD3.pth.tar')
model_step3.load_state_dict(saved_model['state_dict'])

iou_classes_step3_CS, val_acc_step3_CS = eval_finetune3(model_step3, loader_val_CS, criterion_CS, 0, [20, 27, 20])
print(val_acc_step3_CS)

iou_classes_step3_IDD, val_acc_step3_IDD = eval_finetune3(model_step3, loader_val_IDD, criterion_IDD, 1, [20, 27, 20])
print(val_acc_step3_IDD)

iou_classes_step3_BDD, val_acc_step3_BDD = eval_finetune3(model_step3, loader_val_BDD, criterion_BDD, 2, [20, 27, 20])
print(val_acc_step3_BDD)
