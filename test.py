#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""

import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from featurefusion_network import build_featurefusion_network
from PIL import Image
import os
import torch
import argparse
import json
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from os.path import exists
import torch.optim as optim


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='test', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./layer=5/", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

fusion_network = build_featurefusion_network(256,0.1,8,2048,5)
fusion_network.load_state_dict(torch.load(args.model_path + 'Transformer_best.pth'))
if use_gpu: fusion_network.cuda()
fusion_network.eval()

regressor = CountRegressor(256, pool='mean')
regressor.load_state_dict(torch.load(args.model_path + 'FamNet_best.pth'))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)
for im_id in pbar:
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    rects = list()
    for bbox in bboxes:
        x1, y1 = bbox[0][0], bbox[0][1]
        x2, y2 = bbox[2][0], bbox[2][1]
        rects.append([y1, x1, y2, x2])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image, 'lines_boxes': rects}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    if use_gpu:
        image = image.cuda()
        boxes = boxes.cuda()

    with torch.no_grad(): Image_features = extract_features(resnet50_conv, image.unsqueeze(0))

    if not args.adapt:
        with torch.no_grad():
            N, M = image.unsqueeze(0).shape[0], boxes.unsqueeze(0).shape[2]
            print(N,M)
            for ix in range(0,N):
            # boxes = boxes.squeeze(0)
                boxes = boxes.unsqueeze(0)[ix][0]
                cnter = 0
                Cnter1 = 0
                for keys in ['map4']:
                    image_features = Image_features[keys][ix].unsqueeze(0)
                    if keys == 'map1' or keys == 'map2':
                        Scaling = 4.0
                    elif keys == 'map3':
                        Scaling = 8.0
                    elif keys == 'map4':
                        Scaling =  16.0
                    else:
                        Scaling = 32.0
                    boxes_scaled = boxes / Scaling
                    print(boxes_scaled.shape)
                    boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
                    boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
                    boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
                    feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
                    # make sure exemplars don't go out of bound
                    boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
                    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
                    boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
                    box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
                    box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
                    max_h = math.ceil(max(box_hs))
                    max_w = math.ceil(max(box_ws))        
    
                    for j in range(0,M):
                        y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                        y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                        h, w = image_features.shape[2], image_features.shape[3]
                        #print(y1,y2,x1,x2,max_h,max_w)
                        #if j == 0:
                        examples_features = image_features[:,:,y1:y2, x1:x2]
                        if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                            #examples_features = pad_to_size(examples_features, max_h, max_w)
                            examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
                #else:
                #    feat = image_features[:,:,y1:y2, x1:x2]
                #    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                #        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                #        #feat = pad_to_size(feat, max_h, max_w)
                #    examples_features = torch.cat((examples_features,feat),dim=0)
                        if j == 0:
                            combined = fusion_network(examples_features,image_features)
                            combined = combined.reshape(-1,256,h,w)
                            print("combined.shape:{}".format(combined.shape))
                        else:
                            combined1 = fusion_network(examples_features,image_features)
                            combined1 = combined1.reshape(-1,256,h,w)
                            print("combined1.shape:{}".format(combined1.shape))
                            combined = torch.cat((combined,combined1),dim=0)

                    if cnter == 0:
                        Combined = 1.0 * combined
                    else:
                        if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                            combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                        Combined = torch.cat((Combined,combined),dim=1)
                    cnter += 1
                if ix == 0:
                    features = 1.0 * Combined.unsqueeze(0)
                else:
                    features = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)

        

        #features = fuse_features(image_features, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
            output = regressor(features)
            print(output.shape)
            
    else:
        features.required_grad = True
        adapted_regressor = copy.deepcopy(regressor)
        adapted_regressor.train()
        optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
        for step in range(0, args.gradient_steps):
            optimizer.zero_grad()
            output = adapted_regressor(features)
            lCount = args.weight_mincount * MincountLoss(output, boxes)
            lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
            Loss = lCount + lPerturbation
            # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
            # So Perform gradient descent only for non zero cases
            if torch.is_tensor(Loss):
                Loss.backward()
                optimizer.step()
        features.required_grad = False
        output = adapted_regressor(features)

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")

print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
