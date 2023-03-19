"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan,Udbhav, Thu Nguyen, Minh Hoai

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""
import torch.nn as nn
from model import  Resnet50FPN,CountRegressor,weights_normal_init
from utils import MAPS, Scales, Transform,TransformTrain,extract_features, fuse_features,visualize_output_and_save,pad_to_size
from featurefusion_network import build_featurefusion_network
from PIL import Image
import os
import torch
import argparse
import json
import math
import numpy as np
from tqdm import tqdm
from os.path import exists,join
import random
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave", help="./model")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int,default=1500, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int,default=1, help="GPU id")
parser.add_argument("-lr", "--learning-rate", type=float,default=5e-5, help="learning rate")
args = parser.parse_args()


data_path = args.data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

#fusion_network_path = './Transformer_last.pth'
#regressor_path = './FamNet_last.pth'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

resnet50_conv = Resnet50FPN()
resnet50_conv.cuda()
resnet50_conv.eval()

fusion_network = build_featurefusion_network(256,0.1,8,2048,4)
#fusion_network.load_state_dict(torch.load(fusion_network_path))
fusion_network.cuda()

regressor = CountRegressor(256, pool='mean')
#regressor.load_state_dict(torch.load(regressor_path))
weights_normal_init(regressor, dev=0.001)
regressor.train()
regressor.cuda()

optimizer = optim.Adam(chain(fusion_network.parameters(),regressor.parameters()), lr = args.learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

def train():
    print("Training on FSC147 train set data")
    im_ids = data_split['train']
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')    
        sample = {'image':image,'lines_boxes':rects,'gt_density':density}
        sample = TransformTrain(sample)
        image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()

        with torch.no_grad():
            Image_features = extract_features(resnet50_conv, image.unsqueeze(0))
            
        Image_features.requires_grad = True
        optimizer.zero_grad()


        N, M = image.unsqueeze(0).shape[0], boxes.unsqueeze(0).shape[2] # 1 , 3/4...
        
        for ix in range(0,N):
            
            #boxes = boxes.squeeze(0)
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
    
                h, w = image_features.shape[2], image_features.shape[3]
                for j in range(0,M):
                    y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                    y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                             
                    if j == 0:
                        examples_features = image_features[:,:,y1:y2, x1:x2]                
                        #Shaping examples to same size
                        if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                            examples_features = pad_to_size(examples_features, max_h, max_w)
                            #examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')        
                    else:
                        feat = image_features[:,:,y1:y2, x1:x2]
                   
                        if feat.shape[2] != max_h or feat.shape[3] != max_w:
                            #feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                            feat = pad_to_size(feat, max_h, max_w)
                        examples_features = torch.cat((examples_features,feat),dim=0)
                    
                combined = fusion_network(examples_features,image_features)
                combined = combined.reshape(-1,256,h,w)    
                  
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
            
        #print("features.final.shape:{}".format(features.shape))
        
        output = regressor(features)
        #print("output.shape:{}".format(output.shape))
        

        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)
        
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        pbar.set_description('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f} Best VAL MAE: {:5.2f}, RMSE: {:5.2f}'.format( gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae/cnt, (train_rmse/cnt)**0.5,best_mae,best_rmse))
        print("")
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids))**0.5
    return train_loss,train_mae,train_rmse



   
def eval():
    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors

    print("Evaluation on {} data".format(args.test_split))
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image':image,'lines_boxes':rects}
        sample = Transform(sample)
        image, boxes = sample['image'].cuda(), sample['boxes'].cuda()

        with torch.no_grad():
            Image_features = extract_features(resnet50_conv,image.unsqueeze(0))
        
            N, M = image.unsqueeze(0).shape[0], boxes.unsqueeze(0).shape[2]
            for ix in range(0,N):
                #boxes = boxes.squeeze(0)
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
    
                    h, w = image_features.shape[2], image_features.shape[3]
                    for j in range(0,M):
                        y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                        y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                             
                        if j == 0:
                            examples_features = image_features[:,:,y1:y2, x1:x2]                
                            #Shaping examples to same size
                            if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                                examples_features = pad_to_size(examples_features, max_h, max_w)
                                #examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')        
                        else:
                            feat = image_features[:,:,y1:y2, x1:x2]
                   
                            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                                #feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                                feat = pad_to_size(feat, max_h, max_w)
                            examples_features = torch.cat((examples_features,feat),dim=0)
                    
                    combined = fusion_network(examples_features,image_features)
                    combined = combined.reshape(-1,256,h,w)

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
        
            output = regressor(features)

        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
        print("")

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
    return SAE/cnt, (SSE/cnt)**0.5


best_mae, best_rmse = 1e7, 1e7
stats = list()
for epoch in range(0,args.epochs):
    fusion_network.train()
    regressor.train()
    train_loss,train_mae,train_rmse = train()
    fusion_network.eval()
    regressor.eval()
    val_mae,val_rmse = eval()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(args.output_dir, "stats" +  ".txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))
    
    regressor_model_name_last = args.output_dir + '/' + "FamNet_last.pth"
    fusion_model_name_last = args.output_dir + '/' + "Transformer_last.pth" 
    torch.save(regressor.state_dict(), regressor_model_name_last)
    torch.save(fusion_network.state_dict(), fusion_model_name_last)  
    
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        regressor_model_name = args.output_dir + '/' + "FamNet_best.pth"
        fusion_model_name = args.output_dir + '/' + "Transformer_best.pth"
        torch.save(regressor.state_dict(), regressor_model_name)
        torch.save(fusion_network.state_dict(), fusion_model_name)
    print("Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
              epoch+1,  stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))
    




