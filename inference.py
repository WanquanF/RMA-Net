import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init

import math
import numpy as np
import igl
import struct
import os
import sys
import time

#sys.path.append('./code')
import utils.drawer as drawer 
from model.network import Net_PointNR_v2, Net_PointRR_v2, args
from model.loss import Loss, chamfer_dist

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

def save_point_with_RGB(point_tensor, save_path, color_r, color_g, color_b):
    point_np = point_tensor.cpu().numpy().reshape(-1,3)
    fs = open(save_path,'w')
    for vid in range(point_np.shape[0]):
        fs.write('v '+str(point_np[vid][0])+' '+str(point_np[vid][1])+' '+str(point_np[vid][2])+' '+str(color_r)+' '+str(color_g)+' '+str(color_b)+'\n')
    fs.close()

print('\n')
print('##############################################')
print('GPU id: ',args.device_id)
print('Iteration: ',args.iteration)
print('The pre-trained weights: ',args.weight)
print('Source object: ',args.src)
print('Target object: ',args.tgt)
if args.if_nonrigid==0:
    print('The translation is rigid.')
else:
    print('The translation is non-rigid.')

if __name__=='__main__':
    if args.if_nonrigid==1:
        rma_net= Net_PointNR_v2().cuda()
        # Load the pre-trained weights of RMA-Net
        rma_net.load_state_dict(torch.load(args.weight),True)
        # The testing samples
        source_points, _ = igl.read_triangle_mesh(args.src)
        target_points, _ = igl.read_triangle_mesh(args.tgt)
        source_points_tensor = torch.from_numpy(source_points).float().cuda().view(1,-1,3)
        target_points_tensor = torch.from_numpy(target_points).float().cuda().view(1,-1,3)
        print('Start to deform '+args.src+' to '+args.tgt)
        with torch.no_grad():
            phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list = rma_net(source_points_tensor, target_points_tensor, iteration = args.iteration)
        results_path = args.src[:-4]+'_deform_results'
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        for stage in range(args.iteration):
            stage_result_path = results_path+'/stage_'+str(stage+1)+'.obj'
            # save the results of each stage
            save_point_with_RGB(deformation_points_list[stage].transpose(1,2),stage_result_path,1,0.549,0)
        print('Finished. The results are saved in the path: '+results_path)
        print('##############################################')
        print('\n')
    else:
        rma_net= Net_PointRR_v2().cuda()
        # Load the pre-trained weights of RMA-Net
        rma_net.load_state_dict(torch.load(args.weight),True)
        # The testing samples
        source_points, _ = igl.read_triangle_mesh(args.src)
        target_points, _ = igl.read_triangle_mesh(args.tgt)
        source_points_tensor = torch.from_numpy(source_points).float().cuda().view(1,-1,3)
        target_points_tensor = torch.from_numpy(target_points).float().cuda().view(1,-1,3)
        print('Start to deform '+args.src+' to '+args.tgt)
        with torch.no_grad(): 
            rigid_matrix_list, deformation_points_list, rigid_matrix_accumulation_list = rma_net(source_points_tensor, target_points_tensor, iteration = args.iteration)
        results_path = args.src[:-4]+'_deform_results'
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        for stage in range(args.iteration):
            stage_result_path = results_path+'/stage_'+str(stage+1)+'.obj'
            # save the results of each stage
            save_point_with_RGB(deformation_points_list[stage].transpose(1,2),stage_result_path,1,0.549,0)
        print('Finished. The results are saved in the path: '+results_path)
        print('##############################################')
        print('\n')
