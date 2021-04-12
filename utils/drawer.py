import os
import numpy as np
import torch

colormap =[ [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [0.9411764705882353, 0.5215686274509804, 0.09803921568627451]
          ]

rb_colormap_list =[ 0,         0,    0.5625,
         0,         0,    0.6250,
         0,         0,    0.6875,
         0,         0,    0.7500,
         0,         0,    0.8125,
         0,         0,    0.8750,
         0,         0,    0.9375,
         0,         0,    1.0000,
         0,    0.0625,    1.0000,
         0,    0.1250,    1.0000,
         0,    0.1875,    1.0000,
         0,    0.2500,    1.0000,
         0,    0.3125,    1.0000,
         0,    0.3750,    1.0000,
         0,    0.4375,    1.0000,
         0,    0.5000,    1.0000,
         0,    0.5625,    1.0000,
         0,    0.6250,    1.0000,
         0,    0.6875,    1.0000,
         0,    0.7500,    1.0000,
         0,    0.8125,    1.0000,
         0,    0.8750,    1.0000,
         0,    0.9375,    1.0000,
         0,    1.0000,    1.0000,
    0.0625,    1.0000,    0.9375,
    0.1250,    1.0000,    0.8750,
    0.1875,    1.0000,    0.8125,
    0.2500,    1.0000,    0.7500,
    0.3125,    1.0000,    0.6875,
    0.3750,    1.0000,    0.6250,
    0.4375,    1.0000,    0.5625,
    0.5000,    1.0000,    0.5000,
    0.5625,    1.0000,    0.4375,
    0.6250,    1.0000,    0.3750,
    0.6875,    1.0000,    0.3125,
    0.7500,    1.0000,    0.2500,
    0.8125,    1.0000,    0.1875,
    0.8750,    1.0000,    0.1250,
    0.9375,    1.0000,    0.0625,
    1.0000,    1.0000,         0,
    1.0000,    0.9375,         0,
    1.0000,    0.8750,         0,
    1.0000,    0.8125,         0,
    1.0000,    0.7500,         0,
    1.0000,    0.6875,         0,
    1.0000,    0.6250,         0,
    1.0000,    0.5625,         0,
    1.0000,    0.5000,         0,
    1.0000,    0.4375,         0,
    1.0000,    0.3750,         0,
    1.0000,    0.3125,         0,
    1.0000,    0.2500,         0,
    1.0000,    0.1875,         0,
    1.0000,    0.1250,         0,
    1.0000,    0.0625,         0,
    1.0000,         0,         0,
    0.9375,         0,         0,
    0.8750,         0,         0,
    0.8125,         0,         0,
    0.7500,         0,         0,
    0.6875,         0,         0,
    0.6250,         0,         0,
    0.5625,         0,         0,
    0.5000,         0,         0]
rb_colormap = np.array(rb_colormap_list).reshape(64,3)

#point : N,3 numpy ndarray, float32
#savepath : string
def render_points(point,savepath):
    f = open(savepath,'w')
    for i in range(point.shape[0]):
        f.write('v '+str(point[i][0])+' '+str(point[i][1])+' '+str(point[i][2])+'\n')
    f.close()

#point : N,3 numpy ndarray, float32
#seg : N, numpy ndarray, int32
#savepath : string
def render_points_with_seg_info(point, seg, savepath):
    f = open(savepath,'w')
    for i in range(point.shape[0]):
#        print(type(seg[i]),seg[i])
        f.write('v '+str(point[i][0])+' '+str(point[i][1])+' '+str(point[i][2])+' '+str(colormap[seg[i]][0])+' '+str(colormap[seg[i]][1])+' '+str(colormap[seg[i]][2])+'\n')
    f.close()
    
def render_points_with_weight(point, weight, savepath):
    f = open(savepath,'w')
    for i in range(point.shape[0]):
#        print(type(seg[i]),seg[i])
        color_level = int(weight[i]*64)
        if color_level<0:
            color_level=0
        if color_level>63:
            color_level=63
        f.write('v '+str(point[i][0])+' '+str(point[i][1])+' '+str(point[i][2])+' '+str(rb_colormap[color_level][0])+' '+str(rb_colormap[color_level][1])+' '+str(rb_colormap[color_level][2])+'\n')
    f.close()
    
def render_points_with_rgb(point, r,g,b, savepath):
    f = open(savepath,'w')
    for i in range(point.shape[0]):
        f.write('v '+str(point[i][0])+' '+str(point[i][1])+' '+str(point[i][2])+' '+str(r)+' '+str(g)+' '+str(b)+'\n')
    f.close()
    
def render_points_with_weights_info(source_p, point_r_, point_t_, point_weight_, weight_name_list):
    patch_num = len(weight_name_list)
    point_r = point_r_.view(1, 3, 3*patch_num)
    point_t = point_t_.view(1, 1, 3, patch_num)
    point = torch.matmul(source_p, point_r).view(1, -1, 3, patch_num) + point_t
    for i in range(len(weight_name_list)):
        point_one_path = point[0,:,:,i].cpu().numpy()
        weight_one_patch = point_weight_[0,i,:].cpu().numpy()
#        print(point_one_path.shape,weight_one_patch.shape)
#        print(point_one_path,weight_one_patch)
        render_points_with_weight(point_one_path, weight_one_patch, weight_name_list[i])
#    exit()
        
