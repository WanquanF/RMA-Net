import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import sys
import time
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import cv2 as cv
from PIL import Image
import random


import utils.drawer as drawer
#from data.FAUST.code_for_visualization.point_render import rb_colormap ,render_points
#from chamfer_distance import ChamferDistance
#chamfer_dist = ChamferDistance()

from model.exfunc.functions.point_render_func import Render
from model.exfunc.functions.point_masker_func import Masker

import glob
import trimesh

import model.ptlk as ptlk

torch.pi = torch.acos(torch.zeros(1)).item() * 2



resolution_ = 256
point_renderer_ = Render(resolution_, resolution_, 7, 9, 1e-5) # this is an object of point renderer
point_mask_ = Masker(resolution_, resolution_, 5, 1e-5)
threshold_ = 0.1
threshold_2_ = 1e-10
'''
start = time.clock()
elapsed = (time.clock() - start)
print("Time used:",elapsed)
'''


def compute_sqrdis_map(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M * N
    '''
    thisbatchsize = points_x.size()[0]
    pn_x = points_x.size()[1]
    pn_y = points_y.size()[1]
    x_sqr = torch.sum(torch.mul(points_x, points_x), dim=-1).view(thisbatchsize, pn_x, 1).expand(-1,-1,pn_y)
    y_sqr = torch.sum(torch.mul(points_y, points_y), dim=-1).view(thisbatchsize, 1, pn_y).expand(-1,pn_x,-1)
    inner = torch.bmm(points_x, points_y.transpose(1,2))
    sqrdis = x_sqr + y_sqr - 2*inner
    return sqrdis

def chamfer_dist(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M, batchsize * N
    '''
    thisbatchsize = points_x.size()[0]
    sqrdis = compute_sqrdis_map(points_x, points_y)
    dist1 = sqrdis.min(dim = 2)[0].view(thisbatchsize,-1)
    dist2 = sqrdis.min(dim = 1)[0].view(thisbatchsize,-1)
    return dist1, dist2
    
    
def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))



def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        
        
        self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        self.trans_mask = torch.zeros([1,4,4],dtype=torch.float32,device="cuda:0",requires_grad=False)
        self.trans_mask[:, 0:3, 3]+=1.0
        view_divide = args.view_divide
        self.all_rot_matrix_for_lfd = torch.zeros([view_divide**2,3,3], device="cuda:0")
        for view_id in range(view_divide**2):
            euler_x = view_id//(view_divide)
            euler_y = (view_id - euler_x * (view_divide))
            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*euler_x/view_divide, -torch.pi+2*torch.pi*euler_y/view_divide, 0]], dtype=torch.float32).to("cuda:0")
            self.all_rot_matrix_for_lfd[view_id] = euler2rot(euler_angle)
        self.all_rot_matrix_for_lfd = self.all_rot_matrix_for_lfd.view(-1,3)
        
        
        
    def loss_on_depth(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
        ### reset the views
        view_divide = self.args.view_divide
        self.all_rot_matrix_for_lfd = torch.zeros([view_divide**2,3,3], device="cuda:0")
        for view_id in range(view_divide**2):
            euler_x = view_id//(view_divide)
            euler_y = (view_id - euler_x * (view_divide))
            random_x = random.random()
            random_y = random.random()
            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*(euler_x+random_x)/view_divide, -torch.pi+2*torch.pi*(euler_y+random_y)/view_divide, 0]], dtype=torch.float32).to("cuda:0")
            self.all_rot_matrix_for_lfd[view_id] = euler2rot(euler_angle)
        self.all_rot_matrix_for_lfd = self.all_rot_matrix_for_lfd.view(-1,3)
        ### reset the views done.
        deformation_p_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2) ).transpose(1,2)
        p1_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        for view_id in range(self.args.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
            proj_ori_vertex = torch.cat(((proj_ori_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_ori_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
            proj_tar_vertex = torch.cat(((proj_tar_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_tar_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            ori_depth_img, ori_weight_img, _ = point_renderer_(proj_ori_vertex.contiguous(), threshold_)
            ori_img = ori_depth_img / ori_weight_img
            tgt_depth_img, tgt_weight_img, _ = point_renderer_(proj_tar_vertex.contiguous(), threshold_)
            tar_img = tgt_depth_img / tgt_weight_img
            mask = torch.sign(ori_img*tar_img).detach()
            depth_dis_abs =  torch.abs(ori_img-tar_img)
            depth_dis_thres =  -1.0 * (depth_dis_abs.detach() - 0.05)
            mask2 = (torch.sign(torch.sign(depth_dis_thres).detach()+0.5)+1.0)*0.5
            mask = torch.mul(mask2, mask).detach()
            depth_dis_sqr = torch.mul(depth_dis_abs,depth_dis_abs)
            output += torch.sum(torch.mul(mask,depth_dis_sqr))/(self.args.view_divide**2)
        return output

            
    def loss_on_mask(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
        deformation_p_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2) ).transpose(1,2)
        p1_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        for view_id in range(self.args.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
            proj_ori_vertex = (proj_ori_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
            proj_tar_vertex = (proj_tar_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
            ori_mask = point_mask_(proj_ori_vertex.contiguous(), threshold_2_)
            tgt_mask = point_mask_(proj_tar_vertex.contiguous(), threshold_2_)
            output += torch.mean(torch.abs(ori_mask - tgt_mask))/(self.args.view_divide**2)
        return output

        
        
    def loss_on_cd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
        dist1, dist2 = chamfer_dist(deformation_p, p1)
        output += (torch.sum(dist1) + torch.sum(dist2))*0.5
        return output/thisbatchsize
        
    #This is actually the point-to-point dense correspondence L2 term                
    def loss_on_pd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
        dist = deformation_p-p1 
        output += torch.mul(dist,dist)
        return torch.sum(output)/thisbatchsize
    
        
        

    def loss_on_sparse(self, point_weight):
        thisbatchsize = point_weight.size()[0]
        weight_squ_difference = torch.mean(torch.abs(point_weight))
        return weight_squ_difference

        
        
    def loss_on_tran(self, rigid_matrix):
        thisbatchsize = rigid_matrix.size()[0]
        output = 0
        trans_ = torch.mul(self.trans_mask.expand(thisbatchsize,-1,-1), rigid_matrix)
        output += torch.sum(torch.mul(trans_, trans_))/thisbatchsize
        
        return output
        
    def loss_on_arap(self, neighbour_indexes, deformation_p, source_points):
        thisbatchsize = deformation_p.size()[0]
        deformation_neibour_points_ = indexing_neighbor(deformation_p, neighbour_indexes)
        source_neibour_points_ = indexing_neighbor(source_points, neighbour_indexes)
        deformation_neibour_dis_ = deformation_neibour_points_ - deformation_p.unsqueeze(2)
        source_neibour_dis_ = source_neibour_points_ - source_points.unsqueeze(2)
        deformation_neibour_dis_ = torch.sqrt(torch.mul(deformation_neibour_dis_, deformation_neibour_dis_).sum(dim =-1)+0.00001)
        source_neibour_dis_ = torch.sqrt(torch.mul(source_neibour_dis_, source_neibour_dis_).sum(dim =-1)+0.00001)
        difference = deformation_neibour_dis_ - source_neibour_dis_
        squ_difference = torch.sum(torch.mul(difference, difference))
        return squ_difference/thisbatchsize



        
    
    def forward(self, phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list, source_points, target_points):
        thisbatchsize = phi_list[0].size()[0]
        iter_time = len(phi_list)
        
        neighbour_indexes_ = get_neighbor_index(source_points, self.args.neighbour_num)
        
        zero_tensor = torch.mean(torch.zeros((1),dtype = torch.float, device=phi_list[0].device))
        
        loss = torch.mean(torch.zeros((1),dtype = torch.float, device=phi_list[0].device))
        loss_stages=[]
        
        if self.args.weight_cd > 0:
            # L^{cd}
            loss_cd = 0 
            for i in range(iter_time):
                loss_cd += self.args.gamma**(iter_time-i-1) * self.loss_on_cd(deformation_points_list[i].transpose(1,2), target_points)
#            loss_cd += self.loss_on_cd(deformation_points_list[iter_time-1].transpose(1,2), target_points)
            loss += loss_cd * self.args.weight_cd
            loss_stages.append(loss_cd)
        else:
            loss_stages.append(zero_tensor)
              
        
        if self.args.weight_pd > 0:
            # L^{pd}
            loss_pd = 0 
            for i in range(iter_time):
                if i<0:
#                print(i,self.args.gamma**(iter_time-i-1))
                    loss_pd += zero_tensor
                else:
                    loss_pd += self.args.gamma**(iter_time-i-1) * self.loss_on_pd(deformation_points_list[i].transpose(1,2), target_points)
            loss += loss_pd * self.args.weight_pd
            loss_stages.append(loss_pd)
        else:
            loss_stages.append(zero_tensor)
#        exit()
            
            
        
        if self.args.weight_sparse >0:
            # L^{sparse}
            loss_sparse = 0
            for i in range(iter_time):
                loss_sparse += self.args.gamma**(iter_time-i-1) * self.loss_on_sparse(point_weight_list[i])
#            loss_sparse = self.loss_on_sparse(point_weight, p0)
            loss += loss_sparse * self.args.weight_sparse 
            loss_stages.append(loss_sparse)
        else:
            loss_stages.append(zero_tensor)
            
        if self.args.weight_depth >0:
            # L^{depth}
            loss_depth = 0
            for i in range(iter_time):
                loss_depth += self.args.gamma**(iter_time-i-1) * self.loss_on_depth(deformation_points_list[i].transpose(1,2), target_points)
            loss += loss_depth * self.args.weight_depth 
            loss_stages.append(loss_depth)
        else:
            loss_stages.append(zero_tensor)
            
                        
        if self.args.weight_tran >0:
            # L^{tran}
            loss_tran = 0
            for i in range(iter_time):
                loss_tran += self.args.gamma**(iter_time-i-1) * self.loss_on_tran(rigid_matrix_list[i])
            loss += loss_tran * self.args.weight_tran 
            loss_stages.append(loss_tran)
        else:
            loss_stages.append(zero_tensor)
                                    
        if self.args.weight_mask >0:
            # L^{mask}
            loss_mask = 0
            for i in range(iter_time):
                loss_mask += self.args.gamma**(iter_time-i-1) * self.loss_on_mask(deformation_points_list[i].transpose(1,2), target_points)
#            loss_mask = self.loss_on_mask(point_weight, p0)
            loss += loss_mask * self.args.weight_mask 
            loss_stages.append(loss_mask)
        else:
            loss_stages.append(zero_tensor)
            
        
        if self.args.weight_arap >0:
            # L^{arap}
            loss_arap = 0
            for i in range(iter_time):
                if i==0 or i==1:
                    loss_arap += 10000 * self.args.gamma**(iter_time-i-1) * self.loss_on_arap(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
                elif i==2 or i==3:
                    loss_arap += 1000 * self.args.gamma**(iter_time-i-1) * self.loss_on_arap(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
                else:
                    loss_arap += 100 * self.args.gamma**(iter_time-i-1) * self.loss_on_arap(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
            loss += loss_arap * self.args.weight_arap 
            loss_stages.append(loss_arap)
        else:
            loss_stages.append(zero_tensor)
            

            
            
            
            
            
        
        return loss, loss_stages

