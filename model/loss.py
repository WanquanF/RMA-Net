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


#from emd import earth_mover_distance

import utils.drawer as drawer
#from data.FAUST.code_for_visualization.point_render import rb_colormap ,render_points
#from chamfer_distance import ChamferDistance

from model.exfunc.functions.point_render_func import Render
from model.exfunc.functions.point_masker_func import Masker

import glob
import trimesh

import model.ptlk as ptlk
#from network import Net_PointNR_seg

torch.pi = torch.acos(torch.zeros(1)).item() * 2

#chamfer_dist = ChamferDistance()


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
        
#        self.pointnet_seg_pretrained = Net_PointNR_seg().cuda()
#        
#        self.pointnet_seg_pretrained.load_state_dict(torch.load('/WanquanMobile/code_about_points/pnrr_0917/model/pnrr_TrainTest_Faust8192/outseg_baseline_1/sample_100000.pt'),True)
#        for param in self.pointnet_seg_pretrained.parameters():
#            param.requires_grad_(False)
        #print("Init ok.")
        
        view_divide = args.view_divide
        
#        ftest = open('test_rot.obj','w')
#        test_point = torch.zeros([3], dtype=torch.float32).to("cuda:0")
#        test_point+=1
        self.all_rot_matrix_for_lfd = torch.zeros([view_divide**2,3,3], device="cuda:0")
        for view_id in range(view_divide**2):
            euler_x = view_id//(view_divide)
            euler_y = (view_id - euler_x * (view_divide))
#            euler_z = view_id - euler_x * (view_divide**2) - euler_y * view_divide
#            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*euler_x/view_divide, -torch.pi+2*torch.pi*euler_y/view_divide, -torch.pi+2*torch.pi*euler_z/view_divide]], dtype=torch.float32).to("cuda:0")
            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*euler_x/view_divide, -torch.pi+2*torch.pi*euler_y/view_divide, 0]], dtype=torch.float32).to("cuda:0")
            self.all_rot_matrix_for_lfd[view_id] = euler2rot(euler_angle)
#            a_rot_test_point = torch.matmul(self.all_rot_matrix_for_lfd[view_id], test_point)
#            print(euler_x, euler_y, euler_z)
#            print(view_id, a_rot_test_point)
#            ftest.write('v '+str(a_rot_test_point[0].item())+' '+str(a_rot_test_point[1].item())+' '+str(a_rot_test_point[2].item())+'\n')
#            print(euler_x,euler_y,euler_z,self.all_rot_matrix_for_lfd[view_id])
        
        self.all_rot_matrix_for_lfd = self.all_rot_matrix_for_lfd.view(-1,3)
        
        
        
    def loss_on_lfd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
#        print(deformation_p.size())
#        print(p1.size())

        ### reset the views
        view_divide = self.args.view_divide
        
#        ftest = open('test_rot.obj','w')
#        test_point = torch.zeros([3], dtype=torch.float32).to("cuda:0")
#        test_point+=1
        self.all_rot_matrix_for_lfd = torch.zeros([view_divide**2,3,3], device="cuda:0")
        for view_id in range(view_divide**2):
            euler_x = view_id//(view_divide)
            euler_y = (view_id - euler_x * (view_divide))
#            euler_z = view_id - euler_x * (view_divide**2) - euler_y * view_divide
#            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*euler_x/view_divide, -torch.pi+2*torch.pi*euler_y/view_divide, -torch.pi+2*torch.pi*euler_z/view_divide]], dtype=torch.float32).to("cuda:0")
            random_x = random.random()
            random_y = random.random()
            euler_angle = torch.tensor([[-torch.pi+2*torch.pi*(euler_x+random_x)/view_divide, -torch.pi+2*torch.pi*(euler_y+random_y)/view_divide, 0]], dtype=torch.float32).to("cuda:0")
            self.all_rot_matrix_for_lfd[view_id] = euler2rot(euler_angle)
#            a_rot_test_point = torch.matmul(self.all_rot_matrix_for_lfd[view_id], test_point)
#            print(euler_x, euler_y, euler_z)
#            print(view_id, a_rot_test_point)
#            ftest.write('v '+str(a_rot_test_point[0].item())+' '+str(a_rot_test_point[1].item())+' '+str(a_rot_test_point[2].item())+'\n')
#            print(euler_x,euler_y,euler_z,self.all_rot_matrix_for_lfd[view_id])
        
        self.all_rot_matrix_for_lfd = self.all_rot_matrix_for_lfd.view(-1,3)
        ### reset the views done.


        deformation_p_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2) ).transpose(1,2)
        p1_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        for view_id in range(self.args.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
#            proj_ori_vertex = (proj_ori_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
            proj_ori_vertex = torch.cat(((proj_ori_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_ori_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
#            proj_tar_vertex = (proj_tar_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
            proj_tar_vertex = torch.cat(((proj_tar_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_tar_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            ori_depth_img, ori_weight_img, _ = point_renderer_(proj_ori_vertex.contiguous(), threshold_)
            ori_img = ori_depth_img / ori_weight_img
            tgt_depth_img, tgt_weight_img, _ = point_renderer_(proj_tar_vertex.contiguous(), threshold_)
            tar_img = tgt_depth_img / tgt_weight_img
#            print(ori_img.size(), tar_img.size())
#            print(ori_img.max(), tar_img.max())
##            src_rgb_img = np.ones((256, 256, 3),dtype=np.uint8)
#            tar_rgb_img = Image.new('RGB', (256, 256), (255, 255, 255))
#            img_array = np.array(tar_rgb_img)
#            for i in range(256):
#                for j in range(256):
##                    src_value = int(63-ori_img[0,i,j,0]*0.25)
#                    tar_value = int(tar_img[0,i,j,0]*0.25)
#                    img_array[i,j] = (rb_colormap[tar_value,0]*255, rb_colormap[tar_value,1]*255, rb_colormap[tar_value,2]*255)
#            
#            img_array = Image.fromarray(np.uint8(img_array))
#            img_array.save('./imgs/tar'+str(view_id)+'.png',"png")
################# If to use the intersection mask #########################
##### L2
#            mask = torch.sign(ori_img*tar_img).detach()
#            output += torch.sum(torch.mul(mask, torch.mul(ori_img-tar_img, ori_img-tar_img)))/(self.args.view_divide**2)


##### L1
#            output += torch.sum( torch.abs(ori_img-tar_img))/(self.args.view_divide**2)


##### threshold L2
            mask = torch.sign(ori_img*tar_img).detach()
            depth_dis_abs =  torch.abs(ori_img-tar_img)
            depth_dis_thres =  -1.0 * (depth_dis_abs.detach() - 0.05)
            mask2 = (torch.sign(torch.sign(depth_dis_thres).detach()+0.5)+1.0)*0.5
            mask = torch.mul(mask2, mask).detach()
            depth_dis_sqr = torch.mul(depth_dis_abs,depth_dis_abs)
            output += torch.sum(torch.mul(mask,depth_dis_sqr))/(self.args.view_divide**2)

##### threshold L1
#            mask = torch.sign(ori_img*tar_img).detach()
#            depth_dis_abs =  torch.abs(ori_img-tar_img)
#            depth_dis_thres =  -1.0 * (depth_dis_abs.detach() - 0.05)
#            mask2 = (torch.sign(torch.sign(depth_dis_thres).detach()+0.5)+1.0)*0.5
#            mask = torch.mul(mask2, mask).detach()
#            output += torch.sum(torch.mul(mask,depth_dis_abs))/(self.args.view_divide**2)
            
##### robust L2
#            mask = torch.sign(ori_img*tar_img).detach()
#            depth_dis_abs =  torch.abs(ori_img-tar_img)
#            depth_dis_softmask =  torch.exp( -1.0 * (torch.mul(depth_dis_abs.detach()/0.2, depth_dis_abs.detach()/0.2) ) )
#            mask = torch.mul(depth_dis_softmask,mask).detach()
#            depth_dis_sqr = torch.mul(depth_dis_abs,depth_dis_abs)
#            output += torch.sum(torch.mul(mask,depth_dis_sqr))/(self.args.view_divide**2)
            
##### robust L1
#            mask = torch.sign(ori_img*tar_img).detach()
#            depth_dis_abs =  torch.abs(ori_img-tar_img)
#            depth_dis_softmask =  torch.exp( -1.0 * (torch.mul(depth_dis_abs.detach()/0.2, depth_dis_abs.detach()/0.2) ) )
#            mask = torch.mul(depth_dis_softmask,mask).detach()
#            output += torch.sum(torch.mul(mask,depth_dis_abs))/(self.args.view_divide**2)

            
        
        return output

        
    def loss_on_dice(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
#        print(deformation_p.size())
#        print(p1.size())
        deformation_p_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2) ).transpose(1,2)
        p1_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        for view_id in range(self.args.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
            proj_ori_vertex = (proj_ori_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
#            proj_ori_vertex = torch.cat(((proj_ori_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_ori_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
            proj_tar_vertex = (proj_tar_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
#            proj_tar_vertex = torch.cat(((proj_tar_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_tar_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            ori_depth_img, ori_weight_img, _ = point_renderer_(proj_ori_vertex.contiguous(), threshold_)
            ori_img = ori_depth_img / ori_weight_img
            tgt_depth_img, tgt_weight_img, _ = point_renderer_(proj_tar_vertex.contiguous(), threshold_)
            tar_img = tgt_depth_img / tgt_weight_img
################# If to use the intersection mask #########################
            tar_mask = torch.sign(torch.abs(tar_img)).detach()
#            print('tar_mask.max():',tar_mask.max())
#            print('tar_mask.min():',tar_mask.min())
#            print('tar_mask.sum():',tar_mask.sum())
            ori_exp = 1 - torch.exp(-100 * torch.mul(ori_img, ori_img))
            intersection = torch.mul(tar_mask, ori_exp)
            
            output += 1 - (2*torch.sum(intersection)/(torch.sum(tar_mask) + torch.sum(ori_exp)))
        return output
            
    def loss_on_mask(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
#        print(deformation_p.size())
#        print(p1.size())
        deformation_p_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2) ).transpose(1,2)
        p1_views = torch.bmm(self.all_rot_matrix_for_lfd.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        for view_id in range(self.args.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
            proj_ori_vertex = (proj_ori_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
#            proj_ori_vertex = torch.cat(((proj_ori_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_ori_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
            proj_tar_vertex = (proj_tar_vertex + torch.LongTensor([[[1.,1.,-1.]]]).cuda()) * resolution_ / 2. # (B, v_n, 3)
#            proj_tar_vertex = torch.cat(((proj_tar_vertex[..., :2] + 1.) * (resolution_-1) / 2, proj_tar_vertex[..., 2].unsqueeze(-1)-1), -1) # (B, v_n, 3)
            
            ori_mask = point_mask_(proj_ori_vertex.contiguous(), threshold_2_)
            tgt_mask = point_mask_(proj_tar_vertex.contiguous(), threshold_2_)
            
#            ori_depth_img, ori_weight_img, _ = point_renderer_(proj_ori_vertex.contiguous(), threshold_)
#            ori_img = ori_depth_img / ori_weight_img
#            tgt_depth_img, tgt_weight_img, _ = point_renderer_(proj_tar_vertex.contiguous(), threshold_)
#            tar_img = tgt_depth_img / tgt_weight_img

#            print(ori_img.size(), tar_img.size())
#            print(ori_img.max(), tar_img.max())
##            src_rgb_img = np.ones((256, 256, 3),dtype=np.uint8)
#            tar_rgb_img = Image.new('RGB', (256, 256), (255, 255, 255))
#            img_array = np.array(tar_rgb_img)
#            for i in range(256):
#                for j in range(256):
##                    src_value = int(63-ori_img[0,i,j,0]*0.25)
#                    tar_value = int(tar_img[0,i,j,0]*0.25)
#                    img_array[i,j] = (rb_colormap[tar_value,0]*255, rb_colormap[tar_value,1]*255, rb_colormap[tar_value,2]*255)
#            
#            img_array = Image.fromarray(np.uint8(img_array))
#            img_array.save('./imgs/tar'+str(view_id)+'.png',"png")
################# If to use the intersection mask #########################
#            tar_mask = torch.sign(torch.abs(tar_img)).detach()
##            print('tar_mask.max():',tar_mask.max())
##            print('tar_mask.min():',tar_mask.min())
##            print('tar_mask.sum():',tar_mask.sum())
#            ori_exp = 1 - torch.exp(-100 * torch.mul(ori_img, ori_img))
#            intersection = torch.mul(tar_mask, ori_exp)
            
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
#        output += torch.abs(dist)
        return torch.sum(output)/thisbatchsize
    
        
    def loss_on_wsmo(self, neighbour_indexes, point_weight):
        thisbatchsize = point_weight.size()[0]
#        neighbour_indexes = get_neighbor_index(p0, self.args.neighbour_num)
        point_weight_ = point_weight
        neibour_weights = indexing_neighbor(point_weight_, neighbour_indexes)
        neibour_mean_weights = torch.mean(neibour_weights, dim = 2)
        weight_difference = neibour_mean_weights - point_weight_
        weight_squ_difference = torch.mean(torch.mul(weight_difference, weight_difference))
        return weight_squ_difference
        

    def loss_on_wact(self, point_weight):
        thisbatchsize = point_weight.size()[0]
#        neighbour_indexes = get_neighbor_index(p0, self.args.neighbour_num)
        weight_squ_difference = torch.mean(torch.abs(point_weight))
        return weight_squ_difference

        
        
    def loss_on_nc(self, neighbour_indexes, deformation_p):
        thisbatchsize = deformation_p.size()[0]
        
#        neighbour_indexes = get_neighbor_index(p0, self.args.neighbour_num)
        deformation_neibour_points_ = indexing_neighbor(deformation_p, neighbour_indexes)
        neibour_mean_ = torch.mean(deformation_neibour_points_, dim = 2)
        difference = neibour_mean_ - deformation_p
        squ_difference = torch.mean(torch.mul(difference, difference))
        return squ_difference
        
    
        
    
    def loss_on_pcp(self, deformation_p, p0):
        thisbatchsize = deformation_p.size()[0]
#        feature_out_local = self.pointnet_seg_pretrained(deformation_p)
#        feature_in_local = self.pointnet_seg_pretrained(p0)
#        difference_local = feature_out_local - feature_in_local
#        squ_difference = torch.sum(torch.mul(difference_local,difference_local)/thisbatchsize)
#        return squ_difference
        return torch.mean(torch.zeros((1),dtype = torch.float, device=p0.device))
        
    def loss_on_trans(self, rigid_matrix):
        thisbatchsize = rigid_matrix.size()[0]
        output = 0
        trans_ = torch.mul(self.trans_mask.expand(thisbatchsize,-1,-1), rigid_matrix)
        output += torch.sum(torch.mul(trans_, trans_))/thisbatchsize
        
        return output
        
    def loss_on_nn(self, neighbour_indexes, deformation_p, source_points):
        thisbatchsize = deformation_p.size()[0]
#        neighbour_indexes = get_neighbor_index(p0, self.args.neighbour_num)
        deformation_neibour_points_ = indexing_neighbor(deformation_p, neighbour_indexes)
        source_neibour_points_ = indexing_neighbor(source_points, neighbour_indexes)
        deformation_neibour_dis_ = deformation_neibour_points_ - deformation_p.unsqueeze(2)
        source_neibour_dis_ = source_neibour_points_ - source_points.unsqueeze(2)
        deformation_neibour_dis_ = torch.sqrt(torch.mul(deformation_neibour_dis_, deformation_neibour_dis_).sum(dim =-1)+0.00001)
        source_neibour_dis_ = torch.sqrt(torch.mul(source_neibour_dis_, source_neibour_dis_).sum(dim =-1)+0.00001)
#        print(deformation_neibour_dis_.mean())
#        print(source_neibour_dis_.mean())
#        print(neighbour_indexes.max())
        difference = deformation_neibour_dis_ - source_neibour_dis_
        squ_difference = torch.sum(torch.mul(difference, difference))
        return squ_difference/thisbatchsize



    def loss_on_emd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        output = 0
#        d1 = torch.sum(earth_mover_distance(deformation_p, p1, transpose=False))
#        d2 = torch.sum(earth_mover_distance(p1, deformation_p, transpose=False))
#        output += (d1+d2)*0.5
        return output/thisbatchsize
        
        
    def loss_on_spmd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        N_sample_points = 10000
#        thisbatch_samplepoints = self.sample_points.view(1, -1, 3).expand(thisbatchsize, -1, -1)
        thisbatch_samplepoints = torch.rand(thisbatchsize, N_sample_points, 3).cuda() * 2.0 - 1.0
        
        # 1. compute the position of projection points
        dis_deform = compute_sqrdis_map(deformation_p, thisbatch_samplepoints)
        dis_target = compute_sqrdis_map(p1, thisbatch_samplepoints)
        mean_dis_deform = torch.topk(dis_deform, k=5, dim=1, largest=False)[0]
        mean_dis_target = torch.topk(dis_target, k=5, dim=1, largest=False)[0]
        
        sofmax_weights_deform = torch.softmax(mean_dis_deform, dim=1)
        sofmax_weights_target = torch.softmax(mean_dis_target, dim=1)
        
        mean_dis_deform_id = torch.topk(dis_deform, k=5, dim=1, largest=False)[1]
        mean_dis_target_id = torch.topk(dis_target, k=5, dim=1, largest=False)[1]
        
        id_ = torch.arange(thisbatchsize).view(-1, 1, 1)
        closest_points_deform = deformation_p[id_, mean_dis_deform_id]
        closest_points_target = p1[id_, mean_dis_target_id]
        
        closest_points_deform_proj = torch.mul(closest_points_deform, sofmax_weights_deform.unsqueeze(3).expand(-1,-1,-1,3)).sum(dim = 1)
        closest_points_target_proj = torch.mul(closest_points_target, sofmax_weights_target.unsqueeze(3).expand(-1,-1,-1,3)).sum(dim = 1)
        
#        # 2. refine the position of sample points
#        closest_points_middle_proj = 0.5 * (closest_points_deform_proj + closest_points_target_proj)
#        closest_points_projs_len = 0.5 * (closest_points_deform_proj - closest_points_target_proj)
#        closest_points_projs_len = torch.sqrt(torch.mul(closest_points_projs_len, closest_points_projs_len).sum(dim = -1) + 0.0000001).unsqueeze(2).expand(-1, -1, 3)
#        
#        closest_points_middleline = thisbatch_samplepoints - closest_points_middle_proj
#        closest_points_middleline_len = torch.sqrt(torch.mul(closest_points_middleline, closest_points_middleline).sum(dim = -1) + 0.0000001).unsqueeze(2).expand(-1, -1, 3)
#        closest_points_middleline_unit = closest_points_middleline/closest_points_middleline_len
#        
#        thisbatch_samplepoints = closest_points_middle_proj + (1.732/2)*closest_points_middleline_unit*closest_points_projs_len
#        thisbatch_samplepoints = thisbatch_samplepoints.detach()
        
        # 3. compute the position of sample points again
        
        dis_deform = compute_sqrdis_map(deformation_p, thisbatch_samplepoints)
        dis_target = compute_sqrdis_map(p1, thisbatch_samplepoints)
        mean_dis_deform = torch.topk(dis_deform, k=5, dim=1, largest=False)[0]
        mean_dis_target = torch.topk(dis_target, k=5, dim=1, largest=False)[0]
        
        sofmax_weights_deform = torch.softmax(mean_dis_deform, dim=1)
        sofmax_weights_target = torch.softmax(mean_dis_target, dim=1)
        
        mean_dis_deform_id = torch.topk(dis_deform, k=5, dim=1, largest=False)[1]
        mean_dis_target_id = torch.topk(dis_target, k=5, dim=1, largest=False)[1]
        
        id_ = torch.arange(thisbatchsize).view(-1, 1, 1)
        closest_points_deform = deformation_p[id_, mean_dis_deform_id]
        closest_points_target = p1[id_, mean_dis_target_id]
        
        closest_points_deform_proj = torch.mul(closest_points_deform, sofmax_weights_deform.unsqueeze(3).expand(-1,-1,-1,3)).sum(dim = 1)
        closest_points_target_proj = torch.mul(closest_points_target, sofmax_weights_target.unsqueeze(3).expand(-1,-1,-1,3)).sum(dim = 1)
        
        # 4. compute the loss
        
        
        dis_nor_deform = closest_points_deform_proj - thisbatch_samplepoints
        dis_nor_target = closest_points_target_proj - thisbatch_samplepoints
        
        
        dis_nor_deform_norm = torch.mul(dis_nor_deform, dis_nor_deform) + 0.000001
        dis_nor_target_norm = torch.mul(dis_nor_target, dis_nor_target) + 0.000001
        dis_nor_deform_norm = torch.sqrt(torch.sum(dis_nor_deform_norm, dim = -1)).unsqueeze(2)
        dis_nor_target_norm = torch.sqrt(torch.sum(dis_nor_target_norm, dim = -1)).unsqueeze(2)
#        dis_nor_deform = dis_nor_deform/dis_nor_deform_norm
#        dis_nor_target = dis_nor_target/dis_nor_target_norm
        
        
        weight = torch.exp(-1.0 * dis_nor_target_norm)
#        output = torch.mul( torch.mul(mean_dis_deform - mean_dis_target, mean_dis_deform - mean_dis_target) , weight )
        
#        output = torch.mul( torch.mul(dis_nor_deform - dis_nor_target, dis_nor_deform - dis_nor_target), weight )
        output_p = torch.mul(dis_nor_deform - dis_nor_target, dis_nor_deform - dis_nor_target)
#        output = torch.mul(mean_dis_deform - mean_dis_target, mean_dis_deform - mean_dis_target)

        dis_nor_deform = dis_nor_deform/dis_nor_deform_norm
        dis_nor_target = dis_nor_target/dis_nor_target_norm
        
        output_g = torch.mul(dis_nor_deform - dis_nor_target, dis_nor_deform - dis_nor_target)
        
        output_d = torch.mul(dis_nor_deform_norm - dis_nor_target_norm, dis_nor_deform_norm - dis_nor_target_norm)
        
        output_p = torch.mul(output_p , weight)
        output_g = torch.mul(output_g , weight)
        output_d = torch.mul(output_d , weight)

        return (0.000 * torch.sum(output_p) + 0.000 * torch.sum(output_g) + 1.0 * torch.sum(output_d) )/(thisbatchsize*N_sample_points) 
    
    
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
            
        if self.args.weight_wsmo >0:
            # L^{wsmo}
            loss_wsmo = 0
            for i in range(iter_time):
                loss_wsmo += self.args.gamma**(iter_time-i-1) * self.loss_on_wsmo(neighbour_indexes_, point_weight_list[i])
#            loss_wsmo = self.loss_on_wsmo(point_weight, p0)
            loss += loss_wsmo * self.args.weight_wsmo 
            loss_stages.append(loss_wsmo)
        else:
            loss_stages.append(zero_tensor)
            
        if self.args.weight_nc >0:
            # L^{nc}
            loss_nc =0 
            for i in range(iter_time):
                loss_nc += self.args.gamma**(iter_time-i-1) * self.loss_on_nc(neighbour_indexes_, deformation_points_list[i].transpose(1,2))
            loss += loss_nc * self.args.weight_nc
            loss_stages.append(loss_nc)
        else:
            loss_stages.append(zero_tensor)
            
        if self.args.weight_pcp >0:
            # L^{pcp}
            loss_pcp = 0
            for i in range(iter_time):
                loss_pcp += self.args.gamma**(iter_time-i-1) * self.loss_on_pcp(deformation_points_list[i].transpose(1,2), target_points)
            loss += loss_pcp * self.args.weight_pcp
            loss_stages.append(loss_pcp)
        else:
            loss_stages.append(zero_tensor)
        
        
        if self.args.weight_wact >0:
            # L^{wact}
            loss_wact = 0
            for i in range(iter_time):
                loss_wact += self.args.gamma**(iter_time-i-1) * self.loss_on_wact(point_weight_list[i])
#            loss_wact = self.loss_on_wact(point_weight, p0)
            loss += loss_wact * self.args.weight_wact 
            loss_stages.append(loss_wact)
        else:
            loss_stages.append(zero_tensor)
            
        if self.args.weight_lfd >0:
            # L^{lfd}
            loss_lfd = 0
            for i in range(iter_time):
                loss_lfd += self.args.gamma**(iter_time-i-1) * self.loss_on_lfd(deformation_points_list[i].transpose(1,2), target_points)
#            loss_lfd = self.loss_on_lfd(point_weight, p0)
            loss += loss_lfd * self.args.weight_lfd 
            loss_stages.append(loss_lfd)
        else:
            loss_stages.append(zero_tensor)
            
        if self.args.weight_dice >0:
            # L^{dice}
            loss_dice = 0
            for i in range(iter_time):
                loss_dice += self.args.gamma**(iter_time-i-1) * self.loss_on_dice(deformation_points_list[i].transpose(1,2), target_points)
#            loss_dice = self.loss_on_dice(point_weight, p0)
            loss += loss_dice * self.args.weight_dice 
            loss_stages.append(loss_dice)
        else:
            loss_stages.append(zero_tensor)
                        
        if self.args.weight_trans >0:
            # L^{trans}
            loss_trans = 0
            for i in range(iter_time):
                loss_trans += self.args.gamma**(iter_time-i-1) * self.loss_on_trans(rigid_matrix_list[i])
#            loss_trans = self.loss_on_trans(point_weight, p0)
            loss += loss_trans * self.args.weight_trans 
            loss_stages.append(loss_trans)
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
            
        
        if self.args.weight_nn >0:
            # L^{nn}
            loss_nn = 0
            for i in range(iter_time):
                if i==0 or i==1:
                    loss_nn += self.args.gamma**(iter_time-i-1) * self.loss_on_nn(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
                elif i==2 or i==3:
                    loss_nn += 0.1 * self.args.gamma**(iter_time-i-1) * self.loss_on_nn(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
                else:
                    loss_nn += 0.01 * self.args.gamma**(iter_time-i-1) * self.loss_on_nn(neighbour_indexes_, deformation_points_list[i].transpose(1,2), source_points)
#            loss_nn = self.loss_on_nn(point_weight, p0)
            loss += loss_nn * self.args.weight_nn 
            loss_stages.append(loss_nn)
#            exit()
        else:
            loss_stages.append(zero_tensor)
            

            
        if self.args.weight_emd >0:
            # L^{emd}
            loss_emd = 0
            for i in range(iter_time):
                loss_emd += self.args.gamma**(iter_time-i-1) * self.loss_on_emd(deformation_points_list[iter_time-1].transpose(1,2), target_points)
#            loss_emd = self.loss_on_emd(point_weight, p0)
            loss += loss_emd * self.args.weight_emd 
            loss_stages.append(loss_emd)
#            exit()
        else:
            loss_stages.append(zero_tensor)
            
            
        if self.args.weight_spmd >0:
            # L^{spmd}
            loss_spmd = 0
            for i in range(iter_time):
                loss_spmd += self.args.gamma**(iter_time-i-1) * self.loss_on_spmd(deformation_points_list[iter_time-1].transpose(1,2), target_points)
#            loss_spmd = self.loss_on_spmd(point_weight, p0)
            loss += loss_spmd * self.args.weight_spmd 
            loss_stages.append(loss_spmd)
#            exit()
        else:
            loss_stages.append(zero_tensor)
            
            
        
        return loss, loss_stages

