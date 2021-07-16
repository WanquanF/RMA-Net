import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np
import torch.nn.init as init
import struct
import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code/')
from utils import drawer

import time
from network import  Net_PointNR_v2
from utils.config import parse_args
from loss import Loss, chamfer_dist
import time
import datetime


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

over_fitting_id = 0

import igl

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


#from chamfer_distance import ChamferDistance

#chamfer_dist = ChamferDistance()


'''
start = time.clock()
elapsed = (time.clock() - start)
print("Time used:",elapsed)
'''


# parse args
args = parse_args()
#print ('args:')
#print (args)
#cuda_num=args.device_id
#print('cuda_num:',cuda_num)

batch_size=args.batchsize
train_max_samples = args.train_max_samples

pack_path=args.pack_path
train_bin_path = args.train_bin_path
test_bin_path = args.test_bin_path
point_num = args.point_num

result_path = './results/'+args.out_baseline

train_points_pair_path = pack_path+'/'+train_bin_path
test_points_pair_path = pack_path+'/'+test_bin_path

print('The packed data path is : ',pack_path)
print('The training_file path is : ',train_points_pair_path)
print('The testing_file path is : ',test_points_pair_path)

# READ in train_points_pair
train_points_pair=np.fromfile(train_points_pair_path, dtype = np.float32).reshape(-1,point_num*2,3)
# READ in test_points_pair
test_points_pair=np.fromfile(test_points_pair_path, dtype = np.float32).reshape(-1,point_num*2,3)

train_points_pair_num = train_points_pair.shape[0]
test_points_pair_num = test_points_pair.shape[0]

train_points_pair_tensor = torch.from_numpy(train_points_pair)
test_points_pair_tensor = torch.from_numpy(test_points_pair)

train_points_pair_batch = torch.zeros([batch_size,point_num*2,3],dtype=torch.float,requires_grad=False).cuda()
test_points_pair_batch = torch.zeros([batch_size,point_num*2,3],dtype=torch.float,requires_grad=False).cuda()


def stageprint(vlist, text):
    for i in range(len(vlist)):
        print('stage '+str(i)+' '+text+' : '+str(vlist[i]))
    
def update_test_cache(used_samples_num, model, loss_obj, args):
    print('updating cache for used_samples_num = ' + str(used_samples_num))
    test_cache_file=result_path+'/result_cache.txt'
    loss_sum_, loss_stages_=compute_test_loss_values(model, loss_obj, args)
    print('the test loss: ',loss_sum_, loss_stages_)
    cf=open(test_cache_file,'a+')
    # The first number is the iteration times.
    cf.write(str(used_samples_num//batch_size)+' ')
    cf.write(str(loss_sum_)+' ')
    for i in range(len(loss_stages_)):
        cf.write(str(loss_stages_[i])+' ')
    cf.write('\n')
    cf.close()
    update_pics()
    if args.visualization_while_testing:
        update_visualization(model,  args)
#        exit()
    
    
def update_pics():
    test_cache_file=result_path+'/result_cache.txt'
    cf=open(test_cache_file,'r')
    lines=cf.readlines()
    x=[]
    y_sum=[]
    y_cd=[]
    y_pd=[]
    y_sparse=[]
    y_depth=[]
    y_tran=[]
    y_mask=[]
    y_arap=[]
    
    for i in range(len(lines)):
        if i%1==0:
            index = int(lines[i].split(' ')[0])
            sum_loss = float(lines[i].split(' ')[1])
            cd_loss = float(lines[i].split(' ')[2])
            pd_loss = float(lines[i].split(' ')[3])
            sparse_loss = float(lines[i].split(' ')[4])
            depth_loss = float(lines[i].split(' ')[5])
            tran_loss = float(lines[i].split(' ')[6])
            mask_loss = float(lines[i].split(' ')[7])
            arap_loss = float(lines[i].split(' ')[8])
            
            iter_index=index
            x.append(iter_index)
            y_sum.append(sum_loss)
            y_cd.append(cd_loss)
            y_pd.append(pd_loss)
            y_sparse.append(sparse_loss)
            y_depth.append(depth_loss)
            y_tran.append(tran_loss)
            y_mask.append(mask_loss)
            y_arap.append(arap_loss)
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The sum loss')
    plt.xlabel('iteration')
    plt.ylabel('sum loss')
    plt.plot(x, y_sum, c='r', ls='-')
    plt.savefig(result_path+'/loss_sum.png')
    
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on cd')
    plt.xlabel('iteration')
    plt.ylabel('cd loss')
    plt.plot(x, y_cd, c='#526922', ls='-')
#    plt.savefig(result_path+'/loss_cd.png')
    
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on pd')
    plt.xlabel('iteration')
    plt.ylabel('pd loss')
    plt.plot(x, y_pd, c='#526922', ls='-')
#    plt.savefig(result_path+'/loss_pd.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on sparse')
    plt.xlabel('iteration')
    plt.ylabel('sparse loss')
    plt.plot(x, y_sparse, c='#526922', ls='-')
    plt.savefig(result_path+'/loss_sparse.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on depth')
    plt.xlabel('iteration')
    plt.ylabel('depth loss')
    plt.plot(x, y_depth, c='#526922', ls='-')
    plt.savefig(result_path+'/loss_depth.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on tran')
    plt.xlabel('iteration')
    plt.ylabel('tran loss')
    plt.plot(x, y_tran, c='#526922', ls='-')
    plt.savefig(result_path+'/loss_tran.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on mask')
    plt.xlabel('iteration')
    plt.ylabel('mask loss')
    plt.plot(x, y_mask, c='#526922', ls='-')
    plt.savefig(result_path+'/loss_mask.png')
    
    fig = plt.figure(0)
    fig.clear()
    plt.title('The loss on arap')
    plt.xlabel('iteration')
    plt.ylabel('arap loss')
    plt.plot(x, y_arap, c='#526922', ls='-')
    plt.savefig(result_path+'/loss_arap.png')
    
    
def update_visualization(model,  args):
    global test_points_pair_tensor
    global test_points_pair_batch
    
    test_cache_file=result_path+'/result_cache.txt'
    cf=open(test_cache_file,'r')
    lines=cf.readlines()
    last_line = lines[len(lines)-1]
    iter_num = int(last_line.split(' ')[0])
    visual_folder = result_path+'/visual_'+str(iter_num*batch_size)
    if not os.path.exists(visual_folder):
        os.mkdir(visual_folder)
        
            
    print('Satrt to visualize the results now.')
    for idd in range(5):    
        print(idd)
        sample_id = idd + over_fitting_id
        test_points_pair_one_sample = test_points_pair_tensor[sample_id:sample_id+1].cuda()
        
        if True:
            test_points_source_one_sample = test_points_pair_one_sample[:,:point_num,:]
            test_points_target_one_sample = test_points_pair_one_sample[:,point_num:,:]
            
        model.eval()
        with torch.no_grad():
            phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list = model(test_points_source_one_sample, test_points_target_one_sample, iteration = 7)
            iter_time = len(phi_list)
            for stage in range(iter_time):
                deform_rigid_points_list[stage] = deform_rigid_points_list[stage].transpose(1,2)
                deformation_points_list[stage] = deformation_points_list[stage].transpose(1,2) 
            
            visual_folder_one_sample = visual_folder+'/'+str(sample_id)
            if not os.path.exists(visual_folder_one_sample):
                os.mkdir(visual_folder_one_sample)
                
            source_points_name = visual_folder_one_sample+'/source.obj'
            target_points_name = visual_folder_one_sample+'/target.obj'
            
            weight_stages_list = []
            
            for stage in range(iter_time):
#                visual_folder_one_stage = visual_folder_weight_one_sample + '/'+str(stage+1)
#                if not os.path.exists(visual_folder_one_stage):
#                    os.mkdir(visual_folder_one_stage)
                deform_points_name = visual_folder_one_sample+'/deform_stage_'+str(stage+1)+'.obj'
                
                old_dmt_p = None
                if stage == 0:
                    old_dmt_p = test_points_source_one_sample
                else:
                    old_dmt_p = deformation_points_list[stage-1]
                dist1 = None
                cd_list = []
                with torch.no_grad():
                    dist1, dist2 = chamfer_dist(test_points_target_one_sample, deformation_points_list[stage])
                    cd_list.append(torch.mean(dist1+dist2).item()*0.5)

                stageprint(cd_list,'testing chamfer error')
                
                
#                old_p_with_errormap_path = visual_folder_one_stage+'/stage_'+str(stage+1)+'_old_p_with_error.obj'
#                old_error_path = visual_folder_one_stage+'/stage_'+str(stage+1)+'_old_error.pd'
#                new_p_with_segmap_path = visual_folder_one_stage+'/stage_'+str(stage+1)+'_new_p_with_seg.obj'
#                new_error_path = visual_folder_one_stage+'/stage_'+str(stage+1)+'_new_error.pd'
#                weight_save_txt_path = visual_folder_one_stage+'/stage_'+str(stage+1)+'_kdim_weight.txt'
#                target_path = visual_folder_one_stage+'/target.obj'
#                this_rigid_path = visual_folder_one_stage+'/this_rigid.obj'
                if stage==0:
                    drawer.render_points_with_rgb(old_dmt_p.cpu().numpy().reshape(-1,3), 1,0,0,source_points_name)
                    drawer.render_points_with_rgb(test_points_target_one_sample.cpu().numpy().reshape(-1,3),0,0,1,target_points_name)
                drawer.render_points_with_rgb(deformation_points_list[stage].cpu().numpy().reshape(-1,3),0,1,0,deform_points_name)
                
#                drawer.render_points_with_rgb(old_dmt_p.cpu().numpy().reshape(-1,3), 1,0,0, old_p_with_errormap_path)
#                drawer.render_points_with_rgb(deformation_points_list[stage].cpu().numpy().reshape(-1,3),0,1,0,new_p_with_segmap_path)
#                drawer.render_points_with_rgb(test_points_target_one_sample.cpu().numpy().reshape(-1,3),0,0,1,target_path)
#                drawer.render_points_with_weight(deform_rigid_points_list[stage].cpu().numpy().reshape(-1,3),point_weight_list[stage].cpu().numpy().reshape(-1),this_rigid_path)
                

    

def stophere():
    while True:
        continue

def run_train_val(model, optimizer, loss_obj,  args):
    global train_points_pair_tensor
    global test_points_pair_tensor

    global train_points_pair_batch
    global test_points_pair_batch
    


    used_samples_num=args.last_sample_id
    start_pos=used_samples_num % train_points_pair_num

    if used_samples_num==0:
        update_test_cache(used_samples_num, model, loss_obj,  args)
    while used_samples_num<train_max_samples:
        while True:
            end_pos=start_pos+batch_size
            print('Training with pair samples: '+str(start_pos)+'~'+str(end_pos))
            train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args) ############## train one batch
            used_samples_num+=end_pos-start_pos
            if used_samples_num%(args.test_blank)==0:
                update_test_cache(used_samples_num, model, loss_obj, args) ############## test once
                print('Test here, at '+str(used_samples_num))
                torch.save(model.state_dict(), result_path+'/sample_'+str(used_samples_num)+'.pt')
            if end_pos>=train_points_pair_num:
                start_pos=end_pos - train_points_pair_num
            else:
                start_pos=end_pos
            print(used_samples_num,train_max_samples)
            if used_samples_num >= train_max_samples:
                break
    
    
    
def train_one_batch(model, optimizer, loss_obj, start_pos, end_pos, args):
    global train_points_pair_tensor
    global train_points_pair_batch
    
    
    global test_points_pair_tensor
    
    
    print(start_pos, end_pos)
    if end_pos<=train_points_pair_num:
        train_points_pair_batch = train_points_pair_tensor[start_pos:end_pos].cuda()
    else:
        bottom = train_points_pair_num - start_pos
        top = end_pos - train_points_pair_num
        
        train_points_pair_batch[:bottom] = train_points_pair_tensor[start_pos:].cuda()
        
        train_points_pair_batch[bottom:] = train_points_pair_tensor[:top].cuda()
    
    if True:
        train_points_source_batch = train_points_pair_batch[:,:point_num,:]
        train_points_target_batch = train_points_pair_batch[:,point_num:,:]
    for train_times in range(1):
        optimizer.zero_grad()    
        model.train()
        phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list = model(train_points_source_batch,train_points_target_batch)
        regi_loss, regi_loss_stages  = loss_obj(phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list, train_points_source_batch, train_points_target_batch)
        cd_list = []
        for i in range(len(deformation_points_list)):
            with torch.no_grad():
                dist1, dist2 = chamfer_dist(train_points_target_batch, deformation_points_list[i].transpose(1,2))
                cd_list.append(torch.mean(dist1+dist2).item()*0.5)
            
        stageprint(cd_list,'chamfer error')
        model.zero_grad()
        if True:
            scaler.scale(regi_loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            print('optimizer.lr : ',optimizer.state_dict()['param_groups'][0]['lr'])
            scheduler.step()
            scaler.update()
        else:
            regi_loss.backward()
            optimizer.step()

    

    

    
def test_one_batch(model, loss_obj, start_pos, end_pos, args):
    global test_points_pair_tensor
    global test_points_pair_batch
    if end_pos<=test_points_pair_num:
        test_points_pair_batch = test_points_pair_tensor[start_pos:end_pos].cuda()
    else:
        bottom = test_points_pair_num - start_pos
        top = end_pos - test_points_pair_num
        
        test_points_pair_batch[:bottom] = test_points_pair_tensor[start_pos:].cuda()
        
        test_points_pair_batch[bottom:] = test_points_pair_tensor[:top].cuda()
        
    
    if True:
        test_points_source_batch = test_points_pair_batch[:,:point_num,:]
        test_points_target_batch = test_points_pair_batch[:,point_num:,:]
    
    
    model.eval()
    with torch.no_grad():
        phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list = model(test_points_source_batch,test_points_target_batch)
        regi_loss, regi_loss_stages  = loss_obj(phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list, test_points_source_batch, test_points_target_batch)

    return regi_loss, regi_loss_stages
    
    
    

    
def compute_test_loss_values(model, loss_obj, args):

    start_pos=0
    loss_sum=0.0
    loss_stages=[]
    batch_cnt=0.0
    print('Computing the testing loss on the testing set:')
    for s in range(0, 30, batch_size):
        start_pos = s
        end_pos = s + batch_size
        if end_pos > test_points_pair_num:
            end_pos = test_points_pair_num
        this_batch_size = end_pos - start_pos
        lsum,lstages = test_one_batch(model, loss_obj, start_pos, end_pos, args)
        if start_pos==0:
            loss_sum=lsum.item()*this_batch_size
            for i in range(len(lstages)):
                loss_stages.append(lstages[i].item()*this_batch_size)
        else:
            loss_sum+=lsum.item()*this_batch_size
            for i in range(len(lstages)):
                loss_stages[i]+=lstages[i].item()*this_batch_size
        batch_cnt += this_batch_size
    loss_sum/=batch_cnt
    for i in range(len(loss_stages)):
        loss_stages[i]/=batch_cnt
    
    return loss_sum, loss_stages


if __name__=='__main__':
    pnrr_net = Net_PointNR_v2().cuda()
    if args.last_sample_id==0:
        if os.path.exists(result_path):
            os.system('rm -rf '+result_path)
        os.makedirs(result_path)
        os.system('cp '+args.pre_trained+' '+result_path+'/sample_0.pt')
        
    
    pnrr_net.load_state_dict(torch.load(result_path+'/sample_'+str(args.last_sample_id)+'.pt'),True)
    
    
    
    # setup optimizer
    optimizer = optim.AdamW(pnrr_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.epsilon)

    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.learning_rate, total_steps = (args.train_max_samples - args.last_sample_id)//args.batchsize, pct_start=0.03, cycle_momentum=False, anneal_strategy='linear')
    
    scaler = GradScaler(enabled=args.mixed_precision)
    
    # setup loss object
    loss_obj = Loss(args)
    # run train and test
    run_train_val(pnrr_net, optimizer, loss_obj,  args)
    
