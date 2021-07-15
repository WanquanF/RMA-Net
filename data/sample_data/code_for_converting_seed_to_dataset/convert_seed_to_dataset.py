import os
import sys
import random
import numpy as np
import igl

ACAP_INTER_NUM = 100
SAMPLE_NUM = 2048

##### STEP 1 : Convert the seed pair into a track of frames by ACAP interpolation.
print('STEP 1 : Convert the seed pair into a track of frames by ACAP interpolation.')

seed_meshes_path = '../seed'
seed_list_path = '../seed/seed.list'
seed_list_f = open(seed_list_path)
seed_list = seed_list_f.readlines()
for i in range(len(seed_list)):
    if seed_list[i][-1]=='\n':
        seed_list[i] = seed_list[i][:-1]
    seed_list[i] = seed_meshes_path + '/' + seed_list[i]
#print(seed_list)


acap_exe = './vertex2acap/build/acap_interp'

track_path = '../track'

if os.path.exists(track_path)==False:
    os.mkdir(track_path)

for i in range(ACAP_INTER_NUM+1):
    acap_mesh_path = track_path + '/' + 'frame_' + str(i)+'.obj'
#    print(acap_mesh_path)
    if os.path.exists(acap_mesh_path):
        print('The file '+acap_mesh_path+' has already been exsisting! We do not cover it here. Skipping the ACAP process.')
        break
    acap_rate = i/(ACAP_INTER_NUM+1)
    os.system(acap_exe+' '+seed_list[0]+' '+seed_list[1]+' '+acap_mesh_path+' '+str(acap_rate))
    

##### STEP 2 : Extract the training and testing pairs.
print('STEP 2 : Extract the training and testing pairs.')

packed_data_path = '../packed_data'
if os.path.exists(packed_data_path)==False:
    os.mkdir(packed_data_path)
training_list_path = packed_data_path+'/training_pairs.list'
testing_list_path = packed_data_path+'/testing_pairs.list'

all_pairs_list = []
for i in range(ACAP_INTER_NUM+1):
    for j in range(ACAP_INTER_NUM+1):
        if i-j>-50 and i-j<50:
            continue
        if i>j+80 or i<j-80:
            continue
        all_pairs_list.append([i,j])
        
for times in range(3):
    random.shuffle(all_pairs_list)
        
#print(len(all_pairs_list))
training_pairs_num = int(len(all_pairs_list)*0.95)
testing_pairs_num = len(all_pairs_list) - training_pairs_num

training_pairs_list = all_pairs_list[:training_pairs_num]
testing_pairs_list = all_pairs_list[training_pairs_num:]

#print(len(training_pairs_list))
#print(len(testing_pairs_list))
#print(testing_pairs_list)

# saving the index of the training and testing pairs
if os.path.exists(training_list_path) or os.path.exists(testing_list_path):
    print('The dataset index list have been existing ! We do not cover them here! Skipping writing the list files!')
else:
    t_f=open(training_list_path,'w')
    for i in range(len(training_pairs_list)):
        t_f.write(str(training_pairs_list[i][0])+' '+str(training_pairs_list[i][1])+'\n')
    t_f.close()
    t_f=open(testing_list_path,'w')
    for i in range(len(testing_pairs_list)):
        t_f.write(str(testing_pairs_list[i][0])+' '+str(testing_pairs_list[i][1])+'\n')
    t_f.close()
    

##### STEP 3 : Sample points from the meshes.
print('STEP 3 : Sample points from the meshes.')

# The point extraction is implemented with the Poisson Disk Sampling filter in MeshlabServer.

for i in range(ACAP_INTER_NUM+1):
    acap_mesh_path = track_path + '/' + 'frame_' + str(i)+'.obj'
    acap_points_path = track_path + '/' + 'frame_points_' + str(i)+'.obj'
    if os.path.exists(acap_points_path):
        print('The extracted points have been existing ! We do not cover them here! Skipping extracting points!')
        break
    os.system('python ./sample_points_for_one_mesh.py --meshpath '+ acap_mesh_path +' --pointpath '+ acap_points_path +' --pointnum '+str(SAMPLE_NUM))
    

##### STEP 4 : Construct the bin files of the training and testing pairs.
print('STEP 4 : Construct the bin files of the training and testing pairs.')


train_points_pair = [] # all_train_pair_num * (SAMPLE_NUM*2) *3
test_points_pair = [] # all_test_pair_num * (SAMPLE_NUM*2) *3
train_points_pair_path = packed_data_path + '/training_points.bin'
test_points_pair_path = packed_data_path + '/testing_points.bin'


for i in range(len(training_pairs_list)):
    apair = training_pairs_list[i]
    source_p_path = track_path + '/' + 'frame_points_' + str(apair[0])+'.obj'
    target_p_path = track_path + '/' + 'frame_points_' + str(apair[1])+'.obj'
    point0,_  = igl.read_triangle_mesh(source_p_path)
    point1,_  = igl.read_triangle_mesh(target_p_path)
    
    points = np.concatenate((point0,point1),axis=0)
    train_points_pair.append(points)
train_points_pair = np.array(train_points_pair).astype(np.float32)
print('train_points_pair.shape:',train_points_pair.shape)
train_points_pair.tofile(train_points_pair_path)


for i in range(len(testing_pairs_list)):
    apair = testing_pairs_list[i]
    source_p_path = track_path + '/' + 'frame_points_' + str(apair[0])+'.obj'
    target_p_path = track_path + '/' + 'frame_points_' + str(apair[1])+'.obj'
    point0,_  = igl.read_triangle_mesh(source_p_path)
    point1,_  = igl.read_triangle_mesh(target_p_path)
    
    points = np.concatenate((point0,point1),axis=0)
    test_points_pair.append(points)
test_points_pair = np.array(test_points_pair).astype(np.float32)
print('test_points_pair.shape:',test_points_pair.shape)
test_points_pair.tofile(test_points_pair_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    



