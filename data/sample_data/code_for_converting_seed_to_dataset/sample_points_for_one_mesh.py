import os
import sys
import igl
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--meshpath', type=str, default = None)
parser.add_argument('--pointpath', type=str, default=None)
parser.add_argument('--pointnum', type=int, default=2048)

args = parser.parse_args()

def change_mlx_script(pnum, pre_lines, post_lines):
    f = open('./mlx/pds.mlx','w')
    for i in range(len(pre_lines)):
        if i<len(pre_lines)-1:
            f.write(pre_lines[i]+'\n')
        else:
            f.write(pre_lines[i])
    f.write(str(pnum))
    for i in range(len(post_lines)):
        if i<len(post_lines)-1:
            f.write(post_lines[i]+'\n')
        else:
            f.write(post_lines[i])
    f.close()
    
def shellSort(nums, by_index=2, larger_first=True):
    lens = len(nums)
    gap = 1  
    while gap < lens // 3:
        gap = gap * 3 + 1  
    while gap > 0:
        for i in range(gap, lens):
            curNum, preIndex = nums[i], i - gap  
#            print('preIndex:',preIndex)
            if larger_first==True:
                while preIndex >= 0 and curNum[by_index] > nums[preIndex][by_index]:
                    nums[preIndex + gap] = nums[preIndex] 
                    preIndex -= gap
            else:
                while preIndex >= 0 and curNum[by_index] < nums[preIndex][by_index]:
                    nums[preIndex + gap] = nums[preIndex] 
                    preIndex -= gap
            nums[preIndex + gap] = curNum  
        gap //= 3  
    return nums

gpre = open('./mlx/pds.pre')
gpre_lines = gpre.readlines()
#print(gpre_lines)
for i in range(len(gpre_lines)):
    if gpre_lines[i][-1] == '\n':
        gpre_lines[i] = gpre_lines[i][:-1]
#print(gpre_lines)


gpost = open('./mlx/pds.post')
gpost_lines = gpost.readlines()
#print(gpost_lines)
for i in range(len(gpost_lines)):
    if gpost_lines[i][-1] == '\n':
        gpost_lines[i] = gpost_lines[i][:-1]
#print(gpost_lines)

object_pnum = args.pointnum

records = []


d = -5
while True:
    pnum_ = object_pnum + d
    
    if d>30:
        records_sorted = shellSort(records, by_index=1, larger_first=False)
        choose_pos = 0
        while True:
            if records[choose_pos][1]<object_pnum:
                choose_pos += 1
            else:
                break
        change_mlx_script(pnum_, gpre_lines, gpost_lines)
        os.system('/home/wanquan/meshlab/distrib/meshlabserver -i '+args.meshpath+' -o '+args.pointpath+' -s ./mlx/pds.mlx')
        vs, _ = igl.read_triangle_mesh(args.pointpath)
        real_pnum = vs.shape[0]
        vs_sort = vs.tolist().copy()
        shellSort(vs_sort, by_index=1, larger_first=True)
        vs_sort = vs_sort[:object_pnum]
        fobjout = open(args.pointpath,'w')
        for vi in range(len(vs_sort)):
            fobjout.write('v '+str(vs_sort[vi][0])+' '+str(vs_sort[vi][1])+' '+str(vs_sort[vi][2])+'\n')
        fobjout.close()
        print('#################################### SUCCES ####################################')
        print(records[-1])
        print('MESH : ',args.meshpath)
        print('POINT : ',args.pointpath)
        print('PNUM :',len(vs_sort),'of ',real_pnum)
        break
        
    change_mlx_script(pnum_, gpre_lines, gpost_lines)
    os.system('/home/wanquan/meshlab/distrib/meshlabserver -i '+args.meshpath+' -o '+args.pointpath+' -s ./mlx/pds.mlx')
    vs, _ = igl.read_triangle_mesh(args.pointpath)
    real_pnum = vs.shape[0]
    records.append([pnum_, real_pnum])
    if d >= -5 and d <= 10:
        d += 1
    elif d < -5:
        d = 10 + (-5 -d)+1
    else:
        d = -5 - (d-10) 
    if real_pnum >= object_pnum and real_pnum <= object_pnum + 4:
        vs_sort = vs.tolist().copy()
        shellSort(vs_sort, by_index=1, larger_first=True)
        print(vs_sort)
        vs_sort = vs_sort[:object_pnum]
        fobjout = open(args.pointpath,'w')
        for vi in range(len(vs_sort)):
            fobjout.write('v '+str(vs_sort[vi][0])+' '+str(vs_sort[vi][1])+' '+str(vs_sort[vi][2])+'\n')
        fobjout.close()
        print('#################################### SUCCES ####################################')
        print(records[-1])
        print('MESH : ',args.meshpath)
        print('POINT : ',args.pointpath)
        print('PNUM :',len(vs_sort),'of ',real_pnum)
        break





























