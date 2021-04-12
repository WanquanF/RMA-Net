import os

#coarse-net

loss_weight=' '
#loss_weight+=' --weight_cd -0.1'
loss_weight+=' --weight_nc -100'
loss_weight+=' --weight_pcp -0.00001'
loss_weight+=' --weight_wsmo -1.0'
loss_weight+=' --weight_pd 10.0'
loss_weight+=' --weight_wact 1'
loss_weight+=' --weight_lfd 1.0'
loss_weight+=' --weight_dice -1000000.0'
loss_weight+=' --weight_mask -0.1'
loss_weight+=' --weight_trans 0.01'
loss_weight+=' --weight_nn 100'
loss_weight+=' --gamma 1.0'



os.system('CUDA_VISIBLE_DEVICES=1 python train_view.py --visualization_while_testing 1 --last_sample_id 0 --iteration 5 --test_blank 1000 --train_max_samples 50000 --learning_rate 0.001  --batchsize 1  --out_baseline \'out_baseline_5220_eval\' --point_num 2048 --view_divide 5 --pack_path \'../../data/Fusion_packed\' --weight_cd -1.0 '+loss_weight)





