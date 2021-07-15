import os

#coarse-net

loss_weight=' '
loss_weight+=' --weight_depth 1.0'
loss_weight+=' --weight_mask 0.1'
loss_weight+=' --weight_arap 0.01'
loss_weight+=' --weight_tran 0.1'
loss_weight+=' --weight_sparse 10'
loss_weight+=' --gamma 1.0'



os.system('CUDA_VISIBLE_DEVICES=3 python train_view.py '+
          ' --pre_trained \'../pre_trained/deform.pt\' '+
          ' --visualization_while_testing 1 '+
          ' --last_sample_id 0 '+
          ' --iteration 7 '+
          ' --test_blank 100 '+
          ' --train_max_samples 3000 '+
          ' --learning_rate 0.001   '+
          ' --batchsize 4 '+
          ' --out_baseline \'training_results_sample\' '+
          ' --point_num 2048  '+
          ' --view_divide 5  '+
          ' --pack_path \'../data/sample_data/packed_data\' '+
          ' --train_bin_path \'training_points.bin\' '+
          ' --test_bin_path \'testing_points.bin\' '+
          loss_weight)





