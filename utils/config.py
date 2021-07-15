import argparse
import os
from configparser import SafeConfigParser



def parse_args():
    parser = argparse.ArgumentParser()
    
    #basic settings
    parser.add_argument('--device_id',
		         help='Specify the index of the cuda device, e.g. 0, 1 ,2',
                        default=0, type=int)
                        
    
                                            
    parser.add_argument('--max_meet_times',
                 help='the max number of epoches',
                        default=5, type=int)
                        
    parser.add_argument('--train_max_epoch',
                 help='the max number of epoches while training',
                        default=5, type=int)                    
                        
    parser.add_argument('--last_sample_id',
                 help='the id in the last saved trained model',
                        default=0, type=int)                    

    parser.add_argument('--train_max_samples',
                 help='the max number of samples used in the training',
                        default=500000, type=int)
                                                                        
    parser.add_argument('--all_training_samples_num',
                 help='the number of samples used in the training',
                        default=100000, type=int)
                        
    parser.add_argument('--train_data_pair_num',
                 help='the number of training pairs used in the training',
                        default=1000000000000, type=int)
                        
    parser.add_argument('--view_divide',
                 help='the number of training pairs used in the training',
                        default=11, type=int)
                        
                          
                        
    parser.add_argument('--test_cache_file',
                 help='the file of the test loss',
                        default='./output/test_cache.txt', type=str)
                        
    parser.add_argument('--out_baseline',
                 help='the file of the baseline training results',
                        default='./output_baseline', type=str)  
                           
                                     
    # arguments for training process
    parser.add_argument('--batchsize',
                        help='Batch size for training',
                        default=200, type=int)
    parser.add_argument('--test_blank',
                        help='how often the testing process is performed',
                        default=100, type=int)
    

    # parameters for grid searching
    parser.add_argument('--weight_cd',
                        default=1, type=float)
    parser.add_argument('--weight_pd',
                        default=-1, type=float)
    parser.add_argument('--weight_sparse',
                        default=-1, type=float)
    parser.add_argument('--weight_depth',
                        default=-1, type=float)
    parser.add_argument('--weight_tran',
                        default=-1, type=float)
    parser.add_argument('--weight_mask',
                        default=-1, type=float)
    parser.add_argument('--weight_arap',
                        default=-1, type=float)
    parser.add_argument('--weight_rtmat',
                        default=-1, type=float)
    parser.add_argument('--weight_rt',
                        default=-1, type=float)
    parser.add_argument('--weight_emd',
                        default=-1, type=float)
    parser.add_argument('--weight_setzero',
                        default=-1, type=float)
    parser.add_argument('--weight_spmd',
                        default=-1, type=float)
    parser.add_argument('--weight_cyclenn',
                        default=-1, type=float)
    parser.add_argument('--gamma',
                        default=0.8, type=float)
                        
    parser.add_argument('--weight_decay',
                        default=0.00005, type=float)
    parser.add_argument('--learning_rate',
                        default=0.0005, type=float)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--clip', type=float, default=1.0)

    
                    
    parser.add_argument('--patch_num', default=10, type=int,
                        metavar='pn', help='number of patches')
    parser.add_argument('--point_num', default=8192, type=int,
                        metavar='pn', help='number of patches')
    parser.add_argument('--pack_path', type=str, default='None', metavar='None',
                    help='the path of packed_data (default: None)')
    parser.add_argument('--pre_trained', type=str, default='None', metavar='None',
                    help='the path of pretrained model weights (default: None)')
                    
    parser.add_argument('--train_bin_path', type=str, default='None', metavar='None',
                    help='the path of training_bin_file (default: None)')
    parser.add_argument('--test_bin_path', type=str, default='None', metavar='None',
                    help='the path of testing_bin_file (default: None)')
                    
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn_delta', dest='learn_delta', action='store_true',
                        help='flag for training step size delta')
    parser.add_argument('--neighbour_num', default=4, type=int,
                        metavar='nn', help='neighbour_num of weight smoothing term')
    parser.add_argument('--visualization_while_testing', default=1, type=int,
                        metavar='visual', help='1 if visualize; 0 if not')
    parser.add_argument('--iteration', default=3, type=int,
                        metavar='iter', help='iteration times')
    
    
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=1, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
                        
                        
    parser.add_argument('--weight', type=str, default='None',
                        help='the path of pretrained weights')
    parser.add_argument('--src', type=str, default='None',
                        help='the path of the source object')
    parser.add_argument('--tgt', type=str, default='None',
                        help='the path of the target object')
    parser.add_argument('--if_nonrigid', type=int, default='1',
                        help='1 if non-rigid, 0 if rigid')
    
    args = parser.parse_args()

    

    return args
