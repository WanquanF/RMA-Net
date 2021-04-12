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
import glob
import h5py
import copy
sys.path.append('../')
sys.path.append('../../')
from utils.config import parse_args
import model.ptlk as ptlk
import model.LieAlgebra as LieAlgebra
import igl


args = parse_args()
cuda_num=args.device_id



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=5):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())



class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x, if_relu_atlast = True):
        batch_size, num_dims, num_points = x.size()
        
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        if if_relu_atlast == False:
            return self.conv5(x).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x
        
class DGCNN_multi_knn(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x, if_relu_atlast = True):
        batch_size, num_dims, num_points = x.size()
        
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=True)[0]
        
        x = get_graph_feature(x1)
        x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2)
        x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=True)[0]
        
        x = get_graph_feature(x3)
        x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        if if_relu_atlast == False:
            return self.conv5(x).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x
        

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)

def symfn_avg(x):
    a = torch.nn.functional.avg_pool1d(x, x.size(-1))
    #a = torch.sum(x, dim=-1, keepdim=True) / x.size(-1)
    return a
    
def symfn_max(x):
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    #a = torch.sum(x, dim=-1, keepdim=True) / x.size(-1)
    return a


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
        else:
            weights = torch.nn.Linear(last, outp)
            init.xavier_normal_(weights.weight, gain=1.0)
        layers.append(weights)
#        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out



# The network used for point non-rigid registration, recurrent strategy
class Net_PointNR_v2(nn.Module):
    def __init__(self):
        super(Net_PointNR_v2, self).__init__()
        #unused parameters
        self.patch_num = args.patch_num
        self.dim_k = args.dim_k
        self.cycle = args.cycle
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        #basic settings
        self.emb_dims = args.emb_dims
        self.top_k = 1024
        self.state_dim = 1024
        ######################## LAYERS #########################
        
        ###### The lieAlgebra compution
        self.exp = LieAlgebra.se3.Exp # [B, 6] -> [B, 4, 4]
        self.rigid_transform = LieAlgebra.se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
        ###### End part
        
        ###### The layers for the source point cloud
        self.emb_nn_source = DGCNN(emb_dims=self.emb_dims)
        ###### End part
        
        ###### The layers for rigid registration 
        ### 1. Feature extraction
        self.emb_nn_rigid = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.pointer_rigid = Transformer(args=args)
        ### 3. Rotation and translation
        mlp_rt_rigid = [512,512,512,256,128,64]
        self.rt_mlp_rigid = MLPNet(self.top_k, mlp_rt_rigid, b_shared=True).layers
        self.rt_rigid = torch.nn.Conv1d(64, 6, 1)
        init.xavier_normal_(self.rt_rigid.weight, gain=1.0)
        ### 4. Init the state H
        self.emb_nn_source_state = DGCNN(emb_dims=self.state_dim)
        ###### End part
        
        
        ###### The layers for non-rigid registration 
        ### 1. Feature extraction
        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.pointer = Transformer(args=args)
        ### 3. point-wise weight
        mlp_w = [512,256,256,256]
        self.point_wise_weight_mlp = MLPNet(self.state_dim, mlp_w, b_shared=True).layers
        mlp_w_2 = [64]
        self.point_wise_weight_mlp_2 = MLPNet(256, mlp_w_2, b_shared=True).layers
        self.point_wise_weight = torch.nn.Conv1d(64, 1, 1)
        init.xavier_normal_(self.point_wise_weight.weight, gain=1.0)
        ### 3. Rotation and translation
        mlp_rt = [512,256]
        self.rt_mlp = MLPNet(self.state_dim, mlp_rt, b_shared=True).layers
        mlp_rt_2 = [256,128,64]
        self.rt_mlp_2 = MLPNet(512, mlp_rt_2, b_shared=True).layers
        mlp_rt_3 = [64,64,64]
        self.rt_mlp_3 = MLPNet(64, mlp_rt_3, b_shared=True).layers
        self.rt = torch.nn.Conv1d(64, 6, 1)
        init.xavier_normal_(self.rt.weight, gain=1.0)

        
        ###### The layers of GRU
        ### 1. Acquire the Z
        self.points_mlp_z = torch.nn.Conv1d(4*self.emb_dims + self.top_k + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_z.weight, gain=1.0)
        ### 2. Acquire the R
        self.points_mlp_r = torch.nn.Conv1d(4*self.emb_dims + self.top_k + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_r.weight, gain=1.0)
        ### 2. Acquire the H-wave
        self.points_mlp_hwave = torch.nn.Conv1d(4*self.emb_dims + self.state_dim  + self.top_k, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_hwave.weight, gain=1.0)
        
        ###### Others
        self.sigmoid = torch.nn.Sigmoid()
        
    def rigid_registration(self, source_points, target_points, state_h):
        this_batch_size = source_points.size()[0]
        src_embedding = self.emb_nn_rigid(source_points) 
        tgt_embedding = self.emb_nn_rigid(target_points)
        src_tgt, tgt_src = self.pointer_rigid(src_embedding, tgt_embedding)
        phi_src_tgt = src_tgt + src_embedding
        phi_tgt_src = tgt_src + tgt_embedding
#        phi_src_tgt =  src_embedding
#        phi_tgt_src =  tgt_embedding

        d_k = phi_tgt_src.size(1)
        scores = torch.matmul(phi_src_tgt.transpose(2, 1).contiguous(), phi_tgt_src) / math.sqrt(d_k)
        scores_global, _ = scores.topk(k=self.top_k, dim=2)
        scores_global = scores_global.transpose(1,2).view(this_batch_size, self.top_k, -1)
        
        global_phi_src_tgt = symfn_avg(phi_src_tgt).expand(-1,-1, phi_src_tgt.size()[-1])
        global_phi_tgt_src = symfn_avg(phi_tgt_src).expand(-1,-1, phi_tgt_src.size()[-1])
#        print(scores_global.size())
#        print(scores_global.size(), phi_src_tgt.size(), phi_tgt_src.size(), src_embedding.size())
        # Let's view the feature_concate as Xt.
        feature_concat = torch.cat((scores_global, phi_src_tgt, global_phi_src_tgt, global_phi_tgt_src, src_embedding),1)
        # The GRU 
        hx = torch.cat([feature_concat, state_h],dim=1)
        new_z = self.sigmoid(self.points_mlp_z(hx))
        new_r = self.sigmoid(self.points_mlp_r(hx))
        new_hwave = torch.tanh(self.points_mlp_hwave(torch.cat([torch.mul(new_r, state_h), feature_concat], dim=1)))
        state_h = torch.mul(new_hwave, new_z) +torch.mul(state_h, (1-new_z))
        
#        print(state_h.size())
        
        feature_concat_rigid_pooling = symfn_avg(state_h).view(this_batch_size, -1, 1)
        phi_feature = self.rt_mlp_rigid(feature_concat_rigid_pooling)
        phi = self.rt_rigid(phi_feature).view(this_batch_size,6)
        rigid_matrix = self.exp(phi).view(this_batch_size,1,4,4)
        deform_rigid_points = self.rigid_transform(rigid_matrix, source_points.transpose(1,2)).transpose(1,2)
        deformation_points = deform_rigid_points
        
        return phi, deform_rigid_points, deformation_points, rigid_matrix

    def forward(self, source_points, target_points, iteration=args.iteration, if_test=False):
        this_batch_size = source_points.size()[0]
        
        phi_list = []
        rigid_matrix_list = []
        point_weight_list = []
        deform_rigid_points_list = []
        deformation_points_list = []
        source_points = source_points.transpose(1,2)
        target_points = target_points.transpose(1,2)
        
        
        dmt_tgt = None
        tgt_dmt = None
        
        # get the feature of source
        src_embedding = self.emb_nn_source(source_points)
        # get the feature of target
        tgt_embedding = self.emb_nn(target_points)
        # init the hidden state 
        state_h = torch.tanh(self.emb_nn_source_state(source_points, if_relu_atlast = False))
        
        ###### The first iteration.
        phi, deform_rigid_points, deformation_points, rigid_matrix = self.rigid_registration(source_points, target_points, state_h)
        phi_list.append(phi)
        rigid_matrix_list.append(rigid_matrix)
        point_weight_list.append(torch.mean(target_points,dim=1)*0)
        deform_rigid_points_list.append(deform_rigid_points)
        deformation_points_list.append(deformation_points)
        
        
        for iter_stage in range(iteration-1):
            deformation_points = deformation_points.detach()
            dmt_embedding = self.emb_nn(deformation_points)
            dmt_tgt, tgt_dmt = self.pointer(dmt_embedding, tgt_embedding)
            phi_dmt_tgt = dmt_tgt + dmt_embedding 
            phi_tgt_dmt = tgt_dmt + tgt_embedding 
#            phi_dmt_tgt =  dmt_embedding 
#            phi_tgt_dmt =  tgt_embedding 
            
            d_k = phi_tgt_dmt.size(1)
            scores = torch.matmul(phi_dmt_tgt.transpose(2, 1).contiguous(), phi_tgt_dmt) / math.sqrt(d_k)
            scores_global, _ = scores.topk(k=self.top_k, dim=2)
            # this score_global is the correlation
            scores_global = scores_global.transpose(1,2).view(this_batch_size, self.top_k, -1)
            
            
            global_phi_dmt_tgt = symfn_avg(phi_dmt_tgt).expand(-1,-1, phi_dmt_tgt.size()[-1])
            global_phi_tgt_dmt = symfn_avg(phi_tgt_dmt).expand(-1,-1, phi_tgt_dmt.size()[-1])
            # the update_block can work now:
            # INput: ( scores_global, phi_dmt_tgt, phi_tgt_dmt, src_embedding, state_h ) 
            # OUTput: ( state_h, phi, point_weight)
            # The update_block should be writen as a function
#            state_h, phi, point_weight = self.update_block(scores_global, phi_dmt_tgt, phi_tgt_dmt, src_embedding, state_h)
            
            # Let's view the feature_concate as Xt.
            feature_concat = torch.cat((scores_global, phi_dmt_tgt, global_phi_dmt_tgt, global_phi_tgt_dmt, src_embedding),1)
            # The GRU 
            hx = torch.cat([feature_concat, state_h],dim=1)
            new_z = self.sigmoid(self.points_mlp_z(hx))
            new_r = self.sigmoid(self.points_mlp_r(hx))
            new_hwave = torch.tanh(self.points_mlp_hwave(torch.cat([torch.mul(new_r, state_h), feature_concat], dim=1)))
            state_h = torch.mul(new_hwave, new_z) +torch.mul(state_h, (1-new_z))
            
            weight_feature = self.point_wise_weight_mlp(state_h)
            phi_feature = self.rt_mlp(state_h)
            phi_feature = torch.cat([weight_feature, phi_feature], dim=1)
            
            weight_feature = self.point_wise_weight_mlp_2(weight_feature)
            point_weight = self.point_wise_weight(weight_feature)
#            point_weight = torch.sigmoid(self.point_wise_weight(weight_feature))
            
            phi_feature = self.rt_mlp_2(phi_feature)
            
            phi_feature_pooling = symfn_avg(phi_feature).view(this_batch_size, -1, 1)
            
            phi_feature_pooling = self.rt_mlp_3(phi_feature_pooling)
            
            phi = self.rt(phi_feature_pooling).view(this_batch_size,6)
            rigid_matrix = self.exp(phi).view(this_batch_size,1,4,4)
            deform_rigid_points = self.rigid_transform(rigid_matrix, source_points.transpose(1,2)).transpose(1,2)
            deformation_points = torch.mul(point_weight, deform_rigid_points) + torch.mul((1-point_weight), deformation_points)
            #append the tensors
            phi_list.append(phi)
            rigid_matrix_list.append(rigid_matrix)
            point_weight_list.append(point_weight)
            deform_rigid_points_list.append(deform_rigid_points)
            deformation_points_list.append(deformation_points)
            
        return phi_list, point_weight_list, deform_rigid_points_list, deformation_points_list, rigid_matrix_list



class Net_PointRR_v2(nn.Module):
    def __init__(self):
        super(Net_PointRR_v2, self).__init__()
        #unused parameters
        self.patch_num = args.patch_num
        self.dim_k = args.dim_k
        self.cycle = args.cycle
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        #basic settings
        self.emb_dims = args.emb_dims
        self.top_k = 1024
        self.state_dim = 1024
        ######################## LAYERS #########################
        
        ###### The lieAlgebra compution
        self.exp = LieAlgebra.se3.Exp # [B, 6] -> [B, 4, 4]
        self.rigid_transform = LieAlgebra.se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
        ###### End part
        
        ###### The layers for the source point cloud
        self.emb_nn_source = DGCNN(emb_dims=self.emb_dims)
        ###### End part
        
        ###### The layers for rigid registration 
        ### 1. Feature extraction
        self.emb_nn_rigid = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.pointer_rigid = Transformer(args=args)
        ### 3. Rotation and translation
        mlp_rt_rigid = [512,512,512,256,128,64]
        self.rt_mlp_rigid = MLPNet(self.top_k, mlp_rt_rigid, b_shared=True).layers
        self.rt_rigid = torch.nn.Conv1d(64, 6, 1)
        init.xavier_normal_(self.rt_rigid.weight, gain=1.0)
        ### 4. Init the state H
        self.emb_nn_source_state = DGCNN(emb_dims=self.state_dim)
        ###### End part
        
        
        ###### The layers for non-rigid registration 
        ### 1. Feature extraction
        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        ### 2. Transformer
        self.pointer = Transformer(args=args)
        ### 3. point-wise weight
        mlp_w = [512,256,256,256]
        self.point_wise_weight_mlp = MLPNet(self.state_dim, mlp_w, b_shared=True).layers
        mlp_w_2 = [64]
        self.point_wise_weight_mlp_2 = MLPNet(256, mlp_w_2, b_shared=True).layers
        self.point_wise_weight = torch.nn.Conv1d(64, 1, 1)
        init.xavier_normal_(self.point_wise_weight.weight, gain=1.0)
        ### 3. Rotation and translation
        mlp_rt = [512,256]
        self.rt_mlp = MLPNet(self.state_dim, mlp_rt, b_shared=True).layers
        mlp_rt_2 = [256,128,64]
        self.rt_mlp_2 = MLPNet(512, mlp_rt_2, b_shared=True).layers
        mlp_rt_3 = [64,64,64]
        self.rt_mlp_3 = MLPNet(64, mlp_rt_3, b_shared=True).layers
        self.rt = torch.nn.Conv1d(64, 6, 1)
        init.xavier_normal_(self.rt.weight, gain=1.0)

        
        ###### The layers of GRU
        ### 1. Acquire the Z
        self.points_mlp_z = torch.nn.Conv1d(4*self.emb_dims + self.top_k + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_z.weight, gain=1.0)
        ### 2. Acquire the R
        self.points_mlp_r = torch.nn.Conv1d(4*self.emb_dims + self.top_k + self.state_dim, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_r.weight, gain=1.0)
        ### 2. Acquire the H-wave
        self.points_mlp_hwave = torch.nn.Conv1d(4*self.emb_dims + self.state_dim  + self.top_k, self.state_dim, 1)
        init.xavier_normal_(self.points_mlp_hwave.weight, gain=1.0)
        
        ###### Others
        self.sigmoid = torch.nn.Sigmoid()
        
#    def rigid_registration(self, source_points, target_points):
#        this_batch_size = source_points.size()[0]
#        src_embedding = self.emb_nn_rigid(source_points) 
#        tgt_embedding = self.emb_nn_rigid(target_points)
#        src_tgt, tgt_src = self.pointer_rigid(src_embedding, tgt_embedding)
#        phi_src_tgt = src_tgt + src_embedding
#        phi_tgt_src = tgt_src + tgt_embedding
#        d_k = phi_tgt_src.size(1)
#        scores = torch.matmul(phi_src_tgt.transpose(2, 1).contiguous(), phi_tgt_src) / math.sqrt(d_k)
#        scores_global, _ = scores.topk(k=self.top_k, dim=2)
#        scores_global = scores_global.transpose(1,2).view(this_batch_size, self.top_k, -1)
#        feature_concat_rigid = scores_global
#        feature_concat_rigid_pooling = symfn_avg(feature_concat_rigid).view(this_batch_size, -1, 1)
#        phi_feature = self.rt_mlp_rigid(feature_concat_rigid_pooling)
#        phi = self.rt_rigid(phi_feature).view(this_batch_size,6)
#        rigid_matrix = self.exp(phi).view(this_batch_size,1,4,4)
#        deform_rigid_points = self.rigid_transform(rigid_matrix, source_points.transpose(1,2)).transpose(1,2)
#        deformation_points = deform_rigid_points
#        
#        return phi, deform_rigid_points, deformation_points, rigid_matrix

    def forward(self, source_points, target_points, iteration=args.iteration, if_test=False):
        this_batch_size = source_points.size()[0]
        
        rigid_matrix_list = []
        deformation_points_list = []
        rigid_matrix_accumulation_list = []
        
        source_points = source_points.transpose(1,2)
        target_points = target_points.transpose(1,2)
        
        # get the feature of source
        src_embedding = self.emb_nn_source(source_points)
        # get the feature of target
        tgt_embedding = self.emb_nn(target_points)
        # init the hidden state 
        state_h = torch.tanh(self.emb_nn_source_state(source_points, if_relu_atlast = False))
        
        rigid_matrix_accumulation = torch.eye(4).unsqueeze(0).expand(this_batch_size,-1,-1).detach().cuda()
#        print('rigid_matrix_accumulation:',rigid_matrix_accumulation)
        
        ###### The first iteration.
#        phi, deformation_points, rigid_matrix = self.rigid_registration_one_iteration(source_points, target_points, state_h)
#        phi_list.append(phi)
#        rigid_matrix_list.append(rigid_matrix)
#        point_weight_list.append(torch.mean(target_points,dim=1)*0)
#        deform_rigid_points_list.append(deform_rigid_points)
#        deformation_points_list.append(deformation_points)
        
        dmt_tgt = None
        tgt_dmt = None
        
        deformation_points = source_points
        
        for iter_stage in range(iteration):
            deformation_points = deformation_points.detach()
            dmt_embedding = self.emb_nn(deformation_points)
            dmt_tgt, tgt_dmt = self.pointer(dmt_embedding, tgt_embedding)
            phi_dmt_tgt = dmt_tgt + dmt_embedding 
            phi_tgt_dmt = tgt_dmt + tgt_embedding 
            d_k = phi_tgt_dmt.size(1)
            scores = torch.matmul(phi_dmt_tgt.transpose(2, 1).contiguous(), phi_tgt_dmt) / math.sqrt(d_k)
            scores_global, _ = scores.topk(k=self.top_k, dim=2)
            # this score_global is the correlation
            scores_global = scores_global.transpose(1,2).view(this_batch_size, self.top_k, -1)
            
            
            global_phi_dmt_tgt = symfn_avg(phi_dmt_tgt).expand(-1,-1, phi_dmt_tgt.size()[-1])
            global_phi_tgt_dmt = symfn_avg(phi_tgt_dmt).expand(-1,-1, phi_tgt_dmt.size()[-1])
            # the update_block can work now:
            # INput: ( scores_global, phi_dmt_tgt, phi_tgt_dmt, src_embedding, state_h ) 
            # OUTput: ( state_h, phi, point_weight)
            # The update_block should be writen as a function
#            state_h, phi, point_weight = self.update_block(scores_global, phi_dmt_tgt, phi_tgt_dmt, src_embedding, state_h)
            
            # Let's view the feature_concate as Xt.
            feature_concat = torch.cat((scores_global, phi_dmt_tgt,  global_phi_dmt_tgt, global_phi_tgt_dmt, src_embedding),1)
            # The GRU 
            hx = torch.cat([feature_concat, state_h],dim=1)
            new_z = self.sigmoid(self.points_mlp_z(hx))
            new_r = self.sigmoid(self.points_mlp_r(hx))
            new_hwave = torch.tanh(self.points_mlp_hwave(torch.cat([torch.mul(new_r, state_h), feature_concat], dim=1)))
            state_h = torch.mul(new_hwave, new_z) +torch.mul(state_h, (1-new_z))
            
            weight_feature = self.point_wise_weight_mlp(state_h)
            phi_feature = self.rt_mlp(state_h)
            phi_feature = torch.cat([weight_feature, phi_feature], dim=1)
            
            weight_feature = self.point_wise_weight_mlp_2(weight_feature)
            point_weight = self.point_wise_weight(weight_feature)
            
            phi_feature = self.rt_mlp_2(phi_feature)
            
            phi_feature_pooling = symfn_avg(phi_feature).view(this_batch_size, -1, 1)
            
            phi_feature_pooling = self.rt_mlp_3(phi_feature_pooling)
            
            phi = self.rt(phi_feature_pooling).view(this_batch_size,6)
            rigid_matrix = self.exp(phi).view(this_batch_size,1,4,4)
            
            deformation_points = self.rigid_transform(rigid_matrix, deformation_points.transpose(1,2)).transpose(1,2)
            rigid_matrix_accumulation = torch.bmm(rigid_matrix.view(this_batch_size,4,4), rigid_matrix_accumulation)
            #append the tensors
            rigid_matrix_list.append(rigid_matrix)
            deformation_points_list.append(deformation_points)
            rigid_matrix_accumulation_list.append(rigid_matrix_accumulation)
            
        return rigid_matrix_list, deformation_points_list, rigid_matrix_accumulation_list

        

