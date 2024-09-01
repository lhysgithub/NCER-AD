import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from .encoder import ScaleGraphBlock
from .loss_functions import GatheringLossDim, GatheringLoss

# ours
from .ours_memory_module import MemoryModule
# memae
# from .memae_memory_module import MemoryModule
# mnad
# from .mnad_memory_module import MemoryModule

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)    # N x L x C(=d_model)
    
# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''

        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out      # N x L x c_out

class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, config, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial
        self.config = config

        # Encoding
        self.embedding = InputEmbedding(in_dim=config.input_c, d_model=config.input_c, dropout=dropout, device=device) 
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, config.input_c, n_heads, dropout=dropout
                    ), config.input_c, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(config.input_c)
        )

        # MSGEncoder
        self.encoder2 = ScaleGraphBlock(self.config)
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=c_out, fea_dim2=config.seq_len, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        # self.scale_fusion = nn.Linear(self.config.top_k*c_out*2,1*c_out)
        # self.scale_fusion_2 = nn.Linear(self.config.top_k*c_out,c_out)

        self.gathering_loss = GatheringLossDim()
        
        # ours
        self.weak_decoder = Decoder(c_out, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def get_cluster_loss(self,length_list,scale_list,querys,mem_module,x):
        # 计算graph_query与mem_graph的差距
        B, T, N = x.size()
        mems = mem_module.mem # n_mem,n_headnn
        querys = querys.reshape(-1,B,mem_module.fea_dim) # klb,n_headnn
        base = 0
        all_scale_query = []
        all_scale_loss = []
        for j in range(len(length_list)): # 对于每一个尺度
            L = length_list[j]
            S = scale_list[j]
            scale_query = querys[base:base+L].reshape(L,B,-1).permute(1,2,0).unsqueeze(-1).repeat(1,1,1,S).reshape(B,mem_module.fea_dim,-1)[:,:,:self.config.seq_len].permute(0,2,1) # B T n_headnn
            all_scale_query.append(scale_query)
            base += L
        all_scale_query = torch.stack(all_scale_query,dim=-1).reshape(B,T,-1,self.config.top_k).permute(0,3,1,2) # B K T n_headnn
        all_scale_loss = self.gathering_loss(all_scale_query, mems)
        # all_scale_loss = torch.softmax(all_scale_loss/self.config.temperature, dim=-1)
        cluster_loss = all_scale_loss
        return cluster_loss

    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
        B, T, N = x.size()
        x = self.embedding(x)   # embeddin : N x L x C(=d_model)
        out_single_scale = self.encoder(x)
        enc_outputs = self.encoder2(x) # BTNK BTN K,B,n_head,N,N
        out_with_dim = enc_outputs["res_with_dim"]
        out_without_dim = enc_outputs["res"]
        out_adj = enc_outputs["res_adj"]
        scale_list = enc_outputs["scale_list"]
        length_list = enc_outputs["length_list"]
        graph_queries = enc_outputs["res_graph_queries"]
        edge_queries = enc_outputs["res_edge_queries"]
        out_with_dim = out_with_dim.permute(0, 3, 1, 2).reshape(-1,N) # BKTN # todo 根据adj判断异常

        if self.config.fusion_type == "Memory":
            # pass
            queries = out_with_dim # (BKT)N
            outputs = self.mem_module(queries) 
            out, attn = outputs['output'], outputs['attn'] # (BKT)N*2 
            out = out.reshape(B,-1,T,N*2).permute(0,2,3,1).reshape(B,T,-1)
            out = self.scale_fusion(out).reshape(B,T,N)

            queries, mem_items = queries, self.mem_module.mem
            queries = queries.reshape(B,-1,T,N) # B K T N 
            gathering_loss = GatheringLossDim()
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/self.config.temperature, dim=-1)
        else:
            if self.config.fusion_type == "FFT":
                queries = out_without_dim # (BT)N
            elif self.config.fusion_type == "Mean":
                queries = out_with_dim.reshape(B,-1,T,N).mean(1) # B T N 
            elif self.config.fusion_type == "Max":
                queries = out_with_dim.reshape(B,-1,T,N).max(1)[0] # B T N 
            elif self.config.fusion_type == "Cat":
                queries = out_with_dim.reshape(B,-1,T,N).permute(0,2,1,3).reshape(B,T,-1) # B T K N
                queries = self.scale_fusion_2(queries) 
            elif self.config.fusion_type == "old":
                queries = out_single_scale # BTN
            # queries = out_single_scale # BTN
            out = queries
            outputs = self.mem_module(queries) 
            out, attn = outputs['output'], outputs['attn'] # (BKT)N*2 
            out = out[:,:,:N]

            queries, mem_items = queries, self.mem_module.mem
            queries = queries.reshape(B,T,N) # B T N 
            gathering_loss = GatheringLoss(reduce=False)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/self.config.temperature, dim=-1)
        
        out = self.weak_decoder(out)

        # 计算出入度
        # cluster_loss = None
        if self.config.cluster_loss == "degree":
            out_adj = out_adj.reshape(-1,B,N,N)
            in_out_degree = out_adj.transpose(-1,-2).sum(-1) + out_adj.sum(-1) # KL/T,B,N

            base = 0
            res_in_out_degree_distance = []
            for j in range(len(length_list)): # 对于每一个尺度
                L = length_list[j]
                S = scale_list[j]
                scale_in_out_degree_distance = []
                for k in range(L): # 对于每一个path
                    if k == 0: # 若没有之前的出入度，则出入度差为0
                        last_in_out_degree = in_out_degree[base+k]
                        current_in_out_degree = in_out_degree[base+k]
                    else: # 若存在之前的出入度，出入度差为当前出入度与之前出入度的差
                        last_in_out_degree = in_out_degree[base+k-1]
                        current_in_out_degree = in_out_degree[base+k]
                    in_out_degree_distance = F.l1_loss(current_in_out_degree,last_in_out_degree,reduction="none").mean(-1).unsqueeze(-1).repeat(1,S) # B,S  # 将出入度的差扩展至整个patch
                    scale_in_out_degree_distance.append(in_out_degree_distance)
                scale_in_out_degree_distance = torch.cat(scale_in_out_degree_distance,dim=-1)[:,:T] # 聚合所有patch的出入度差，得到整合时间窗口的出入度差
                res_in_out_degree_distance.append(scale_in_out_degree_distance)
                base += L
            res_in_out_degree_distance = torch.stack(res_in_out_degree_distance,-1).mean(-1) # 聚合所有尺度的出入度差，并取平均
            cluster_loss = res_in_out_degree_distance
        elif self.config.cluster_loss == "graph":
            # 计算graph_query与mem_graph的差距
            cluster_loss = self.get_cluster_loss(length_list,scale_list,graph_queries,self.encoder2.mem_module,x)
        elif self.config.cluster_loss == "embedding":
            cluster_loss = self.get_cluster_loss(length_list,scale_list,edge_queries,self.encoder2.mem_module,x)

        '''
        out (reconstructed input time window) : N x L x enc_in
        enc_in == c_out
        '''
        return {"out":out, 
                "latent_score":latent_score,
                "cluster_loss":cluster_loss,
                }
            
