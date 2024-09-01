import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model.gat import GAT,GraphAttentionLayerOriginal,GraphAttentionLayerOriginalv2
import torch.fft
from math import sqrt
from .mem import MemoryModule
# from layers.Embed import DataEmbedding

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,n_heads=1):
        super(GraphAttentionLayer, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features,out_features)
        self.attention = self_attention(FullAttention, out_features, n_heads=n_heads)#todo fix
        self.conv1 = nn.Conv1d(in_channels=out_features, out_channels=out_features*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=out_features*4, out_channels=out_features, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # h: B N d_model     adj: B N N
        _,e = self.attention(Wh,Wh,Wh)

        zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0.1, e, zero_vec)
        attention = e.mean(1)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.einsum("ben,bnd->bed",attention,Wh)
        y = x = self.norm1(h_prime+h)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None,return_attn = False):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if return_attn:
            return self.norm2(x + y), attn
        return self.norm2(x + y)

class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        # out = [x,h]
        
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class GraphBlock(nn.Module):
    def __init__(self, c_out , d_model , conv_channel, skip_channel,
                        gcn_depth , dropout, propalpha ,seq_len , node_dim):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1 , conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len , (1, seq_len ))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3) # B 1 d_model T
        out = self.start_conv(out) # B 32 38 T
        out = self.gelu(self.gconv1(out , adp)) # B 32 38 T
        out = self.end_conv(out).squeeze() # B T 38 1 -> B T 38
        out = self.linear(out) # 

        return self.norm(x + out)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class GraphModule(nn.Module):
    def __init__(self,configs,d_model) -> None:
        super(GraphModule,self).__init__()
        self.n_heads = configs.n_heads
        self.configs = configs
        self.att = Attention_Block(d_model, None, n_heads=self.n_heads, dropout=configs.dropout, activation="gelu")
        self.t = 0.00001 # lhy: 不确定这个是否设置的合适
        if self.configs.gat_type == 'qkv':
            self.gat = GraphAttentionLayer(in_features=d_model, out_features=d_model,dropout=configs.dropout, alpha=0.2, concat=False)
        elif self.configs.gat_type == 'line':
            self.gat = GraphAttentionLayerOriginal(in_features=d_model, out_features=d_model,dropout=configs.dropout, alpha=0.2, concat=False)
        pass
    
    def forward(self, x):
        out, attn = self.att(x,return_attn = True) # b(l),n_head,n,n
        adj = attn.permute(0,2,3,1)
        adj = F.gumbel_softmax(adj,self.t)
        adj = adj.permute(3,0,1,2)
        adj = adj[:self.n_heads-1].max(0)[0]
        out1 = self.gat(x,adj) # x ?
        return out1,adj
        
class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.configs = configs
        
        # todo: 使用特征压缩/映射提取深层特征

        # todo: 为每一个expert训练一个模型 # 卧槽，这个居然一直没干
        self.att0 = Attention_Block(configs.input_c, None,n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.att1 = Attention_Block(configs.input_c, None,n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.att2 = Attention_Block(configs.seq_len, None,n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.att3 = Attention_Block(configs.input_c, None,n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")


        self.adj_build = Attention_Block(configs.seq_len, None, n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        
        if self.configs.gat_type == 'qkv':
            self.t = 0.00001 # lhy: 不确定这个是否设置的合适
            self.gat = GraphAttentionLayer(in_features=configs.seq_len, out_features=configs.seq_len,dropout=configs.dropout, alpha=0.2, concat=False)
            self.mem_module = MemoryModule(n_memory=configs.n_memory, fea_dim=configs.input_c*configs.input_c*configs.n_heads)
        elif self.configs.gat_type == 'line':
            self.mem_module = MemoryModule(n_memory=configs.n_memory, fea_dim=configs.input_c*configs.input_c*configs.n_heads)
            self.t = 0.00001 # lhy: 不确定这个是否设置的合适
            self.gat = GraphAttentionLayerOriginal(in_features=configs.seq_len, out_features=configs.seq_len,dropout=configs.dropout, alpha=0.2, concat=False)
        elif self.configs.gat_type == 'linev2':
            self.mem_module = MemoryModule(n_memory=configs.n_memory, fea_dim=self.configs.d_gat)
            self.t = 0.00001 # lhy: 不确定这个是否设置的合适
            self.gat = GraphAttentionLayerOriginalv2(in_features=configs.seq_len, out_features=configs.seq_len,d_gat=self.configs.d_gat,dropout=configs.dropout, alpha=0.2, concat=False)

        # self.att_inter = self_attention(FullAttention, configs.seq_len, n_heads=1)
        # todo: 将指标间相关性固化为边，然后再根据图结构差距判断异常

        self.norm = nn.LayerNorm(configs.input_c)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=configs.dropout)
        # self.graph_module = nn.ModuleList()
        # for i in range(self.k):
        #     self.graph_module.append(
        #         GraphModule(configs,configs.seq_len)
        #     )

    def forward(self, x):
        B, T, N = x.size()
        if self.configs.multi_scale_type == "true":
            scale_list, scale_weight = FFT_for_Period(x, self.k)
        elif self.configs.multi_scale_type == "one":
            scale_list, scale_weight = np.array([self.configs.specific_scale_size,]*self.configs.top_k),torch.ones((B,self.configs.top_k)).to(x.device)
        max_scale = self.seq_len
        res = []
        res_attn = []
        res_time_attn = []
        res_adj = []
        length_list =[]
        res_graph_queries = []
        res_edge_queries = []
        for i in range(self.k):
            scale = scale_list[i]
            # Gconv
            # x = self.gconv[i](x)
            # paddng
            if (self.seq_len) % scale != 0:
                L = ((self.seq_len) // scale) + 1
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                L = (self.seq_len) // scale
                length = self.seq_len
                out = x
            length_list.append(L)
            out = out.reshape(B, L, scale, N)  #
            padding2 = torch.zeros([B, L, (max_scale - scale), N]).to(out.device)
            out = torch.cat([out, padding2], dim=2) # B, length // scale, max_scale, N

            if self.configs.gat_type == 'linev2':
                out4 = out.reshape(-1,T,N).permute(0,2,1) # BL,T,N -> BL,N,T
                out4,edges = self.gat(out4)  # b(l),n,n,g

                edge_queries = edges.reshape(-1,N,N,self.configs.d_gat).mean(-2).mean(-2) # bl,n,n,g -> bl,g
                outputs = self.mem_module(edge_queries)
                # _, _ = outputs['output'], outputs['attn']
                edge_queries = edge_queries.reshape(B,-1,self.configs.d_gat).permute(1,0,2) # lb,g
                res_edge_queries.append(edge_queries)

                out4 = out4.reshape(B,-1,N,T).permute(0,1,3,2) # B,L,N,T -> B,L,T,N

            else:
                # 学习指标见相关性
                # adj = torch.zeros([L,B,N,N])
                out4 = out.reshape(-1,T,N).permute(0,2,1) # BL,T,N -> BL,N,T
                _, adj = self.adj_build(out4,return_attn = True) # b(l),n_head,n,n
                
                if self.configs.adj_fusion_source == "adj":
                    adj = adj.permute(0,2,3,1) # b(l),n,n,n_head
                    adj = F.gumbel_softmax(adj,self.t)
                    adj = adj.permute(0,3,1,2) # b(l),n_head,n,n
                graph_queries = adj.reshape(-1,self.configs.n_heads*N*N) # bl,n_head,n,n
                outputs = self.mem_module(graph_queries) # mem out 出来的图是浮点权重，而不是01边，需要从新
                adj, attn = outputs['output'], outputs['attn'] # Bl,NN*2
                graph_queries = graph_queries.reshape(B,-1,self.configs.n_heads,N,N).permute(1,0,2,3,4)
                res_graph_queries.append(graph_queries)
                
                if self.configs.adj_fusion_type == "mem":
                    adj = adj[:,self.configs.n_heads*N*N:].reshape(-1,self.configs.n_heads,N,N) #bl,n_head,n,n
                elif self.configs.adj_fusion_type == "none":
                    adj = adj[:,:self.configs.n_heads*N*N].reshape(-1,self.configs.n_heads,N,N) #bl,n,n
                adj = adj.permute(0,2,3,1) # b(l),n,n,n_head
                adj = F.gumbel_softmax(adj,self.t)
                adj = adj[:,:,:,:self.configs.n_heads-1].max(-1)[0] # bl, n, n

                out4 = self.gat(out4,adj) 
                # adj = adj.reshape(self.configs.n_heads,B,L,N,N).permute(0,1,3,4,2).unsqueeze(-1).repeat(scale).reshape(self.configs.n_heads,B,N,N,-1)[:,:,:,:,:self.seq_len]
                # adj = adj.permute(1,0,4,2,3) # adj: n_heads,B,N,N,T->B,n_heads,T,N,N
                adj = adj.reshape(B,L,N,N).permute(1,0,2,3) # L,B,N,N
                out4 = out4.reshape(B,-1,N,T).permute(0,1,3,2) # B,L,N,T -> B,L,T,N

                res_adj.append(adj)
            
            
            # for whole time window time attention [gpu boom]
            
            # out0 = out.reshape(B, -1, N)
            # out0 = self.att3(out0)
            # out0 = self.gelu(out0)
            # out0 = out0.reshape(B,-1, max_scale, N)
            
            # for time-patch-intra-attetion

            out1 = out.reshape(-1, max_scale, N) # B L T N
            # out1 = self.norm(self.n2d(out1))
            out1,time_attn = self.att0(out1,return_attn = True)  # B L T N | B L n_head T T
            out1 = self.gelu(out1) # BL,S,N
            out1 = out1.reshape(B,-1, max_scale, N) # B,L,S,N
            time_attn = time_attn.reshape(-1,T*T) # (B L n_head) (T T)
            
            # for inter-metric correlation
            
            out3 = out.permute(0, 1, 3, 2)
            out3 = out3.reshape(-1, N, max_scale) # BL,N,T
            # out3 = self.normt(self.t2d(out3))
            out3,inter_attn = self.att2(out3,return_attn = True) # todo: 固化指标间依赖
            out3 = self.gelu(out3) # BS,L,N
            out3 = out3.reshape(B, -1, N, max_scale).permute(0, 1, 3, 2) # B,L,S,N
          
            # for time-patch-inter-attetion
            
            out2 = out.permute(0, 2, 1, 3) # B T L N 
            out2 = out2.reshape(B * max_scale, -1, N)
            # out2 = self.norm(self.n2d2(out2))
            out2 = self.att1(out2)
            out2 = self.gelu(out2) # BS,L,N
            out2 = out2.reshape(B, max_scale, -1, N).permute(0, 2, 1, 3) # B,L,T,N
           

            # for feature fusion

            # out = out0 + out1 + out2 + out3 # B,L,S,N # todo：尝试利用拼接+全连接网络来融合这些特征
            if self.configs.encode_feature_type == "embedding":
                out = out
            elif self.configs.encode_feature_type == "patch_time":
                out = out1 + out 
            elif self.configs.encode_feature_type == "inter_patch_time":
                out = out1 + out + out2
            elif self.configs.encode_feature_type == "patch_inter_metric":
                out = out1 + out + out2 + out3
            elif self.configs.encode_feature_type == "patch_inter_metric_gat":
                out = out1 + out + out2 + out4
            # out = out
            out = self.norm(out)
            out = out[:,:,:scale,:].reshape(B,-1,N)[:,:self.seq_len,:] # B,T,N # inverse padding
            res.append(out)
            


        res = torch.stack(res, dim=-1) # (B,T,N,k)
        res_with_dim = res
        if self.configs.gat_type == 'linev2':
            res_edge_queries = torch.cat(res_edge_queries,dim=0).reshape(-1,self.configs.d_gat) # klb,nng
            res_graph_queries  = None
            res_adj = None
        else:
            res_edge_queries = None
            res_adj = torch.cat(res_adj,dim=0).reshape(-1,N*N) # k,L,B,N,N
            res_graph_queries = torch.cat(res_graph_queries,dim=0).reshape(-1,self.configs.n_heads*N*N) # klb,n_headnn
        # todo: 思考如何融合多尺度下的指标间相关性 # 求和在softmax？

        
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        
        return {"res_with_dim":res_with_dim,
                "res":res,
                "res_adj":res_adj,
                "scale_list":scale_list,
                "length_list":length_list,
                "res_graph_queries":res_graph_queries,
                "res_edge_queries":res_edge_queries
                }