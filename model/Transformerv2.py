import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from .encoder import ScaleGraphBlock
from .loss_functions import GatheringLossDim, GatheringLoss

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
    
class SpatioTemporalEncoderLayerParallel(nn.Module):
    def __init__(self, config):
        super(SpatioTemporalEncoderLayerParallel, self).__init__()
        self.encoder_layer = EncoderLayer(
                    AttentionLayer(config.input_c, config.win_size, config.n_heads, dropout=config.dropout), 
                    config.win_size, config.d_ff, dropout=config.dropout, activation='gelu')
        self.encoder_layer2 = EncoderLayer(
                    AttentionLayer(config.win_size, config.input_c, config.n_heads, dropout=config.dropout), 
                    config.input_c, config.d_ff, dropout=config.dropout, activation='gelu') 
        self.norm = nn.LayerNorm(config.input_c)

    def forward(self, x):
        x_t = self.encoder_layer2(x)

        x = x.transpose(-1, 1)
        x = self.encoder_layer(x)
        x = x.transpose(-1, 1)
        
        return self.norm(x+x_t) 
    
class SpatioTemporalEncoderLayerSerial(nn.Module): # 不做了
    def __init__(self, config):
        super(SpatioEncoderLayer, self).__init__()
        self.encoder_layer = EncoderLayer(
                    AttentionLayer(config.input_c, config.win_size, config.n_heads, dropout=config.dropout), 
                    config.win_size, config.d_ff, dropout=config.dropout, activation='gelu')
        self.encoder_layer2 = EncoderLayer(
                    AttentionLayer(config.win_size, config.input_c, config.n_heads, dropout=config.dropout), 
                    config.input_c, config.d_ff, dropout=config.dropout, activation='gelu') 
        self.norm = nn.LayerNorm(config.input_c)

    def forward(self, x):
        x_t = self.encoder_layer2(x)

        x = x.transpose(-1, 1)
        x = self.encoder_layer(x)
        x = x.transpose(-1, 1)
        
        return self.norm(x+x_t) 

class SpatioEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SpatioEncoderLayer, self).__init__()
        self.encoder_layer = EncoderLayer(
                    AttentionLayer(config.input_c, config.win_size, config.n_heads, dropout=config.dropout), 
                    config.win_size, config.d_ff, dropout=config.dropout, activation='gelu')

    def forward(self, x):
        x = x.transpose(-1, 1)
        x = self.encoder_layer(x)
        x = x.transpose(-1, 1)
        return x  
    
class TemporalEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TemporalEncoderLayer, self).__init__()
        self.encoder_layer = EncoderLayer(
                    AttentionLayer(config.win_size, config.input_c, config.n_heads, dropout=config.dropout), 
                    config.input_c, config.d_ff, dropout=config.dropout, activation='gelu') 

    def forward(self, x):
        x = self.encoder_layer(x)
        return x  
    
# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        embeds = []
        embeds.append(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            if self.norm is not None:
                x = self.norm(x)
            embeds.append(x)

        embeds = torch.stack(embeds,dim=0)

        return x,embeds
    
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
    def __init__(self, config, device=None):
        super(TransformerVar, self).__init__()
        self.config = config

        # Encoding
        self.embedding = InputEmbedding(in_dim=config.input_c, d_model=config.input_c, dropout=config.dropout, device=device) 
        self.layer_norm = nn.LayerNorm(config.input_c)
        
        # Encoder
        if config.encoder_type == "Temporal":
            self.encoder = Encoder([TemporalEncoderLayer(config) for _ in range(config.encoder_layer_num)],norm_layer = self.layer_norm)
        elif config.encoder_type == "Spatio":
            self.encoder = Encoder([SpatioEncoderLayer(config)  for _ in range(config.encoder_layer_num)],norm_layer = self.layer_norm)
        elif config.encoder_type == "SpatioTemporal":
            self.encoder = Encoder([SpatioTemporalEncoderLayerParallel(config) for _ in range(config.encoder_layer_num)],norm_layer = self.layer_norm)

        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(config.win_size, config.input_c, config.n_heads, dropout=config.dropout), 
        #             config.input_c, config.d_ff, dropout=config.dropout, activation='gelu') 
        #         for _ in range(config.encoder_layer_num)
        #     ],
        #     norm_layer = nn.LayerNorm(config.input_c)
        # )
        
        # ours
        self.weak_decoder = Decoder(config.input_c, config.input_c, d_ff=config.d_ff, activation='gelu', dropout=config.dropout)
        self.mse_loss = nn.MSELoss(reduction="none")
        
    def forward(self, x, c = None):
        '''
        x (input time window) : N x L x enc_in
        c (cluster center): encoder_layer_num x enc_in
        '''
        B, T, N = x.size()

        embed_x = self.embedding(x)   # embeddin : B x T x N
        embed, layer_embeds = self.encoder(embed_x)  # layer_embeds: encoder_layer_num x B x T x N

        layer_embeds = layer_embeds[:self.config.layer_cluster_num]
        if c is None:
            c = layer_embeds.mean(1).mean(1) 
        c = c.unsqueeze(1).unsqueeze(1).repeat(1,B,T,1)
        cluster_loss = self.mse_loss(layer_embeds,c)

        out = self.weak_decoder(embed)
        recon_loss = self.mse_loss(out,x)
        
        '''
        out (reconstructed input time window) : N x L x enc_in
        enc_in == c_out
        '''
        return {"out":out, 
                "layer_embeds":layer_embeds,
                "recon_loss":recon_loss,
                "cluster_loss":cluster_loss
                }
            
