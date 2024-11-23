import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from .multihead_attention import MultiheadAttention, MultiheadAttention_with_flash
from .jax_attention import MultiheadAttention_with_jax
from .feedforward_network import FeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim,
                 ffn_dim,
                 activation_fn,
                 ffn_dropout,
                 num_head,
                 dropout

                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadAttention(embed_dim,num_head)
        self.ffn = FeedForwardNetwork(
                embed_dim, ffn_dim, activation_fn, ffn_dropout
            )
        self.dropout_module = nn.Dropout(dropout)  

    def forward(self, x):

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
        )

        x = self.dropout_module(x)

        x += residual
        residual = x

        x = self.final_layer_norm(x)

        x = self.ffn(x)
        x += residual

        return x
    

class Encoder(nn.Module):
    def __init__(
        self,
        combine_channel_depth, 
        emb_dim, 
        num_head, 
        comb_ffn_dim, 
        activation_fn,
        ffn_dropout = 0.1, 
        dropout = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(combine_channel_depth,):
            self.layers.append(
                EncoderLayer(
                    emb_dim,
                    comb_ffn_dim,
                    activation_fn,
                    ffn_dropout,
                    num_head,
                    dropout
                )
            )


    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            x  = layer(x)
        
        return x
    

class EncoderLayer_with_flash(nn.Module):
    def __init__(self, 
                 embed_dim,
                 ffn_dim,
                 activation_fn,
                 ffn_dropout,
                 num_head,
                 dropout

                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadAttention_with_flash(embed_dim,num_head)
        self.ffn = FeedForwardNetwork(
                embed_dim, ffn_dim, activation_fn, ffn_dropout
            )
        self.dropout_module = nn.Dropout(dropout)  


    def forward(self, x):

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
        )

        x = self.dropout_module(x)

        x += residual
        residual = x

        x = self.final_layer_norm(x)

        x = self.ffn(x)
        x += residual

        return x
    

class Encoder_with_flash(nn.Module):
    def __init__(
        self,
        combine_channel_depth, 
        emb_dim, 
        num_head, 
        comb_ffn_dim, 
        activation_fn,
        ffn_dropout = 0.1, 
        dropout = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(combine_channel_depth,):
            self.layers.append(
                EncoderLayer(
                    emb_dim,
                    comb_ffn_dim,
                    activation_fn,
                    ffn_dropout,
                    num_head,
                    dropout
                )
            )


    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            x  = layer(x)
        
        return x
    
class EncoderLayer_with_jax(nn.Module):
    def __init__(self, 
                 embed_dim,
                 ffn_dim,
                 activation_fn,
                 ffn_dropout,
                 num_head,
                 dropout

                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        # self.self_attn = MultiheadAttention(embed_dim,num_head)
        self.self_attn = MultiheadAttention_with_jax(embed_dim,num_head) # jax로 행렬 연산 최적화
        self.ffn = FeedForwardNetwork(
                embed_dim, ffn_dim, activation_fn, ffn_dropout
            )
        self.dropout_module = nn.Dropout(dropout)  


    def forward(self, x):
        print(f'x = {x.shape}')
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
        )

        x = self.dropout_module(x)

        x += residual
        residual = x

        x = self.final_layer_norm(x)

        x = self.ffn(x)
        x += residual

        return x
    

class Encoder_with_jax(nn.Module):
    def __init__(
        self,
        combine_channel_depth, 
        emb_dim, 
        num_head, 
        comb_ffn_dim, 
        activation_fn,
        ffn_dropout = 0.1, 
        dropout = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(combine_channel_depth,):
            self.layers.append(
                EncoderLayer_with_jax(
                    emb_dim,
                    comb_ffn_dim,
                    activation_fn,
                    ffn_dropout,
                    num_head,
                    dropout
                )
            )


    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            x  = layer(x)
        
        return x