import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def get_sinusoid_encoding(num_tokens, token_len):
            """ Make Sinusoid Encoding Table

                Args:
                    num_tokens (int): number of tokens
                    token_len (int): length of a token
                    
                Returns:
                    (torch.FloatTensor) sinusoidal position encoding table
            """
            def get_position_angle_vec(i):
                return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

            sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

            return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    
class Custom_pat_Embedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x, masked_position=None, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B,C,H,W) -> (B,dim,H_win,W_win) -> (B,dim,H_win*W_win) -> (B,H_win*W_win,dim) 

        x += self.pos_embedding

        return x
    
class PositionalEmbedding(nn.Module):

    def __init__(self, seq_len, emb_dim, num_channels, relative = False):
        super().__init__()
        # channel_length = emb_dim//num_channels
        self.num_channels = num_channels
        self.seq_len = seq_len
        if relative:
            self.pos_embedding = nn.Parameter(get_sinusoid_encoding(num_tokens=seq_len, token_len=emb_dim),requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
    def forward(self, x):     

        x+=self.pos_embedding #.repeat(1,1,self.num_channels).type_as(x)
        return x