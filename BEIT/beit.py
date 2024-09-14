import torch
import torch.nn as nn
import torch.nn.functional as F


from .encoder import Encoder, Encoder_with_flash
from .embedding import (
    PositionalEmbedding,
    Custom_pat_Embedding
)
from .multiway_network import MutliwayEmbedding
# from .multihead_attention import MultiheadAttention, MultiheadAttention_with_flash

class BEiT3(nn.Module):
    def __init__(self, 
                num_channels = 3,
                img_size=224,
                patch_size=16,
                in_chans=3,
                emb_dim=1024,
                num_head=16,
                comb_ffn_dim=3072,
                combine_channel_depth = 9,
                ffn_dropout = 0.1,
                relative_pos = False,
                dropout = 0.1,
                num_classes = 21,
                activation_fn = 'gelu',
                ):
        super().__init__()
        self.num_channels = num_channels

        seq_len = (img_size//patch_size)**2

        self.patch_embedding = Custom_pat_Embedding(img_size,
                                                    patch_size,
                                                    in_chans,
                                                    emb_dim)
        self.positional_embedding = PositionalEmbedding(seq_len, 
                                                   emb_dim, 
                                                   num_channels, 
                                                   relative=relative_pos)
        
        self.encoder = Encoder(
            combine_channel_depth, 
            emb_dim, 
            num_head, 
            comb_ffn_dim, 
            activation_fn,
            ffn_dropout, 
            dropout
        )
        
        self.upsample1 = nn.ConvTranspose2d(emb_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

    def view_and_permute(self, x):
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)  # [b, n, c] -> [b, c, h, w]
            return x

    def forward(self, x = None):

        x = self.patch_embedding(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)

        x = self.view_and_permute(x) # [b, 196, 1024] -> [b, 1024, 14, 14]
        x = self.upsample1(x)        # [b, 512, 28, 28]

        return x



class BEiT3_with_flash(nn.Module):
    def __init__(self, 
                num_channels = 3,
                img_size=224,
                patch_size=16,
                in_chans=3,
                emb_dim=1024,
                num_head=16,
                comb_ffn_dim=3072,
                combine_channel_depth = 9,
                ffn_dropout = 0.1,
                relative_pos = False,
                dropout = 0.1,
                num_classes = 21,
                activation_fn = 'gelu',
                ):
        super().__init__()
        self.num_channels = num_channels

        seq_len = (img_size//patch_size)**2

        self.patch_embedding = Custom_pat_Embedding(img_size,
                                                    patch_size,
                                                    in_chans,
                                                    emb_dim)
        self.positional_embedding = PositionalEmbedding(seq_len, 
                                                   emb_dim, 
                                                   num_channels, 
                                                   relative=relative_pos)
        
        self.encoder = Encoder_with_flash(
            combine_channel_depth, 
            emb_dim, 
            num_head, 
            comb_ffn_dim, 
            activation_fn,
            ffn_dropout, 
            dropout
        )
        
        self.upsample1 = nn.ConvTranspose2d(emb_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

    def view_and_permute(self, x):
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)  # [b, n, c] -> [b, c, h, w]
            return x

    def forward(self, x = None):

        x = self.patch_embedding(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)

        x = self.view_and_permute(x) # [b, 196, 1024] -> [b, 1024, 14, 14]
        x = self.upsample1(x)        # [b, 512, 28, 28]

        return x