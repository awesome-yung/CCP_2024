import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        # x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_fn(x)  # autocast가 type 관리
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )
        self.dropout_module = nn.Dropout(dropout)
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query,
        key,
        value,
    ):
        bsz, tgt_len, _ = query.size()
       
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(bsz,tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        attn_weights = torch.bmm(q,k.transpose(1,2))
        
        attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs,v)
        attn = attn.transpose(0,1).reshape(tgt_len, bsz, self.embed_dim).transpose(0,1)
            
        attn = self.out_proj(attn)
        
        return attn

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

class BEiT3(nn.Module):
    def __init__(self, 
                num_channels = 3,
                img_size=512,
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
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=150 ):
        super(ASPP, self).__init__()
        # atrous 3x3, rate=6
        self.conv_3x3_r6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        # atrous 3x3, rate=12
        self.conv_3x3_r12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        # atrous 3x3, rate=18
        self.conv_3x3_r18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        # atrous 3x3, rate=24
        self.conv_3x3_r24 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=24, dilation=24)
        self.drop_conv_3x3 = nn.Dropout2d(0.5)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.drop_conv_1x1 = nn.Dropout2d(0.5)

        self.conv_1x1_out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # 1번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        # print("feature_map = ",feature_map.shape)
        # print("feature map_1 = ,", feature_map.shape)
        out_3x3_r6 = self.drop_conv_3x3(F.relu(self.conv_3x3_r6(feature_map)))
        # print("feature map_2 = ,", out_3x3_r6.shape)
        out_img_r6 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r6)))
        # print("feature map_3 = ,", out_img_r6.shape)
        out_img_r6 = self.conv_1x1_out(out_img_r6)
        
        # 2번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r12 = self.drop_conv_3x3(F.relu(self.conv_3x3_r12(feature_map)))
        out_img_r12 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r12)))
        out_img_r12 = self.conv_1x1_out(out_img_r12)
        # 3번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r18 = self.drop_conv_3x3(F.relu(self.conv_3x3_r18(feature_map)))
        out_img_r18 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r18)))
        out_img_r18 = self.conv_1x1_out(out_img_r18)
        # 4번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r24 = self.drop_conv_3x3(F.relu(self.conv_3x3_r24(feature_map)))
        out_img_r24 = self.drop_conv_1x1(F.relu(self.conv_1x1(out_3x3_r24)))
        out_img_r24 = self.conv_1x1_out(out_img_r24)

        out = sum([out_img_r6, out_img_r12, out_img_r18, out_img_r24])

        return out

class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self,  backbone, classifier):
        super(DeepLabV2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.upsampling = upsampling #  upsampling = 8

    def forward(self, x):
        # x = self.backbone(x)
        _, _, feature_map_h, feature_map_w = x.size()
        out = self.classifier(x)
        out = F.softmax(out,dim=1)
        # print("\n feature = ", feature_map_h, feature_map_w, self.upsampling)
        return out
    # out = F.softmax(out,dim=1)
    
def get_temp_model(num_classes = 150):
    backbone = BEiT3()
    aspp_module = ASPP(in_channels=512, out_channels=256, num_classes=num_classes)
    model = DeepLabV2(backbone=backbone, classifier=aspp_module)
    num_params = count_parameters(model)
    print(f'Total number of parameters: {num_params}')
    return model