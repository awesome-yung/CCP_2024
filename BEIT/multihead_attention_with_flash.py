import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
# from flash_attn.flash_attention import FlashAttention
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    

class MultiheadAttention_with_flash(nn.Module):
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

        # Q, K, V projection layers
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Dropout module
        self.dropout_module = nn.Dropout(dropout)

        # FlashAttention module (causal=False for non-autoregressive models)
        # self.flash_attention = FlashAttention(causal=False)

    def forward(self, query, key, value):
        bsz, tgt_len, _ = query.size()

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        # Reshape to [batch_size, seq_len, num_heads, head_dim] and then merge heads
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # FlashAttention expects Q, K, V in shape [batch_size * num_heads, seq_len, head_dim]
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, tgt_len, self.head_dim)

        # Use FlashAttention to compute attention
        attn_output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # Reshape back to original size and apply output projection
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
