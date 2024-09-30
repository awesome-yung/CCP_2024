import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import functools
import numpy as np

class MultiheadAttention_with_jax(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        key_chunk_size=64,  #  chunk size 조절
        query_chunk_size=64
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

        self.key_chunk_size = key_chunk_size
        self.query_chunk_size = query_chunk_size

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _query_chunk_attention(self, query, key, value, precision=jax.lax.Precision.HIGHEST):
        """JAX 기반의 메모리 효율적인 어텐션 계산."""
        num_kv, num_heads, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / jnp.sqrt(k_features)

        @functools.partial(jax.checkpoint, prevent_cse=False)
        def summarize_chunk(query, key, value):
            attn_weights = jnp.einsum('qhd,khd->qhk', query, key, precision=precision)
            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum('vhf,qhv->qhf', value, exp_weights, precision=precision)
            return (exp_values, exp_weights.sum(axis=-1))

        def chunk_scanner(chunk_idx):
            key_chunk = jax.lax.dynamic_slice(
                key, (chunk_idx, 0, 0),
                slice_sizes=(key_chunk_size, num_heads, k_features))
            value_chunk = jax.lax.dynamic_slice(
                value, (chunk_idx, 0, 0),
                slice_sizes=(key_chunk_size, num_heads, v_features))
            return summarize_chunk(query, key_chunk, value_chunk)

        chunk_values, chunk_weights = jax.lax.map(
            chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

        all_values = chunk_values.sum(axis=0)
        all_weights = chunk_weights.sum(axis=0)
        # return all_values / all_weights
        return all_values / all_weights[..., None]


    def forward(self, query, key, value):
        bsz, tgt_len, _ = query.size()

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        # Reshape to [batch_size, seq_len, num_heads, head_dim] and then merge heads
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use JAX-based attention calculation
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, tgt_len, self.head_dim)

        # Convert to JAX arrays
        q_jax = jnp.array(q.detach().cpu().numpy())
        k_jax = jnp.array(k.detach().cpu().numpy())
        v_jax = jnp.array(v.detach().cpu().numpy())

        # JAX 어텐션 계산
        attn_output = self._query_chunk_attention(q_jax, k_jax, v_jax)

        # Convert back to PyTorch tensor
        attn_output = torch.from_numpy(np.array(attn_output)).to(query.device)

        # Reshape back to original size and apply output projection
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
