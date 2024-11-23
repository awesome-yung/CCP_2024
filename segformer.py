from math import sqrt
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn, einsum
from timm.models import register_model
from einops import rearrange
import jax
import jax.numpy as jnp
import math
import numpy as np
import functools


# cast the val of num_layers, see the class SegFormer parameters
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super(DsConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in,
                      stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super(EfficientSelfAttention, self).__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        # print(f"2 : x = {x.shape}, q = {q.shape}")
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        # q @ k.transpose(-2, -1) * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # attn @ V
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        out = self.to_out(out)
        return out

#  MultiheadAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio
class MultiheadAttention_with_jax(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout=0.0,
        key_chunk_size=4096,  #  chunk size 조절
        query_chunk_size=1024,
        reduction_ratio=0
    ):
        super().__init__()
        self.embed_dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scaling = self.head_dim**-0.5

        # Q, K, V projection layers
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=True)

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


    def forward(self, x):
        bsz, tgt_len, _, _ = x.size()

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q *= self.scaling
        
        # print(f"3 : x = {x.shape}, q = {q.shape}")
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

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MiT_with_fp32(nn.Module):
    def __init__(self, *, channels, dims, heads, ff_expansion, reduction_ratio, num_layers):
        super().__init__()

        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
                dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, MultiheadAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))


    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class MiT(nn.Module):
    def __init__(self, *, channels, dims, heads, ff_expansion, reduction_ratio, num_layers):
        super().__init__()

        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
                dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))


    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret
    
class MiT_with_jax(nn.Module):
    def __init__(self, *, channels, dims, heads, ff_expansion, reduction_ratio, num_layers):
        super().__init__()

        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
                dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, MultiheadAttention_with_jax(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))


    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class Segformer_b3(nn.Module):
    def __init__(
        self,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (3, 4, 18, 3),
        channels = 3,
        decoder_dim = 768,
        num_classes = 21
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )


    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        out = self.to_segmentation(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
    
class Segformer_b1(nn.Module):
    def __init__(
        self,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (2, 2 ,2 ,2),
        channels = 3,
        decoder_dim = 256,
        num_classes = 21
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )


    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        out = self.to_segmentation(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


class Segformer_b1_with_jax(nn.Module):
    def __init__(
        self,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (2, 2 ,2 ,2),
        channels = 3,
        decoder_dim = 256,
        num_classes = 21
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT_with_jax(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )


    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        out = self.to_segmentation(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
    

class Segformer_b1_with_fp32(nn.Module):
    def __init__(
        self,
        dims = (64, 128, 320, 512),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (2, 2 ,2 ,2),
        channels = 3,
        decoder_dim = 256,
        num_classes = 21
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT_with_fp32(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )


    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        out = self.to_segmentation(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out