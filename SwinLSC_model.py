import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode
        
    Returns:
        Output after drop path
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is a regularization technique that randomly drops entire paths in residual networks.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Partition the feature map into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Size of the window (M)

    Returns:
        windows: Partitioned windows of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Restore windows back to the original feature map.
    
    Args:
        windows: Windows tensor of shape (num_windows*B, window_size, window_size, C)
        window_size: Size of the window (M)
        H: Height of the original feature map
        W: Width of the original feature map

    Returns:
        x: Restored feature map of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding layer.
    This layer splits an image into patches and projects them into an embedding space.
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        """
        Args:
            patch_size: Size of the patch (patch_size x patch_size)
            in_c: Number of input channels
            embed_dim: Dimension of the embedding
            norm_layer: Normalization layer
        """
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            x: Embedded patches of shape (B, L, embed_dim)
            H: Height of feature map after embedding
            W: Width of feature map after embedding
        """
        _, _, H, W = x.shape
        """
        padding
        If the H and W values of the input image are not integer multiples of patch_2, padding is required
        """
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            """
            to pad the last 3 dimensions,
            (W_left, W_right, H_top,H_bottom, C_front, C_back)
            """
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        """Down sampling patch_2 times"""
        x = self.proj(x)
        _, _, H, W = x.shape
        """flatten: [B, C, H, W] -> [B, C, HW],transpose: [B, C, HW] -> [B, HW, C]"""
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    This layer merges multiple patches into one, reducing the resolution by a factor of 2.
    It also incorporates a channel attention mechanism to enhance feature representation.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        """
        Args:
            dim: Input feature dimension
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        """
        Add channel attention mechanism, mainly by weighting the channels with linear layers and activation functions, 
        with fixed parameters.
        Channel attention mechanism to weight channels adaptively.
        """
        self.channel_attention = nn.Sequential(
            nn.Linear(4 * dim, 4 * dim // 4),
            nn.ReLU(),
            nn.Linear(4 * dim // 4, 4 * dim),
            nn.Sigmoid()
        )

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map
            
        Returns:
            x: Output tensor after patch merging with shape (B, H/2*W/2, 2*C)
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Get features from 2x2 neighboring patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left patches
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left patches
        x2 = x[:, 0::2, 1::2, :]  # Top-right patches
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right patches

        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        # Apply channel attention mechanism
        attn = self.channel_attention(x)
        x = x * attn

        x = self.norm(x)
        x = self.reduction(x) # Reduce dimension from 4*C to 2*C

        return x


class Mlp(nn.Module):
    """
    MLP module with two fully connected layers.
    This implements the Feed-Forward Network (FFN) in Transformer blocks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            act_layer: Activation layer
            drop: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor
            
        Returns:
            x: Output tensor after MLP
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CompleteAttention(nn.Module):
    """
    Complete Attention module.
    This is a custom attention mechanism that processes the entire feature map,
    providing a global context to complement the local window attention.
    """
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            window_size: Size of the window
            qkv_bias: Whether to use bias in qkv projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        # Linear projections for different feature map sizes
        self.E = nn.Linear(3136, 128)  # For 56x56 feature maps
        self.F = nn.Linear(3136, 128)
        self.E2 = nn.Linear(784, 128)  # For 28x28 feature maps
        self.F2 = nn.Linear(784, 128)
        self.E3 = nn.Linear(196, 128)  # For 14x14 feature maps
        self.F3 = nn.Linear(196, 128)

    def forward(self, x, h, w):
        """
        Args:
            x: Input tensor from window partition
            h: Height of the original feature map
            w: Width of the original feature map
            
        Returns:
            x: Output tensor after complete attention
        """
        # x shape:[nW*B, Mh*Mw, C]
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        # Reshape and reverse window partition to get the full feature map
        x = x.reshape(-1, self.window_size[0], self.window_size[1], x.shape[2])
        x = window_reverse(windows=x, window_size=self.window_size[0], H=h, W=w)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        # qkv(): -> [B, H, W, 3 * C]
        # reshape: -> [batch_size, H*W, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, H*W, embed_dim_per_head]
        # Calculate query, key, value projections
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, H*W, embed_dim_per_head]
        q, k, v = qkv.unbind(0)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, H*W]
        # @: multiply -> [batch_size, num_heads, H*W, H*W]
        # Scale query
        q = q * self.scale
        # [batch_size, num_heads, embed_dim_per_head, H*W]
        # Transpose key for matrix multiplication
        k_t = k.transpose(-2, -1)
        # [batch_size, num_heads,128,embed_dim_per_head]
        # k = nn.Linear(H * W, 128).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))(
        #     k_t).transpose(-2, -1)
        # k = self.E(k_t).transpose(-2, -1)
        # Apply dimension reduction based on feature map size
        if k_t.shape[3] == 3136:
            k = self.E(k_t).transpose(-2, -1)
        elif k_t.shape[3] == 784:
            k = self.E2(k_t).transpose(-2, -1)
        elif k_t.shape[3] == 196:
            k = self.E3(k_t).transpose(-2, -1)
        # [batch_size, num_heads,H*W,128]
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, H*W, embed_dim_per_head]
        # transpose: -> [batch_size, H*W, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, H*W, total_embed_dim]
        # [batch_size, num_heads,H*W,embed_dim_per_head]
        # transpose:[batch_size, num_heads,embed_dim_per_head,H*W]->F:[batch_size, num_heads,embed_dim_per_head,128]
        # transpose: [batch_size, num_heads,128,embed_dim_per_head]
        # v = nn.Linear(H * W, 128).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))(
        #     v.transpose(-2, -1)).transpose(-2, -1)
        # v = self.F(v.transpose(-2, -1)).transpose(-2, -1)
        # if v.shape[2] == 3136:
        #     v = self.F(v.transpose(-2, -1)).transpose(-2, -1)
        # else:
        #     v = self.F2(v.transpose(-2, -1)).transpose(-2, -1)
        if v.shape[2] == 3136:
            v = self.F(v.transpose(-2, -1)).transpose(-2, -1)
        elif v.shape[2] == 784:
            v = self.F2(v.transpose(-2, -1)).transpose(-2, -1)
        elif v.shape[2] == 196:
            v = self.F3(v.transpose(-2, -1)).transpose(-2, -1)
        # Apply attention to value
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention module.
    This is the core attention mechanism in Swin Transformer that operates within local windows.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: Input feature dimension
            window_size: Size of the window (Mh, Mw)
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in qkv projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        # Calculate relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (nW*B, Mh*Mw, C)
            mask: Attention mask (optional)
            
        Returns:
            x: Output tensor after window attention
        """
        # x shape:[nW*B, Mh*Mw, C]

        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    This is the basic building block of the Swin Transformer architecture, combining window attention
    with shifted window attention and MLP layers.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            window_size: Size of the window
            shift_size: Size of the shift (0 for W-MSA, window_size//2 for SW-MSA)
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.fusion = nn.Linear(2 * dim, dim) # Fusion layer for combining local and global attention
        # Window attention module
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        # Complete (global) attention module
        self.attn_comp = CompleteAttention(dim=dim,
                                           window_size=(self.window_size, self.window_size),
                                           num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        """
        Args:
            x: Input tensor of shape (B, L, C)
            attn_mask: Attention mask for shifted window attention
            
        Returns:
            x: Output tensor after transformer block
        """
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        # Apply cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows for attention computation
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
        x_windows2 = window_partition(x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        # Compute window attention (W-MSA or SW-MSA)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]
        # reshape: [B, H*W, C]=>>[B, H, W, C]
        # window_partition: [B, H, W, C] =>>[nW*B, Mh*Mw, C]
        # Compute complete (global) attention
        attn_comp_cal = window_partition(self.attn_comp(x=x_windows2, h=self.H, w=self.W).reshape(B, H, W, C),
                                         window_size=self.window_size)
        attn_comp_cal = attn_comp_cal.view(attn_comp_cal.shape[0], -1, C)
        # attn_windows = attn_windows + attn_comp_cal
        # merge windows
        # Reverse window partitioning
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]
        attn_comp_cal = window_reverse(attn_comp_cal, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        # Reverse cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # Remove padding if necessary
            x = x[:, :H, :W, :].contiguous()
            attn_comp_cal = attn_comp_cal[:, :H, :W, :].contiguous()
        # Reshape and fuse local and global attention results
        x = x.view(B, H * W, C)
        attn_comp_cal = attn_comp_cal.view(B, H * W, C)
        x = torch.cat([x, attn_comp_cal], dim=-1)
        x = self.fusion(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer Layer.
    This layer consists of multiple Swin Transformer blocks and an optional patch merging layer.
    """
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        """
        Args:
            dim: Input feature dimension
            depth: Number of Swin Transformer blocks
            num_heads: Number of attention heads
            window_size: Size of the window
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            norm_layer: Normalization layer
            downsample: Downsample layer (PatchMerging)
            use_checkpoint: Whether to use checkpointing to save memory
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        """
        Create attention mask for SW-MSA.
        
        Args:
            x: Input tensor
            H: Height of feature map
            W: Width of feature map
            
        Returns:
            attn_mask: Attention mask for SW-MSA
        """
        # calculate attention mask for SW-MSA
        # Ensure Hp and Wp are multiples of window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # Create mask with same layout as feature map
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # Define slices for cyclic shift
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        """
        Forward pass for the basic Swin Transformer layer.
        
        Args:
            x: Input tensor of shape (B, L, C)
            H: Height of feature map"""
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer.
    This layer performs channel-wise feature scaling based on global information.
    """
    def __init__(self, channel, reduction=16):
        """
        Args:
            channel: Input feature dimension
            reduction: Reduction ratio for channel scaling
        """
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        # Used to compress the length of the last one-dimensional signal input to 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the SE Layer.
        
        Args:
            x: Input tensor of shape (B, L, C)
        """
        b, l, c = x.shape
        # Ensure channel dimension matches
        # if c != self.channel:
        #     # If input channel number does not match expected, adjust fc layer
        #     self.fc = nn.Sequential(
        #         nn.Linear(c, c // self.reduction, bias=False),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(c // self.reduction, c, bias=False),
        #         nn.Sigmoid()
        #     ).to(x.device)
        #     self.channel = c
        # [b, l, c] -> [b, c, l] -> [b, c, 1] -> [b, c] squeeze is used to remove the dimension of size 1.
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)  # [b, c]
        y = self.fc(y)  # [b, c]
        # [b, c] -> [b, 1, c] -> [b, l, c]
        y = y.unsqueeze(1).expand(-1, l, -1)  # [b, l, c]
        return x * y


class SwinLSC(nn.Module):
    """
    SwinLSC Model.
    This is the main model class for the SwinLSC architecture.
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        """
        Args:
            patch_size: Size of the patch
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dim: Dimension of the embedding
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            window_size: Size of the window in each stage
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to use bias in qkv projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            norm_layer: Normalization layer
            patch_norm: Whether to normalize the patches
            use_checkpoint: Whether to use checkpointing to save memory
        """
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # Channels for outputting feature matrix in Stage 4
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            """
            Build the layers of the SwinLSC model.
            
            Args:
                i_layer: Index of the current layer
                dim: Dimension of the input feature
            """
            # Note that the stages built here differ from the stages in the paper figure
            # Here, the stage does not include the patch_merging layer of that stage, but includes the patch_merging layer of the next stage
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        # Pool the last dimension to 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Add SE attention layer
        self.se = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 3:
                self.se.append(SELayer(int(embed_dim * 2 ** i_layer), reduction=8))
            else:
                self.se.append(SELayer(int(embed_dim * 2 ** (i_layer + 1)), reduction=8))

        # Add extra feature extraction layer
        self.extra_conv = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass for the SwinLSC model.
        
        Args:
            x: Input tensor of shape (B, L, C)
        """ 
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # Adding residual connections and SE attention, 
        # residual connections are added after each Transformer layer,
        #  which helps alleviate the gradient vanishing problem and improve the training stability of the model
        for i, layer in enumerate(self.layers):
            identity = x
            x, H, W = layer(x, H, W)
            x = self.se[i](x)
            if x.shape == identity.shape:  # Only add residual connections when dimension matching接
                x = x + identity

        x = self.norm(x)

        # Reshaping tensors to use additional convolutional layers
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.extra_conv(x)
        x = x.view(B, C, -1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=7,
                    embed_dim=96,
                    depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=7,
                    embed_dim=96,
                    depths=(2, 2, 18, 2),
                    num_heads=(3, 6, 12, 24),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=7,
                    embed_dim=128,
                    depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=12,
                    embed_dim=128,
                    depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=7,
                    embed_dim=128,
                    depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=12,
                    embed_dim=128,
                    depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=7,
                    embed_dim=192,
                    depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48),
                    num_classes=num_classes,
                    **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):

    model = SwinLSC(in_chans=3,
                    patch_size=4,
                    window_size=12,
                    embed_dim=192,
                    depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48),
                    num_classes=num_classes,
                    **kwargs)
    return model
