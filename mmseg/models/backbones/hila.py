import math
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InterLevelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kv_dim=None,
                 store_attn=False, top_down_update=False, use_euc_dist=False, stride=2, patch_size=(4, 4), dim_reduc=1):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.attn = None
        self.store_attn = store_attn
        self.top_down_update = top_down_update
        self.use_euc_dist = use_euc_dist

        self.patch_size = patch_size
        self.stride = stride

        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.attn_dim = dim // dim_reduc if kv_dim > dim else kv_dim // dim_reduc

        self.q = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, 2 * self.attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_init = nn.Parameter(torch.zeros(patch_size[0] * patch_size[1]))
        self.relative_position_bias_table = nn.Parameter(torch.zeros(patch_size[0]*patch_size[1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):

        B, _, C = Xb.shape
        patch_size, stride, num_heads = self.patch_size, self.stride, self.num_heads
        Xbp = Xb.permute(0, 2, 1).view(Xb.shape[0], -1, Hb, Wb)
        Xbp = F.unfold(Xbp, kernel_size=patch_size, stride=stride,
                       padding=((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)).contiguous()
        Xbp = Xbp.reshape(B, C, patch_size[0] ** 2, Ht * Wt).permute(0, 3, 2, 1)
        Xbp = Xbp.reshape(B * Ht * Wt, -1, C)
        Xt = Xt.view(-1, 1, Xt.shape[-1])

        if self.top_down_update:
            x, y = Xbp, Xt
            H, W = Hb, Wb
        else:
            x, y = Xt, Xbp
            H, W = Ht, Wt

        BHW, N, _ = x.shape
        C = self.attn_dim
        q = self.q(x).reshape(BHW, N, num_heads, C // num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(BHW, -1, 2, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if not self.use_euc_dist:
            m = torch.matmul(q, k.transpose(-2, -1))
        else:
            m = torch.cdist(q, k, compute_mode='use_mm_for_euclid_dist')
        attn = m * self.scale

        if self.top_down_update:
            relative_position_init = self.relative_position_init.unsqueeze(1)
            relative_position_bias = self.relative_position_bias_table.unsqueeze(1)
            attn = attn + relative_position_init + relative_position_bias
            attn = self.patch_wise_softmax(attn, Ht, Wt, Hb, Wb)
        else:
            relative_position_init = self.relative_position_init.unsqueeze(0)
            relative_position_bias = self.relative_position_bias_table.unsqueeze(0)
            attn = attn + relative_position_init + relative_position_bias
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        if self.store_attn:
            self.attn = attn

        x = (attn @ v).transpose(1, 2).reshape(BHW, N, C)

        B = BHW // Ht // Wt
        if self.top_down_update:
            x = x.view(B, Ht * Wt, -1, C)
            x = x.permute(0, 3, 2, 1).reshape(B, C * patch_size[0] ** 2, Ht * Wt)
            x = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=stride,
                       padding=((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)).contiguous()
            x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def patch_wise_softmax(self, attn, Ht, Wt, Hb, Wb):
        patch_size, stride, num_heads = self.patch_size, self.stride, self.num_heads
        padding = ((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)

        attn = torch.exp(attn)
        attn_sum = attn.reshape(-1, Ht * Wt, num_heads, patch_size[0] ** 2)
        attn_sum = attn_sum.permute(0, 2, 3, 1).reshape(-1, patch_size[0] ** 2, Ht * Wt)
        attn_sum = F.fold(attn_sum, output_size=(Hb, Wb), kernel_size=patch_size, stride=stride, padding=padding)
        attn_sum = F.unfold(attn_sum, kernel_size=patch_size, stride=stride, padding=padding).permute(0, 2, 1)

        attn_sum[attn_sum == 0] = 1.0
        return attn / attn_sum.reshape(-1, num_heads, patch_size[0] ** 2, 1)


class TopDownAttn(nn.Module):

    def __init__(self, top_dim, bot_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, store_attn=False,
                 patch_size=4, stride=2, use_euc_dist=False, info_weights=(1, 1), dim_reduc=1):
        super().__init__()

        self.info_weights = info_weights
        self.topdown_attn = InterLevelAttention(
            bot_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=to_2tuple(patch_size),
            stride=stride, attn_drop=attn_drop, proj_drop=drop, kv_dim=top_dim, store_attn=store_attn,
            dim_reduc=dim_reduc, top_down_update=True, use_euc_dist=use_euc_dist)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(bot_dim)
        self.norm2 = norm_layer(top_dim)
        self.norm3 = norm_layer(bot_dim)

        mlp_hidden_dim = int(bot_dim * mlp_ratio)
        self.bottom_mlp = Mlp(in_features=bot_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):
        Xb = Xb + self.drop_path(self.topdown_attn(self.norm2(Xt), self.norm1(Xb), Ht, Wt, Hb, Wb))
        Xb_ffn = self.drop_path(self.bottom_mlp(self.norm3(Xb), Hb, Wb))
        Xb = self.info_weights[0]*Xb + self.info_weights[1]*Xb_ffn
        return Xt, Xb


class BottomUpAttn(nn.Module):
    def __init__(self, top_dim, bot_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, store_attn=False,
                 patch_size=4, stride=2, use_euc_dist=False, info_weights=(1, 1), dim_reduc=1):
        super().__init__()

        self.info_weights = info_weights
        self.bottomup_attn = InterLevelAttention(
            top_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=to_2tuple(patch_size),
            stride=stride, attn_drop=attn_drop, proj_drop=drop, kv_dim=bot_dim, store_attn=store_attn,
            dim_reduc=dim_reduc, use_euc_dist=use_euc_dist)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm4 = norm_layer(top_dim)
        self.norm5 = norm_layer(bot_dim)
        self.norm6 = norm_layer(top_dim)

        mlp_hidden_dim = int(top_dim * mlp_ratio)
        self.top_mlp = Mlp(in_features=top_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):
        # TODO: check result different for other models
        Xt = Xt + self.drop_path(self.bottomup_attn(self.norm4(Xt), self.norm5(Xb), Ht, Wt, Hb, Wb))
        Xt_ffn = self.drop_path(self.top_mlp(self.norm6(Xt), Ht, Wt))
        Xt = self.info_weights[0] * Xt + self.info_weights[1] * Xt_ffn
        return Xt, Xb


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
