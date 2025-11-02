# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from .utils import MLP
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)


        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=qv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.fill_(0)

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            return_attention=False
    ):
        B, Nq, C = query.shape
        Nk = key.size(1)
        Nv = value.size(1)

        q = self.q_proj(query).view(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if key_padding_mask is None:
            attn = attn.softmax(dim=-1)
        else:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = attn.masked_fill(mask, 0)

        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn

        return x


class MotionFusionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, drop_path=0.1, num_frames=11, n_levels=4, rank=32,
                 init_values=1e-4):
        super().__init__()
        # motion encoding
        self.spatial_attention = MSDeformAttn(
            embed_dim=hidden_dim,
            num_levels=1,
            num_heads=num_heads,
            num_points=4,
            batch_first=True,
            img2col_step=132
        )
        self.temporal_attention = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim*4, hidden_dim, 2)
        self.temp_pos = nn.Parameter(torch.zeros(num_frames, hidden_dim))
        self.in_proj = nn.Linear(2, hidden_dim)

        # cross-fusion
        conv_layers = []
        cross_layers = []
        for l in range(n_levels):
            cross_layers.append(Attention(hidden_dim, num_heads, attn_drop=dropout, proj_drop=dropout))
            if l == 0:
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            else:
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.cross_attentions = nn.ModuleList(cross_layers)

        # gating
        self.gate = nn.Linear(hidden_dim, rank, bias=False)
        self.down_proj = nn.Linear(hidden_dim, rank)
        self.up_proj = nn.Linear(rank, hidden_dim)
        self.gamma = nn.Parameter(init_values * torch.ones((hidden_dim)))
        self.norm = nn.LayerNorm(hidden_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.temp_pos, std=.02)
        nn.init.trunc_normal_(self.gate.weight, std=.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.constant_(self.down_proj.bias, 0)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.constant_(self.up_proj.bias, 0)
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv[0].weight, gain=1)
            nn.init.constant_(conv[0].bias, 0)

    def forward(self, src, motion, motion_mask, pos, reference_points, spatial_shapes, level_start_index):
        motion = self.encode_motion(motion, motion_mask, pos, reference_points, spatial_shapes[0])
        b, t, c = motion.shape[:3]
        motion = motion.flatten(0, 1)

        feat_list = []
        for l, start in enumerate(level_start_index):
            h, w = spatial_shapes[l]
            feat = src[:, start:start+h*w]
            mask = F.interpolate(motion_mask.float(), size=(h, w)).bool()
            motion = self.conv_layers[l](motion)
            motion = rearrange(motion, '(b t) c h w -> (b h w) t c', b=b)
            feat = rearrange(feat, 'b n c -> (b n) 1 c')
            mask = rearrange(mask, 'b t h w -> (b h w) t')
            feat = self.cross_attentions[l](feat, motion, motion, key_padding_mask=mask)
            feat = rearrange(feat, '(b n) 1 c -> b n c', b=b)
            motion = rearrange(motion, '(b h w) t c -> (b t) c h w', h=h, w=w)
            feat_list.append(feat)
        feat = torch.cat(feat_list, dim=1)

        # spatial gating
        gate = torch.sigmoid(self.gate(src))
        feat = gate * self.down_proj(feat)
        feat = self.up_proj(feat)

        # channel gating
        feat = self.gamma * torch.square(torch.relu(feat))

        src = src + self.drop_path(feat)
        src = self.norm(src)
        return src

    def encode_motion(self, x, mask, pos_embeds, reference_points, spatial_shapes):
        bs, t, in_channel, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2)  # bs, t, h, w, c
        x = self.in_proj(x)

        seq_len = h * w
        dim = x.shape[-1]
        x = x.reshape(bs, t, seq_len, dim)
        mask = mask.reshape(bs, t, seq_len)

        sp_pos = pos_embeds.permute(0, 2, 3, 1).reshape(bs, seq_len, dim)
        xt = x + sp_pos.unsqueeze(1)
        x = x.reshape(bs * t, seq_len, dim)
        xt = xt.reshape(bs * t, seq_len, dim)
        mask = mask.reshape(bs * t, seq_len)

        xt = self.spatial_attention(
            query=xt,
            reference_points=reference_points,
            value=x,
            spatial_shapes=spatial_shapes.unsqueeze(0),
            level_start_index=torch.tensor(0, device=x.device),
            key_padding_mask=mask,
        )
        x = self.norm0(x + xt)

        x = x.reshape(bs, t, seq_len, dim).permute(0, 2, 1, 3)
        mask = mask.reshape(bs, t, seq_len).permute(0, 2, 1)

        # temporal attention
        x = x.reshape(bs * seq_len, t, dim)
        mask = mask.reshape(bs * seq_len, t)
        xt = x + self.temp_pos[:t].unsqueeze(0)

        xt = self.temporal_attention(xt, x, x, key_padding_mask=mask)

        x = self.norm1(x + xt)

        x = x + self.ffn(x)
        x = self.norm2(x)

        x = x.reshape(bs, h, w, t, dim).permute(0, 3, 4, 1, 2)
        return x    # bs, t, c, h, w