import netron
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models import create_model
from timm.models.layers import DropPath, trunc_normal_, create_classifier, DropBlock2d
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import numpy as np
from utils.torch_utils import load_dict
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


class Convchor(nn.Module):
    def __init__(self, ch_in, f, ch_out, stride, r, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(Convchor, self).__init__()
        self.pointwise_conv1 = nn.Conv2d(ch_in, f, kernel_size=1, stride=1)
        nn.init.zeros_(self.pointwise_conv1.weight)
        nn.init.zeros_(self.pointwise_conv1.bias)
        self.bn1 = norm_layer(f)
        self.act1 = act_layer(inplace=True)

        self.depthwise_conv = nn.Conv2d(f, f, kernel_size=3, stride=stride, padding=1, groups=r)
        nn.init.zeros_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)
        self.bn2 = norm_layer(f)
        self.act2 = act_layer(inplace=True)

        self.pointwise_conv2 = nn.Conv2d(f, ch_out, kernel_size=1, stride=1)
        nn.init.zeros_(self.pointwise_conv2.weight)
        nn.init.zeros_(self.pointwise_conv2.bias)

        self.bn3 = norm_layer(ch_out)
        self.act3 = act_layer(inplace=True)
        self.r = r

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.pointwise_conv2(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.r * x
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class CBAMLayer(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAMLayer, self).__init__()
#
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.mlp(self.max_pool(x))
#         avg_out = self.mlp(self.avg_pool(x))
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x
#
#
# class CE_Block(nn.Module):
#     def __init__(self, ch_in, reduction=16):
#         super(CE_Block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
#         self.fc = nn.Sequential(
#             nn.Linear(ch_in, ch_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch_in // reduction, ch_in, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)  # squeeze操作
#         y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
#         return x * y.expand_as(x)  # 注意力作用每一个通道上
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_blocks(drop_block_rate=0.):
    return DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), h=8, s=0.1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adapter_attn = Transchor(h)
        self.adapter_mlp = Transchor(h)
        self.s = s

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm1(x))) * self.s
        return x


class Block_att(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), s=0.1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.s = s

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(y) * self.s
        return x


class Block_mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), s=0.1):
        super().__init__()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.s = s

    def forward(self, x, y):
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(y) * self.s
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, f=1, r=1, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv_adapter = Convchor(med_planes, f, med_planes, stride, r)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(outplanes)
            )

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x1 = self.conv_adapter(x)
        if self.drop_block is not None:
            x1 = self.drop_block(x1)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x2 = x2 + x1
        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        return x, x2


class Transchor(nn.Module):
    def __init__(self, h=8, xavier_init=False, cat=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, groups=768)

        if xavier_init:
            nn.init.xavier_uniform_(self.depthwise_conv.weight)
        else:
            nn.init.zeros_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)

        self.bn = nn.BatchNorm2d(768)

        if cat == False:
            self.adapter_down = nn.Linear(768, h)  # equivalent to 1 * 1 Conv
        else:
            self.adapter_down = nn.Linear(768 * 2, h)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(h, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.layer1 = norm_layer(8)
        self.layer2 = norm_layer(768)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.h = h

    def forward(self, x):
        B, N, C = x.shape

        x_patch = x[:, 1:].reshape(B, 14, 14, 768).permute(0, 3, 1, 2)
        x_cls = x[:, :1]

        x_patch = self.depthwise_conv(x_patch)
        x_patch = self.bn(x_patch)
        x_patch = self.act(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, 768)

        x = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        # x_down = self.layer1(x_down)
        x_down = self.act(x_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        # x_up = self.layer2(x_up)
        # x_up = self.act(x_up)
        return x_up


class Bridge(nn.Module):
    def __init__(self, outplanes, m=8, h=8, a=1, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 xavier_init=False):
        super(Bridge, self).__init__()
        dw_stride = 1024 // outplanes

        # 降维
        self.adapter_down = nn.Conv2d(outplanes // 4, m, 1, 1, 0)
        nn.init.zeros_(self.adapter_down.weight)

        self.bn1 = nn.BatchNorm2d(m)
        self.act1 = act_layer()
        # 卷积
        if dw_stride == 4:
            self.adapter_conv = nn.Sequential(
                nn.Conv2d(m, m, 3, 1, 1),
                nn.MaxPool2d(4),
                nn.ReLU(),
            )

        if dw_stride == 2:
            self.adapter_conv = nn.Sequential(
                nn.Conv2d(m, m, 3, 1, 1),
                nn.AvgPool2d(2),
                nn.ReLU(),
            )

        if dw_stride == 1:
            self.adapter_conv = nn.Sequential(
                nn.Conv2d(m, m, 3, 1, 1),
                nn.ReLU()
            )

        if dw_stride == 0:
            self.adapter_conv = nn.Sequential(
                nn.ConvTranspose2d(m, m, 2, 2),
                nn.Conv2d(m, m, 3, 1, 1),
                nn.ReLU()
            )

        # 升维
        self.adapter_up = nn.Conv2d(m, 768, 1, 1, 0)
        nn.init.zeros_(self.adapter_up.weight)
        self.bn2 = nn.BatchNorm2d(768)
        self.act2 = act_layer()

        self.adapter_convpass = Transchor(h=h)

        self.norm = norm_layer(768)

        self.a = a

        # self.ce = ChannelAttention(768)

    def forward(self, x_t, x_r):
        B, _, _ = x_t.shape
        x_r = self.adapter_down(x_r)
        x_r = self.bn1(x_r)
        x_r = self.act1(x_r)

        x_r = self.adapter_conv(x_r)

        x_r = self.adapter_up(x_r)

        x_r = self.bn2(x_r)

        x_r = self.act2(x_r)

        x_t = x_t[:, 1:].reshape(B, 14, 14, 768).permute(0, 3, 1, 2)
        x_t = x_t + self.a * x_r
        x_t = x_t.permute(0, 2, 3, 1).reshape(B, 14 * 14, 768)
        x_t = torch.cat([x_t[:, 0:1, :], x_t], dim=1)
        x_t = self.norm(x_t)
        x_t = self.adapter_convpass(x_t)

        return x_t


class Base_50(nn.Module):

    def __init__(self, f=8, m=8, h=8, r=1, a=1, s=0.1, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., class_token=True,
                 no_embed_class=False):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        self.patch_embed = PatchEmbed()
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.drop_block = drop_blocks(drop_block_rate=drop_rate)

        for i in range(8):
            self.add_module(
                'blocks' + str(i) + '_att',
                Block_att(
                    dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path=self.dpr[i], s=s
                )
            )
            self.add_module(
                'blocks' + str(i) + '_mlp',
                Block_mlp(
                    dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=self.dpr[i], s=s
                )
            )

        self.blocks8 = nn.Sequential(*[Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[i], h=h, s=s) for i in range(8, 12)])

        # CNN stage
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1~3 stage
        stage_1_channel = int(base_channel * channel_ratio)
        self.layer1_0 = ConvBlock(64, stage_1_channel, 1, res_conv=True, f=f, r=r, drop_block=self.drop_block)
        self.layer1_1 = ConvBlock(stage_1_channel, stage_1_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer1_2 = ConvBlock(stage_1_channel, stage_1_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)

        # 4~7 stage
        stage_2_channel = int(base_channel * channel_ratio * 2)
        self.layer2_0 = ConvBlock(stage_1_channel, stage_2_channel, 2, res_conv=True, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer2_1 = ConvBlock(stage_2_channel, stage_2_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer2_2 = ConvBlock(stage_2_channel, stage_2_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer2_3 = ConvBlock(stage_2_channel, stage_2_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)

        # 8~13 stage
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        self.layer3_0 = ConvBlock(stage_2_channel, stage_3_channel, 2, res_conv=True, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer3_1 = ConvBlock(stage_3_channel, stage_3_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer3_2 = ConvBlock(stage_3_channel, stage_3_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer3_3 = ConvBlock(stage_3_channel, stage_3_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer3_4 = ConvBlock(stage_3_channel, stage_3_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer3_5 = ConvBlock(stage_3_channel, stage_3_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)

        # 14~16 stage
        stage_4_channel = int(base_channel * channel_ratio * 2 * 2 * 2)
        self.layer4_0 = ConvBlock(stage_3_channel, stage_4_channel, 2, res_conv=True, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer4_1 = ConvBlock(stage_4_channel, stage_4_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)
        self.layer4_2 = ConvBlock(stage_4_channel, stage_4_channel, 1, res_conv=False, f=f, r=r,
                                  drop_block=self.drop_block)

        self.adapter1 = Bridge(stage_1_channel, m=m, h=h, a=a)
        self.adapter2 = Bridge(stage_1_channel, m=m, h=h, a=a)
        self.adapter3 = Bridge(stage_1_channel, m=m, h=h, a=a)
        self.adapter4 = Bridge(stage_2_channel, m=m, h=h, a=a)
        self.adapter5 = Bridge(stage_2_channel, m=m, h=h, a=a)
        self.adapter6 = Bridge(stage_2_channel, m=m, h=h, a=a)
        self.adapter7 = Bridge(stage_2_channel, m=m, h=h, a=a)
        self.adapter8 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter9 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter10 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter11 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter12 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter13 = Bridge(stage_3_channel, m=m, h=h, a=a)
        self.adapter14 = Bridge(stage_4_channel, m=m, h=h, a=a)
        self.adapter15 = Bridge(stage_4_channel, m=m, h=h, a=a)
        self.adapter16 = Bridge(stage_4_channel, m=m, h=h, a=a)

        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.num_features = int(512 * channel_ratio)
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type='avg')

        trunc_normal_(self.cls_token, std=.02)

        # self.apply(self._init_weights)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # 0 stage

        # transformer embeding
        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)

        # CNN
        x_r_stem = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 ~2
        x_r, x2_at = self.layer1_0(x_r_stem)
        x_r, x2_mlp = self.layer1_1(x_r)
        x_t_at = self.adapter1(x_t, x2_at)
        x_t = self.blocks0_att(x_t, x_t_at)
        x_t_mlp = self.adapter2(x_t, x2_mlp)
        x_t = self.blocks0_mlp(x_t, x_t_mlp)

        # 3 ~4
        x_r, x2_at = self.layer1_2(x_r)
        x_r, x2_mlp = self.layer2_0(x_r)
        x_t_at = self.adapter3(x_t, x2_at)
        x_t = self.blocks1_att(x_t, x_t_at)
        x_t_mlp = self.adapter4(x_t, x2_mlp)
        x_t = self.blocks1_mlp(x_t, x_t_mlp)

        # 5 ~6
        x_r, x2_at = self.layer2_1(x_r)
        x_r, x2_mlp = self.layer2_2(x_r)
        x_t_at = self.adapter5(x_t, x2_at)
        x_t = self.blocks2_att(x_t, x_t_at)
        x_t_mlp = self.adapter6(x_t, x2_mlp)
        x_t = self.blocks2_mlp(x_t, x_t_mlp)

        # 7 ~8
        x_r, x2_at = self.layer2_3(x_r)
        x_r, x2_mlp = self.layer3_0(x_r)
        x_t_at = self.adapter7(x_t, x2_at)
        x_t = self.blocks3_att(x_t, x_t_at)
        x_t_mlp = self.adapter8(x_t, x2_mlp)
        x_t = self.blocks3_mlp(x_t, x_t_mlp)

        # 9 ~10
        x_r, x2_at = self.layer3_1(x_r)
        x_r, x2_mlp = self.layer3_2(x_r)
        x_t_at = self.adapter9(x_t, x2_at)
        x_t = self.blocks4_att(x_t, x_t_at)
        x_t_mlp = self.adapter10(x_t, x2_mlp)
        x_t = self.blocks4_mlp(x_t, x_t_mlp)

        # 11 ~12
        x_r, x2_at = self.layer3_3(x_r)
        x_r, x2_mlp = self.layer3_4(x_r)
        x_t_at = self.adapter11(x_t, x2_at)
        x_t = self.blocks5_att(x_t, x_t_at)
        x_t_mlp = self.adapter12(x_t, x2_mlp)
        x_t = self.blocks5_mlp(x_t, x_t_mlp)

        # 13 ~14
        x_r, x2_at = self.layer3_5(x_r)
        x_r, x2_mlp = self.layer4_0(x_r)
        x_t_at = self.adapter13(x_t, x2_at)
        x_t = self.blocks6_att(x_t, x_t_at)
        x_t_mlp = self.adapter14(x_t, x2_mlp)
        x_t = self.blocks6_mlp(x_t, x_t_mlp)

        # 15 ~16
        x_r, x2_at = self.layer4_1(x_r)
        x_r, x2_mlp = self.layer4_2(x_r)
        x_t_at = self.adapter15(x_t, x2_at)
        x_t = self.blocks7_att(x_t, x_t_at)
        x_t_mlp = self.adapter16(x_t, x2_mlp)
        x_t = self.blocks7_mlp(x_t, x_t_mlp)

        # 9~12
        x_t = self.blocks8(x_t)

        # trans classification
        x_t = self.norm(x_t)
        tran_cls = self.head(x_t[:, 0])

        return tran_cls


if __name__ == '__main__':
    import pprint

    model = Base_50(f=8, m=8, h=8, r=1, a=1, s=0.1, num_classes=100, drop_path_rate=0.1)
    # summary(model, input_size=(1, 3, 224, 224))
    model_dict = model.state_dict()


    def test(model_dict):
        replace_dict = {'layer1.0': 'layer1_0',
                        'layer1.1': 'layer1_1',
                        'layer1.2': 'layer1_2',
                        'layer2.0': 'layer2_0',
                        'layer2.1': 'layer2_1',
                        'layer2.2': 'layer2_2',
                        'layer2.3': 'layer2_3',
                        'layer3.0': 'layer3_0',
                        'layer3.1': 'layer3_1',
                        'layer3.2': 'layer3_2',
                        'layer3.3': 'layer3_3',
                        'layer3.4': 'layer3_4',
                        'layer3.5': 'layer3_5',
                        'layer4.0': 'layer4_0',
                        'layer4.1': 'layer4_1',
                        'layer4.2': 'layer4_2',
                        'blocks.0.norm1': 'blocks0_att.norm1',
                        'blocks.0.attn': 'blocks0_att.attn',
                        'blocks.0.norm2': 'blocks0_mlp.norm2',
                        'blocks.0.mlp': 'blocks0_mlp.mlp',

                        'blocks.1.norm1': 'blocks1_att.norm1',
                        'blocks.1.attn': 'blocks1_att.attn',
                        'blocks.1.norm2': 'blocks1_mlp.norm2',
                        'blocks.1.mlp': 'blocks1_mlp.mlp',

                        'blocks.2.norm1': 'blocks2_att.norm1',
                        'blocks.2.attn': 'blocks2_att.attn',
                        'blocks.2.norm2': 'blocks2_mlp.norm2',
                        'blocks.2.mlp': 'blocks2_mlp.mlp',

                        'blocks.3.norm1': 'blocks3_att.norm1',
                        'blocks.3.attn': 'blocks3_att.attn',
                        'blocks.3.norm2': 'blocks3_mlp.norm2',
                        'blocks.3.mlp': 'blocks3_mlp.mlp',

                        'blocks.4.norm1': 'blocks4_att.norm1',
                        'blocks.4.attn': 'blocks4_att.attn',
                        'blocks.4.norm2': 'blocks4_mlp.norm2',
                        'blocks.4.mlp': 'blocks4_mlp.mlp',

                        'blocks.5.norm1': 'blocks5_att.norm1',
                        'blocks.5.attn': 'blocks5_att.attn',
                        'blocks.5.norm2': 'blocks5_mlp.norm2',
                        'blocks.5.mlp': 'blocks5_mlp.mlp',

                        'blocks.6.norm1': 'blocks6_att.norm1',
                        'blocks.6.attn': 'blocks6_att.attn',
                        'blocks.6.norm2': 'blocks6_mlp.norm2',
                        'blocks.6.mlp': 'blocks6_mlp.mlp',

                        'blocks.7.norm1': 'blocks7_att.norm1',
                        'blocks.7.attn': 'blocks7_att.attn',
                        'blocks.7.norm2': 'blocks7_mlp.norm2',
                        'blocks.7.mlp': 'blocks7_mlp.mlp',

                        'blocks.8': 'blocks8.0',
                        'blocks.9': 'blocks8.1',
                        'blocks.10': 'blocks8.2',
                        'blocks.11': 'blocks8.3',
                        }
        model_dict_new = {}
        for key, value in model_dict.items():
            new_key = key
            for old, new in replace_dict.items():
                new_key = new_key.replace(new, old)
            model_dict_new[new_key] = value

        model_dict.clear()
        model_dict.update(model_dict_new)

        for key, value in model_dict.items():
            print(key)

        return model_dict

    def dict2new(model_dict):
        replace_dict = {'layer1.0': 'layer1_0',
                        'layer1.1': 'layer1_1',
                        'layer1.2': 'layer1_2',
                        'layer2.0': 'layer2_0',
                        'layer2.1': 'layer2_1',
                        'layer2.2': 'layer2_2',
                        'layer2.3': 'layer2_3',
                        'layer3.0': 'layer3_0',
                        'layer3.1': 'layer3_1',
                        'layer3.2': 'layer3_2',
                        'layer3.3': 'layer3_3',
                        'layer3.4': 'layer3_4',
                        'layer3.5': 'layer3_5',
                        'layer4.0': 'layer4_0',
                        'layer4.1': 'layer4_1',
                        'layer4.2': 'layer4_2',
                        'blocks.0.norm1': 'blocks0_att.norm1',
                        'blocks.0.attn': 'blocks0_att.attn',
                        'blocks.0.norm2': 'blocks0_mlp.norm2',
                        'blocks.0.mlp': 'blocks0_mlp.mlp',

                        'blocks.1.norm1': 'blocks1_att.norm1',
                        'blocks.1.attn': 'blocks1_att.attn',
                        'blocks.1.norm2': 'blocks1_mlp.norm2',
                        'blocks.1.mlp': 'blocks1_mlp.mlp',

                        'blocks.2.norm1': 'blocks2_att.norm1',
                        'blocks.2.attn': 'blocks2_att.attn',
                        'blocks.2.norm2': 'blocks2_mlp.norm2',
                        'blocks.2.mlp': 'blocks2_mlp.mlp',

                        'blocks.3.norm1': 'blocks3_att.norm1',
                        'blocks.3.attn': 'blocks3_att.attn',
                        'blocks.3.norm2': 'blocks3_mlp.norm2',
                        'blocks.3.mlp': 'blocks3_mlp.mlp',

                        'blocks.4.norm1': 'blocks4_att.norm1',
                        'blocks.4.attn': 'blocks4_att.attn',
                        'blocks.4.norm2': 'blocks4_mlp.norm2',
                        'blocks.4.mlp': 'blocks4_mlp.mlp',

                        'blocks.5.norm1': 'blocks5_att.norm1',
                        'blocks.5.attn': 'blocks5_att.attn',
                        'blocks.5.norm2': 'blocks5_mlp.norm2',
                        'blocks.5.mlp': 'blocks5_mlp.mlp',

                        'blocks.6.norm1': 'blocks6_att.norm1',
                        'blocks.6.attn': 'blocks6_att.attn',
                        'blocks.6.norm2': 'blocks6_mlp.norm2',
                        'blocks.6.mlp': 'blocks6_mlp.mlp',

                        'blocks.7.norm1': 'blocks7_att.norm1',
                        'blocks.7.attn': 'blocks7_att.attn',
                        'blocks.7.norm2': 'blocks7_mlp.norm2',
                        'blocks.7.mlp': 'blocks7_mlp.mlp',

                        'blocks.8': 'blocks8.0',
                        'blocks.9': 'blocks8.1',
                        'blocks.10': 'blocks8.2',
                        'blocks.11': 'blocks8.3',
                        }
        model_dict_new = {}
        for key, value in model_dict.items():
            new_key = key
            for old, new in replace_dict.items():
                new_key = new_key.replace(old, new)
            model_dict_new[new_key] = value

        model_dict.clear()
        model_dict.update(model_dict_new)


    def load_model_weights(model, model_path):
        state = torch.load(model_path, map_location='cpu')
        for key in model.state_dict():
            if 'num_batches_tracked' in key:
                continue
            p = model.state_dict()[key]
            if key in state['state_dict']:
                ip = state['state_dict'][key]
                if p.shape == ip.shape:
                    p.data.copy_(ip.data)  # Copy the data of parameters
        return model


    def load_dict(model, class_num):
        vit_model = create_model('vit_base_patch16_224_in21k', checkpoint_path='./pre_weight/ViT-B_16.npz',
                                 drop_path_rate=0.1)
        vit_model.reset_classifier(class_num)
        torch.save(vit_model.state_dict(), './load_weight/vit-b_16.pth')
        vit_dict = torch.load('./load_weight/vit-b_16.pth')
        dict2new(vit_dict)

        resnet50 = create_model('resnet50', num_classes=11221)
        resnet50 = load_model_weights(resnet50, './pre_weight/resnet50_21k.pth')
        resnet50.reset_classifier(class_num)
        torch.save(resnet50.state_dict(), './load_weight/resnet50_21k_train.pth')
        resnet50_dict = torch.load('./load_weight/resnet50_21k_train.pth')
        dict2new(resnet50_dict)

        model.load_state_dict(vit_dict, strict=False)
        model.load_state_dict(resnet50_dict, strict=False)


    model_dict = test(model_dict)

    # input_zero = torch.ones((1, 3, 224, 224))
    # output = model(input_zero)
    # print(output[0].shape,output[1].shape)

    trainable = []
    # for n, p in model.named_parameters():
    #     print(n)
    #     if 'blocks' in n:
    #         p.requires_grad = False
    #     else:
    #         trainable.append(p)

    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable.append(p)
        else:
            p.requires_grad = False

    # model_dict = test(model_dict)
    # for key, value in model_dict.items():
    #     print(key)

    summary(model, input_size=(1, 3, 224, 224))

    # pprint.pprint(model)
    # input = torch.randn(1, 3, 224, 224)
    # modelpath = './demo.onnx'
    # torch.onnx.export(model, input, modelpath)
    # netron.start(modelpath)
