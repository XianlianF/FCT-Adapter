import timm.models.vision_transformer
import torch
import torch.nn as nn
from functools import partial
from timm.models import create_model
from timm.models.layers import DropBlock2d
from torchinfo import summary


class Convchor(nn.Module):
    def __init__(self, ch_in, f, ch_out, stride, r, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(Convchor, self).__init__()
        self.pointwise_conv1 = nn.Conv2d(ch_in, f, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.pointwise_conv1.weight)
        nn.init.zeros_(self.pointwise_conv1.bias)
        self.bn1 = norm_layer(f)
        self.act1 = act_layer(inplace=True)

        self.depthwise_conv = nn.Conv2d(f, f, kernel_size=3, stride=stride, padding=1, groups=r)
        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)
        self.bn2 = norm_layer(f)
        self.act2 = act_layer(inplace=True)

        self.pointwise_conv2 = nn.Conv2d(f, ch_out, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.pointwise_conv2.weight)
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


def drop_blocks(drop_block_rate=0.):
    return DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None


class Transchor(nn.Module):
    def __init__(self, embed_dim, h=8, xavier_init=False, cat=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.embed_dim = embed_dim
        self.depthwise_conv = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1,
                                        groups=self.embed_dim)

        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)

        self.bn = nn.BatchNorm2d(self.embed_dim)

        if cat == False:
            self.adapter_down = nn.Linear(self.embed_dim, h)  # equivalent to 1 * 1 Conv
        else:
            self.adapter_down = nn.Linear(self.embed_dim * 2, h)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(h, self.embed_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.h = h

    def forward(self, x):
        B, N, C = x.shape

        x_patch = x[:, 1:].reshape(B, 14, 14, self.embed_dim).permute(0, 3, 1, 2)
        x_cls = x[:, :1]

        x_patch = self.depthwise_conv(x_patch)
        x_patch = self.bn(x_patch)
        x_patch = self.act(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.embed_dim)

        x = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Bridge(nn.Module):
    def __init__(self, outplanes, embed_dim, m=8, a=1, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Bridge, self).__init__()
        self.embed_dim = embed_dim
        dw_stride = 1024 // outplanes

        # 降维
        self.adapter_down = nn.Conv2d(outplanes // 4, m, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_down.weight)

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
        self.adapter_up = nn.Conv2d(m, self.embed_dim, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        self.bn2 = nn.BatchNorm2d(self.embed_dim)
        self.act2 = act_layer()

        self.a = a

    def forward(self, x_r):
        B, _, _, _ = x_r.shape
        x_r = self.adapter_down(x_r)
        x_r = self.bn1(x_r)
        x_r = self.act1(x_r)

        x_r = self.adapter_conv(x_r)

        x_r = self.adapter_up(x_r)

        x_r = self.bn2(x_r)

        x_r = self.act2(x_r)

        x_r = x_r.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.embed_dim)
        x_r = torch.cat([x_r[:, 0:1, :], x_r], dim=1)
        x_t = self.a * x_r

        return x_t


def forward_conv(self, x):
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    x1 = self.conv_adapter(x)
    if self.drop_block is not None:
        x1 = self.drop_block(x1)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.drop_block(x)
    x = self.act2(x)
    x = self.aa(x)

    x2 = x1 + x

    x = self.conv3(x2)
    x = self.bn3(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x += shortcut
    x = self.act3(x)
    return x, x2


def forward_trans(self, x, y1=None, y2=None):
    if y1 != None:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))) + self.drop_path1(
            self.ls1(self.adapter_attn(self.norm1(x) + y1))) * self.s
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.drop_path2(
            self.ls2(self.adapter_mlp(self.norm2(x) + y2))) * self.s
    if y1 == None:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def set_adapter(model, adapter, embed_dim, f=8, h=8, r=1, s=0.1):
    if adapter == "Convchor":
        for _ in model.modules():
            if type(_) == timm.models.resnet.Bottleneck:
                ch_in = _.conv2.in_channels
                ch_out = _.conv2.out_channels
                stride = _.conv2.stride
                _.conv_adapter = Convchor(ch_in=ch_in, f=f, ch_out=ch_out, stride=stride, r=r)
                bound_method = forward_conv.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)

    if adapter == "Transchor":
        for _ in model.modules():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Transchor(embed_dim=embed_dim, h=h)
                _.adapter_mlp = Transchor(embed_dim=embed_dim, h=h)
                _.s = s
                bound_method = forward_trans.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)


class Base_50_adapter(nn.Module):
    def __init__(self, cnn, vit, m=8, a=1, num_classes=100, embed_dim=768):
        # Transformer
        super().__init__()

        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.act1 = cnn.act1
        self.maxpool = cnn.maxpool

        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop

        self.patch_embed = vit.patch_embed
        self._pos_embed = vit._pos_embed

        # 获取 ResNet50 的 4 个阶段
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4

        # 获取 ViT-b/16 的 12 个块
        self.blocks = nn.ModuleList(list(vit.blocks.children()))

        self.adapter1 = Bridge(256, embed_dim=embed_dim, m=m, a=a)
        self.adapter2 = Bridge(256, embed_dim=embed_dim, m=m, a=a)
        self.adapter3 = Bridge(256, embed_dim=embed_dim, m=m, a=a)
        self.adapter4 = Bridge(512, embed_dim=embed_dim, m=m, a=a)
        self.adapter5 = Bridge(512, embed_dim=embed_dim, m=m, a=a)
        self.adapter6 = Bridge(512, embed_dim=embed_dim, m=m, a=a)
        self.adapter7 = Bridge(512, embed_dim=embed_dim, m=m, a=a)
        self.adapter8 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter9 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter10 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter11 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter12 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter13 = Bridge(1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter14 = Bridge(2048, embed_dim=embed_dim, m=m, a=a)
        self.adapter15 = Bridge(2048, embed_dim=embed_dim, m=m, a=a)
        self.adapter16 = Bridge(2048, embed_dim=embed_dim, m=m, a=a)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x_r = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)

        # 1 ~2
        x_r, x2_at = self.layer1[0](x_r)
        x_r, x2_mlp = self.layer1[1](x_r)
        x_t_at = self.adapter1(x2_at)
        x_t_mlp = self.adapter2(x2_mlp)
        x_t = self.blocks[0](x_t, x_t_at, x_t_mlp)

        # 3 ~4
        x_r, x2_at = self.layer1[2](x_r)
        x_r, x2_mlp = self.layer2[0](x_r)
        x_t_at = self.adapter3(x2_at)
        x_t_mlp = self.adapter4(x2_mlp)
        x_t = self.blocks[1](x_t, x_t_at, x_t_mlp)

        # 5 ~6
        x_r, x2_at = self.layer2[1](x_r)
        x_r, x2_mlp = self.layer2[2](x_r)
        x_t_at = self.adapter5(x2_at)
        x_t_mlp = self.adapter6(x2_mlp)
        x_t = self.blocks[2](x_t, x_t_at, x_t_mlp)

        # 7 ~8
        x_r, x2_at = self.layer2[3](x_r)
        x_r, x2_mlp = self.layer3[0](x_r)
        x_t_at = self.adapter7(x2_at)
        x_t_mlp = self.adapter8(x2_mlp)
        x_t = self.blocks[3](x_t, x_t_at, x_t_mlp)

        # 9 ~10
        x_r, x2_at = self.layer3[1](x_r)
        x_r, x2_mlp = self.layer3[2](x_r)
        x_t_at = self.adapter9(x2_at)
        x_t_mlp = self.adapter10(x2_mlp)
        x_t = self.blocks[4](x_t, x_t_at, x_t_mlp)

        # 11 ~12
        x_r, x2_at = self.layer3[3](x_r)
        x_r, x2_mlp = self.layer3[4](x_r)
        x_t_at = self.adapter11(x2_at)
        x_t_mlp = self.adapter12(x2_mlp)
        x_t = self.blocks[5](x_t, x_t_at, x_t_mlp)

        # 13 ~14
        x_r, x2_at = self.layer3[5](x_r)
        x_r, x2_mlp = self.layer4[0](x_r)
        x_t_at = self.adapter13(x2_at)
        x_t_mlp = self.adapter14(x2_mlp)
        x_t = self.blocks[6](x_t, x_t_at, x_t_mlp)

        # 15 ~16
        x_r, x2_at = self.layer4[1](x_r)
        x_r, x2_mlp = self.layer4[2](x_r)
        x_t_at = self.adapter15(x2_at)
        x_t_mlp = self.adapter16(x2_mlp)
        x_t = self.blocks[7](x_t, x_t_at, x_t_mlp)

        # 9~12
        x_t = self.blocks[8](x_t)
        x_t = self.blocks[9](x_t)
        x_t = self.blocks[10](x_t)
        x_t = self.blocks[11](x_t)

        x_t = self.norm(x_t)
        tran_cls = self.head(x_t[:, 0])
        return tran_cls


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


class Resnet_ViT(nn.Module):
    """
    Resnet50和ViT-B/16

    参数：
    - f: Convhor隐藏层维数
    - h: Tranchor隐藏层维数
    - m: Bridge隐藏层维数
    - r: Conchor输出系数
    - s: Tranchor输出系数
    - a: Bridge输出系数
    """

    def __init__(self, num_classes, f, h, m, r, s, a):
        super().__init__()
        self.resnet50 = create_model('resnet50', num_classes=11221, drop_path_rate=0.1)
        self.resnet50 = load_model_weights(self.resnet50, '../pre_weight/resnet50_21k.pth')
        self.vit_model = create_model('vit_base_patch16_224_in21k',
                                      checkpoint_path='../pre_weight/ViT-B_16.npz',
                                      drop_path_rate=0.1)
        embed_dim = self.vit_model.embed_dim
        set_adapter(self.resnet50, "Convchor", embed_dim=embed_dim, f=f, h=h, r=r, s=s)
        set_adapter(self.vit_model, "Transchor", f=f, h=h, embed_dim=embed_dim, r=r, s=s)

        self.model = Base_50_adapter(cnn=self.resnet50, vit=self.vit_model, m=m, a=a, num_classes=num_classes,
                                     embed_dim=embed_dim)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = Resnet_ViT(num_classes=100, f=8, h=8, m=8, r=1, s=0.1, a=1)
    model_dict = model.state_dict()
    # print(model_dict)
    for key, weight in model_dict.items():
        if key == 'blocks.4.8.mlp.fc2.weight':
            print(weight)

    trainable = []

    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable.append(p)
        else:
            p.requires_grad = False

    summary(model, input_size=(1, 3, 224, 224))
