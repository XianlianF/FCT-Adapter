# 快速选择模型
from .FCT_Adapter.Base_50_adapter import Resnet_ViT
from .FCT_Adapter.Small_34_adapter import Resnet34_ViT_S
from .Conformer.conformer import Conformer
from .Convpass.convpass import Convpass_model
from timm.models import create_model


def select_model(args, config):
    type2model = {
        'Resnet_ViT': Resnet_ViT(num_classes=config['class_num'], f=8, h=8, m=8, r=1, s=0.1, a=1),
        'Resnet34_ViT_S': Resnet34_ViT_S(num_classes=config['class_num'], f=8, h=8, m=8, r=1, s=0.1, a=1),
        'Conformer_B': Conformer(patch_size=16, num_classes=config['class_num'], channel_ratio=6, embed_dim=576,
                                 depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True),
        'Conformer_S': Conformer(patch_size=16, num_classes=config['class_num'], channel_ratio=4, embed_dim=384,
                                 depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True),
        'Convpass': Convpass_model(num_classes=config['class_num'], s=config['scale'],
                                   xavier_init=config['xavier_init']),
        'ViT-S/16': create_model('vit_small_patch16_224',
                                 checkpoint_path='./pre_weight/ViT-S_16.npz',
                                 num_classes=config['class_num'], drop_path_rate=0.1),
        'ViT-B/16': create_model('vit_base_patch16_224_in21k',
                                 checkpoint_path='./pre_weight/ViT-B_16.npz',
                                 num_classes=config['class_num'], drop_path_rate=0.1),
        'Swin-B': create_model('vit_base_patch16_224_in21k',
                               checkpoint_path='./pre_weight/ViT-B_16.npz',
                               num_classes=config['class_num'], drop_path_rate=0.1),
        'Convnext': create_model("convnext_base_in22k",
                                 checkpoint_path='./pre_weight/convnext_base_22k_224.pth',
                                 num_classes=config['class_num'],
                                 drop_path_rate=0.1)
    }
    model = type2model[args.model_type]
    return model
