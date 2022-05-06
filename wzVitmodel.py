"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
from torchsummary import summary
import torch
import torch.nn as nn


# 原论文中在多头注意力之后 使用的是dropout层，有其他大神用droppath实现，效果更好
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
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
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 将图像块
class PatchEmbed(nn.Module):
    """
    将一张2D图像划分成多个图像块，这个过程实际是卷积实现的。之后将每个图像块映射成一维向量，作为transformer的真正输入
    """
    # TODO norm_layer本来是None，我改成了nn.LayerNorm
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else: self.norm = nn.Identity() # 我改成了self.norm = nn.Identity()，原来是nn.Identity()
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# 多头注意力
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads # 注意力头的数量
        head_dim = dim // num_heads # 每个注意力头对应的维度大小

        # 缩放点积的分母
        self.scale = qk_scale or head_dim ** -0.5

        # 这里作者直接用 dim * 3 得到qkv三个向量，推测是为了并行化计算加快训练速度。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        # 每个多头注意力输出拼接，得到最终的输出
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # 每一批图片数，加1是因为class token， 对应768
        print(x.shape)
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        attn = (q @ k.transpose(-2, -1)) * self.scale # q和k进行多维矩阵乘法时，实际只有最后两个维度相乘
        attn = attn.softmax(dim=-1)# 似乎是二维矩阵，按行进行柔性最大值处理。
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 和v矩阵相乘 加权求和
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# transformer编码器块中的MLP块
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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


# transformer编码器模块，将自注意力和MLP结合起来
class Block(nn.Module):
    def __init__(self,
                 dim, # 输入向量的特征维度
                 num_heads, # 注意力头的数量
                 mlp_ratio=4., # 第一个全连接层的节点是输入节点的四倍，对应768变为2304
                 qkv_bias=False, # 是否使用偏差
                 qk_scale=None, # 缩放因子
                 drop_ratio=0., # 对应多头注意力模块中最后一个全连接层
                 attn_drop_ratio=0., # qkv矩阵计算输出之后经过softmax层后的dropout
                 drop_path_ratio=0., # 对应的编码块中droppath
                 act_layer=nn.GELU, # 默认激活函数
                 norm_layer=nn.LayerNorm): # 默认归一化方式
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) # 编码块中的第一个LN层
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # TODO 如果drop_path_ratio大于0，就采用droppath否则直接用恒等映射。
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # 对应MLP块中最前面的的LN层
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # MLP模块中第一个全连接层对应的隐藏层节点数量，这里相当于维度增加。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): transformer编码器重复的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            # TODO representation_size 对应最后进行分类时的MLP分类头，None表示没有
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 这里没传入类“PatchEmbed”中的“norm_layer”参数，我直接修改了PatchEmbed定义
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 第一个维度的1对应的是批量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 位置嵌入是随机生成的？
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio) # 加上位置嵌入向量之后的drop层

        # 构建了一个等差数列，保存后续transformer编码块中的drop率
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # transformer编码块重复指定的次数
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim) # 在所有的transformer编码块之后的LN层

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() # 不使用pre_logits层

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() # 最后一个全连接层
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # 得到图像块对应的嵌入向量
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            # 默认情况。拼接分类嵌入向量和x
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed) # 加上位置嵌入信息，准备输入transformer编码器

        x = self.blocks(x)
        x = self.norm(x) # 输出[1, 197, 768]

        # 下面似乎是和具体的分类任务相关的
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    # 下面计算中出错
    def forward(self, x):
        x = self.forward_features(x)
        print(x.shape)
        if self.head_dist is not None:
            print('head_dist')
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            print('head_dist是none')
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

if __name__ == '__main__':
    测试模型 = vit_base_patch16_224_in21k(10)
    测试模型.to(torch.device("cuda:0"))
    summary(测试模型, input_size=(3, 224, 224), batch_size=3, device='cuda')

    # 测试模型 = PatchEmbed()
    # 测试模型.to(torch.device("cuda:0"))
    # summary(测试模型, input_size=(3, 224, 224), batch_size=3, device='cuda')

'''
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [1, 768, 14, 14]         590,592
         LayerNorm-2              [1, 196, 768]           1,536
        PatchEmbed-3              [1, 196, 768]               0
           Dropout-4              [1, 197, 768]               0
         LayerNorm-5              [1, 197, 768]           1,536
            Linear-6             [1, 197, 2304]       1,771,776
           Dropout-7          [1, 12, 197, 197]               0
            Linear-8              [1, 197, 768]         590,592
           Dropout-9              [1, 197, 768]               0
        Attention-10              [1, 197, 768]               0
         Identity-11              [1, 197, 768]               0
        LayerNorm-12              [1, 197, 768]           1,536
           Linear-13             [1, 197, 3072]       2,362,368
             GELU-14             [1, 197, 3072]               0
          Dropout-15             [1, 197, 3072]               0
           Linear-16              [1, 197, 768]       2,360,064
          Dropout-17              [1, 197, 768]               0
              Mlp-18              [1, 197, 768]               0
         Identity-19              [1, 197, 768]               0
            Block-20              [1, 197, 768]               0
            
        LayerNorm-21              [1, 197, 768]           1,536
           Linear-22             [1, 197, 2304]       1,771,776
          Dropout-23          [1, 12, 197, 197]               0
           Linear-24              [1, 197, 768]         590,592
          Dropout-25              [1, 197, 768]               0
        Attention-26              [1, 197, 768]               0
        
         Identity-27              [1, 197, 768]               0
        LayerNorm-28              [1, 197, 768]           1,536
           Linear-29             [1, 197, 3072]       2,362,368
             GELU-30             [1, 197, 3072]               0
          Dropout-31             [1, 197, 3072]               0
           Linear-32              [1, 197, 768]       2,360,064
          Dropout-33              [1, 197, 768]               0
              Mlp-34              [1, 197, 768]               0
              
         Identity-35              [1, 197, 768]               0
            Block-36              [1, 197, 768]               0
            
        LayerNorm-37              [1, 197, 768]           1,536
           Linear-38             [1, 197, 2304]       1,771,776
          Dropout-39          [1, 12, 197, 197]               0
           Linear-40              [1, 197, 768]         590,592
          Dropout-41              [1, 197, 768]               0
        Attention-42              [1, 197, 768]               0
         Identity-43              [1, 197, 768]               0
        LayerNorm-44              [1, 197, 768]           1,536
           Linear-45             [1, 197, 3072]       2,362,368
             GELU-46             [1, 197, 3072]               0
          Dropout-47             [1, 197, 3072]               0
           Linear-48              [1, 197, 768]       2,360,064
          Dropout-49              [1, 197, 768]               0
              Mlp-50              [1, 197, 768]               0
         Identity-51              [1, 197, 768]               0
            Block-52              [1, 197, 768]               0
        LayerNorm-53              [1, 197, 768]           1,536
           Linear-54             [1, 197, 2304]       1,771,776
          Dropout-55          [1, 12, 197, 197]               0
           Linear-56              [1, 197, 768]         590,592
          Dropout-57              [1, 197, 768]               0
        Attention-58              [1, 197, 768]               0
         Identity-59              [1, 197, 768]               0
        LayerNorm-60              [1, 197, 768]           1,536
           Linear-61             [1, 197, 3072]       2,362,368
             GELU-62             [1, 197, 3072]               0
          Dropout-63             [1, 197, 3072]               0
           Linear-64              [1, 197, 768]       2,360,064
          Dropout-65              [1, 197, 768]               0
              Mlp-66              [1, 197, 768]               0
         Identity-67              [1, 197, 768]               0
            Block-68              [1, 197, 768]               0
        LayerNorm-69              [1, 197, 768]           1,536
           Linear-70             [1, 197, 2304]       1,771,776
          Dropout-71          [1, 12, 197, 197]               0
           Linear-72              [1, 197, 768]         590,592
          Dropout-73              [1, 197, 768]               0
        Attention-74              [1, 197, 768]               0
         Identity-75              [1, 197, 768]               0
        LayerNorm-76              [1, 197, 768]           1,536
           Linear-77             [1, 197, 3072]       2,362,368
             GELU-78             [1, 197, 3072]               0
          Dropout-79             [1, 197, 3072]               0
           Linear-80              [1, 197, 768]       2,360,064
          Dropout-81              [1, 197, 768]               0
              Mlp-82              [1, 197, 768]               0
         Identity-83              [1, 197, 768]               0
            Block-84              [1, 197, 768]               0
        LayerNorm-85              [1, 197, 768]           1,536
           Linear-86             [1, 197, 2304]       1,771,776
          Dropout-87          [1, 12, 197, 197]               0
           Linear-88              [1, 197, 768]         590,592
          Dropout-89              [1, 197, 768]               0
        Attention-90              [1, 197, 768]               0
         Identity-91              [1, 197, 768]               0
        LayerNorm-92              [1, 197, 768]           1,536
           Linear-93             [1, 197, 3072]       2,362,368
             GELU-94             [1, 197, 3072]               0
          Dropout-95             [1, 197, 3072]               0
           Linear-96              [1, 197, 768]       2,360,064
          Dropout-97              [1, 197, 768]               0
              Mlp-98              [1, 197, 768]               0
         Identity-99              [1, 197, 768]               0
           Block-100              [1, 197, 768]               0
       LayerNorm-101              [1, 197, 768]           1,536
          Linear-102             [1, 197, 2304]       1,771,776
         Dropout-103          [1, 12, 197, 197]               0
          Linear-104              [1, 197, 768]         590,592
         Dropout-105              [1, 197, 768]               0
       Attention-106              [1, 197, 768]               0
        Identity-107              [1, 197, 768]               0
       LayerNorm-108              [1, 197, 768]           1,536
          Linear-109             [1, 197, 3072]       2,362,368
            GELU-110             [1, 197, 3072]               0
         Dropout-111             [1, 197, 3072]               0
          Linear-112              [1, 197, 768]       2,360,064
         Dropout-113              [1, 197, 768]               0
             Mlp-114              [1, 197, 768]               0
        Identity-115              [1, 197, 768]               0
           Block-116              [1, 197, 768]               0
       LayerNorm-117              [1, 197, 768]           1,536
          Linear-118             [1, 197, 2304]       1,771,776
         Dropout-119          [1, 12, 197, 197]               0
          Linear-120              [1, 197, 768]         590,592
         Dropout-121              [1, 197, 768]               0
       Attention-122              [1, 197, 768]               0
        Identity-123              [1, 197, 768]               0
       LayerNorm-124              [1, 197, 768]           1,536
          Linear-125             [1, 197, 3072]       2,362,368
            GELU-126             [1, 197, 3072]               0
         Dropout-127             [1, 197, 3072]               0
          Linear-128              [1, 197, 768]       2,360,064
         Dropout-129              [1, 197, 768]               0
             Mlp-130              [1, 197, 768]               0
        Identity-131              [1, 197, 768]               0
           Block-132              [1, 197, 768]               0
       LayerNorm-133              [1, 197, 768]           1,536
          Linear-134             [1, 197, 2304]       1,771,776
         Dropout-135          [1, 12, 197, 197]               0
          Linear-136              [1, 197, 768]         590,592
         Dropout-137              [1, 197, 768]               0
       Attention-138              [1, 197, 768]               0
        Identity-139              [1, 197, 768]               0
       LayerNorm-140              [1, 197, 768]           1,536
          Linear-141             [1, 197, 3072]       2,362,368
            GELU-142             [1, 197, 3072]               0
         Dropout-143             [1, 197, 3072]               0
          Linear-144              [1, 197, 768]       2,360,064
         Dropout-145              [1, 197, 768]               0
             Mlp-146              [1, 197, 768]               0
        Identity-147              [1, 197, 768]               0
           Block-148              [1, 197, 768]               0
       LayerNorm-149              [1, 197, 768]           1,536
          Linear-150             [1, 197, 2304]       1,771,776
         Dropout-151          [1, 12, 197, 197]               0
          Linear-152              [1, 197, 768]         590,592
         Dropout-153              [1, 197, 768]               0
       Attention-154              [1, 197, 768]               0
        Identity-155              [1, 197, 768]               0
       LayerNorm-156              [1, 197, 768]           1,536
          Linear-157             [1, 197, 3072]       2,362,368
            GELU-158             [1, 197, 3072]               0
         Dropout-159             [1, 197, 3072]               0
          Linear-160              [1, 197, 768]       2,360,064
         Dropout-161              [1, 197, 768]               0
             Mlp-162              [1, 197, 768]               0
        Identity-163              [1, 197, 768]               0
           Block-164              [1, 197, 768]               0
       LayerNorm-165              [1, 197, 768]           1,536
          Linear-166             [1, 197, 2304]       1,771,776
         Dropout-167          [1, 12, 197, 197]               0
          Linear-168              [1, 197, 768]         590,592
         Dropout-169              [1, 197, 768]               0
       Attention-170              [1, 197, 768]               0
        Identity-171              [1, 197, 768]               0
       LayerNorm-172              [1, 197, 768]           1,536
          Linear-173             [1, 197, 3072]       2,362,368
            GELU-174             [1, 197, 3072]               0
         Dropout-175             [1, 197, 3072]               0
          Linear-176              [1, 197, 768]       2,360,064
         Dropout-177              [1, 197, 768]               0
             Mlp-178              [1, 197, 768]               0
        Identity-179              [1, 197, 768]               0
           Block-180              [1, 197, 768]               0
       LayerNorm-181              [1, 197, 768]           1,536
          Linear-182             [1, 197, 2304]       1,771,776
         Dropout-183          [1, 12, 197, 197]               0
          Linear-184              [1, 197, 768]         590,592
         Dropout-185              [1, 197, 768]               0
       Attention-186              [1, 197, 768]               0
        Identity-187              [1, 197, 768]               0
       LayerNorm-188              [1, 197, 768]           1,536
          Linear-189             [1, 197, 3072]       2,362,368
            GELU-190             [1, 197, 3072]               0
         Dropout-191             [1, 197, 3072]               0
          Linear-192              [1, 197, 768]       2,360,064
         Dropout-193              [1, 197, 768]               0
             Mlp-194              [1, 197, 768]               0
        Identity-195              [1, 197, 768]               0
           Block-196              [1, 197, 768]               0
           
       LayerNorm-197              [1, 197, 768]           1,536 所有transformer编码块之后的LN层
       
          Linear-198                   [1, 768]         590,592
            Tanh-199                   [1, 768]               0
            
          Linear-200                    [1, 10]           7,690
================================================================
Total params: 86,246,410
Trainable params: 86,246,410
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 408.54
Params size (MB): 329.00
Estimated Total Size (MB): 738.12
----------------------------------------------------------------
'''
