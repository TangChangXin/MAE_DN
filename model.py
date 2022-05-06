import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from torchsummary import summary
from tensorboardX import SummaryWriter

'''
B站论文解说：MAE中掩码的部分也是要经过处理后成为嵌入向量的？不是直接分块后遮盖住就不管的。
'''

'''
3维图像融合成2维图像使用MAE的方式，自监督阶段也使用MAE的方式。
3M图像256*304*304，按照16*16*16划分，共5776块，是196块的29倍；如何随机丢块。每个块大小像素点4096个，是768的5.3倍。768是三通道算出来的，4096是单通道算出来的
# TODO 方法1。图像块按照16*16*16划分之后沿着深度方向向下遍历；然后沿着高度方向再向右遍历；最后沿着宽度方向向前遍历。每个16*16*16的块输出1*16*16的平面图像，
纵向的所有块最终输出16个1*16*16的平面图像，将这些拼接成一个16*16*16的块再输入。
把沿着深度方向形状为256*16*16的图像块作为一组数据输入，模型定义的时候用一个层把16*16的形状转为256*256。
假设换个思路，对256*304的图像编码输出1*304的向量，然后将304张图片的输出结果拼接在一起作为融合图像。但是这样似乎不算充分利用体数据。
transformer编码器的输入似乎是768，所以
如果图像块是16*16*16，那
# TODO 方法2。图像块按照256*16*16划分，在256*16*16的块里随机选取某些层进行训练，输出1*16*16。
'''



"""
MAE随机选择图像块应该在MAE类中实现
"""


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


class 图像块嵌入(nn.Module):
    """
    如何将2维和3维整合在一起。3维输入的应该是
    现在认为输入的图像都是已经划分成图像块之后的结果。有监督训练的时候在另一个类里对有标签的图像划分。
    """
    def __init__(self, 图像形状, 图像块嵌入向量的维度):
        r"""
        加if分支处理3维和2维的区别
        输入通道数暂时没用到，有监督训练时如何对输入的图像划分
        3维输入数据形状BCDHW 1*2*256*16*16，在256中随机选取部分用来训练，输出1*256的向量，然后整形为1*16*16。

        :param 图像形状:
        :param 图像块嵌入向量的维度:
        """
        # todo 现在是3维为例
        super(图像块嵌入, self).__init__()
        self.图像形状 = 图像形状
        self.图像块的大小 = 16
        self.图像块数量 = 256 # TODO 注意修改
        self.标准化 = nn.LayerNorm(图像块嵌入向量的维度)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() 原代码

    def forward(self, x):
        # 输入1*2*256*16*16，展平后1*2*256*256。形状[B, C, D, H, W] -> [B, C, D , HW].交换维度后，1*256*256*2，形状[B, D, HW, C]
        x = torch.flatten(x, start_dim=3).permute(0, 2, 3, 1)
        # 再展平后输出1*256*512，形状[B, D, HWC]
        x = torch.flatten(x, 2)
        x = self.标准化(x)
        return x


class 自注意力(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self,
                 嵌入向量维度,
                 注意力头数量,
                 qkv_偏差=False,
                 qk_缩小因子=None,
                 自注意力丢弃率=0.,
                 全连接层丢弃率=0.):
        super(自注意力, self).__init__()
        self.注意力头数量 = 注意力头数量
        注意力头维度 = 嵌入向量维度 // 注意力头数量  # 每个注意力头的维度大小
        self.qk_缩小因子 = qk_缩小因子 or 注意力头维度 ** -0.5
        # 通过一个全连接层生成qkv三个向量，有助于并行化计算
        # 小数据集中qkv直接false，但是原版用参数控制true
        self.qkv = nn.Linear(嵌入向量维度, 嵌入向量维度 * 3, bias=qkv_偏差)
        self.自注意力丢弃 = nn.Dropout(自注意力丢弃率) # 自注意力信息经过softmax后的Dropout

        # 多头注意力的输出拼接后与Wo相乘得到最终的输出。Wo矩阵通过全连接层实现
        self.线性投影 = nn.Linear(嵌入向量维度, 嵌入向量维度)
        self.线性投影丢弃 = nn.Dropout(全连接层丢弃率)

    def forward(self, x):
        批量大小, 图像块数量, 嵌入向量维度 = x.shape
        # [batch_size, num_patches + 1, embed_dim] 每一批图片数。图像块的数量加1是因为算上class token，我的方法按照纵向的深度计算
        # qkv(): -> [批量大小, 图像块数量 + 1, 3 * 嵌入向量维度] 我不分类可能不加分类嵌入向量。
        # 3对应qkv三个向量。后面两个维度的数据可以理解为将嵌入向量按照注意力头的数量平均划分之后分别送入不同的注意力头中计算
        # reshape: -> [批量大小, 图像块数量 + 1, 3, 注意力头数量, 每个注意力头的嵌入向量维度]。
        # permute: -> [3, 批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        # TODO 报错
        qkv = self.qkv(x).reshape(批量大小, 图像块数量, 3, self.注意力头数量, 嵌入向量维度 // self.注意力头数量).permute(2, 0, 3, 1, 4)

        # 分别取出qkv向量。向量的形状[批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # k最后两个维度调换位置，k.transpose：-> [批量大小, 注意力头数量, 每个注意力头的嵌入向量维度, 图像块数量 + 1]
        # q和k矩阵相乘: -> [批量大小, 注意力头数量, 图像块数量 + 1, 图像块数量 + 1]
        自注意力信息 = (q @ k.transpose(-2, -1)) * self.qk_缩小因子  # q和k进行多维矩阵乘法时，实际只有最后两个维度相乘
        自注意力信息 = 自注意力信息.softmax(dim=-1)  # 沿着最后一个维度进行柔性最大值处理。
        自注意力信息 = self.自注意力丢弃(自注意力信息)

        # 自注意力信息和v矩阵相乘：-> [批量大小, 注意力头数量, 图像块数量 + 1, 每个注意力头的嵌入向量维度]
        # transpose: -> [批量大小, 图像块数量 + 1, 注意力头数量, 每个注意力头的嵌入向量维度]
        # reshape: -> [批量大小, 图像块数量 + 1, 嵌入向量维度]
        自注意力信息 = (自注意力信息 @ v).transpose(1, 2).reshape(批量大小, 图像块数量, 嵌入向量维度)  # 和v矩阵相乘 加权求和
        自注意力信息 = self.线性投影(自注意力信息)
        自注意力信息 = self.线性投影丢弃(自注意力信息)
        return 自注意力信息


class 多层感知机(nn.Module):
    """
    transformer编码块中的MLP模块
    """
    def __init__(self, 输入大小, 隐藏层大小=None, 激活函数=nn.GELU, 丢弃率=0.):
        """

        :param 输入大小: (int) 第一个全连接层的输入大小,默认和嵌入向量维度大小一致。
        :param 隐藏层大小: (int)
        :param 激活函数:
        :param 丢弃率:
        """
        super(多层感知机, self).__init__()
        隐藏层大小 = 隐藏层大小 or 输入大小
        self.全连接1 = nn.Linear(输入大小, 隐藏层大小)
        self.激活函数 = 激活函数()
        self.全连接2 = nn.Linear(隐藏层大小, 输入大小)
        self.随机丢弃 = nn.Dropout(丢弃率)

    def forward(self, x):
        x = self.全连接1(x)
        x = self.激活函数(x)
        x = self.随机丢弃(x)
        x = self.全连接2(x)
        x = self.随机丢弃(x)
        return x


class 编码块(nn.Module):
    """
    transformer编码器模块，将自注意力模块和MLP模块结合起来
    """
    def __init__(self,
                 特征维度, # 输入向量的维度
                 注意力头数量, # 注意力头的数量
                 多层感知机扩增率=4., # 第一个全连接层的节点是输入节点的四倍，对应768变为2304
                 qkv_偏差=False, # 是否使用偏差
                 qk_缩小因子=None, # 缩放因子
                 自注意力丢弃率=0.,  # qkv矩阵计算输出之后经过softmax层后的dropout
                 全连接层丢弃率=0., # 对应多头注意力模块中最后一个全连接层
                 drop_path_ratio=0., # 对应的编码块中droppath
                 激活函数=nn.GELU, # 默认激活函数
                 标准化=nn.LayerNorm): # 默认归一化方式
        super(编码块, self).__init__()
        self.标准化1 = 标准化(特征维度) # 编码块中的第一个LN层
        self.多头自注意力 = 自注意力(特征维度, 注意力头数量=注意力头数量, qkv_偏差=qkv_偏差, qk_缩小因子=qk_缩小因子, 自注意力丢弃率=自注意力丢弃率, 全连接层丢弃率=全连接层丢弃率)
        # TODO 如果drop_path_ratio大于0，就采用droppath否则直接用恒等映射。作者认为droppath比dropout效果好
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.标准化2 = 标准化(特征维度) # 对应MLP块中最前面的的LN层
        隐藏层维度 = int(特征维度 * 多层感知机扩增率) # MLP模块中第一个全连接层对应的隐藏层节点数量，这里相当于维度增加。
        self.多层感知机 = 多层感知机(输入大小=特征维度, 隐藏层大小=隐藏层维度, 激活函数=激活函数, 丢弃率=全连接层丢弃率)

    def forward(self, x):
        x = x + self.drop_path(self.多头自注意力(self.标准化1(x)))
        x = x + self.drop_path(self.多层感知机(self.标准化2(x))) # 不能从这里直接修改多层感知机的输出维度，因为要和原输入x相加
        return x


class 融合网络无cls(nn.Module):
    def __init__(self, 图像大小=224, 图像块大小=16, 输入通道数=3, 嵌入向量维度=512, 深度=12, 注意力头数量=8, 多层感知机扩增率=4.0,
                 qkv_偏差=True, qk_缩小因子=None, 全连接丢弃率=0., 自注意力丢弃率=0., drop_path_ratio=0., 图像嵌入层=图像块嵌入, 标准化=None,):
        """

        :param 图像大小:
        :param 图像块大小:
        :param 输入通道数: 卷积生成图像块时用到
        :param 嵌入向量维度: (int) 大小为图像块的宽度乘高度再乘2。也可以自己通过卷积输出通道数来指定。
        :param 深度: transformer编码器重复的次数。
        :param 注意力头数量: (int): 必须是嵌入向量维度的因数。
        :param 多层感知机扩增率: (int): 多层感知机模块中第一个全连接层的隐藏层节点数目和嵌入向量维度的比值
        :param qkv_偏差:
        :param qk_缩小因子:
        :param 全连接丢弃率:
        :param 自注意力丢弃率: 多头自注意力中输出经过softmax层之后的dropout层的丢弃率。
        :param drop_path_ratio:
        :param 图像嵌入层: (nn.Module): 将图像划分成若干块之后生成嵌入向量
        :param 标准化:
        """
        super(融合网络无cls, self).__init__()
        self.num_features = 嵌入向量维度  # num_features for consistency with other models
        标准化 = 标准化 or partial(nn.LayerNorm, eps=1e-6)
        self.图像块嵌入向量 = 图像嵌入层(图像形状=304, 图像块嵌入向量的维度=嵌入向量维度)
        图像块数量 = self.图像块嵌入向量.图像块数量

        # 第一个维度的1对应的是批量
        self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量, 嵌入向量维度))
        self.位置嵌入丢弃 = nn.Dropout(p=全连接丢弃率) # 加上位置嵌入向量之后的drop层

        # 构建了一个等差数列，保存后续transformer编码块中的drop率。最后一个transformer编码块需要令输出形状为1*256，方便调整为16*16
        丢弃率 = [x.item() for x in torch.linspace(0, drop_path_ratio, 深度)]  # stochastic depth decay rule
        # transformer编码块重复指定的次数
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=qkv_偏差, qk_缩小因子=qk_缩小因子,
                自注意力丢弃率=自注意力丢弃率, 全连接层丢弃率=全连接丢弃率, drop_path_ratio=丢弃率[i], 标准化=标准化)
            for i in range(深度)
        ])
        self.标准化 = 标准化(嵌入向量维度) # 在所有的transformer编码块之后的LN层。输出形状[1, 256， 512]
        self.通道融合 = nn.Linear(嵌入向量维度, 256) # 输出形状[1, 256， 256]
        self.激活函数 = nn.GELU()
        # TODO 这里可能需要修改
        self.特征降维 = nn.Linear(256, 1) # pytorch如何对指定维度的信息进行融合

        # 权重初始化
        nn.init.trunc_normal_(self.位置嵌入向量, std=0.02)
        self.apply(权重初始化)

    def forward(self, x):
        # [B, C, D, H, W] -> [批量, 图像块数量, 嵌入向量维度] 我的图像深度相当于图像块数量
        # 得到图像块对应的嵌入向量
        x = self.图像块嵌入向量(x)  # 输出形状[B, 256, 512]
        # 这里报错说明实际输入的嵌入向量维度和参数传入的嵌入向量维度大小不一致。
        x = self.位置嵌入丢弃(x + self.位置嵌入向量) # 加上位置嵌入信息，准备输入transformer编码器
        x = self.编码块堆叠(x)
        x = self.标准化(x) # 输出形状[1, 512， 256]
        x = self.通道融合(x) # 输出形状[1, 256， 256]
        x = self.激活函数(x)
        x = torch.transpose(x, 1, 2)
        x = self.特征降维(x) # 输出形状[1, 256， 1]
        x = torch.reshape(x, (-1, 16, 16)) # 输出形状[批量, 16, 16]
        # print(x.shape)
        return x


class 融合网络(nn.Module):
    def __init__(self, 图像大小=224, 图像块大小=16, 输入通道数=3, 分类数目=1000,
                 嵌入向量维度=512, 深度=12, 注意力头数量=8, 多层感知机扩增率=4.0, qkv_偏差=True,
                 qk_缩小因子=None, representation_size=None, 蒸馏=False, 全连接丢弃率=0.,
                 自注意力丢弃率=0., drop_path_ratio=0., 图像嵌入层=图像块嵌入, 标准化=None,):
        """

        :param 图像大小:
        :param 图像块大小:
        :param 输入通道数: 卷积生成图像块时用到
        :param 分类数目:
        :param 嵌入向量维度: (int) 大小为图像块的宽度乘高度再乘2。也可以自己通过卷积输出通道数来指定。
        :param 深度: transformer编码器重复的次数。
        :param 注意力头数量: (int): 必须是嵌入向量维度的因数。
        :param 多层感知机扩增率: (int): 多层感知机模块中第一个全连接层的隐藏层节点数目和嵌入向量维度的比值
        :param qkv_偏差:
        :param qk_缩小因子:
        :param representation_size:
        :param 蒸馏: (bool): model includes a distillation token and head as in DeiT models # TODO 蒸馏Vit模型
        :param 全连接丢弃率:
        :param 自注意力丢弃率: 多头自注意力中输出经过softmax层之后的dropout层的丢弃率。
        :param drop_path_ratio:
        :param 图像嵌入层: (nn.Module): 将图像划分成若干块之后生成嵌入向量
        :param 标准化:
        """
        """
        Args:
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            # TODO representation_size 对应最后进行分类时的MLP分类头，None表示没有
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models # TODO 蒸馏Vit模型
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(融合网络, self).__init__()
        self.num_features = self.embed_dim = 嵌入向量维度  # num_features for consistency with other models
        self.num_tokens = 2 if 蒸馏 else 1
        标准化 = 标准化 or partial(nn.LayerNorm, eps=1e-6)
        self.图像块嵌入向量 = 图像嵌入层(图像形状=304, 图像块嵌入向量的维度=嵌入向量维度)
        图像块数量 = self.图像块嵌入向量.图像块数量

        # 第一个维度的1对应的是批量
        self.分类嵌入向量 = nn.Parameter(torch.zeros(1, 1, 嵌入向量维度))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, 嵌入向量维度)) if 蒸馏 else None
        # 位置嵌入是随机生成的？
        # TODO
        # self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量 + self.num_tokens, 嵌入向量维度))
        self.位置嵌入向量 = nn.Parameter(torch.zeros(1, 图像块数量 + self.num_tokens, 嵌入向量维度))
        self.位置嵌入丢弃 = nn.Dropout(p=全连接丢弃率) # 加上位置嵌入向量之后的drop层

        # 构建了一个等差数列，保存后续transformer编码块中的drop率。最后一个transformer编码块需要令输出形状为1*256，方便调整为16*16
        丢弃率 = [x.item() for x in torch.linspace(0, drop_path_ratio, 深度)]  # stochastic depth decay rule
        # transformer编码块重复指定的次数
        self.编码块堆叠 = nn.Sequential(*[
            编码块(特征维度=嵌入向量维度, 注意力头数量=注意力头数量, 多层感知机扩增率=多层感知机扩增率, qkv_偏差=qkv_偏差, qk_缩小因子=qk_缩小因子,
                自注意力丢弃率=自注意力丢弃率, 全连接层丢弃率=全连接丢弃率, drop_path_ratio=丢弃率[i], 标准化=标准化)
            for i in range(深度)
        ])
        self.标准化 = 标准化(嵌入向量维度) # 在所有的transformer编码块之后的LN层

        # Representation layer
        # """
        if representation_size and not 蒸馏:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("全连接层", nn.Linear(嵌入向量维度, representation_size)),
                ("激活函数", nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # """

        # Classifier head(s)
        # """
        self.head = nn.Linear(self.num_features, 分类数目) if 分类数目 > 0 else nn.Identity()
        self.head_dist = None
        if 蒸馏:
            self.head_dist = nn.Linear(嵌入向量维度, self.分类数目) if 分类数目 > 0 else nn.Identity()
        # """

        # 权重初始化 Weight init
        nn.init.trunc_normal_(self.位置嵌入向量, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.分类嵌入向量, std=0.02)
        self.apply(权重初始化)


    # def forward_features(self, x):
    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # 得到图像块对应的嵌入向量
        x = self.图像块嵌入向量(x)  # [B, 196, 768] [B, 256, 512]
        # [1, 1, 768] -> [B, 1, 768]
        分类嵌入向量 = self.分类嵌入向量.expand(x.shape[0], -1, -1)
        # print(分类嵌入向量.shape)
        if self.dist_token is None:
            # 默认情况。拼接分类嵌入向量和x
            x = torch.cat((分类嵌入向量, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((分类嵌入向量, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 这里报错说明实际输入的嵌入向量维度和参数传入的嵌入向量维度大小不一致。
        x = self.位置嵌入丢弃(x + self.位置嵌入向量) # 加上位置嵌入信息，准备输入transformer编码器
        x = self.编码块堆叠(x)
        x = self.标准化(x)
        return x
        # 下面似乎是和具体的分类任务相关的
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

    # 下面计算中出错
    """
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    """



class 自监督重建OCTA图像(nn.Module):
    """
    现在考虑传进来的数据是已经划分块之后的结果
    """
    def __init__(self):
        super(自监督重建OCTA图像, self).__init__()





def 权重初始化(模块):
    """
    ViT 权重初始化
    :param 模块: module
    """
    if isinstance(模块, nn.Linear):
        nn.init.trunc_normal_(模块.weight, std=.01)
        if 模块.bias is not None:
            nn.init.zeros_(模块.bias)
    elif isinstance(模块, nn.Conv2d):
        nn.init.kaiming_normal_(模块.weight, mode="fan_out")
        if 模块.bias is not None:
            nn.init.zeros_(模块.bias)
    elif isinstance(模块, nn.LayerNorm):
        nn.init.zeros_(模块.bias)
        nn.init.ones_(模块.weight)

# '''
if __name__ == '__main__':
    # 测试模型 = 图像块嵌入(304, 512)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(2, 256, 16, 16), batch_size=3, device='cuda') # [3, 256, 512]

    # 测试模型 = 自注意力(嵌入向量维度=512, 注意力头数量=8, 自注意力丢弃率=0.2, 全连接层丢弃率=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 512]

    # 测试模型 = 多层感知机(512, 200, 2048, 丢弃率=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 200]

    # 测试模型 = 编码块(特征维度=512, 注意力头数量=8, 自注意力丢弃率=0.2, 全连接层丢弃率=0.2, drop_path_ratio=0.2)
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(256,512), batch_size=3, device='cuda') # [3, 256, 512]

    # 测试模型 = 融合网络()
    # 测试模型.to(torch.device('cuda:0'))
    # print('\n')
    # summary(测试模型, input_size=(2, 256, 16, 16), batch_size=3, device='cuda')

    测试模型 = 融合网络无cls() # 输出[3, 256, 512]
    测试模型.to(torch.device('cuda:0'))
    print('\n')
    summary(测试模型, input_size=(2, 256, 16, 16), batch_size=1, device='cuda')

    # print(测试模型)

    # with SummaryWriter(comment='测试模型') as 可视化:
    #     可视化.add_graph(测试模型, torch.randn(1, 2, 256, 16, 16))
# '''
