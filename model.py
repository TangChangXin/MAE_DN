import torch
import torch.nn as nn
from torchsummary import summary

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
class 多头自己注意力(nn.Module):
    def __init__(self):
        super(多头自己注意力, self).__init__()


class 图像块嵌入向量(nn.Module):
    """
    如何将2维和3维整合在一起。3维输入的应该是
    现在认为输入的图像都是已经划分成图像块之后的结果。有监督训练的时候在另一个类里对有标签的图像划分。
    """
    def __init__(self, 图像形状, 图像块嵌入向量的维度, 标准化层=None):
        """
        加if分支处理3维和2维的区别
        输入通道数暂时没用到，有监督训练时如何对输入的图像划分
        3维输入数据形状BCDHW 1*2*256*16*16，在256中随机选取部分用来训练，输出1*256的向量，然后整形为1*16*16。
        """
        # todo 现在是3维为例
        super(图像块嵌入向量, self).__init__()
        self.图像形状 = 图像形状
        self.图像块的大小 = 16
        if 标准化层:
            self.标准化层 = 标准化层(图像块嵌入向量的维度)
        else: self.标准化层 = nn.Identity()
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() 原代码

    def forward(self, x):
        # 输入1*2*256*16*16，展平后1*2*256*256。形状[B, C, D, H, W] -> [B, C, D , HW]
        # 交换维度后，1*256*256*2，形状[B, D, HW, C]
        x = torch.flatten(x, start_dim=3).permute(0, 2, 3, 1)
        x = torch.flatten(x, 2) # 再展平后输出1*256*512，形状[B, D, HWC]
        x = self.标准化层(x)
        return x


class 多头注意力(nn.Module):
    def __init__(self,
                 嵌入向量的维度,
                 注意力头的数量,
                 注意力丢弃率=0.,
                 全连接层丢弃率=0.):
        super(多头注意力, self).__init__()
        self.注意力头的数量 = 注意力头的数量
        注意力头维度 = 嵌入向量的维度 // 注意力头的数量 # 每个注意力头的维度大小
        self.缩小 = 注意力头维度 ** -0.5
        # 通过一个全连接层生成qkv三个向量，有助于并行化计算
        self.qkv = nn.Linear(嵌入向量的维度, 嵌入向量的维度 * 3, bias=False)
        self.注意力丢弃 = nn.Dropout(注意力丢弃率)

        # 多头注意力的输出拼接后与Wo相乘得到最终的输出。Wo矩阵通过全连接层实现
        self.proj = nn.Linear(嵌入向量的维度, 嵌入向量的维度)
        self.proj_drop = nn.Dropout(全连接层丢弃率)

        def forward(self, x):
            # [batch_size, num_patches + 1, embed_dim]
            # 每一批图片数。图像块的数量加1是因为算上class token，我的方法按照纵向的深度计算           B, N, C = x.shape
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

            attn = (q @ k.transpose(-2, -1)) * self.scale  # q和k进行多维矩阵乘法时，实际只有最后两个维度相乘
            attn = attn.softmax(dim=-1)  # 似乎是二维矩阵，按行进行柔性最大值处理。
            attn = self.attn_drop(attn)

            # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
            # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 和v矩阵相乘 加权求和
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

if __name__ == '__main__':
    测试模型 = 图像块嵌入向量(1, 512, nn.LayerNorm)
    测试模型.to(torch.device('cuda:0'))
    summary(测试模型, input_size=(2, 256, 16, 16), batch_size=1, device='cuda')  #
