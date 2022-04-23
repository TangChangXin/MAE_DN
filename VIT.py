## from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) ## 224*224
        patch_height, patch_width = pair(patch_size)## 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img) ## img 1 3 224 224  输出形状x : 1 196 1024 
        b, n, _ = x.shape ## 

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



if __name__ == '__main__':
    测试模型 = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    测试模型.to(torch.device('cuda:0'))
    summary(测试模型, input_size=(3, 224, 224), batch_size=1, device='cuda')  #




'''
        Layer (type)               Output Shape         Param #
================================================================
         Rearrange-1              [1, 196, 768]               0
            Linear-2             [1, 196, 1024]         787,456
           Dropout-3             [1, 197, 1024]               0
         LayerNorm-4             [1, 197, 1024]           2,048
            Linear-5             [1, 197, 3072]       3,145,728
           Softmax-6          [1, 16, 197, 197]               0
            Linear-7             [1, 197, 1024]       1,049,600
           Dropout-8             [1, 197, 1024]               0
         Attention-9             [1, 197, 1024]               0
          PreNorm-10             [1, 197, 1024]               0
        LayerNorm-11             [1, 197, 1024]           2,048
           Linear-12             [1, 197, 2048]       2,099,200
             GELU-13             [1, 197, 2048]               0
          Dropout-14             [1, 197, 2048]               0
           Linear-15             [1, 197, 1024]       2,098,176
          Dropout-16             [1, 197, 1024]               0
      FeedForward-17             [1, 197, 1024]               0
          PreNorm-18             [1, 197, 1024]               0
        LayerNorm-19             [1, 197, 1024]           2,048
           Linear-20             [1, 197, 3072]       3,145,728
          Softmax-21          [1, 16, 197, 197]               0
           Linear-22             [1, 197, 1024]       1,049,600
          Dropout-23             [1, 197, 1024]               0
        Attention-24             [1, 197, 1024]               0
          PreNorm-25             [1, 197, 1024]               0
        LayerNorm-26             [1, 197, 1024]           2,048
           Linear-27             [1, 197, 2048]       2,099,200
             GELU-28             [1, 197, 2048]               0
          Dropout-29             [1, 197, 2048]               0
           Linear-30             [1, 197, 1024]       2,098,176
          Dropout-31             [1, 197, 1024]               0
      FeedForward-32             [1, 197, 1024]               0
          PreNorm-33             [1, 197, 1024]               0
        LayerNorm-34             [1, 197, 1024]           2,048
           Linear-35             [1, 197, 3072]       3,145,728
          Softmax-36          [1, 16, 197, 197]               0
           Linear-37             [1, 197, 1024]       1,049,600
          Dropout-38             [1, 197, 1024]               0
        Attention-39             [1, 197, 1024]               0
          PreNorm-40             [1, 197, 1024]               0
        LayerNorm-41             [1, 197, 1024]           2,048
           Linear-42             [1, 197, 2048]       2,099,200
             GELU-43             [1, 197, 2048]               0
          Dropout-44             [1, 197, 2048]               0
           Linear-45             [1, 197, 1024]       2,098,176
          Dropout-46             [1, 197, 1024]               0
      FeedForward-47             [1, 197, 1024]               0
          PreNorm-48             [1, 197, 1024]               0
        LayerNorm-49             [1, 197, 1024]           2,048
           Linear-50             [1, 197, 3072]       3,145,728
          Softmax-51          [1, 16, 197, 197]               0
           Linear-52             [1, 197, 1024]       1,049,600
          Dropout-53             [1, 197, 1024]               0
        Attention-54             [1, 197, 1024]               0
          PreNorm-55             [1, 197, 1024]               0
        LayerNorm-56             [1, 197, 1024]           2,048
           Linear-57             [1, 197, 2048]       2,099,200
             GELU-58             [1, 197, 2048]               0
          Dropout-59             [1, 197, 2048]               0
           Linear-60             [1, 197, 1024]       2,098,176
          Dropout-61             [1, 197, 1024]               0
      FeedForward-62             [1, 197, 1024]               0
          PreNorm-63             [1, 197, 1024]               0
        LayerNorm-64             [1, 197, 1024]           2,048
           Linear-65             [1, 197, 3072]       3,145,728
          Softmax-66          [1, 16, 197, 197]               0
           Linear-67             [1, 197, 1024]       1,049,600
          Dropout-68             [1, 197, 1024]               0
        Attention-69             [1, 197, 1024]               0
          PreNorm-70             [1, 197, 1024]               0
        LayerNorm-71             [1, 197, 1024]           2,048
           Linear-72             [1, 197, 2048]       2,099,200
             GELU-73             [1, 197, 2048]               0
          Dropout-74             [1, 197, 2048]               0
           Linear-75             [1, 197, 1024]       2,098,176
          Dropout-76             [1, 197, 1024]               0
      FeedForward-77             [1, 197, 1024]               0
          PreNorm-78             [1, 197, 1024]               0
        LayerNorm-79             [1, 197, 1024]           2,048
           Linear-80             [1, 197, 3072]       3,145,728
          Softmax-81          [1, 16, 197, 197]               0
           Linear-82             [1, 197, 1024]       1,049,600
          Dropout-83             [1, 197, 1024]               0
        Attention-84             [1, 197, 1024]               0
          PreNorm-85             [1, 197, 1024]               0
        LayerNorm-86             [1, 197, 1024]           2,048
           Linear-87             [1, 197, 2048]       2,099,200
             GELU-88             [1, 197, 2048]               0
          Dropout-89             [1, 197, 2048]               0
           Linear-90             [1, 197, 1024]       2,098,176
          Dropout-91             [1, 197, 1024]               0
      FeedForward-92             [1, 197, 1024]               0
          PreNorm-93             [1, 197, 1024]               0
      Transformer-94             [1, 197, 1024]               0
         Identity-95                  [1, 1024]               0
        LayerNorm-96                  [1, 1024]           2,048
           Linear-97                  [1, 1000]       1,025,000
================================================================
Total params: 52,195,304
Trainable params: 52,195,304
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 209.66
Params size (MB): 199.11
Estimated Total Size (MB): 409.34
----------------------------------------------------------------

进程已结束,退出代码0

'''




'''

v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
'''