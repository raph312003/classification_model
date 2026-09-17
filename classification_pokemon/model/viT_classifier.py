import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels = 1, 
        patch_size = 8, 
        embed_dim = 128
    ):
        super().__init__()

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.projection(x)
        return x

# Test Patch Embedding
# x_dummy = torch.randn(2, 1, 128, 128)
# print("Initial shape", x_dummy.shape)
# embedding = PatchEmbedding()
# out = embedding(x_dummy)
# print("Patches shape", out.shape)

class Attention(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout
    ):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True
        )

        self.q = torch.nn.Linear(embed_dim, embed_dim)
        self.k = torch.nn.Linear(embed_dim, embed_dim)
        self.v = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_output, attn_output_weights = self.att(q, k, v)

        return attn_output
        
class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Test 
# norm = PreNorm(128,Attention(embed_dim=128, num_heads=4, dropout=0.2))
# print(norm(out).shape)

class FeedForward(nn.Sequential):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout
    ):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
# ff = FeedForward(dim=128, hidden_dim=256, dropout = 0.2)
# print(ff(out).shape)

class ResBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# res_att = ResBlock(Attention(embed_dim=128, num_heads=4, dropout=0.2))
# print(res_att(out).shape)

class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_heads: int = 4,
        img_size: int = 128,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        patch_size: int = 8,
        n_layers: int = 4,
        num_class: int = 2,
        dropout: float = 0.2
    
    ):
        super().__init__()
        self.n_layers = n_layers

        self.embedding = PatchEmbedding(
            in_channels, 
            patch_size, 
            embed_dim
        )

        num_patchs = (img_size // patch_size)**2

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patchs+1, embed_dim)
        )
        self.cls_token = nn.Parameter(
            torch.randn(1,1,embed_dim)
        )

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResBlock(PreNorm(embed_dim, Attention(embed_dim, num_heads, dropout))),
                ResBlock(PreNorm(embed_dim, FeedForward(embed_dim, hidden_dim, dropout)))
            )
            self.layers.append(transformer_block)

        # output
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim, num_class))

    def forward(self, x):

        x = self.embedding(x)
        b, n, _ = x.shape

        # Add cls token to input
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        # go from [1, 1, 128] -> [2, 1, 128] because 2 batch_size

        x = torch.cat([cls_tokens, x], dim=1)
        # [2, 1, 128] + [2, 256, 128] -> [2, 257, 128]

        x += self.pos_embedding[:,:(n+1)]
        # [2, 257, 128] -> Image1 [1, 257, 128] + [1, 257, 128] + Image2 [1, 257, 128] + [1, 257, 128]

        for i in range(self.n_layers):
            x = self.layers[i](x)

        x = self.classifier(x[:,0,:])
        # x[:,0,:] -> CLS
        # [2, 257, 128] -> [2, 0, 128] -> classifier [2, 0, 128] -> [2, 0, 2]
        return x

# test viT
# model = viT()
# model(x_dummy)


        
        












        
        


