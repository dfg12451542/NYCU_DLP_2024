import torch.nn as nn
import torch
import math
import torch.nn.functional as F

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.activation = F.relu

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_o = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=attn_drop)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # q = self.activation(q)
        # k = self.activation(k)
        # v = self.activation(v)

        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        # out_dim = in_feature * self.num_heads

        q = q.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3)#.reshape(batch_size * self.num_heads, seq_len, sub_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3)#.reshape(batch_size * self.num_heads, seq_len, sub_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3)#.reshape(batch_size * self.num_heads, seq_len, sub_dim)

        y = self.scaled_dot_product(q, k, v)
        #y = y.reshape(batch_size, self.num_heads, seq_len, in_feature)
        y = y.permute(0, 2, 1, 3).reshape(batch_size, seq_len, in_feature)
        y = self.linear_o(y)
        # y = self.activation(y)

        return y
    
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = self.dropout(F.softmax(attn_logits, dim=-1))
        values = torch.matmul(attention, v)

        return values
#         #raise Exception('TODO1!')


# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim=768, heads=8):
#         super(MultiHeadAttention, self).__init__()
#         self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
#         self.projector = nn.Linear(dim, dim)

#     def forward(self, x):
#         for i, sa_head in enumerate(self.self_attention_heads):
#             if i == 0:
#                 out = sa_head(x)
#             else:
#                 out = torch.cat((out, sa_head(x)), axis=-1)
#         out = self.projector(out)
#         return out
    
class Attention(nn.Module):
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn
    
# class ScaledDotProductAttention(nn.Module):
#     def forward(self, query, key, value):
#         dk = query.size()[-1]
#         scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
#         attention = F.softmax(scores, dim=-1)
#         return attention.matmul(value)

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    