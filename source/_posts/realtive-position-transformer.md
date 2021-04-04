---
title: transformer相对编码
date: 2021-04-04 23:16:54
category: 深度学习
tags: [Attention,Transformer]
---
[论文地址](https://arxiv.org/pdf/1803.02155.pdf)
[参考博客1](https://wyydsb.xin/other/relativepositionembed.html)
[参考博客1](https://blog.csdn.net/weixin_41089007/article/details/91477253)

#原理
该论文的考虑出发点为原始的编码方式仅仅考虑了位置的**距离**关系，没有考虑位置的**先后**关系，本论文增加了这种关系。
公示很简单：
原始Attention：
$$
e_ij=\frac{x_iW_q x_iW_k^T}{\sqrt{d_{model}}}  \\
a_ij = softmax(w_ij) \\
z_i = \sum_{j=1}^n a_{ij}x_jW_v
$$
Relative Position Attention:
$$
e_ij=\frac{x_iW_q (x_iW_k+a_{ij}^k)^T}{\sqrt{d_{model}}}  \\
a_ij = softmax(w_{ij}) \\
z_i = \sum_{j=1}^n a_{ij}(x_jW_v+a_{ij}^v)
$$
>其中，$a\_{ij}^k$和$a\_{ij}^v$分别表示两个可学习的位置信息,至于为什么加在这两个地方，自然是因为这两个地方计算了相对位置。
同时，作者发现如果两个单词距离超过某个阈值$k$提升不大，因此在此限制了位置最大距离，即超过$k$的距离也按照$k$距离的位置信息计算。
位置信息本质是在训练两个矩阵$W^K=(w\_{-k}^K,...,w\_k^K)$和$W^V=(w\_{-k}^V,...,w\_k^V)$
$$
a_{ij}^K=W_{clip(j-i,k)}^K \\
a_{ij}^V=W_{clip(j-i,k)}^V \\
clip(x,k)=max(-k,min(k,x))
$$

#代码
```
import torch
import torch.nn as nn

torch.manual_seed(2021)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))  # q对k元素的attention

        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.einsum('bhqe,qke->bhqk', r_q1, r_k2)  # q对k位置的attention
        attn = (attn1 + attn2) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)  # qk对v元素的attention
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = torch.einsum('bhav,ave->bhae', attn, r_v2)  # qk对v位置的attention
        x = weight1 + weight2
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x


if __name__ == '__main__':
    multiHeadAttentionLayer = MultiHeadAttentionLayer(128, 8, 0.5, 'cpu')
    x = torch.randn(4, 43, 128)
    result = multiHeadAttentionLayer(x, x, x)
    print(result)
    # x = torch.randn(64, 8, 43, 16)
    # y = torch.randn(43, 43, 16)
    # print(torch.einsum('bhqe,qke->bhqk', [x, y]))

```