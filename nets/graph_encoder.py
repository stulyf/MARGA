import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None,
            use_rope=True,
            rope_theta=1e4
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # RoPE options
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def _rotary_embed_2d(self, q, k, pos_xy):
        """
        q, k: [H, B, N, D]  (heads头数, batch批大小, length点序列长度, head_dim隐藏层维度)
        pos_xy: [B, N, 2]   coordinates in [0,1]
        returns rotated q, k with same shape as input
        """
        assert q.dim() == 4 and k.dim() == 4
        H, B, Nq, D = q.shape
        _, _, Nk, Dk = k.shape
        assert D % 4 == 0, "head_dim must be divisible by 4 for 2D-RoPE"
        assert Nq == Nk, "Self-attention expected: query/key lengths must match when applying RoPE"
        assert pos_xy.size(1) == Nq, "pos_xy length must match sequence length"

        # Move to [B, H, N, D]
        q_bhnd = q.permute(1, 0, 2, 3).contiguous()
        k_bhnd = k.permute(1, 0, 2, 3).contiguous()

        B2, H2, N, D2 = q_bhnd.shape
        P = D2 // 2  # number of 2-dim pairs
        assert P % 2 == 0, "D/2 must be even so that x/y pairs split equally"
        Px = P // 2
        Py = P - Px

        # reshape last dim into pairs
        def to_pairs(t):
            return t.contiguous().view(B2, H2, N, P, 2)

        q_pairs = to_pairs(q_bhnd)
        k_pairs = to_pairs(k_bhnd)

        # frequencies for x and y halves
        kx = torch.arange(Px, device=q_bhnd.device, dtype=q_bhnd.dtype)
        ky = torch.arange(Py, device=q_bhnd.device, dtype=q_bhnd.dtype)
        omega_x = self.rope_theta ** (-2 * kx / (2 * Px))
        omega_y = self.rope_theta ** (-2 * ky / (2 * Py))

        x = pos_xy[..., 0].unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]
        y = pos_xy[..., 1].unsqueeze(-1).unsqueeze(-1)

        phase_x = (omega_x.view(1, 1, 1, -1, 1)) * x.unsqueeze(1)
        phase_y = (omega_y.view(1, 1, 1, -1, 1)) * y.unsqueeze(1)

        def rotate(t, phase):
            cos, sin = phase.cos(), phase.sin()
            # t: [B,H,N,Px_or_Py,2]
            u, v = t[..., 0], t[..., 1]                 # [B,H,N,P]
            u = u.unsqueeze(-1)                          # [B,H,N,P,1]
            v = v.unsqueeze(-1)
            u2 = u * cos - v * sin                       # [B,H,N,P,1]
            v2 = u * sin + v * cos
            u2 = u2.squeeze(-1)
            v2 = v2.squeeze(-1)
            return torch.stack([u2, v2], dim=-1)

        # split into x/y halves on the pairs axis
        qx, qy = q_pairs[..., :Px, :], q_pairs[..., Px:, :]
        kx_p, ky_p = k_pairs[..., :Px, :], k_pairs[..., Px:, :]

        qx = rotate(qx, phase_x)
        qy = rotate(qy, phase_y)
        kx_p = rotate(kx_p, phase_x)
        ky_p = rotate(ky_p, phase_y)

        # merge back pairs
        def from_pairs(ax, ay):
            t = torch.cat([ax, ay], dim=3)
            return t.contiguous().view(B2, H2, N, D2)

        q_rot = from_pairs(qx, qy)
        k_rot = from_pairs(kx_p, ky_p)

        # Back to [H, B, N, D]
        q_out = q_rot.permute(1, 0, 2, 3).contiguous()
        k_out = k_rot.permute(1, 0, 2, 3).contiguous()
        return q_out, k_out


    def forward(self, q, h=None, mask=None, pos_xy=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)


        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Apply RoPE to Q/K for self-attention if enabled
        if self.use_rope and pos_xy is not None:
            # Expect self-attention: n_query == graph_size, and pos for all nodes
            assert Q.size(2) == K.size(2), "RoPE expects self-attention (n_query == graph_size)"
            assert pos_xy.size(1) == Q.size(2), "pos_xy length must equal n_query"
            Q, K = self._rotary_embed_2d(Q, K, pos_xy)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        # 输出为注意力得分和V相乘
        heads = torch.matmul(attn, V)

        # 对不同头的结果进行拼接
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
            use_rope=False,
            rope_theta=1e4,
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.norm1 = Normalization(embed_dim, normalization)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, h, pos_xy=None, mask=None):
        h = h + self.mha(h, h, mask=mask, pos_xy=pos_xy)
        h = self.norm1(h)
        h = h + self.ff(h)
        h = self.norm2(h)
        return h


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            use_rope=False,
            rope_theta=1e4
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.embed_dim = embed_dim

        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(
                n_heads, embed_dim, feed_forward_hidden, normalization,
                use_rope=use_rope, rope_theta=rope_theta
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None, pos_xy=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        for layer in self.layers:
            h = layer(h, pos_xy=pos_xy, mask=mask)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
