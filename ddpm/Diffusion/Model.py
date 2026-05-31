
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from Diffusion.Train import CosineLinear_PEDCC


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        # T=1000, d_model=64, dim=256
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        # tensor([0.0000, 0.2878, 0.5756, 0.8635, 1.1513, 1.4391, 1.7269, 2.0148, 2.3026,
        #         2.5904, 2.8782, 3.1661, 3.4539, 3.7417, 4.0295, 4.3173, 4.6052, 4.8930,
        #         5.1808, 5.4686, 5.7565, 6.0443, 6.3321, 6.6199, 6.9078, 7.1956, 7.4834,
        #         7.7712, 8.0590, 8.3469, 8.6347, 8.9225])
        emb = torch.exp(-emb)   # e^次方  e^0=1, e^(-8.99925) = 1.3335x 10^-4
        # tensor([1.0000e+00, 7.4989e-01, 5.6234e-01, 4.2170e-01, 3.1623e-01, 2.3714e-01,
        #         1.7783e-01, 1.3335e-01, 1.0000e-01, 7.4989e-02, 5.6234e-02, 4.2170e-02,
        #         3.1623e-02, 2.3714e-02, 1.7783e-02, 1.3335e-02, 1.0000e-02, 7.4989e-03,
        #         5.6234e-03, 4.2170e-03, 3.1623e-03, 2.3714e-03, 1.7783e-03, 1.3335e-03,
        #         1.0000e-03, 7.4989e-04, 5.6234e-04, 4.2170e-04, 3.1623e-04, 2.3714e-04,
        #         1.7783e-04, 1.3335e-04])
        pos = torch.arange(T).float()
        # pos[:, None]列向量 emb[None, :]行向量
        emb = pos[:, None] * emb[None, :]             # torch.Size([1000, 32])
        assert list(emb.shape) == [T, d_model // 2]
        # tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
        #           0.0000e+00,  0.0000e+00],
        #         [ 8.4147e-01,  6.8156e-01,  5.3317e-01,  ...,  2.3714e-04,
        #           1.7783e-04,  1.3335e-04],
        #         [ 9.0930e-01,  9.9748e-01,  9.0213e-01,  ...,  4.7427e-04,
        #           3.5566e-04,  2.6670e-04],
        #         ...,
        #         [-8.9797e-01, -5.4493e-02,  9.9281e-01,  ...,  2.3423e-01,
        #           1.7637e-01,  1.3256e-01],
        #         [-8.5547e-01,  6.4066e-01,  9.0373e-01,  ...,  2.3446e-01,
        #           1.7654e-01,  1.3269e-01],
        #         [-2.6461e-02,  9.9213e-01,  5.3634e-01,  ...,  2.3469e-01,
        #           1.7672e-01,  1.3283e-01]])
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # 拼接成torch.Size([1000, 32, 2])
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)    # torch.Size([1000, 64]

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),   # Embedding(1000, 64)
            nn.Linear(d_model, dim),            # [64,256]
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t) # [8 256]
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        # T=1000, ch=64, ch_mult=[1 2 3 4], num_res_blocks=2, drop_out =0.15
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4    # tdim = 256
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):  # ch_mult=[1.2.3.4]
            out_ch = ch * mult              # out_ch依次是64 128 256 512
            for _ in range(num_res_blocks):   # num_res_blocks = 2
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,  # out_ch  tdim = 256
                    dropout=dropout, attn=(i in attn)))     # attn=2
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.linear_1 = nn.Linear(now_ch * 16, now_ch * 4)            # [b 256 4 4] => [b 4096] => [b 1024]
        # self.linear_2 = nn.Linear(now_ch * 8, now_ch * 4)
        # self.linear_2_reverse = nn.Linear(now_ch * 4, now_ch * 8)
        self.linear_1_reverse = nn.Linear(now_ch * 4, now_ch * 16)    # PEDCC feature dim: 1024
        self.out = CosineLinear_PEDCC(1024, 10)   # in_features:512, out_features:10
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def l2_norm(self, input):                          # According to amsoftmax, we have to normalize the feature, which is x here
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x, t, PEDCC_feature_vector=None):
        # Timestep embedding
        # temb、cemb同维度[b 512]; labels,t同维度[b]
        temb = self.time_embedding(t)           # temb [b 512]
        # cemb = self.cond_embedding(labels)     # cemb [b 512]
        # Downsampling
        h = self.head(x)            # h [b 128 32 32]
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # print(f'h.shape: {h.shape}')
        [b, c, H, W] = [h.size(0), h.size(1), h.size(2), h.size(3)]  # [b 256 4 4]
        # b = h.size(0)
        # print(b, c, H, W)]
        h1 = self.linear_1(h.view(b, -1))

        #####################################PEDCC Layer
        feature_norm = self.l2_norm(h1)
        cosine_out = self.out(feature_norm)  # out = self.out2(x)
        #####################################################

        # h: [b 4096] ==> PEDCC feature h1 [b 1024]
        assert h1.shape == (b, 1024)
        # h2 = self.linear_2(h1)                               # h2: [b 1024]

        if PEDCC_feature_vector is not None:
            h1 = torch.from_numpy(PEDCC_feature_vector).float().to(x.device)
        # h3 = self.linear_2_reverse(h2)
        h4 = self.linear_1_reverse(h1)
        # h = h4.view(b, c, H, W)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return [h, feature_norm, cosine_out]             # h2: PEDCC feature vector [b 1024]


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=64, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    print(t.shape)
    y = model(x, t)
    print(y.shape)

