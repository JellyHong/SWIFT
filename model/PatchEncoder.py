import torch
from torch import nn

from model.layers.SelfAttention import FullAttention, AttentionLayer
from model.layers.Embed import PatchEmbedding


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        # B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.selfatt = AttentionLayer(
            FullAttention(False, 3, attention_dropout=dropout, output_attention=True), d_model, n_heads
        )
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            DWConv(d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, enc_out):
        residual = enc_out
        enc_out, _ = self.selfatt(enc_out, enc_out, enc_out, None)
        enc_out = self.norm_layer1(enc_out + residual)

        residual = enc_out
        enc_out = self.ffn(enc_out)
        enc_out = self.norm_layer2(enc_out + residual)

        return enc_out


class TSPatchLayer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, n_heads, patch_len=16, stride=8, num_layers=4, dropout=0.1):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = seq_len // stride
        self.d_model = d_model
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        for encoder_layer in self.encoder:
            enc_out = encoder_layer(enc_out)

        bsz = enc_out.size(0) // 3
        enc_out = torch.reshape(enc_out, (bsz, 3, enc_out.shape[1], enc_out.shape[2]))

        output = enc_out.reshape(bsz, -1, enc_out.shape[3])

        return output
