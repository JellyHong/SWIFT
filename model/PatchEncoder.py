import torch
from torch import nn

from model.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        # B, N, C = x.shape
        x = x.transpose(1, 2)  # 变形为 [B, C, N]
        x = self.dwconv(x)  # 进行一维卷积操作，输出形状为 [B, C, N]
        x = x.transpose(1, 2)  # 变形为 [B, N, C]
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
        # Apply self-attention
        enc_out, _ = self.selfatt(enc_out, enc_out, enc_out, None)
        enc_out = self.norm_layer1(enc_out + residual)  # Residual connection

        residual = enc_out
        # Apply feed-forward network
        enc_out = self.ffn(enc_out)
        enc_out = self.norm_layer2(enc_out + residual)  # Residual connection

        return enc_out


class TSPatchLayer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

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
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        for encoder_layer in self.encoder:
            enc_out = encoder_layer(enc_out)

        # Apply reshaping to original format
        bsz = enc_out.size(0) // 3
        enc_out = torch.reshape(enc_out,
                                (bsz, 3, enc_out.shape[1],
                                 enc_out.shape[2]))  # [bs , nvars , patch_num , d_model]

        # output = self.head(enc_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        output = enc_out.reshape(bsz, -1, enc_out.shape[3])

        return output