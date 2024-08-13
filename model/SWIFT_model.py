import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FSA_GAT import GAT
from model.PatchEncoder import TSPatchLayer


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, dropout):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Conv1d(in_dim1, k_dim * num_heads, kernel_size=1, bias=False)
        self.proj_k2 = nn.Conv1d(in_dim2, k_dim * num_heads, kernel_size=1, bias=False)
        self.proj_v2 = nn.Conv1d(in_dim2, v_dim * num_heads, kernel_size=1, bias=False)
        self.proj_o = nn.Conv1d(v_dim * num_heads, in_dim1, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, weight_matrix=None, mask=None):
        batch_size, seq_len1, _ = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1.permute(0, 2, 1)).view(batch_size, self.num_heads, self.k_dim, seq_len1).permute(0, 1, 3,
                                                                                                              2)
        k2 = self.proj_k2(x2.permute(0, 2, 1)).view(batch_size, self.num_heads, self.k_dim, seq_len2).permute(0, 1, 3,
                                                                                                              2)
        v2 = self.proj_v2(x2.permute(0, 2, 1)).view(batch_size, self.num_heads, self.v_dim, seq_len2).permute(0, 1, 3,
                                                                                                              2)

        attn = torch.matmul(q1, k2.transpose(-2, -1)) / self.k_dim ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_len1, 1]
            mask = mask.expand(batch_size, self.num_heads, seq_len1,
                               seq_len2)  # [batch_size, num_heads, seq_len1, seq_len2]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        if weight_matrix is not None:
            attn = attn * weight_matrix  # Apply weight matrix to attention scores

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # Apply dropout to attention weights

        output = torch.matmul(attn, v2).permute(0, 1, 3, 2).contiguous().view(batch_size, self.num_heads * self.v_dim,
                                                                              seq_len1)
        return self.proj_o(output).permute(0, 2, 1)


class DualCrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, out_dim, dropout=0.1):
        super(DualCrossAttention, self).__init__()
        self.cross_attention1 = CrossAttention(in_dim1, in_dim2, k_dim, v_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(in_dim1)

        self.cross_attention2 = CrossAttention(in_dim1, in_dim1, k_dim, v_dim, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(in_dim1, in_dim1 * 4),
            nn.GELU(),
            nn.Linear(in_dim1 * 4, out_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(in_dim1)
        self.ln3 = nn.LayerNorm(out_dim)

    def forward(self, cords_features, spatial_features, speed_features, mask=None):
        batch_size, nvars_patch_num, d_model = cords_features.size()
        patch_num = nvars_patch_num // 3
        patch_num_v = speed_features.size(1) // 3
        sl = spatial_features.size(1) // 2
        w = 1.25

        weight_matrix1 = torch.ones(batch_size, self.cross_attention1.num_heads, nvars_patch_num, sl + sl).to(
            cords_features.device)
        weight_matrix1[:, :, :2 * patch_num, :sl] *= w
        weight_matrix1[:, :, 2 * patch_num:, sl:] *= w

        # First cross attention block with LayerNorm and Residual Connection
        residual1 = cords_features
        x = self.cross_attention1(cords_features, spatial_features, weight_matrix1, None)
        x = self.ln1(x + residual1)

        weight_matrix2 = torch.ones(batch_size, self.cross_attention2.num_heads, nvars_patch_num,
                                    3 * patch_num_v).to(cords_features.device)
        weight_matrix2[:, :, :patch_num, :patch_num_v] *= w
        weight_matrix2[:, :, patch_num:2 * patch_num,
        patch_num_v:2 * patch_num_v] *= w
        weight_matrix2[:, :, 2 * patch_num:, 2 * patch_num_v:] *= w

        # Second cross attention block with LayerNorm and Residual Connection
        residual2 = x
        x = self.cross_attention2(x, speed_features, weight_matrix2, None)
        x = self.ln2(x + residual2)

        residual3 = x
        x = self.ffn(x)
        x = self.ln3(x + residual3)

        return x


class StackedDualCrossAttention(nn.Module):
    def __init__(self, num_layers, in_dim1, in_dim2, k_dim, v_dim, num_heads, out_dim, dropout=0.1):
        super(StackedDualCrossAttention, self).__init__()
        self.layers = nn.ModuleList(
            [DualCrossAttention(in_dim1, in_dim2, k_dim, v_dim, num_heads, out_dim, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, cords_features, spatial_features, speed_features, mask=None):
        x = cords_features
        for layer in self.layers:
            x = layer(x, spatial_features, speed_features, mask)

        return x


class SWIFTModel(nn.Module):
    def __init__(self, in_feature1, in_feature2, num_heads, d_model, in_dim1, in_dim2, out_dim, patch_len, stride,
                 seq_len, pred_len, num_layers):
        super(SWIFTModel, self).__init__()
        self.staticInfoLayer1 = GAT(d_model=d_model,
                                    in_feature=in_feature1,
                                    num_heads_per_layer=[8, 1],
                                    num_features_per_layer=[16, 256],
                                    add_skip_connection=True,
                                    bias=True,
                                    dropout=0.1,
                                    load_flight_scene=False,
                                    avg_last=True)

        self.staticInfoLayer2 = GAT(d_model=d_model,
                                    in_feature=in_feature2,
                                    num_heads_per_layer=[8, 1],
                                    num_features_per_layer=[16, 256],
                                    add_skip_connection=True,
                                    bias=True,
                                    dropout=0.1,
                                    load_flight_scene=False,
                                    avg_last=True)

        self.dynamicInfoLayer = TSPatchLayer(seq_len=seq_len,
                                             pred_len=pred_len,
                                             d_model=d_model,
                                             n_heads=num_heads,
                                             patch_len=patch_len,
                                             stride=stride,
                                             num_layers=num_layers,
                                             dropout=0.1)

        self.stacked_dualct = StackedDualCrossAttention(num_layers=1,
                                                        in_dim1=in_dim1,
                                                        in_dim2=in_dim2,
                                                        k_dim=in_dim2,
                                                        v_dim=in_dim2,
                                                        num_heads=num_heads,
                                                        out_dim=out_dim,
                                                        dropout=0.1)

        self.head = FlattenHead(in_feature1 + in_feature2, out_dim * ((seq_len - pred_len - stride) // stride + 1),
                                pred_len, head_dropout=0.1)

    def forward(self, src_cords, src_sp_lon_lat, src_sp_alt, src_v, masks, lon_lat_node_features,
                lon_lat_edge_index_input,
                lon_lat_edge_prob_input, alt_node_features,
                alt_edge_index_input, alt_edge_prob_input):
        src_lon_lat = self.staticInfoLayer1(lon_lat_node_features, lon_lat_edge_index_input, lon_lat_edge_prob_input,
                                            src_sp_lon_lat)
        src_alt = self.staticInfoLayer2(alt_node_features, alt_edge_index_input, alt_edge_prob_input, src_sp_alt)

        src_gat = torch.cat((src_lon_lat, src_alt), dim=1)

        cords = self.dynamicInfoLayer(src_cords)

        velocity = self.dynamicInfoLayer(src_v)

        output = self.stacked_dualct(cords, src_gat, velocity, masks)

        output = torch.reshape(output, (output.shape[0], 3, -1, output.shape[2]))  # [bs , nvars , patch_num , d_model]

        output = self.head(output.permute(0, 1, 3, 2))
        result = output.permute(0, 2, 1)

        return result
