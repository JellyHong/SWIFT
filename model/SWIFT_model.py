import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.FSA_GAT import GAT
from model.PatchEncoder import TSPatchLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough P matrix with sinusoidal values
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Register as buffer to avoid being updated during training

    def forward(self, x):
        """
        Add sinusoidal positional encoding to input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.pe.size(1)}")
        pe = self.pe[:, :seq_len, :]  # [1, seq_len, d_model]
        x = x + pe  # [batch_size, seq_len, d_model]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_rel_pos=50):
        """
        Enhanced relative positional encoding with depthwise convolution
        Args:
            d_model:      Input feature dimension
            max_rel_pos:  Maximum relative distance to consider (bidirectional)
        """
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.d_model = d_model

        # Relative position embeddings [2*max_rel_pos+1, d_model]
        self.rel_pos_emb = nn.Embedding(2*max_rel_pos+1, d_model)

        # Depthwise separable convolution parameters
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model  # Depthwise separation
        )
        self.activation = nn.GELU()

        # Initialization for stable training
        nn.init.xavier_uniform_(self.rel_pos_emb.weight)
        nn.init.kaiming_normal_(self.depthwise_conv.weight,
                              mode='fan_in',
                              nonlinearity='linear')

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Enhanced tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # 1. Generate relative position matrix
        # Create [seq_len, seq_len] matrix of relative positions
        range_vec = torch.arange(seq_len, device=x.device)
        rel_pos_mat = range_vec[None, :] - range_vec[:, None]  # [seq_len, seq_len]

        # Clamp values to [-max_rel_pos, max_rel_pos] range
        rel_pos_mat = torch.clamp(rel_pos_mat,
                                -self.max_rel_pos,
                                self.max_rel_pos)
        rel_pos_mat += self.max_rel_pos  # Shift to [0, 2*max_rel_pos]

        # 2. Embed relative positions
        # [seq_len, seq_len, d_model]
        rel_pos_emb = self.rel_pos_emb(rel_pos_mat)

        # 3. Feature fusion
        # Add positional embeddings to input features
        x_expanded = x.unsqueeze(1)  # [batch, 1, seq_len, d_model]
        pos_emb_expanded = rel_pos_emb.unsqueeze(0)  # [1, seq_len, seq_len, d_model]
        fused_features = x_expanded + pos_emb_expanded  # [batch, seq_len, seq_len, d_model]

        # 4. Depthwise convolution processing
        # Reshape for convolution: [batch*seq_len, d_model, seq_len]
        conv_input = fused_features.view(-1, seq_len, self.d_model)  # [batch*seq_len, seq_len, d_model]
        conv_input = conv_input.permute(0, 2, 1)  # [batch*seq_len, d_model, seq_len]

        # Apply depthwise convolution
        conv_output = self.depthwise_conv(conv_input)  # [batch*seq_len, d_model, seq_len]
        conv_output = conv_output.permute(0, 2, 1)  # [batch*seq_len, seq_len, d_model]

        # 5. Feature recombination
        # Reshape back to original dimensions
        conv_output = conv_output.view(batch_size, seq_len, seq_len, self.d_model)

        # Create diagonal mask for position-specific features
        diag_mask = torch.eye(seq_len, dtype=torch.bool, device=x.device)
        diag_mask = diag_mask.unsqueeze(-1).expand(-1, -1, self.d_model)

        # Combine convolved features with original features
        output = torch.where(diag_mask,
                           conv_output,     # Use convolved features for diagonal
                           fused_features   # Use original features elsewhere
                           ).mean(dim=1)    # Average over sequence dimension

        return output



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass for FlattenHead.

        Args:
            x: Tensor of shape [batch_size, n_vars, d_model, patch_num]

        Returns:
            Tensor of shape [batch_size, target_window, ...]
        """
        x = self.flatten(x)  # Flatten the last two dimensions
        x = self.linear(x)  # Apply linear transformation
        x = self.dropout(x)  # Apply dropout
        return x


class StaticDynamicCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(StaticDynamicCrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Convolution layers for Query, Key, and Value projections
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

        # Learnable scaling factor and balancing coefficient
        self.gamma = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.lambda_sync = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x_dy_pe, u_st_pe):
        """
        Forward pass for Stage 1: Static-Dynamic Cross-Attention.

        Args:
            x_dy_pe: Dynamic temporal features with positional encoding [batch, M*N, d_model]
            u_st_pe: Static spatial features with positional encoding [batch, 2L, d_model]

        Returns:
            Tensor of shape [batch, M*N, d_model]
        """
        batch_size, M_N, d = x_dy_pe.size()
        _, L, _ = u_st_pe.size()

        # Compute spatio-temporal similarity matrix W_sync
        # Shape: [batch, 2L, M*N]
        pe_st = u_st_pe.unsqueeze(2)  # [batch, 2L, 1, d_model]
        pe_dy = x_dy_pe.unsqueeze(1)  # [batch, 1, M*N, d_model]
        diff = pe_st - pe_dy  # [batch, 2L, M*N, d_model]
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [batch, 2L, M*N]
        W_sync = torch.exp(-dist_sq / self.gamma)  # [batch, 2L, M*N]

        # Project inputs to Query, Key, and Value
        Q = self.conv_q(x_dy_pe.permute(0, 2, 1))  # [batch, d_model, M*N]
        K = self.conv_k(u_st_pe.permute(0, 2, 1))  # [batch, d_model, 2L]
        V = self.conv_v(u_st_pe.permute(0, 2, 1))  # [batch, d_model, 2L]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.n_heads, self.d_k, M_N).permute(0, 1, 3, 2)  # [batch, heads, M*N, d_k]
        K = K.view(batch_size, self.n_heads, self.d_k, L).permute(0, 1, 3, 2)  # [batch, heads, 2L, d_k]
        V = V.view(batch_size, self.n_heads, self.d_v, L).permute(0, 1, 3, 2)  # [batch, heads, 2L, d_v]

        # Scaled Dot-Product Attention with W_sync
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch, heads, M*N, 2L]
        W_sync = W_sync.unsqueeze(1).permute(0, 1, 3, 2)  # [batch, 1, M*N, 2L]
        scores = scores + self.lambda_sync * W_sync  # Incorporate W_sync
        attn = F.softmax(scores, dim=-1)  # [batch, heads, M*N, 2L]
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [batch, heads, M*N, d_v]
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.n_heads * self.d_v,
                                                        M_N)  # [batch, d_model, M*N]
        out = self.fc_out(out.permute(0, 2, 1))  # [batch, M*N, d_model]

        return out


class SpatialVelocityCrossAttention(nn.Module):
    def __init__(self, spatial_dim, velocity_dim, d_model, n_heads, dropout=0.1):
        super(SpatialVelocityCrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Gating mechanism
        self.conv_gamma = nn.Conv1d(velocity_dim, d_model, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv_gamma.weight)

        # Multi-scale convolutions with different kernel sizes
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(velocity_dim, velocity_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])
        for conv in self.multi_scale_convs:
            nn.init.xavier_uniform_(conv.weight)

        # self.rel_pos_enc = PositionalEncoding(velocity_dim * 3)

        # Relative positional encoding
        self.rel_pos_enc = RelativePositionalEncoding(d_model=velocity_dim * 3, max_rel_pos=30)

        # Convolution layers for Query, Key, and Value projections
        self.conv_q = nn.Conv1d(spatial_dim, d_model, kernel_size=1, bias=False)
        self.conv_k = nn.Conv1d(velocity_dim * 3, d_model, kernel_size=1, bias=False)
        self.conv_v = nn.Conv1d(velocity_dim * 3, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

        # Layer Normalization
        self.ln = nn.LayerNorm(d_model)

        # Activation function
        self.sigmoid = nn.Sigmoid()

        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x_spatial, x_velocity):
        """
        Forward pass for Stage 2: Spatial-Velocity Cross-Attention.

        Args:
            x_spatial: Spatial features [batch, 2L, spatial_dim]
            x_velocity: Velocity features [batch, M*N, velocity_dim]

        Returns:
            Tensor after Spatial-Velocity Cross-Attention [batch, 2L, d_model]
        """
        batch_size, M_N, velocity_dim = x_velocity.size()
        _, L, spatial_dim = x_spatial.size()

        # Gating mechanism to modulate velocity features
        Gamma = self.sigmoid(self.conv_gamma(x_velocity.permute(0, 2, 1)))  # [batch, d_model, M*N]
        Gamma = Gamma.permute(0, 2, 1)  # [batch, M*N, d_model]

        # Multi-scale convolution to capture velocity information at different temporal scales
        x_v_ms = [conv(x_velocity.permute(0, 2, 1)).permute(0, 2, 1) for conv in
                  self.multi_scale_convs]  # List of [batch, M*N, velocity_dim]
        x_v_ms = torch.cat(x_v_ms, dim=-1)  # [batch, M*N, velocity_dim * 3]

        # Add relative positional encoding
        x_v_pe = self.rel_pos_enc(x_v_ms)  # [batch, M*N, velocity_dim * 3]

        # Project spatial and velocity features to Query, Key, and Value
        Q = self.conv_q(x_spatial.permute(0, 2, 1))  # [batch, d_model, 2L]
        K = self.conv_k(x_v_pe.permute(0, 2, 1))  # [batch, d_model, M*N]
        V = self.conv_v(x_v_pe.permute(0, 2, 1))  # [batch, d_model, M*N]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.n_heads, self.d_k, L).permute(0, 1, 3, 2)  # [batch, heads, 2L, d_k]
        K = K.view(batch_size, self.n_heads, self.d_k, M_N).permute(0, 1, 3, 2)  # [batch, heads, M*N, d_k]
        V = V.view(batch_size, self.n_heads, self.d_v, M_N).permute(0, 1, 3, 2)  # [batch, heads, M*N, d_v]

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch, heads, 2L, M*N]
        attn = F.softmax(scores, dim=-1)  # [batch, heads, 2L, M*N]
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [batch, heads, 2L, d_v]
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.n_heads * self.d_v, L)  # [batch, d_model, 2L]
        out = self.fc_out(out.permute(0, 2, 1))  # [batch, 2L, d_model]

        # Repeat Gamma to match the 2L dimension
        Gamma = Gamma.repeat(1, 2, 1)  # [batch, 2L, d_model]

        # Apply gating to the attention output
        out = Gamma * out  # [batch, 2L, d_model]

        # Residual connection and layer normalization
        x_sv = self.ln(x_spatial + out)  # [batch, 2L, d_model]

        return x_sv


class SWIFTModel(nn.Module):
    def __init__(self, in_feature1, in_feature2, num_heads, d_model, out_dim, patch_len, stride, seq_len, pred_len,
                 num_layers):
        super(SWIFTModel, self).__init__()
        self.d_model = d_model

        # Stage 1 Positional Encodings
        self.pos_enc_stage1_static = PositionalEncoding(d_model)
        self.pos_enc_stage1_dynamic = PositionalEncoding(d_model)

        # Stage 2 Positional Encoding for multi-scale velocity features
        self.pos_enc_stage2_velocity = PositionalEncoding(d_model * 3)

        # Graph Attention Networks for processing static features
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
                                    load_flight_scene=True,
                                    avg_last=True)

        # Dynamic Temporal Features processing
        self.dynamicInfoLayer = TSPatchLayer(seq_len=seq_len,
                                             pred_len=pred_len,
                                             d_model=d_model,
                                             n_heads=num_heads,
                                             patch_len=patch_len,
                                             stride=stride,
                                             num_layers=num_layers,
                                             dropout=0.1)

        # Stage 1: Static-Dynamic Cross-Attention
        self.cross_att_stage1 = StaticDynamicCrossAttention(d_model=d_model, n_heads=num_heads, dropout=0.1)

        # Stage 2: Spatial-Velocity Cross-Attention
        self.cross_att_stage2 = SpatialVelocityCrossAttention(
            spatial_dim=d_model,
            velocity_dim=d_model,
            d_model=d_model,
            n_heads=num_heads,
            dropout=0.1
        )

        # Prediction Head
        self.head = FlattenHead(n_vars=3,
                                nf=9728,
                                target_window=pred_len,
                                head_dropout=0.1)

    def forward(self, src_cords, src_sp_lon_lat, src_sp_alt, src_v, masks, lon_lat_node_features,
                lon_lat_edge_index_input,
                lon_lat_edge_prob_input, alt_node_features,
                alt_edge_index_input, alt_edge_prob_input):
        """
        Forward pass for SWIFTModel.

        Args:
            src_cords: Coordinate sequences [batch, M*N, d_model]
            src_sp_lon_lat: Static longitude-latitude features [batch, 2L, d_model]
            src_sp_alt: Static altitude features [batch, 2L, d_model]
            src_v: Velocity features [batch, M*N, d_model]
            masks: Optional masks [batch, ...]
            lon_lat_node_features: Node features for longitude-latitude [batch, M*N, in_feature1]
            lon_lat_edge_index_input: Edge indices for longitude-latitude graphs [optional]
            lon_lat_edge_prob_input: Edge probabilities for longitude-latitude graphs [optional]
            alt_node_features: Node features for altitude [batch, M*N, in_feature2]
            alt_edge_index_input: Edge indices for altitude graphs [optional]
            alt_edge_prob_input: Edge probabilities for altitude graphs [optional]

        Returns:
            Prediction tensor [batch, ..., pred_len]
        """
        # Process static longitude-latitude features through GAT
        src_lon_lat = self.staticInfoLayer1(lon_lat_node_features, lon_lat_edge_index_input, lon_lat_edge_prob_input,
                                            src_sp_lon_lat)  # [batch, L, d_model]

        # Process static altitude features through GAT
        src_alt = self.staticInfoLayer2(alt_node_features, alt_edge_index_input, alt_edge_prob_input,
                                        src_sp_alt)  # [batch, L, d_model]

        # Concatenate static features
        src_gat = torch.cat((src_lon_lat, src_alt), dim=1)  # [batch, 2L, d_model]

        # Add spatio-temporal positional encoding for static features
        src_gat_pe = self.pos_enc_stage1_static(src_gat)  # [batch, 2L, d_model]

        # Process dynamic coordinate features through TSPatchLayer
        cords = self.dynamicInfoLayer(src_cords)  # [batch, M*N, d_model]

        # Add spatio-temporal positional encoding for dynamic features
        cords_pe = self.pos_enc_stage1_dynamic(cords)  # [batch, M*N, d_model]

        # Stage 1: Static-Dynamic Cross-Attention
        x_sd = self.cross_att_stage1(cords_pe, src_gat_pe)  # [batch, M*N, d_model]

        # Process velocity features through TSPatchLayer
        velocity = self.dynamicInfoLayer(src_v)  # [batch, M*N, d_model]

        # Stage 2: Spatial-Velocity Cross-Attention
        x_sv = self.cross_att_stage2(src_gat_pe, velocity)  # [batch, 2L, d_model]

        # Combine Stage 1 and Stage 2 outputs
        combined = torch.cat((x_sd, x_sv), dim=1)  # [batch, M*N + 2L, d_model]

        # Reshape for FlattenHead
        # n_vars (x, y, z), patch_num = (M*N + 2L) // 3
        n_vars = 3
        patch_num = combined.size(1) // n_vars
        combined = combined.view(combined.size(0), n_vars, patch_num,
                                 combined.size(2))  # [batch, 3, patch_num, d_model]

        # Apply the prediction head
        output = self.head(combined.permute(0, 1, 3, 2))  # [batch, pred_len, ...]
        result = output.permute(0, 2, 1)  # [batch, ..., pred_len]

        return result
