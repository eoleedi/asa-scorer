import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from prosody_scorer.models.util import mean_pooling, create_mask
from prosody_scorer.models.mine import MINEModule


# adapt: tanh -> GELU
class BiLSTMScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        """
        BiLSTM(input_size, hidden_size, num_layers=num_layers)
        """
        super().__init__()
        self.blstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        # self.fc = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, 1))

        self.activations = nn.ModuleDict(
            [
                ["tanh", nn.Tanh()],
                ["GELU", nn.GELU()],
            ]
        )

    def forward(self, x, act=None, seq_lengths=None):
        x_nopadded = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True)
        output, hidden = self.blstm(x_nopadded)
        BiLSTM_embedding, out_len = pad_packed_sequence(output, batch_first=True)
        mask = create_mask(BiLSTM_embedding, seq_lengths)

        output = mean_pooling(BiLSTM_embedding, mask)

        score = self.fc(output)
        # score = self.activations['GELU'](score)

        return score


class NonClusterScorer(nn.Module):
    """
    A model for score prediction on multiple aspects.
    @param scorers: a list of aspects to be evaluated (e.g., ['fluency', 'pronunciation'])
    """

    def __init__(self, input_dim: int, embed_dim: int, scorers: list):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self.scorers = nn.ModuleDict(
            {aspect: BiLSTMScorer(embed_dim, embed_dim, 2) for aspect in scorers}
        )

    def forward(self, x) -> torch.Tensor | list:
        """
        x: extract audio features
        return: a pred score (if multiple scorers, return a list of pred scores)
        """
        device = x.device
        # step 1: audio features preprocessing
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device)
        new_audio_embedding_tensor = self.preprocessing(x)

        # create mask
        mask = create_mask(new_audio_embedding_tensor, seq_lengths)

        new_audio_embedding_tensor = new_audio_embedding_tensor * mask

        # step 2: make a score directly
        pred = [
            scorer(x=new_audio_embedding_tensor, seq_lengths=seq_lengths)
            for scorer in self.scorers.values()
        ]
        pred = torch.cat(pred, dim=1) if len(pred) > 1 else pred[0]

        return pred


class SimpleRegressionScorer(nn.Module):
    """Simple BLSTM regressor for sequence features."""

    def __init__(self, input_dim: int, hidden_dim: int, scorers: list):
        super().__init__()
        self.num_outputs = len(scorers)
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device).clamp(min=1)

        x = self.preprocessing(x)
        mask = create_mask(x, seq_lengths)
        x = x * mask

        packed = pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.blstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        blstm_mask = create_mask(output, seq_lengths)
        pooled = mean_pooling(output, blstm_mask)

        pred = self.regressor(pooled)
        return pred if self.num_outputs > 1 else pred[:, :1]


class ClusterScorer(nn.Module):
    """
    The main model for fluency score prediction with using cluster.
    """

    def __init__(
        self, input_dim, embed_dim, scorers: list, clustering_dim=6, num_clusters=50
    ):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self.cluster_embed = nn.Embedding(
            num_clusters + 1, clustering_dim, padding_idx=0
        )
        self.scorers = nn.ModuleDict(
            {
                aspect: BiLSTMScorer(embed_dim + clustering_dim, embed_dim, 2)
                for aspect in scorers
            }
        )

    def forward(self, x, cluster_id):
        """
        x: extract audio features
        return: a pred score
        """
        device = x.device
        # step 1: audio features preprocessing
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device)
        new_audio_embedding_tensor = self.preprocessing(x)
        cluster_embed = self.cluster_embed(cluster_id).float()

        # step 2: concat audio and cluster embedding
        audio_features = torch.concat(
            (new_audio_embedding_tensor, cluster_embed), dim=2
        )
        # create mask
        mask = create_mask(audio_features, seq_lengths)

        audio_features = audio_features * mask

        # step 3: make a score
        pred = [
            scorer(x=audio_features, seq_lengths=seq_lengths)
            for scorer in self.scorers.values()
        ]
        # Stack all predictions into a single tensor
        pred = torch.cat(pred, dim=1) if len(pred) > 1 else pred[0]
        return pred


class TransformerScorer(nn.Module):
    def __init__(
        self,
        num_heads,
        depth=3,
        input_dim=84,
        dropout_prob=0.1,
        activation="gelu",
        norm_first=True,
        clustering_dim=6,
        hidden_dim=32,
        scorers=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim
        self.clustering_dim = clustering_dim
        self.num_outputs = len(scorers) if scorers is not None else 1
        self.model_dim = self.hidden_dim + self.clustering_dim
        self.proj_layer = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout_prob,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        self.mlp_head_utt1 = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_outputs),
        )

    def forward(self, x, cluster_idx=None):
        nonzero_mask = x.abs().sum(dim=2) != 0
        x = self.proj_layer(x)

        if cluster_idx is not None:
            if cluster_idx.dim() == 2:
                cluster_idx = cluster_idx.unsqueeze(-1)
            cluster_idx = cluster_idx.float()
            x = torch.cat([x, cluster_idx], dim=2)
        elif self.clustering_dim > 0:
            zeros = x.new_zeros(x.size(0), x.size(1), self.clustering_dim)
            x = torch.cat([x, zeros], dim=2)

        mask = ~nonzero_mask
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.mean_pooling(x, nonzero_mask)
        x = self.mlp_head_utt1(x)
        return x if self.num_outputs > 1 else x[:, :1]

    def mean_pooling(self, feature_tensor: torch.Tensor, valid_mask: torch.Tensor):
        mask = valid_mask.unsqueeze(-1).float()
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean = (feature_tensor * mask).sum(dim=1) / count
        return mean

    def create_mask(self, x: torch.Tensor, padding_value: float = 0.0):
        # return mask [batch_size, seq_len]
        mask_2d = torch.all(x == padding_value, dim=-1)
        return mask_2d


class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.attn(x)
        if mask is not None:
            safe_mask = mask.clone()
            invalid_rows = ~safe_mask.any(dim=1)
            if invalid_rows.any():
                safe_mask[invalid_rows, 0] = True
            scores = scores.masked_fill(~safe_mask.unsqueeze(-1), -1e9)
            weights = torch.softmax(scores, dim=1)
            weights = weights * safe_mask.unsqueeze(-1)
            weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        else:
            weights = torch.softmax(scores, dim=1)
        mean = torch.sum(weights * x, dim=1)
        second = torch.sum(weights * (x**2), dim=1)
        std = torch.sqrt(torch.clamp(second - mean**2, min=1e-6))
        return torch.cat([mean, std], dim=-1)


class TemporalBranchEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x * valid_mask.unsqueeze(-1)
        y = x.transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = self.act(y)
        y = y * valid_mask.unsqueeze(-1)
        return y


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment

        if dim != self.codebook_dim:
            self.in_proj = weight_norm(nn.Linear(dim, self.codebook_dim))
            self.out_proj = weight_norm(nn.Linear(self.codebook_dim, dim))
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self._codebook = nn.Embedding(codebook_size, self.codebook_dim)

    @property
    def codebook(self):
        return self._codebook

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        # latents: [B, D, T]
        bsz, dim, tlen = latents.shape
        encodings = latents.transpose(1, 2).reshape(bsz * tlen, dim)
        codebook = self.codebook.weight

        encodings = F.normalize(encodings, dim=-1)
        codebook = F.normalize(codebook, dim=-1)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].view(bsz, tlen)
        z_q = self.decode_code(indices)
        return z_q, indices

    def forward(self, z: torch.Tensor):
        # z: [B, D, T]
        z = z.transpose(1, 2)
        z_e = self.in_proj(z)
        z_e = z_e.transpose(1, 2)
        z_q, indices = self.decode_latents(z_e)

        if self.training:
            commitment_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
            commit_loss = commitment_loss + codebook_loss
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.transpose(1, 2)
        z_q = self.out_proj(z_q)
        z_q = z_q.transpose(1, 2)
        return z_q, indices, commit_loss


class CrossAttnHCSSLScorer(nn.Module):
    """
    Cross-attention scorer that fuses SSL and handcrafted (HC) features
    without using MINE.
    """

    def __init__(
        self,
        ssl_input_dim: int,
        hc_input_dim: int,
        hidden_dim: int,
        scorers: list,
        num_heads: int = 4,
        depth: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_outputs = len(scorers)
        self.depth = depth

        self.ssl_proj = nn.Sequential(
            nn.Linear(ssl_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.hc_proj = nn.Sequential(
            nn.Linear(hc_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.ssl_from_hc_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_prob,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.hc_from_ssl_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_prob,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.ssl_attn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        )
        self.hc_attn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        )

        self.ssl_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.hc_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.ssl_ffn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        )
        self.hc_ffn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        )

        self.ssl_pool = AttentiveStatsPooling(hidden_dim)
        self.hc_pool = AttentiveStatsPooling(hidden_dim)

        fusion_dim = hidden_dim * 2 * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, self.num_outputs),
        )

    def _valid_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum(dim=-1) > 0

    def forward(self, ssl_feats: torch.Tensor, hc_feats: torch.Tensor) -> torch.Tensor:
        ssl_mask = self._valid_mask(ssl_feats)
        hc_mask = self._valid_mask(hc_feats)

        ssl = self.ssl_proj(ssl_feats)
        hc = self.hc_proj(hc_feats)
        ssl = ssl * ssl_mask.unsqueeze(-1)
        hc = hc * hc_mask.unsqueeze(-1)

        for i in range(self.depth):
            ssl_cross, _ = self.ssl_from_hc_layers[i](
                query=ssl,
                key=hc,
                value=hc,
                key_padding_mask=~hc_mask,
                need_weights=False,
            )
            hc_cross, _ = self.hc_from_ssl_layers[i](
                query=hc,
                key=ssl,
                value=ssl,
                key_padding_mask=~ssl_mask,
                need_weights=False,
            )

            ssl = self.ssl_attn_norms[i](ssl + ssl_cross)
            hc = self.hc_attn_norms[i](hc + hc_cross)

            ssl = self.ssl_ffn_norms[i](ssl + self.ssl_ffns[i](ssl))
            hc = self.hc_ffn_norms[i](hc + self.hc_ffns[i](hc))

            ssl = ssl * ssl_mask.unsqueeze(-1)
            hc = hc * hc_mask.unsqueeze(-1)

        ssl_global = self.ssl_pool(ssl, ssl_mask)
        hc_global = self.hc_pool(hc, hc_mask)
        fused = torch.cat([ssl_global, hc_global], dim=-1)
        pred = self.head(fused)
        return pred if self.num_outputs > 1 else pred[:, :1]


class LayerWeightedHCSSLScorer(nn.Module):
    """
    HC-guided scorer that learns subconstruct-specific HuBERT layer mixtures.

    Expected inputs:
      - ssl_layers: [B, L, T_ssl, D_ssl]
      - hc_feats:   [B, T_hc, 50]

    The three branches keep the FDMPA naming/order: int, rhy, pro.
    """

    BRANCH_HC_SLICE = {
        "int": slice(8, 10),
        "rhy": slice(10, 50),
        "pro": slice(0, 8),
    }
    BRANCH_HC_DIM = {"int": 2, "rhy": 40, "pro": 8}

    def __init__(
        self,
        ssl_input_dim: int,
        num_layers: int,
        hidden_dim: int,
        scorers: list,
        num_tokens: int = 16,
        dropout_prob: float = 0.1,
        hc_aux_weight: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_outputs = len(scorers)
        self.hc_aux_weight = hc_aux_weight
        self.branches = ["int", "rhy", "pro"]

        self.layer_logits = nn.Parameter(torch.zeros(len(self.branches), num_layers))

        self.ssl_encoders = nn.ModuleDict(
            {
                branch: TemporalBranchEncoder(ssl_input_dim, hidden_dim)
                for branch in self.branches
            }
        )
        self.ssl_stats_pool = nn.ModuleDict(
            {branch: AttentiveStatsPooling(hidden_dim) for branch in self.branches}
        )
        self.hc_predictors = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim, self.BRANCH_HC_DIM[branch]),
                )
                for branch in self.branches
            }
        )

        fusion_dim = hidden_dim * 2 * len(self.branches)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, self.num_outputs),
        )

    # HC auxiliary normalization removed: only keep necessary HC transforms

    def _valid_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum(dim=-1) > 0

    def _all_layer_valid_mask(self, ssl_layers: torch.Tensor) -> torch.Tensor:
        # ssl_layers: [B, L, T, D]. A frame is valid if any layer has non-zero values.
        return ssl_layers.abs().sum(dim=(1, 3)) > 0

    def _adaptive_pool_tokens(
        self, x: torch.Tensor, valid_mask: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = x.shape
        out_tokens = x.new_zeros(bsz, num_tokens, dim)
        out_mask = torch.zeros(bsz, num_tokens, device=x.device, dtype=torch.bool)
        for b in range(bsz):
            valid_len = int(valid_mask[b].sum().item())
            if valid_len <= 0:
                continue
            seq = x[b, :valid_len].transpose(0, 1).unsqueeze(0)
            pooled = nn.AdaptiveAvgPool1d(num_tokens)(seq)
            out_tokens[b] = pooled.squeeze(0).transpose(0, 1)
            out_mask[b] = True
        return out_tokens, out_mask

    def _normalize_hc_branch(
        self, branch: str, hc_branch: torch.Tensor
    ) -> torch.Tensor:
        # Remove mean/std normalization; keep only deterministic transforms
        if branch == "pro":
            return hc_branch
        if branch == "int":
            pitch = torch.log(torch.clamp(hc_branch[:, :, 0:1], min=0.0) + 1.0)
            periodicity = hc_branch[:, :, 1:2]
            return torch.cat([pitch, periodicity], dim=-1)
        return hc_branch

    def compute_branch_diversity(self, ssl_global: dict) -> dict:
        branches_list = ["int", "rhy", "pro"]
        similarities = {}
        sim_values = []
        for i, b1 in enumerate(branches_list):
            for j, b2 in enumerate(branches_list):
                if i >= j:
                    continue
                g1 = F.normalize(ssl_global[b1], p=2, dim=1)
                g2 = F.normalize(ssl_global[b2], p=2, dim=1)
                sim = (g1 * g2).sum(dim=1).mean()
                similarities[f"{b1}-{b2}"] = sim.item()
                sim_values.append(sim)
        similarities["mean"] = (
            torch.stack(sim_values).mean().item() if sim_values else 0.0
        )
        return similarities

    def forward(self, ssl_layers: torch.Tensor, hc_feats: torch.Tensor | None = None):
        if ssl_layers.dim() != 4:
            raise ValueError(
                f"LayerWeightedHCSSLScorer expects ssl_layers [B, L, T, D], got {tuple(ssl_layers.shape)}"
            )
        if ssl_layers.size(1) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} SSL layers, got {ssl_layers.size(1)}"
            )

        ssl_mask = self._all_layer_valid_mask(ssl_layers)
        layer_weights = torch.softmax(self.layer_logits, dim=-1)

        ssl_tokens = {}
        ssl_token_mask = {}
        ssl_global = {}
        hc_aux_preds = {}
        hc_aux_targets = {}
        hc_aux_mask = {}

        hc_mask = self._valid_mask(hc_feats) if hc_feats is not None else None
        if hc_feats is not None and hc_feats.size(1) != ssl_layers.size(2):
            raise ValueError(
                "LayerWeightedHCSSLScorer expects HC features to be resampled "
                f"to SSL frame length: got T_hc={hc_feats.size(1)}, "
                f"T_ssl={ssl_layers.size(2)}"
            )
        batch_min_len = int(torch.min(ssl_mask.sum(dim=1)).item())
        if self.num_tokens > 0:
            target_tokens = (
                min(self.num_tokens, batch_min_len)
                if batch_min_len > 0
                else self.num_tokens
            )
        else:
            target_tokens = max(1, batch_min_len)

        for branch_idx, branch in enumerate(self.branches):
            weighted_ssl = torch.einsum(
                "l,bltd->btd", layer_weights[branch_idx], ssl_layers
            )
            ssl_seq = self.ssl_encoders[branch](weighted_ssl, ssl_mask)
            s_tok, s_tok_mask = self._adaptive_pool_tokens(
                ssl_seq, ssl_mask, target_tokens
            )
            ssl_tokens[branch] = s_tok
            ssl_token_mask[branch] = s_tok_mask
            ssl_global[branch] = self.ssl_stats_pool[branch](ssl_seq, ssl_mask)

            if hc_feats is not None:
                hc_branch = hc_feats[:, :, self.BRANCH_HC_SLICE[branch]]
                hc_branch = self._normalize_hc_branch(branch, hc_branch)
                hc_aux_preds[branch] = self.hc_predictors[branch](ssl_seq)
                hc_aux_targets[branch] = hc_branch
                hc_aux_mask[branch] = ssl_mask & hc_mask

        fused = torch.cat([ssl_global[b] for b in self.branches], dim=-1)
        pred = self.head(fused)
        if self.num_outputs == 1:
            pred = pred[:, :1]

        aux = {
            "layer_weights": layer_weights,
            "ssl_tokens": ssl_tokens,
            "ssl_token_mask": ssl_token_mask,
            "ssl_global": ssl_global,
            "hc_aux_preds": hc_aux_preds,
            "hc_aux_targets": hc_aux_targets,
            "hc_aux_mask": hc_aux_mask,
            "branch_diversity": self.compute_branch_diversity(ssl_global),
        }
        return pred, aux


class SingleLayerHCSSLScorer(nn.Module):
    """
    Single-layer HuBERT scorer with HC-group prediction auxiliary losses.

    Expected inputs:
      - ssl_feats: [B, T_ssl, D_ssl]
      - hc_feats:  [B, T_hc, 50]

    Unlike FDMPAScorer, this model does not use FVQ or MINE. Each SSL branch
    predicts its matching handcrafted feature group as an auxiliary task:
      - int -> pitch + periodicity
      - rhy -> PPG
      - pro -> loudness
    """

    BRANCH_HC_SLICE = {
        "int": slice(8, 10),
        "rhy": slice(10, 50),
        "pro": slice(0, 8),
    }
    BRANCH_HC_DIM = {"int": 2, "rhy": 40, "pro": 8}

    def __init__(
        self,
        ssl_input_dim: int,
        hidden_dim: int,
        scorers: list,
        num_tokens: int = 16,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_outputs = len(scorers)
        self.branches = ["int", "rhy", "pro"]

        self.ssl_encoders = nn.ModuleDict(
            {
                branch: TemporalBranchEncoder(ssl_input_dim, hidden_dim)
                for branch in self.branches
            }
        )
        self.ssl_stats_pool = nn.ModuleDict(
            {branch: AttentiveStatsPooling(hidden_dim) for branch in self.branches}
        )
        self.hc_predictors = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim, self.BRANCH_HC_DIM[branch]),
                )
                for branch in self.branches
            }
        )

        fusion_dim = hidden_dim * 2 * len(self.branches)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, self.num_outputs),
        )

    # HC auxiliary normalization removed: keep only deterministic transforms

    def _valid_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum(dim=-1) > 0

    def _adaptive_pool_tokens(
        self, x: torch.Tensor, valid_mask: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = x.shape
        out_tokens = x.new_zeros(bsz, num_tokens, dim)
        out_mask = torch.zeros(bsz, num_tokens, device=x.device, dtype=torch.bool)
        for b in range(bsz):
            valid_len = int(valid_mask[b].sum().item())
            if valid_len <= 0:
                continue
            seq = x[b, :valid_len].transpose(0, 1).unsqueeze(0)
            pooled = nn.AdaptiveAvgPool1d(num_tokens)(seq)
            out_tokens[b] = pooled.squeeze(0).transpose(0, 1)
            out_mask[b] = True
        return out_tokens, out_mask

    def _normalize_hc_branch(
        self, branch: str, hc_branch: torch.Tensor
    ) -> torch.Tensor:
        # Remove mean/std normalization; keep only deterministic transforms
        if branch == "pro":
            return hc_branch
        if branch == "int":
            pitch = torch.log(torch.clamp(hc_branch[:, :, 0:1], min=0.0) + 1.0)
            periodicity = hc_branch[:, :, 1:2]
            return torch.cat([pitch, periodicity], dim=-1)
        return hc_branch

    def compute_branch_diversity(self, ssl_global: dict) -> dict:
        branches_list = ["int", "rhy", "pro"]
        similarities = {}
        sim_values = []
        for i, b1 in enumerate(branches_list):
            for j, b2 in enumerate(branches_list):
                if i >= j:
                    continue
                g1 = F.normalize(ssl_global[b1], p=2, dim=1)
                g2 = F.normalize(ssl_global[b2], p=2, dim=1)
                sim = (g1 * g2).sum(dim=1).mean()
                similarities[f"{b1}-{b2}"] = sim.item()
                sim_values.append(sim)
        similarities["mean"] = (
            torch.stack(sim_values).mean().item() if sim_values else 0.0
        )
        return similarities

    def forward(self, ssl_feats: torch.Tensor, hc_feats: torch.Tensor):
        if ssl_feats.dim() != 3:
            raise ValueError(
                f"SingleLayerHCSSLScorer expects ssl_feats [B, T, D], got {tuple(ssl_feats.shape)}"
            )
        if hc_feats is None:
            raise ValueError("SingleLayerHCSSLScorer requires handcrafted features.")

        ssl_mask = self._valid_mask(ssl_feats)
        hc_mask = self._valid_mask(hc_feats)
        if hc_feats.size(1) != ssl_feats.size(1):
            raise ValueError(
                "SingleLayerHCSSLScorer expects HC features to be resampled "
                f"to SSL frame length: got T_hc={hc_feats.size(1)}, "
                f"T_ssl={ssl_feats.size(1)}"
            )
        batch_min_len = int(torch.min(ssl_mask.sum(dim=1)).item())
        if self.num_tokens > 0:
            target_tokens = (
                min(self.num_tokens, batch_min_len)
                if batch_min_len > 0
                else self.num_tokens
            )
        else:
            target_tokens = max(1, batch_min_len)

        ssl_tokens = {}
        ssl_token_mask = {}
        ssl_global = {}
        hc_aux_preds = {}
        hc_aux_targets = {}
        hc_aux_mask = {}

        for branch in self.branches:
            ssl_seq = self.ssl_encoders[branch](ssl_feats, ssl_mask)
            s_tok, s_tok_mask = self._adaptive_pool_tokens(
                ssl_seq, ssl_mask, target_tokens
            )
            ssl_tokens[branch] = s_tok
            ssl_token_mask[branch] = s_tok_mask
            ssl_global[branch] = self.ssl_stats_pool[branch](ssl_seq, ssl_mask)

            hc_branch = hc_feats[:, :, self.BRANCH_HC_SLICE[branch]]
            hc_branch = self._normalize_hc_branch(branch, hc_branch)
            hc_aux_preds[branch] = self.hc_predictors[branch](ssl_seq)
            hc_aux_targets[branch] = hc_branch
            hc_aux_mask[branch] = ssl_mask & hc_mask

        fused = torch.cat([ssl_global[b] for b in self.branches], dim=-1)
        pred = self.head(fused)
        if self.num_outputs == 1:
            pred = pred[:, :1]

        aux = {
            "ssl_tokens": ssl_tokens,
            "ssl_token_mask": ssl_token_mask,
            "ssl_global": ssl_global,
            "hc_aux_preds": hc_aux_preds,
            "hc_aux_targets": hc_aux_targets,
            "hc_aux_mask": hc_aux_mask,
            "branch_diversity": self.compute_branch_diversity(ssl_global),
        }
        return pred, aux


class FDMPAScorer(nn.Module):
    """
    Factorized Domain-agnostic Mutual Information Maximization for Prosody Assessment.

    This model decomposes SSL features into three prosodic branches:
    - Intonation (int): Related to Pitch and Periodicity (HC slice 8-10)
    - Rhythm (rhy): Related to PPG (HC slice 10-50)
    - Prominence (pro): Related to Loudness (HC slice 0-8)
    """

    BRANCH_HC_SLICE = {
        "int": slice(8, 10),
        "rhy": slice(10, 50),
        "pro": slice(0, 8),
    }

    def __init__(
        self,
        ssl_input_dim: int,
        hidden_dim: int,
        scorers: list,
        mine_hidden_dim: int = 64,
        mine_ema_decay: float = 0.99,
        num_tokens: int = 16,
        dropout_prob: float = 0.1,
        hc_codebook_size: int = 1024,
        hc_codebook_dim: int = 8,
        hc_commitment: float = 0.25,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_outputs = len(scorers)
        self.branches = ["int", "rhy", "pro"]

        # 1. SSL Encoders for each branch
        self.ssl_encoders = nn.ModuleDict(
            {
                branch: TemporalBranchEncoder(ssl_input_dim, hidden_dim)
                for branch in self.branches
            }
        )

        # 2. HC Projections
        self.hc_input_proj = nn.ModuleDict(
            {
                "int": nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ),
                "rhy": nn.Sequential(
                    nn.Linear(40, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ),
                "pro": nn.Sequential(
                    nn.Linear(8, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ),
            }
        )

        # 3. FVQ Bottleneck for each branch (applied to SSL latents)
        self.ssl_bottleneck = nn.ModuleDict(
            {
                branch: FactorizedVectorQuantize(
                    dim=hidden_dim,
                    codebook_size=hc_codebook_size,
                    codebook_dim=hc_codebook_dim,
                    commitment=hc_commitment,
                )
                for branch in self.branches
            }
        )

        # 4. Pooling for global representations
        self.ssl_stats_pool = nn.ModuleDict(
            {branch: AttentiveStatsPooling(hidden_dim) for branch in self.branches}
        )
        self.hc_stats_pool = nn.ModuleDict(
            {branch: AttentiveStatsPooling(hidden_dim) for branch in self.branches}
        )

        # 5. MINE Modules for MI estimation (Local and Global)
        self.mine_local = nn.ModuleDict(
            {
                branch: MINEModule(
                    hidden_dim, hidden_dim, mine_hidden_dim, mine_ema_decay
                )
                for branch in self.branches
            }
        )

        # 6. Final Scorer Head
        fusion_dim = hidden_dim * 2 * len(self.branches)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, self.num_outputs),
        )

        # Optional branch-specific auxiliary heads
        self.branch_heads = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.num_outputs),
                )
                for branch in self.branches
            }
        )

    def _valid_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum(dim=-1) > 0

    def _adaptive_pool_tokens(
        self, x: torch.Tensor, valid_mask: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = x.shape
        out_tokens = x.new_zeros(bsz, num_tokens, dim)
        out_mask = torch.zeros(bsz, num_tokens, device=x.device, dtype=torch.bool)
        for b in range(bsz):
            valid_len = int(valid_mask[b].sum().item())
            if valid_len <= 0:
                continue
            seq = x[b, :valid_len].transpose(0, 1).unsqueeze(0)
            pooled = nn.AdaptiveAvgPool1d(num_tokens)(seq)
            out_tokens[b] = pooled.squeeze(0).transpose(0, 1)
            out_mask[b] = True
        return out_tokens, out_mask

    def forward(self, ssl_feats: torch.Tensor, hc_feats: torch.Tensor):
        ssl_mask = self._valid_mask(ssl_feats)
        hc_mask = self._valid_mask(hc_feats)

        # Determine the target number of tokens based on the minimal sequence length in the current batch
        ssl_lens = ssl_mask.sum(dim=1)
        hc_lens = hc_mask.sum(dim=1)
        batch_min_len = int(torch.min(torch.min(ssl_lens), torch.min(hc_lens)).item())

        # Determine target number of tokens
        if self.num_tokens > 0:
            target_tokens = (
                min(self.num_tokens, batch_min_len)
                if batch_min_len > 0
                else self.num_tokens
            )
        else:
            target_tokens = max(1, batch_min_len)

        ssl_tokens = {}
        ssl_token_mask = {}
        ssl_global = {}
        ssl_commit_losses = {}

        hc_tokens = {}
        hc_token_mask = {}
        hc_global = {}

        for branch in self.branches:
            # --- SSL Path ---
            # Encode
            ssl_seq = self.ssl_encoders[branch](ssl_feats, ssl_mask)
            # Quantize (FVQ)
            quant_ssl_seq, _, commit_loss = self.ssl_bottleneck[branch](
                ssl_seq.transpose(1, 2)
            )
            ssl_seq = quant_ssl_seq.transpose(1, 2) * ssl_mask.unsqueeze(-1)
            ssl_commit_losses[branch] = commit_loss.mean()

            # Pool to tokens using adaptive target length
            s_tok, s_tok_mask = self._adaptive_pool_tokens(
                ssl_seq, ssl_mask, target_tokens
            )
            ssl_tokens[branch] = s_tok
            ssl_token_mask[branch] = s_tok_mask
            ssl_global[branch] = self.ssl_stats_pool[branch](s_tok, s_tok_mask)

            # --- HC Path ---
            sl = self.BRANCH_HC_SLICE[branch]
            hc_branch = hc_feats[:, :, sl]

            # Specific normalization for Intonation branch (Pitch)
            if branch == "int":
                pitch = hc_branch[:, :, 0:1]
                periodicity = hc_branch[:, :, 1:2]
                pitch = torch.log(pitch + 1.0)
                hc_branch = torch.cat([pitch, periodicity], dim=-1)

            hc_seq = self.hc_input_proj[branch](hc_branch) * hc_mask.unsqueeze(-1)

            # Pool to tokens using adaptive target length
            h_tok, h_tok_mask = self._adaptive_pool_tokens(
                hc_seq, hc_mask, target_tokens
            )
            hc_tokens[branch] = h_tok
            hc_token_mask[branch] = h_tok_mask
            hc_global[branch] = self.hc_stats_pool[branch](h_tok, h_tok_mask)

        # Prediction
        fused = torch.cat([ssl_global[b] for b in self.branches], dim=-1)
        pred = self.head(fused)
        if self.num_outputs == 1:
            pred = pred[:, :1]

        total_ssl_commit_loss = torch.stack(list(ssl_commit_losses.values())).mean()

        # Auxiliary branch predictions
        branch_preds = {}
        for branch in self.branches:
            branch_preds[branch] = self.branch_heads[branch](ssl_global[branch])
            if self.num_outputs == 1:
                branch_preds[branch] = branch_preds[branch][:, :1]

        branch_diversity = self.compute_branch_diversity(ssl_global)

        aux = {
            "ssl_tokens": ssl_tokens,
            "ssl_token_mask": ssl_token_mask,
            "hc_tokens": hc_tokens,
            "hc_token_mask": hc_token_mask,
            "ssl_global": ssl_global,
            "hc_global": hc_global,
            "ssl_commit_loss": total_ssl_commit_loss,
            "branch_preds": branch_preds,
            "branch_diversity": branch_diversity,
        }
        return pred, aux

    def compute_mi_terms(
        self,
        ssl_tokens: dict,
        hc_tokens: dict,
        ssl_global: dict,
        hc_global: dict,
        update_ema: bool = True,
    ):
        """Used in stage 1 and stage 2 training to get MI estimates."""
        local_terms = []
        mi_local_dict = {}

        for branch in self.branches:
            # Local MI: I(z_tok; hc_tok)
            mi_local = self.mine_local[branch].mi_lower_bound(
                ssl_tokens[branch], hc_tokens[branch], update_ema=update_ema
            )
            local_terms.append(mi_local)
            mi_local_dict[branch] = mi_local

        # Since global MINE was removed, return zero tensors for global terms for compatibility
        device = None
        for v in ssl_tokens.values():
            device = v.device
            break
        if device is None:
            device = torch.device("cpu")

        mi_global_sum = torch.tensor(0.0, device=device)
        mi_global_dict = {
            branch: torch.tensor(0.0, device=device) for branch in self.branches
        }

        return (
            torch.stack(local_terms).sum(),
            mi_global_sum,
            mi_local_dict,
            mi_global_dict,
        )

    def mine_parameters(self):
        for module in [self.mine_local]:
            for p in module.parameters():
                yield p

    def non_mine_parameters(self):
        mine_param_ids = {id(p) for p in self.mine_parameters()}
        for p in self.parameters():
            if id(p) not in mine_param_ids:
                yield p

    def compute_branch_diversity(self, ssl_global: dict) -> dict:
        """
        Compute cosine similarity between branch representations.
        Returns dict with pairwise similarities and mean.
        """
        branches_list = ["int", "rhy", "pro"]
        similarities = {}
        sim_values = []

        for i, b1 in enumerate(branches_list):
            for j, b2 in enumerate(branches_list):
                if i >= j:
                    continue
                g1 = F.normalize(ssl_global[b1], p=2, dim=1)
                g2 = F.normalize(ssl_global[b2], p=2, dim=1)
                sim = (g1 * g2).sum(dim=1).mean()
                similarities[f"{b1}-{b2}"] = sim.item()
                sim_values.append(sim)

        mean_sim = torch.stack(sim_values).mean().item() if sim_values else 0.0
        similarities["mean"] = mean_sim
        return similarities
