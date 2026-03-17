import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from prosody_scorer.models.util import mean_pooling, create_mask


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
