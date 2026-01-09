"""
Proxy Feature Extraction and Loss Computation for Prosodic Branches

This module implements teacher signal generation and proxy losses for:
1. Intonation branch - F0 contour based quality
2. Rhythm branch - Syllable duration and rhythm metrics
3. Prominence branch - Syllable-level prominence ranking
4. Boundary branch - Prosodic boundary detection

Each branch has:
- Teacher signal computation (from learner + reference)
- Proxy loss computation (predicted vs teacher)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ========================================================================
# 1. INTONATION BRANCH - F0 Contour Quality
# ========================================================================


class IntonationProxyFeatures:
    """
    Compute intonation quality based on F0 contour similarity to reference.

    Teacher signal: DTW-based distance between normalized F0 contours
    Target: exp(-alpha * dtw_distance) -> [0, 1]

    Calibrated alpha=0.001 based on TTS experiments (DTW distance ~168 for similar speech)
    """

    def __init__(self, alpha: float = 0.001):
        """
        Args:
            alpha: Scaling factor for DTW distance -> quality conversion (default 0.001)
        """
        self.alpha = alpha

    def normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        Z-score normalize F0 contour, handling unvoiced regions (zeros).

        Args:
            f0: Raw F0 contour, shape (T,), zeros indicate unvoiced

        Returns:
            Normalized F0 contour, same shape
        """
        # Extract voiced regions (non-zero F0)
        voiced_mask = f0 > 0
        if voiced_mask.sum() == 0:
            return f0  # All unvoiced, return as-is

        voiced_f0 = f0[voiced_mask]
        mean_f0 = np.mean(voiced_f0)
        std_f0 = np.std(voiced_f0)

        if std_f0 > 0:
            f0_norm = f0.copy()
            f0_norm[voiced_mask] = (voiced_f0 - mean_f0) / std_f0
        else:
            f0_norm = f0.copy()

        return f0_norm

    def align_f0_contours(
        self, f0_learner: np.ndarray, f0_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align learner and reference F0 contours to same length via interpolation.

        Args:
            f0_learner: Learner F0 contour, shape (T1,)
            f0_ref: Reference F0 contour, shape (T2,)

        Returns:
            Aligned (f0_learner, f0_ref), both shape (max(T1, T2),)
        """
        len_learner = len(f0_learner)
        len_ref = len(f0_ref)

        if len_learner == len_ref:
            return f0_learner, f0_ref

        # Interpolate to the longer length
        target_len = max(len_learner, len_ref)

        # Interpolate learner
        x_learner = np.linspace(0, 1, len_learner)
        x_target = np.linspace(0, 1, target_len)
        f_learner = interp1d(
            x_learner, f0_learner, kind="linear", fill_value="extrapolate"
        )
        f0_learner_aligned = f_learner(x_target)

        # Interpolate reference
        x_ref = np.linspace(0, 1, len_ref)
        f_ref = interp1d(x_ref, f0_ref, kind="linear", fill_value="extrapolate")
        f0_ref_aligned = f_ref(x_target)

        return f0_learner_aligned, f0_ref_aligned

    def compute_dtw_distance(self, f0_learner: np.ndarray, f0_ref: np.ndarray) -> float:
        """
        Compute DTW distance between F0 contours.

        Args:
            f0_learner: Normalized learner F0, shape (T,)
            f0_ref: Normalized reference F0, shape (T,)

        Returns:
            DTW distance (scalar)
        """
        # Use fastdtw for efficient DTW computation
        f0_learner_2d = f0_learner.reshape(-1, 1)
        f0_ref_2d = f0_ref.reshape(-1, 1)
        distance, _ = fastdtw(f0_learner_2d, f0_ref_2d, dist=euclidean)

        return distance

    def compute_teacher_quality(
        self, f0_learner: np.ndarray, f0_ref: np.ndarray
    ) -> float:
        """
        Compute intonation quality score from F0 contours.

        Args:
            f0_learner: Raw learner F0 contour, shape (T1,)
            f0_ref: Raw reference F0 contour, shape (T2,)

        Returns:
            Quality score in [0, 1], higher is better
        """
        # Normalize F0 contours
        f0_learner_norm = self.normalize_f0(f0_learner)
        f0_ref_norm = self.normalize_f0(f0_ref)

        # Align to same length
        f0_learner_aligned, f0_ref_aligned = self.align_f0_contours(
            f0_learner_norm, f0_ref_norm
        )

        # Compute DTW distance
        dtw_dist = self.compute_dtw_distance(f0_learner_aligned, f0_ref_aligned)

        # Convert to quality score: exp(-alpha * distance)
        quality = np.exp(-self.alpha * dtw_dist)

        return quality


class IntonationProxyLoss(nn.Module):
    """
    Proxy loss for intonation branch.

    Intonation Quality Score Definition:
    =====================================
    The teacher quality score q_int ∈ [0, 1] measures F0 contour similarity:

        q_int = exp(-α · DTW(F0_learner, F0_ref))

    where:
    - F0_learner, F0_ref: Z-score normalized F0 contours (voiced regions only)
    - DTW: Dynamic Time Warping distance between aligned contours
    - α: Scaling factor (default 0.001, calibrated on TTS data)

    Quality Score Interpretation:
    - q_int ≈ 1.0: Nearly identical F0 contours (excellent intonation)
    - q_int ≈ 0.7-0.9: Similar prosodic patterns (good intonation)
    - q_int ≈ 0.4-0.7: Moderate similarity (acceptable)
    - q_int < 0.4: Significantly different contours (needs improvement)

    The model learns to predict q_int from learned acoustic features.

    Loss Formulation:
    =================
    L_int = ||q_pred - q_int||²  (MSE)

    or optionally:

    L_int = Huber(q_pred, q_int, δ)  (robust to outliers)

    Args:
        q_pred: Model's predicted intonation quality, shape (B,), range [0, 1]
        q_int: Teacher quality computed from DTW, shape (B,), range [0, 1]
    """

    def __init__(self, use_huber: bool = False, huber_delta: float = 1.0):
        """
        Args:
            use_huber: If True, use Huber loss instead of MSE
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        if use_huber:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self, pred_quality: torch.Tensor, teacher_quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intonation proxy loss.

        Args:
            pred_quality: Predicted quality from branch (after sigmoid), shape (B,)
                         Should be in [0, 1] range
            teacher_quality: Teacher quality scores from DTW, shape (B,)
                           Computed as exp(-α · DTW_distance)

        Returns:
            Scalar loss value
        """
        return self.loss_fn(pred_quality, teacher_quality)


# ========================================================================
# 2. RHYTHM BRANCH - Duration and Rhythm Metrics
# ========================================================================


class RhythmProxyFeatures:
    """
    Compute rhythm quality based on syllable durations and rhythm metrics.

    Teacher signal: Distance to reference rhythm statistics
    Metrics: PVI (Pairwise Variability Index), speech rate

    Calibrated beta=0.1 based on TTS experiments
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Scaling factor for rhythm distance -> quality conversion (default 0.1)
        """
        self.beta = beta

    def compute_pvi(self, durations: np.ndarray) -> float:
        """
        Compute normalized Pairwise Variability Index (nPVI).

        nPVI = 100 * (1/(m-1)) * sum_k |d_k - d_{k+1}| / ((d_k + d_{k+1})/2)

        Args:
            durations: Syllable durations, shape (N,)

        Returns:
            nPVI value (scalar)
        """
        if len(durations) < 2:
            return 0.0

        m = len(durations)
        pvi_sum = 0.0

        for k in range(m - 1):
            d_k = durations[k]
            d_k1 = durations[k + 1]
            mean_d = (d_k + d_k1) / 2.0

            if mean_d > 0:
                pvi_sum += np.abs(d_k - d_k1) / mean_d

        nPVI = 100.0 * pvi_sum / (m - 1)

        return nPVI

    def compute_speech_rate(
        self, durations: np.ndarray, total_duration: float
    ) -> float:
        """
        Compute speech rate (syllables per second).

        Args:
            durations: Syllable durations, shape (N,)
            total_duration: Total utterance duration in seconds

        Returns:
            Speech rate (syllables/sec)
        """
        n_syllables = len(durations)
        if total_duration > 0:
            rate = n_syllables / total_duration
        else:
            rate = 0.0

        return rate

    def compute_duration_ratio(
        self, durations: np.ndarray, stress_labels: np.ndarray
    ) -> float:
        """
        Compute ratio of stressed to unstressed syllable durations.

        Args:
            durations: Syllable durations, shape (N,)
            stress_labels: Binary stress labels, shape (N,), 1=stressed, 0=unstressed

        Returns:
            Duration ratio (stressed_mean / unstressed_mean)
        """
        stressed_mask = stress_labels == 1
        unstressed_mask = stress_labels == 0

        if stressed_mask.sum() == 0 or unstressed_mask.sum() == 0:
            return 1.0  # Default if one category missing

        stressed_mean = np.mean(durations[stressed_mask])
        unstressed_mean = np.mean(durations[unstressed_mask])

        if unstressed_mean > 0:
            ratio = stressed_mean / unstressed_mean
        else:
            ratio = 1.0

        return ratio

    def compute_rhythm_features(
        self,
        durations: np.ndarray,
        total_duration: float,
        stress_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute rhythm feature vector.

        Args:
            durations: Syllable durations, shape (N,)
            total_duration: Total utterance duration
            stress_labels: Optional stress labels for duration ratio

        Returns:
            Rhythm feature vector, shape (M,)
            [nPVI, speech_rate, duration_ratio (if stress_labels provided)]
        """
        features = []

        # nPVI
        nPVI = self.compute_pvi(durations)
        features.append(nPVI)

        # Speech rate
        rate = self.compute_speech_rate(durations, total_duration)
        features.append(rate)

        # Duration ratio (if stress labels provided)
        if stress_labels is not None:
            ratio = self.compute_duration_ratio(durations, stress_labels)
            features.append(ratio)

        return np.array(features)

    def compute_teacher_quality(
        self,
        durations_learner: np.ndarray,
        total_duration_learner: float,
        durations_ref: np.ndarray,
        total_duration_ref: float,
        stress_labels_learner: Optional[np.ndarray] = None,
        stress_labels_ref: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute rhythm quality score.

        Args:
            durations_learner: Learner syllable durations
            total_duration_learner: Learner total duration
            durations_ref: Reference syllable durations
            total_duration_ref: Reference total duration
            stress_labels_learner: Optional learner stress labels
            stress_labels_ref: Optional reference stress labels

        Returns:
            Quality score in [0, 1], higher is better
        """
        # Compute feature vectors
        features_learner = self.compute_rhythm_features(
            durations_learner, total_duration_learner, stress_labels_learner
        )
        features_ref = self.compute_rhythm_features(
            durations_ref, total_duration_ref, stress_labels_ref
        )

        # Compute L2 distance
        distance = np.linalg.norm(features_learner - features_ref)

        # Convert to quality
        quality = np.exp(-self.beta * distance)

        return quality


class RhythmProxyLoss(nn.Module):
    """
    Proxy loss for rhythm branch.

    Loss: MSE between predicted quality and teacher quality
    """

    def __init__(self, use_huber: bool = False, huber_delta: float = 1.0):
        super().__init__()
        self.use_huber = use_huber

        if use_huber:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self, pred_quality: torch.Tensor, teacher_quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rhythm proxy loss.

        Args:
            pred_quality: Predicted quality, shape (B,)
            teacher_quality: Teacher quality, shape (B,)

        Returns:
            Scalar loss
        """
        return self.loss_fn(pred_quality, teacher_quality)


# ========================================================================
# 3. PROMINENCE BRANCH - Syllable Prominence Ranking
# ========================================================================


class ProminenceProxyFeatures:
    """
    Compute prominence teacher signals from lexical stress and acoustic cues.

    Teacher signal: Per-syllable prominence scores
    Loss: Pairwise ranking + optional accent classification
    """

    def __init__(
        self,
        f0_weight: float = 0.4,
        intensity_weight: float = 0.3,
        duration_weight: float = 0.3,
    ):
        """
        Args:
            f0_weight: Weight for F0 cue in prominence
            intensity_weight: Weight for intensity cue
            duration_weight: Weight for duration cue
        """
        self.f0_weight = f0_weight
        self.intensity_weight = intensity_weight
        self.duration_weight = duration_weight

    def compute_prominence_scores(
        self,
        f0_peaks: np.ndarray,
        intensities: np.ndarray,
        durations: np.ndarray,
        lexical_stress: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-syllable prominence scores.

        Args:
            f0_peaks: F0 peak values per syllable, shape (N,)
            intensities: Mean intensity per syllable, shape (N,)
            durations: Syllable durations, shape (N,)
            lexical_stress: Binary lexical stress labels, shape (N,)

        Returns:
            Prominence scores, shape (N,)
        """
        # Normalize acoustic features to [0, 1]
        f0_norm = self._normalize(f0_peaks)
        intensity_norm = self._normalize(intensities)
        duration_norm = self._normalize(durations)

        # Weighted combination of acoustic cues
        acoustic_score = (
            self.f0_weight * f0_norm
            + self.intensity_weight * intensity_norm
            + self.duration_weight * duration_norm
        )

        # Combine with lexical stress (boosting effect)
        # Lexical stress adds a baseline prominence
        prominence = acoustic_score + lexical_stress * 0.5

        return prominence

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val > min_val:
            return (values - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(values)

    def generate_pairwise_preferences(
        self, prominence_scores: np.ndarray, margin: float = 0.1
    ) -> List[Tuple[int, int]]:
        """
        Generate pairwise preference pairs where i should be > j.

        Args:
            prominence_scores: Teacher prominence scores, shape (N,)
            margin: Minimum difference to create a preference pair

        Returns:
            List of (i, j) tuples where prominence[i] > prominence[j]
        """
        n = len(prominence_scores)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                diff = prominence_scores[i] - prominence_scores[j]
                if diff > margin:
                    pairs.append((i, j))
                elif diff < -margin:
                    pairs.append((j, i))

        return pairs


class ProminenceProxyLoss(nn.Module):
    """
    Proxy loss for prominence branch.

    Loss: Pairwise hinge loss + optional accent classification
    """

    def __init__(
        self,
        margin: float = 1.0,
        use_accent_loss: bool = True,
        accent_weight: float = 0.5,
    ):
        """
        Args:
            margin: Margin for pairwise hinge loss
            use_accent_loss: Whether to include accent classification loss
            accent_weight: Weight for accent classification loss (gamma)
        """
        super().__init__()
        self.margin = margin
        self.use_accent_loss = use_accent_loss
        self.accent_weight = accent_weight

    def pairwise_ranking_loss(
        self, pred_scores: torch.Tensor, pairs: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compute pairwise ranking hinge loss.

        Args:
            pred_scores: Predicted prominence scores, shape (N,)
            pairs: List of (i, j) where i should be > j

        Returns:
            Scalar loss
        """
        if len(pairs) == 0:
            return torch.tensor(0.0, device=pred_scores.device)

        loss = 0.0
        for i, j in pairs:
            # We want pred_scores[i] > pred_scores[j] by margin
            loss += torch.clamp(
                self.margin - (pred_scores[i] - pred_scores[j]), min=0.0
            )

        loss = loss / len(pairs)

        return loss

    def accent_classification_loss(
        self, pred_logits: torch.Tensor, accent_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy for accent classification.

        Args:
            pred_logits: Predicted accent logits, shape (N,)
            accent_labels: Binary accent labels, shape (N,)

        Returns:
            Scalar loss
        """
        return F.binary_cross_entropy_with_logits(pred_logits, accent_labels)

    def forward(
        self,
        pred_scores: torch.Tensor,
        pairs: List[Tuple[int, int]],
        pred_accent_logits: Optional[torch.Tensor] = None,
        accent_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total prominence proxy loss.

        Args:
            pred_scores: Predicted prominence scores, shape (N,)
            pairs: Pairwise preference pairs
            pred_accent_logits: Optional accent logits, shape (N,)
            accent_labels: Optional accent labels, shape (N,)

        Returns:
            Total loss
        """
        # Pairwise ranking loss
        ranking_loss = self.pairwise_ranking_loss(pred_scores, pairs)

        total_loss = ranking_loss

        # Optional accent classification loss
        if (
            self.use_accent_loss
            and pred_accent_logits is not None
            and accent_labels is not None
        ):
            accent_loss = self.accent_classification_loss(
                pred_accent_logits, accent_labels
            )
            total_loss = total_loss + self.accent_weight * accent_loss

        return total_loss


# ========================================================================
# 4. BOUNDARY BRANCH - Prosodic Boundary Detection
# ========================================================================


class BoundaryProxyFeatures:
    """
    Compute prosodic boundary teacher signals.

    Teacher signal: Break indices at word boundaries
    Classes: 0 (no break), 1 (minor break), 2 (major break), etc.
    """

    def __init__(
        self,
        n_classes: int = 3,
        pause_threshold_minor: float = 0.1,
        pause_threshold_major: float = 0.3,
    ):
        """
        Args:
            n_classes: Number of break index classes
            pause_threshold_minor: Pause duration threshold for minor break (sec)
            pause_threshold_major: Pause duration threshold for major break (sec)
        """
        self.n_classes = n_classes
        self.pause_threshold_minor = pause_threshold_minor
        self.pause_threshold_major = pause_threshold_major

    def compute_break_indices(
        self, pause_durations: np.ndarray, f0_resets: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute break indices from pause durations and F0 resets.

        Args:
            pause_durations: Pause duration at each boundary position, shape (M,)
            f0_resets: Optional F0 reset indicators, shape (M,)

        Returns:
            Break indices, shape (M,), values in [0, n_classes-1]
        """
        break_indices = np.zeros(len(pause_durations), dtype=np.int64)

        for i, pause in enumerate(pause_durations):
            if pause < self.pause_threshold_minor:
                # No break
                break_indices[i] = 0
            elif pause < self.pause_threshold_major:
                # Minor break
                break_indices[i] = 1
            else:
                # Major break
                break_indices[i] = 2

        # Optionally adjust based on F0 resets
        if f0_resets is not None:
            for i, reset in enumerate(f0_resets):
                if reset and break_indices[i] == 0:
                    # F0 reset suggests at least a minor break
                    break_indices[i] = 1

        # Clip to valid range
        break_indices = np.clip(break_indices, 0, self.n_classes - 1)

        return break_indices


class BoundaryProxyLoss(nn.Module):
    """
    Proxy loss for boundary branch.

    Loss: Cross-entropy + optional pause duration deviation penalty
    """

    def __init__(self, use_pause_loss: bool = True, pause_weight: float = 0.1):
        """
        Args:
            use_pause_loss: Whether to include pause duration penalty
            pause_weight: Weight for pause deviation loss (eta)
        """
        super().__init__()
        self.use_pause_loss = use_pause_loss
        self.pause_weight = pause_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def pause_duration_loss(
        self,
        pred_pauses: torch.Tensor,
        ref_pauses: torch.Tensor,
        break_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 loss on pause durations for positions with breaks.

        Args:
            pred_pauses: Predicted pause durations, shape (M,)
            ref_pauses: Reference pause durations, shape (M,)
            break_mask: Binary mask for break positions, shape (M,)

        Returns:
            Scalar loss
        """
        if break_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_pauses.device)

        # Only compute loss at break positions
        loss = torch.abs(pred_pauses[break_mask] - ref_pauses[break_mask]).mean()

        return loss

    def forward(
        self,
        pred_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
        pred_pauses: Optional[torch.Tensor] = None,
        ref_pauses: Optional[torch.Tensor] = None,
        break_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total boundary proxy loss.

        Args:
            pred_logits: Predicted break logits, shape (M, K)
            teacher_labels: Teacher break indices, shape (M,)
            pred_pauses: Optional predicted pause durations, shape (M,)
            ref_pauses: Optional reference pause durations, shape (M,)
            break_mask: Optional mask for break positions, shape (M,)

        Returns:
            Total loss
        """
        # Cross-entropy loss for break classification
        ce_loss = self.ce_loss(pred_logits, teacher_labels)

        total_loss = ce_loss

        # Optional pause duration loss
        if (
            self.use_pause_loss
            and pred_pauses is not None
            and ref_pauses is not None
            and break_mask is not None
        ):
            pause_loss = self.pause_duration_loss(pred_pauses, ref_pauses, break_mask)
            total_loss = total_loss + self.pause_weight * pause_loss

        return total_loss


# ========================================================================
# 5. COMBINED PROXY LOSS
# ========================================================================


class CombinedProxyLoss(nn.Module):
    """
    Combined proxy loss for all prosodic branches.

    L_proxy = λ_int * L_int + λ_rhy * L_rhy + λ_pro * L_pro + λ_bnd * L_bnd
    """

    def __init__(
        self,
        lambda_int: float = 1.0,
        lambda_rhy: float = 1.0,
        lambda_pro: float = 1.0,
        lambda_bnd: float = 1.0,
        intonation_loss_kwargs: Optional[Dict] = None,
        rhythm_loss_kwargs: Optional[Dict] = None,
        prominence_loss_kwargs: Optional[Dict] = None,
        boundary_loss_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            lambda_int: Weight for intonation loss
            lambda_rhy: Weight for rhythm loss
            lambda_pro: Weight for prominence loss
            lambda_bnd: Weight for boundary loss
            *_loss_kwargs: Keyword arguments for each loss module
        """
        super().__init__()

        self.lambda_int = lambda_int
        self.lambda_rhy = lambda_rhy
        self.lambda_pro = lambda_pro
        self.lambda_bnd = lambda_bnd

        # Initialize individual loss modules
        self.intonation_loss = IntonationProxyLoss(**(intonation_loss_kwargs or {}))
        self.rhythm_loss = RhythmProxyLoss(**(rhythm_loss_kwargs or {}))
        self.prominence_loss = ProminenceProxyLoss(**(prominence_loss_kwargs or {}))
        self.boundary_loss = BoundaryProxyLoss(**(boundary_loss_kwargs or {}))

    def forward(
        self,
        # Intonation inputs
        pred_int_quality: Optional[torch.Tensor] = None,
        teacher_int_quality: Optional[torch.Tensor] = None,
        # Rhythm inputs
        pred_rhy_quality: Optional[torch.Tensor] = None,
        teacher_rhy_quality: Optional[torch.Tensor] = None,
        # Prominence inputs
        pred_pro_scores: Optional[torch.Tensor] = None,
        pro_pairs: Optional[List[Tuple[int, int]]] = None,
        pred_accent_logits: Optional[torch.Tensor] = None,
        accent_labels: Optional[torch.Tensor] = None,
        # Boundary inputs
        pred_bnd_logits: Optional[torch.Tensor] = None,
        teacher_bnd_labels: Optional[torch.Tensor] = None,
        pred_pauses: Optional[torch.Tensor] = None,
        ref_pauses: Optional[torch.Tensor] = None,
        break_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined proxy loss.

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # Intonation loss
        if pred_int_quality is not None and teacher_int_quality is not None:
            loss_int = self.intonation_loss(pred_int_quality, teacher_int_quality)
            losses["intonation"] = loss_int
            total_loss += self.lambda_int * loss_int

        # Rhythm loss
        if pred_rhy_quality is not None and teacher_rhy_quality is not None:
            loss_rhy = self.rhythm_loss(pred_rhy_quality, teacher_rhy_quality)
            losses["rhythm"] = loss_rhy
            total_loss += self.lambda_rhy * loss_rhy

        # Prominence loss
        if pred_pro_scores is not None and pro_pairs is not None:
            loss_pro = self.prominence_loss(
                pred_pro_scores, pro_pairs, pred_accent_logits, accent_labels
            )
            losses["prominence"] = loss_pro
            total_loss += self.lambda_pro * loss_pro

        # Boundary loss
        if pred_bnd_logits is not None and teacher_bnd_labels is not None:
            loss_bnd = self.boundary_loss(
                pred_bnd_logits, teacher_bnd_labels, pred_pauses, ref_pauses, break_mask
            )
            losses["boundary"] = loss_bnd
            total_loss += self.lambda_bnd * loss_bnd

        losses["total"] = total_loss

        return losses


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================


def extract_f0_contour(
    audio_input: Union[str, torch.Tensor, np.ndarray],
    sr: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Extract F0 contour from audio file or array using autocorrelation.

    Args:
        audio_input: Path to audio file, or audio tensor/array
        sr: Sample rate to resample to (if input is path) or assumed sample rate
        hop_length: Hop length in samples

    Returns:
        F0 contour array, zeros indicate unvoiced frames
    """
    if isinstance(audio_input, str):
        waveform, sample_rate = torchaudio.load(audio_input)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
        audio = waveform.squeeze().numpy()
    elif isinstance(audio_input, torch.Tensor):
        audio = audio_input.squeeze().cpu().numpy()
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
    else:
        raise ValueError("audio_input must be str, torch.Tensor, or np.ndarray")

    frame_length = 400  # 25ms at 16kHz

    f0_frames = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i : i + frame_length]

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]

        # Find peak in pitch range (80-400 Hz)
        min_lag = int(sr / 400)
        max_lag = int(sr / 80)

        if max_lag < len(corr):
            peak_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
            f0 = sr / peak_lag if peak_lag > 0 else 0
            f0_frames.append(f0)
        else:
            f0_frames.append(0)

    return np.array(f0_frames)


def extract_syllable_features(
    audio_input: Union[str, torch.Tensor, np.ndarray],
    alignment_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract syllable-level acoustic features.

    If alignment_path is None, uses energy-based segmentation to estimate syllables.

    Args:
        audio_input: Path to audio file, or audio tensor/array
        alignment_path: Path to forced alignment file (optional)

    Returns:
        Dictionary with:
            - durations: Syllable durations
            - f0_peaks: F0 peak per syllable
            - intensities: Mean intensity per syllable
            - pause_durations: Pause after each syllable
    """
    sr = 16000
    if isinstance(audio_input, str):
        waveform, sample_rate = torchaudio.load(audio_input)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
        audio = waveform.squeeze().numpy()
        f0_input = audio
    elif isinstance(audio_input, torch.Tensor):
        audio = audio_input.squeeze().cpu().numpy()
        f0_input = audio
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        f0_input = audio
    else:
        raise ValueError("audio_input must be str, torch.Tensor, or np.ndarray")

    # Extract F0
    f0_contour = extract_f0_contour(f0_input, sr=sr)

    # Compute energy/intensity
    frame_length = 400  # 25ms
    hop_length = 160  # 10ms

    n_frames = len(f0_contour)
    intensities = []

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            break
        frame = audio[start:end]
        energy = np.sum(frame**2)
        intensities.append(energy)

    intensities = np.array(intensities)

    # Syllable segmentation (Energy based)
    # Note: This is a heuristic fallback when alignment is not available
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    # Smooth energy
    energy_smooth = gaussian_filter1d(intensities, sigma=3)

    # Find peaks (syllables)
    peaks, _ = find_peaks(
        energy_smooth, distance=10, prominence=np.max(energy_smooth) * 0.05
    )

    # Find valleys (boundaries)
    valleys = []
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i + 1]
        # Find minimum between peaks
        valley = p1 + np.argmin(energy_smooth[p1:p2])
        valleys.append(valley)

    # Define segments
    segments = []
    if len(peaks) > 0:
        # First segment
        start = 0
        end = valleys[0] if len(valleys) > 0 else len(energy_smooth)
        segments.append((start, end))

        # Middle segments
        for i in range(len(valleys) - 1):
            segments.append((valleys[i], valleys[i + 1]))

        # Last segment
        if len(valleys) > 0:
            segments.append((valleys[-1], len(energy_smooth)))

    # Extract features
    syll_durations = []
    syll_f0_peaks = []
    syll_intensities = []
    syll_pauses = []

    for start, end in segments:
        # Duration
        dur_sec = (end - start) * hop_length / sr
        syll_durations.append(dur_sec)

        # F0 Peak
        seg_f0 = f0_contour[start:end]
        if len(seg_f0) > 0 and np.max(seg_f0) > 0:
            syll_f0_peaks.append(np.max(seg_f0))
        else:
            syll_f0_peaks.append(0.0)

        # Mean Intensity
        seg_int = intensities[start:end]
        if len(seg_int) > 0:
            syll_intensities.append(np.mean(seg_int))
        else:
            syll_intensities.append(0.0)

        # Pause (placeholder 0 for contiguous segments)
        syll_pauses.append(0.0)

    # Convert segments to seconds
    segments_sec = [(s * hop_length / sr, e * hop_length / sr) for s, e in segments]

    return {
        "durations": np.array(syll_durations),
        "f0_peaks": np.array(syll_f0_peaks),
        "intensities": np.array(syll_intensities),
        "pause_durations": np.array(syll_pauses),
        "segments": segments_sec,
    }


def generate_proxy_scores(
    audio: torch.Tensor,
    reference_audio: torch.Tensor,
) -> Dict[str, float]:
    """
    Generate proxy feature scores measuring prosodic quality.
    
    Strategy: Combine TTS-similarity with intrinsic quality metrics.
    Good prosody should be:
    1. Similar to native-like TTS patterns (but not identical)
    2. Have natural F0 variation (not monotone)
    3. Have appropriate rhythm patterns (not too fast/slow)
    4. Have clear prominence patterns (not flat)

    Args:
        audio: Audio tensor (learner)
        reference_audio: Reference audio tensor (TTS, represents native baseline)
    Returns:
        Dictionary with three proxy scores [0, 1] (higher = better prosody):
            - intonation_score: F0 naturalness + TTS similarity
            - rhythm_score: Rhythm appropriateness + TTS similarity
            - prominence_score: Prominence clarity + TTS similarity
    """
    # Initialize extractors with balanced parameters
    intonation_extractor = IntonationProxyFeatures(alpha=0.005)
    rhythm_extractor = RhythmProxyFeatures(beta=0.2)
    prominence_extractor = ProminenceProxyFeatures()

    sr = 16000

    # === 1. INTONATION SCORE ===
    # Combines: (1) TTS similarity, (2) F0 variation quality
    f0_learner = extract_f0_contour(audio, sr=sr)
    f0_ref = extract_f0_contour(reference_audio, sr=sr)

    # A) TTS similarity component
    tts_similarity = intonation_extractor.compute_teacher_quality(f0_learner, f0_ref)
    
    # B) Intrinsic F0 quality: variation (good) vs monotone (bad)
    voiced_f0 = f0_learner[f0_learner > 0]
    if len(voiced_f0) > 10:
        # Coefficient of variation: std/mean (normalized variation)
        f0_variation = np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-8)
        # Map to [0, 1]: good variation is around 0.1-0.3
        f0_quality = 1.0 - np.exp(-5.0 * f0_variation)  # sigmoid-like
        f0_quality = np.clip(f0_quality, 0, 1)
    else:
        f0_quality = 0.0
    
    # Combine: 60% TTS similarity + 40% intrinsic quality
    intonation_score = 0.6 * tts_similarity + 0.4 * f0_quality

    # === 2. RHYTHM SCORE ===
    # Combines: (1) TTS similarity, (2) rhythm naturalness
    syll_learner = extract_syllable_features(audio)
    syll_ref = extract_syllable_features(reference_audio)

    dur_learner = len(audio.squeeze()) / sr
    dur_ref = len(reference_audio.squeeze()) / sr

    # A) TTS similarity component
    tts_rhythm_similarity = rhythm_extractor.compute_teacher_quality(
        syll_learner["durations"], dur_learner, syll_ref["durations"], dur_ref
    )
    
    # B) Intrinsic rhythm quality: natural variation in syllable durations
    if len(syll_learner["durations"]) > 2:
        # nPVI: Pairwise Variability Index (natural speech has moderate PVI ~40-60)
        pvi = rhythm_extractor.compute_pvi(syll_learner["durations"])
        # Map PVI to quality: too low (monotone) or too high (erratic) is bad
        # Optimal PVI around 40-60 for English
        pvi_quality = np.exp(-0.001 * (pvi - 50)**2)  # Gaussian centered at 50
        pvi_quality = np.clip(pvi_quality, 0, 1)
    else:
        pvi_quality = 0.5
    
    # Combine: 50% TTS similarity + 50% intrinsic rhythm quality
    rhythm_score = 0.5 * tts_rhythm_similarity + 0.5 * pvi_quality

    # === 3. PROMINENCE SCORE ===
    # Combines: (1) TTS pattern similarity, (2) prominence clarity
    ref_stress = np.zeros_like(syll_ref["durations"])
    ref_prominence = prominence_extractor.compute_prominence_scores(
        syll_ref["f0_peaks"],
        syll_ref["intensities"],
        syll_ref["durations"],
        ref_stress,
    )

    learner_stress = np.zeros_like(syll_learner["durations"])
    learner_prominence = prominence_extractor.compute_prominence_scores(
        syll_learner["f0_peaks"],
        syll_learner["intensities"],
        syll_learner["durations"],
        learner_stress,
    )

    # A) TTS pattern similarity
    if len(learner_prominence) == len(ref_prominence) and len(learner_prominence) > 1:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(learner_prominence, ref_prominence)
        pattern_similarity = float((corr + 1) / 2)  # [-1,1] -> [0,1]
    else:
        if len(learner_prominence) > 0 and len(ref_prominence) > 0:
            min_len = min(len(learner_prominence), len(ref_prominence))
            learner_norm = learner_prominence[:min_len]
            ref_norm = ref_prominence[:min_len]
            # Normalize to [0,1]
            if learner_norm.max() > learner_norm.min():
                learner_norm = (learner_norm - learner_norm.min()) / (learner_norm.max() - learner_norm.min())
            if ref_norm.max() > ref_norm.min():
                ref_norm = (ref_norm - ref_norm.min()) / (ref_norm.max() - ref_norm.min())
            l2_dist = np.linalg.norm(learner_norm - ref_norm)
            pattern_similarity = float(np.exp(-0.5 * l2_dist))
        else:
            pattern_similarity = 0.5
    
    # B) Intrinsic prominence clarity: dynamic range (clear stress vs flat)
    if len(learner_prominence) > 1:
        prominence_range = np.max(learner_prominence) - np.min(learner_prominence)
        # Good prominence has clear contrasts (range > 0.3)
        clarity = 1.0 - np.exp(-3.0 * prominence_range)
        clarity = np.clip(clarity, 0, 1)
    else:
        clarity = 0.0
    
    # Combine: 50% TTS pattern + 50% intrinsic clarity
    prominence_score = 0.5 * pattern_similarity + 0.5 * clarity

    return {
        "intonation_score": float(np.clip(intonation_score, 0, 1)),
        "rhythm_score": float(np.clip(rhythm_score, 0, 1)),
        "prominence_score": float(np.clip(prominence_score, 0, 1)),
    }
