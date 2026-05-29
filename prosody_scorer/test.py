import numpy as np
from argparse import ArgumentParser
from datasets import load_dataset
import os
import pickle
import re
import joblib
import torch
import torchaudio
import promonet
from prosody_scorer.models import (
    ClusterScorer,
    NonClusterScorer,
    TransformerScorer,
    FDMPAScorer,
)
from scipy.stats import spearmanr

from tqdm import tqdm


def load_file(path):
    file = np.loadtxt(path, delimiter=",", dtype=str)
    return file


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset from huggingface (e.g., L2Arctic, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default="ClusterScorer", help="name of the model"
    )
    parser.add_argument(
        "--kmeans_model", type=str, help="path to the trained kmeans model"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="path to the trained model checkpoint"
    )
    parser.add_argument(
        "--aspect",
        nargs="+",
        default=["fluency"],
        help="aspect to evaluate (e.g., fluency)",
    )
    # Sliding window options (ms). window_ms=0 disables sliding (use full utterance)
    parser.add_argument(
        "--window-ms",
        type=int,
        default=5000,
        help="window length in milliseconds for sliding-window inference (0 = full utterance)",
    )
    parser.add_argument(
        "--hop-ms",
        type=int,
        default=4000,
        help="hop length in milliseconds for sliding-window inference (0 -> equals window-ms)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="hidden dim for FDMPAScorer (must match checkpoint)",
    )
    parser.add_argument(
        "--fdmpa-num-tokens",
        type=int,
        default=-1,
        help="number of tokens per FDMPA branch (-1 = adaptive)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="dataset split to evaluate (train/test)",
    )
    args = parser.parse_args()
    return args


def valid_predictions(audio_output, target):
    """
    Validate predictions, supporting multiple aspects.

    Args:
        audio_output: (batch_size, num_aspects) or (batch_size, 1)
        target: (batch_size, num_aspects) or (batch_size, 1)

    Returns:
        mse: average MSE across all aspects
        pcc: average Pearson correlation across all aspects
        spc: average Spearman correlation across all aspects
        mse_list: list of MSE for each aspect (always a list)
        pcc_list: list of Pearson correlation for each aspect (always a list)
        spc_list: list of Spearman correlation for each aspect (always a list)
    """
    mse_list = []
    pcc_list = []
    spc_list = []

    # Handle both single and multi-aspect cases
    num_aspects = audio_output.shape[1] if audio_output.dim() == 2 else 1

    # Calculate MSE, Pearson correlation, and Spearman correlation for each aspect
    for i in range(num_aspects):
        if num_aspects == 1:
            pred = audio_output.view(-1).numpy()
            tgt = target.view(-1).numpy()
        else:
            pred = audio_output[:, i].numpy()
            tgt = target[:, i].numpy()

        aspect_mse = np.mean((pred - tgt) ** 2)

        # Pearson correlation
        corr_matrix = np.corrcoef(pred, tgt)
        aspect_pcc = corr_matrix[0, 1].item()

        # Spearman correlation
        aspect_spc, _ = spearmanr(pred, tgt)

        mse_list.append(aspect_mse)
        pcc_list.append(aspect_pcc)
        spc_list.append(aspect_spc)

    # Return average metrics across all aspects, plus individual lists
    valid_token_mse = np.mean(mse_list)
    avg_pcc = np.mean(pcc_list)
    avg_spc = np.mean(spc_list)

    return valid_token_mse, avg_pcc, avg_spc, mse_list, pcc_list, spc_list


def main():
    args = get_arguments()

    # Handcrafted feature cache (for FDMPA testing)
    hc_cache = {}
    hc_cache_path = None
    hc_cache_dirty = False
    if args.model == "FDMPAScorer":
        dataset_leaf = args.dataset.split("/")[-1]
        safe_dataset = re.sub(r"[^a-zA-Z0-9_.-]", "_", dataset_leaf)
        prefix = "tr" if getattr(args, "split", "train") == "train" else "te"
        cache_dir = os.path.join("data", safe_dataset)
        os.makedirs(cache_dir, exist_ok=True)
        hc_cache_path = os.path.join(cache_dir, f"{prefix}_handcrafted_feats.pkl")
        if os.path.exists(hc_cache_path):
            with open(hc_cache_path, "rb") as f:
                hc_cache = pickle.load(f)
            print(f"Loaded handcrafted cache from: {hc_cache_path}")
        else:
            print(f"No handcrafted cache found. Will create: {hc_cache_path}")

    # Load the test data
    data = load_dataset(args.dataset, split=getattr(args, "split", "train"))

    # Load the model
    if args.model == "NonClusterScorer":
        audio_model = NonClusterScorer(
            input_dim=1024, embed_dim=32, scorers=args.aspect
        )
    elif args.model == "TransformerScorer":
        audio_model = TransformerScorer(num_heads=3)
    elif args.model == "ClusterScorer":
        audio_model = ClusterScorer(
            input_dim=1024, embed_dim=32, clustering_dim=6, scorers=args.aspect
        )
    elif args.model == "FDMPAScorer":
        audio_model = FDMPAScorer(
            ssl_input_dim=1024,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
            num_tokens=args.fdmpa_num_tokens,
        )
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    sd = torch.load(args.checkpoint, map_location="cpu")
    audio_model.load_state_dict(sd, strict=True)

    # Load the kmeans model only when needed (FDMPAScorer does not use kmeans)
    kmeans_model = None
    if args.model != "FDMPAScorer":
        if not getattr(args, "kmeans_model", None):
            raise ValueError("kmeans_model is required for non-FDMPAScorer models")
        kmeans_model = joblib.load(args.kmeans_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    # Inform penn about preferred GPU (if any)
    try:
        import penn as _penn
        if device.type == 'cuda':
            _penn._DEFAULT_GPU = device.index if device.index is not None else 0
        else:
            _penn._DEFAULT_GPU = None
    except Exception:
        pass

    feature_extractor = torchaudio.pipelines.HUBERT_LARGE.get_model()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    predictions = []

    # Helper: sliding windows over waveform
    # Returns list of tuples: (segment_waveform_1d, start_sample, end_sample)
    def make_windows(waveform_1d: torch.Tensor, sr: int, window_ms: int, hop_ms: int):
        # waveform_1d: [N]
        N = waveform_1d.shape[0]
        if window_ms <= 0:
            return [(waveform_1d, 0, N)]
        win = int(window_ms * sr / 1000)
        hop = int(hop_ms * sr / 1000) if hop_ms > 0 else win
        if win <= 0:
            return [(waveform_1d, 0, N)]
        windows = []
        start = 0
        while start < N:
            end = start + win
            if end <= N:
                seg = waveform_1d[start:end]
            else:
                # pad
                pad = torch.zeros(
                    end - N, dtype=waveform_1d.dtype, device=waveform_1d.device
                )
                seg = torch.cat((waveform_1d[start:N], pad), dim=0)
            windows.append((seg, start, end))
            if end >= N:
                break
            start += hop
        return windows

    def _to_time_major(x, expected_channels=None):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.detach().cpu().float().squeeze()
        if x.ndim == 0:
            return None
        if x.ndim == 1:
            return x.unsqueeze(-1)
        if x.ndim == 2:
            c0, c1 = x.shape[0], x.shape[1]
            if expected_channels is not None and c0 == expected_channels:
                return x.transpose(0, 1)
            if expected_channels is not None and c1 == expected_channels:
                return x
            return x if c0 > c1 else x.transpose(0, 1)
        return x.reshape(x.shape[-1], -1)

    def _align_or_zero(x, channels, target_len):
        if x is None:
            return torch.zeros(target_len, channels, dtype=torch.float32)
        if x.shape[1] != channels:
            if x.shape[1] > channels:
                x = x[:, :channels]
            else:
                x = torch.cat(
                    [x, torch.zeros(x.shape[0], channels - x.shape[1], dtype=x.dtype)],
                    dim=1,
                )
        L = x.shape[0]
        if L == target_len:
            return x
        if L == 1:
            return x.repeat(target_len, 1)
        x_ = x.transpose(0, 1).unsqueeze(0)  # (1, C, L)
        x_up = torch.nn.functional.interpolate(
            x_, size=target_len, mode="linear", align_corners=False
        )
        return x_up.squeeze(0).transpose(0, 1)

    def _extract_full_hc(wav_1d_cpu: torch.Tensor, sr: int):
        try:
            hc_tuple = promonet.preprocess.from_audio(
                wav_1d_cpu.unsqueeze(0),
                sample_rate=sr,
                features=["loudness", "pitch", "periodicity", "ppg"],
            )
        except AttributeError as e:
            if "'int' object has no attribute 'device'" not in str(e):
                raise RuntimeError(f"promonet preprocessing failed: {e}")
            hc_tuple_no_ppg = promonet.preprocess.from_audio(
                wav_1d_cpu.unsqueeze(0),
                sample_rate=sr,
                features=["loudness", "pitch", "periodicity"],
            )
            hc_tuple = (*hc_tuple_no_ppg, None)
        except Exception as e:
            raise RuntimeError(f"promonet preprocessing failed: {e}")

        loudness = hc_tuple[0] if len(hc_tuple) > 0 else None
        pitch = hc_tuple[1] if len(hc_tuple) > 1 else None
        periodicity = hc_tuple[2] if len(hc_tuple) > 2 else None
        ppg = hc_tuple[3] if len(hc_tuple) > 3 else None

        loudness_tm = _to_time_major(
            loudness, expected_channels=promonet.LOUDNESS_BANDS
        )
        pitch_tm = _to_time_major(pitch)
        periodicity_tm = _to_time_major(periodicity)
        ppg_tm = _to_time_major(ppg, expected_channels=promonet.PPG_CHANNELS)

        avail = [
            x for x in [loudness_tm, pitch_tm, periodicity_tm, ppg_tm] if x is not None
        ]
        target_len = max(x.shape[0] for x in avail) if len(avail) > 0 else 1

        loudness_aligned = _align_or_zero(
            loudness_tm, promonet.LOUDNESS_BANDS, target_len
        )
        pitch_aligned = _align_or_zero(pitch_tm, 1, target_len)
        periodicity_aligned = _align_or_zero(periodicity_tm, 1, target_len)
        ppg_aligned = _align_or_zero(ppg_tm, promonet.PPG_CHANNELS, target_len)

        return torch.cat(
            [loudness_aligned, pitch_aligned, periodicity_aligned, ppg_aligned], dim=1
        )

    # Run inference (with optional sliding-window)
    for sample_idx, item in enumerate(tqdm(data)):
        audio = item["audio"]
        target = [item[aspect] for aspect in args.aspect]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        # Apply same scaling as used during training (from train.py BaseDataset)
        target = target * 0.2

        # waveform array from datasets audio feature
        array = audio["array"]
        sr = int(audio["sampling_rate"])
        wav = torch.tensor(array, dtype=torch.float32).to(device)
        # Ensure mono 1D waveform for windowing/inference.
        # HuggingFace audio arrays may be shaped (channels, time) or (time, channels).
        if wav.dim() == 2:
            channel_dim = 0 if wav.shape[0] <= wav.shape[1] else 1
            wav = wav.mean(dim=channel_dim)

        window_ms = getattr(args, "window_ms", 5000)
        hop_ms = getattr(args, "hop_ms", 4000)
        windows = make_windows(wav, sr, window_ms, hop_ms)
        audio_id = str(item.get("id", f"sample_{sample_idx}"))

        hc_full = None
        N = wav.shape[0]
        if args.model == "FDMPAScorer":
            full_cache_key = f"{audio_id}__full_hc"
            hc_full = hc_cache.get(full_cache_key, None)
            if hc_full is None:
                # Extract handcrafted features once per utterance (much faster than per-window extraction)
                hc_full = _extract_full_hc(wav.detach().cpu(), sr)
                hc_cache[full_cache_key] = hc_full.detach().cpu()
                hc_cache_dirty = True

        window_preds = []
        with torch.no_grad():
            for w, start, end in windows:
                # HuBERT expects shape (batch, time)
                w_batch = w.unsqueeze(0)
                audio_embedding, _ = feature_extractor.extract_features(w_batch)
                features = audio_embedding[14]
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                B, T, D = features.shape
                if args.model == "FDMPAScorer":
                    L = max(1, hc_full.shape[0])
                    start_eff = min(start, max(0, N - 1))
                    end_eff = min(max(start_eff + 1, end), N)
                    if N <= 1 or L <= 1:
                        hc_seg = hc_full[:1]
                    else:
                        start_f = int(round(start_eff * (L - 1) / (N - 1)))
                        end_f = int(round((end_eff - 1) * (L - 1) / (N - 1))) + 1
                        start_f = max(0, min(start_f, L - 1))
                        end_f = max(start_f + 1, min(end_f, L))
                        hc_seg = hc_full[start_f:end_f]

                    if hc_seg.shape[0] == T:
                        hc_resampled = hc_seg
                    elif hc_seg.shape[0] == 1:
                        hc_resampled = hc_seg.repeat(T, 1)
                    else:
                        hct = hc_seg.transpose(0, 1).unsqueeze(0)
                        hct_up = torch.nn.functional.interpolate(
                            hct, size=T, mode="linear", align_corners=False
                        )
                        hc_resampled = hct_up.squeeze(0).transpose(0, 1)

                    hc_feats = hc_resampled.unsqueeze(0).to(device)
                    pred = audio_model(features, hc_feats)
                else:
                    flat_features = features.reshape(-1, D).cpu().numpy()
                    if kmeans_model is None:
                        raise ValueError(
                            "kmeans_model must be provided for non-FDMPAScorer models"
                        )
                    cluster_ids_np = kmeans_model.predict(flat_features)
                    cluster_ids = (
                        torch.tensor(cluster_ids_np, dtype=torch.long)
                        .reshape(B, T)
                        .to(device)
                    )
                    pred = audio_model(features, cluster_ids)

                # Some models (e.g., FDMPAScorer) return (pred, aux)
                if isinstance(pred, tuple):
                    pred = pred[0]

                # pred: (1, num_aspects) or (1,) -> ensure 2D
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                window_preds.append(pred.cpu())

        # Aggregate window predictions
        if len(window_preds) == 0:
            # fallback: zero prediction
            agg_pred = torch.zeros(1, len(args.aspect), dtype=torch.float32)
        else:
            agg_pred = torch.mean(torch.cat(window_preds, dim=0), dim=0, keepdim=True)

        predictions.append((agg_pred, target))

    # Validate predictions
    all_preds = torch.cat([p[0] for p in predictions], dim=0)
    all_targets = torch.cat([p[1] for p in predictions], dim=0)

    avg_mse, avg_pcc, avg_spc, mse_list, pcc_list, spc_list = valid_predictions(
        all_preds, all_targets
    )

    for i, aspect in enumerate(args.aspect):
        print(
            f"Aspect: {aspect} - MSE: {mse_list[i]:.4f}, PCC: {pcc_list[i]:.4f}, SPC: {spc_list[i]:.4f}"
        )
    print(
        f"Average MSE: {avg_mse:.4f}, Average PCC: {avg_pcc:.4f}, Average SPC: {avg_spc:.4f}"
    )

    if args.model == "FDMPAScorer" and hc_cache_path is not None and hc_cache_dirty:
        with open(hc_cache_path, "wb") as f:
            pickle.dump(hc_cache, f)
        print(f"Saved handcrafted cache to: {hc_cache_path}")


if __name__ == "__main__":
    main()
