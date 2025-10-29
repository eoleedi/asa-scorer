import numpy as np
from argparse import ArgumentParser
from datasets import load_dataset
import joblib
import torch
import torchaudio
from models import ClusterScorer, NonClusterScorer, TransformerScorer

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
        corr: average correlation across all aspects
        mse_list: list of MSE for each aspect (always a list)
        corr_list: list of correlation for each aspect (always a list)
    """
    mse_list = []
    corr_list = []

    # Handle both single and multi-aspect cases
    num_aspects = audio_output.shape[1] if audio_output.dim() == 2 else 1

    # Calculate MSE and correlation for each aspect
    for i in range(num_aspects):
        if num_aspects == 1:
            pred = audio_output.view(-1).numpy()
            tgt = target.view(-1).numpy()
        else:
            pred = audio_output[:, i].numpy()
            tgt = target[:, i].numpy()

        aspect_mse = np.mean((pred - tgt) ** 2)
        corr_matrix = np.corrcoef(pred, tgt)
        aspect_corr = corr_matrix[0, 1].item()

        mse_list.append(aspect_mse)
        corr_list.append(aspect_corr)

    # Return average MSE and correlation across all aspects, plus individual lists
    valid_token_mse = np.mean(mse_list)
    corr = np.mean(corr_list)

    return valid_token_mse, corr, mse_list, corr_list


def main():
    args = get_arguments()

    # Load the test data
    data = load_dataset(args.dataset, split="train")

    # Load the fluency model
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
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    sd = torch.load(args.checkpoint, map_location="cpu")
    audio_model.load_state_dict(sd, strict=True)

    # Load the kmeans model
    kmeans_model = joblib.load(args.kmeans_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    feature_extractor = torchaudio.pipelines.HUBERT_LARGE.get_model()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    predictions = []

    # Helper: sliding windows over waveform (returns list of 1D tensors)
    def make_windows(waveform_1d: torch.Tensor, sr: int, window_ms: int, hop_ms: int):
        # waveform_1d: [N]
        N = waveform_1d.shape[0]
        if window_ms <= 0:
            return [waveform_1d]
        win = int(window_ms * sr / 1000)
        hop = int(hop_ms * sr / 1000) if hop_ms > 0 else win
        if win <= 0:
            return [waveform_1d]
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
            windows.append(seg)
            if end >= N:
                break
            start += hop
        return windows

    # Run inference (with optional sliding-window)
    for item in tqdm(data):
        audio = item["audio"]
        target = [item[aspect] for aspect in args.aspect]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        # Apply same scaling as used during training (from train.py fluDataset)
        target = target * 0.2

        # waveform array from datasets audio feature
        array = audio["array"]
        sr = int(audio["sampling_rate"])
        wav = torch.tensor(array, dtype=torch.float32).to(device)
        # ensure 1d
        if wav.dim() == 2:
            wav = wav.squeeze(0)

        window_ms = getattr(args, "window_ms", 5000)
        hop_ms = getattr(args, "hop_ms", 4000)
        windows = make_windows(wav, sr, window_ms, hop_ms)

        window_preds = []
        with torch.no_grad():
            for w in windows:
                # HuBERT expects shape (batch, time)
                w_batch = w.unsqueeze(0)
                audio_embedding, _ = feature_extractor.extract_features(w_batch)
                features = audio_embedding[14]
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                B, T, D = features.shape
                flat_features = features.reshape(-1, D).cpu().numpy()
                cluster_ids_np = kmeans_model.predict(flat_features)
                cluster_ids = (
                    torch.tensor(cluster_ids_np, dtype=torch.long)
                    .reshape(B, T)
                    .to(device)
                )
                pred = audio_model(features, cluster_ids)
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

    avg_mse, avg_corr, mse_list, corr_list = valid_predictions(all_preds, all_targets)

    for i, aspect in enumerate(args.aspect):
        print(
            f"Aspect: {aspect} - MSE: {mse_list[i]:.4f}, Correlation: {corr_list[i]:.4f}"
        )
    print(f"Average MSE: {avg_mse:.4f}, Average Correlation: {avg_corr:.4f}")


if __name__ == "__main__":
    main()
