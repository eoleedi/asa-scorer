# -*- coding: utf-8 -*-
# train and test the models
import sys
import os
import time
import random
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prosody_scorer.models import (
    ClusterScorer,
    CrossAttnHCSSLScorer,
    FDMPAScorer,
    LayerWeightedHCSSLScorer,
    NonClusterScorer,
    SingleLayerHCSSLScorer,
    SimpleRegressionScorer,
    TransformerScorer,
)
from prosody_scorer.speech_datasets import (
    OnTheFlyFeatureCollator,
    create_dataset,
)

aspect_name_map = {
    "acc": "accuracy",
    "cpn": "completeness",
    "flu": "fluency",
    "psd": "prosodic",
    "ttl": "total",
}

aspect2abbr = {
    "accuracy": "acc",
    "completeness": "cpn",
    "fluency": "flu",
    "prosodic": "psd",
    "total": "ttl",
}


class bcolors:
    HEADER = "\033[95m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def set_arg(parser):
    parser.add_argument(
        "--exp-dir", type=str, default="./exp/", help="directory to dump experiments"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--mine_epochs",
        type=int,
        default=5,
        help="Number of epochs to train only MINE before training scorer",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of maximum training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="training batch size"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="training hidden dimension"
    )
    parser.add_argument(
        "--model", type=str, default="ClusterScorer", help="name of the model"
    )
    parser.add_argument("--use_device", type=str, default="cpu", help="device to use")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="number of heads in transformer"
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="number of layers in transformer"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="dropout probability"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="so762",
        help="Type of dataset: 'so762', 'huggingface', or HuggingFace dataset name (e.g., 'eoleedi/ezai-championship2023')",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/speechocean762",
        help="Directory of the dataset (for SO762 dataset)",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="ssl",
        choices=["ssl", "handcrafted", "fdmpa", "all_layers_fdmpa"],
        help="Feature source to use for SO762 datasets",
    )
    parser.add_argument(
        "--mine_weight",
        type=float,
        default=0.01,
        help="Weight for local token-level MI objective",
    )
    # Global MINE removed; this parameter is deprecated and ignored.
    # parser.add_argument(
    #     "--mine_global_weight",
    #     type=float,
    #     default=0.002,
    #     help="Weight for global MI objective",
    # )
    parser.add_argument(
        "--mine_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for MINE networks",
    )
    parser.add_argument(
        "--mine_steps",
        type=int,
        default=1,
        help="Number of MINE-only updates per training step",
    )
    parser.add_argument(
        "--mine_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for MINE denominator stabilization",
    )
    parser.add_argument(
        "--mine_lr_scale",
        type=float,
        default=0.02,
        help="Scale factor for MINE optimizer lr relative to main lr",
    )
    parser.add_argument(
        "--mi_neg_penalty_weight",
        type=float,
        default=0.5,
        help="Penalty weight for negative MI values (encourages MI >= 0)",
    )
    parser.add_argument(
        "--mi_main_scale_max",
        type=float,
        default=50.0,
        help="Max adaptive scale for MI term inside main loss",
    )
    parser.add_argument(
        "--mine_lr_patience",
        type=int,
        default=3,
        help="Epoch patience before reducing MINE lr when mine_loss worsens",
    )
    parser.add_argument(
        "--mine_lr_rel_tol",
        type=float,
        default=0.02,
        help="Relative tolerance for mine_loss worsening check",
    )
    parser.add_argument(
        "--fdmpa_num_tokens",
        type=int,
        default=-1,
        help="Number of temporal tokens per FDMPA branch. Set to -1 to use the full sequence length (minimal of SSL/HC).",
    )
    parser.add_argument(
        "--hc_aux_weight",
        type=float,
        default=0.1,
        help="Weight for HC auxiliary prediction loss in HC-guided SSL models",
    )
    parser.add_argument(
        "--hc_aux_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for normalizing HC auxiliary branch losses",
    )
    parser.add_argument(
        "--layer_weight_entropy",
        type=float,
        default=0.0,
        help="Optional entropy regularization weight for learned layer distributions",
    )
    parser.add_argument(
        "--layer_weight_diversity",
        type=float,
        default=0.0,
        help="Optional penalty weight for cosine similarity between branch layer weights",
    )
    parser.add_argument(
        "--print_loss_details",
        action="store_true",
        help="Print detailed loss components each batch for FDMPAScorer",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of optimizer steps used for linear learning-rate warmup",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training (default: train)",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Dataset split to use for testing (default: test)",
    )
    parser.add_argument(
        "--load_cluster_index",
        type=bool,
        default=False,
        help="load cluster index (deprecated - auto-detected)",
    )
    parser.add_argument(
        "--kmeans_model",
        type=str,
        default="exp/kmeans/so762/kmeans_model.joblib",
        help="kmeans model path",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=50,
        help="number of clusters for ClusterScorer",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--aspect", nargs="+", default=["fluency"])
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=30.0,
        help="Max audio duration in seconds (for HuggingFace datasets)",
    )
    return parser


def convert_bin(x, num_binary=6):
    # Convert each number to its binary representation
    binary_representations = [
        list(map(int, bin(num)[2:].zfill(num_binary))) for num in x
    ]

    # Convert to a PyTorch tensor
    tensor_2d = torch.tensor(binary_representations)
    return tensor_2d


def cluster_pred(feats, model):
    feats = feats.cpu().numpy()
    cluster_index_list = []
    for feat in feats:
        pred = model.predict(feat)
        pred_bin = convert_bin(pred)
        cluster_index_list.append(pred_bin)
    cluster_index_tensor = torch.stack(cluster_index_list, dim=0)
    return cluster_index_tensor


def draw_train_fig(
    train_mse_values,
    val_mse_values,
    train_corr_values,
    val_corr_values,
    epochs_list,
    exp_dir,
    aspect_names,
):
    """
    Draw training and validation curves.

    Args:
        train_mse_values: list of average training MSE per epoch
        val_mse_values: list of average validation MSE per epoch
        train_corr_values: list of average training correlation per epoch
        val_corr_values: list of average validation correlation per epoch
        epochs_list: list of epoch numbers
        exp_dir: experiment directory to save the figure
        aspect_names: list of aspect names (e.g., ['flu', 'psd'])
    """
    aspect_title = (
        " + ".join([a.upper() for a in aspect_names])
        if len(aspect_names) > 1
        else aspect_names[0].upper()
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_mse_values, label="Training MSE")
    plt.plot(epochs_list, val_mse_values, label="Validation MSE")
    plt.title(f"Training and Validation MSE ({aspect_title})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_corr_values, label="Training Correlation")
    plt.plot(epochs_list, val_corr_values, label="Validation Correlation")
    plt.title(f"Training and Validation Correlation ({aspect_title})")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.legend()

    # Annotate the PCC in the end
    ylast, xlast = val_corr_values[-1], epochs_list[-1]
    plt.text(xlast, ylast, f"{ylast:.3f}", ha="right", color="red", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{exp_dir}/train.jpg")
    plt.close()


def gen_result_header(aspect_names):
    """
    Generate CSV header based on the aspects being trained.

    Args:
        aspect_names: list of aspect names (e.g., ['fluency', 'prosodic'])
    """

    utt_header_set = ["utt_train_mse", "utt_train_pcc", "utt_test_mse", "utt_test_pcc"]
    utt_header_scores = aspect_names

    # Generate headers for each aspect
    utt_header = []
    for dset in utt_header_set:
        utt_header = utt_header + [dset + "_" + x for x in utt_header_scores]

    # Add average headers if multiple aspects
    if len(aspect_names) > 1:
        avg_header = [
            "utt_train_mse_avg",
            "utt_train_pcc_avg",
            "utt_test_mse_avg",
            "utt_test_pcc_avg",
        ]
        header = ["epoch", "learning_rate"] + avg_header + utt_header
    else:
        header = ["epoch", "learning_rate"] + utt_header

    return header


def unpack_batch(data):
    if len(data) == 4:
        audio_paths, utt_label, feats, indexs = data
        return audio_paths, utt_label, feats, None, indexs
    if len(data) == 5:
        audio_paths, utt_label, feats, hc_feats, indexs = data
        return audio_paths, utt_label, feats, hc_feats, indexs
    raise ValueError("Unexpected number of elements in data")


def _continuous_hc_aux_targets(hc_feats: torch.Tensor) -> torch.Tensor:
    continuous = hc_feats[:, :, :10].float().clone()
    continuous[:, :, 8:9] = torch.log(
        torch.clamp(continuous[:, :, 8:9], min=0.0) + 1.0
    )
    return continuous


def _normalize_ppg_distribution(ppg: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ppg = torch.clamp(ppg, min=0.0)
    return ppg / ppg.sum(dim=-1, keepdim=True).clamp_min(eps)


def _valid_ppg_mask(
    target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    ppg_valid = target.detach().sum(dim=-1) > 1e-8
    if mask is None:
        return ppg_valid
    return mask.to(device=target.device, dtype=torch.bool) & ppg_valid


def compute_hc_aux_loss(
    aux: dict,
    loss_fn,
    device: torch.device,
    ema_state: dict | None = None,
    ema_decay: float = 0.99,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    preds = aux.get("hc_aux_preds", {}) if aux is not None else {}
    targets = aux.get("hc_aux_targets", {}) if aux is not None else {}
    masks = aux.get("hc_aux_mask", {}) if aux is not None else {}
    losses = []
    details = {}
    for branch, pred in preds.items():
        target = targets.get(branch)
        if target is None:
            continue
        mask = masks.get(branch)
        if mask is None:
            if branch == "rhy":
                target_prob = _normalize_ppg_distribution(target.detach())
                frame_loss = -(
                    target_prob * F.log_softmax(pred, dim=-1)
                ).sum(dim=-1)
                ppg_mask = _valid_ppg_mask(target).to(dtype=frame_loss.dtype)
                branch_loss = (
                    (frame_loss * ppg_mask).sum() / ppg_mask.sum().clamp_min(1.0)
                )
            else:
                branch_loss = loss_fn(pred, target.detach())
        else:
            if branch == "rhy":
                target_prob = _normalize_ppg_distribution(target.detach())
                frame_loss = -(target_prob * F.log_softmax(pred, dim=-1)).sum(dim=-1)
                mask = _valid_ppg_mask(target, mask).to(dtype=frame_loss.dtype)
            else:
                frame_loss = (pred - target.detach()).pow(2).mean(dim=-1)
                mask = mask.to(device=pred.device, dtype=frame_loss.dtype)
            branch_loss = (frame_loss * mask).sum() / mask.sum().clamp_min(1.0)

        ema_scale = branch_loss.detach().clamp_min(1e-8)
        if ema_state is not None:
            previous = ema_state.get(branch)
            if previous is None:
                ema_state[branch] = ema_scale
            else:
                ema_state[branch] = (
                    ema_decay * previous.to(device=ema_scale.device)
                    + (1.0 - ema_decay) * ema_scale
                ).detach()
            ema_scale = ema_state[branch].to(device=branch_loss.device).clamp_min(1e-8)

        normalized_loss = branch_loss / ema_scale
        losses.append(normalized_loss)
        details[branch] = {
            "raw": float(branch_loss.detach().cpu().item()),
            "ema": float(ema_scale.detach().cpu().item()),
            "normalized": float(normalized_loss.detach().cpu().item()),
        }
    if not losses:
        loss = torch.zeros((), device=device)
    else:
        loss = torch.stack(losses).mean()
    if return_details:
        return loss, details
    return loss


def compute_hc_aux_metrics(aux: dict) -> dict:
    preds = aux.get("hc_aux_preds", {}) if aux is not None else {}
    targets = aux.get("hc_aux_targets", {}) if aux is not None else {}
    masks = aux.get("hc_aux_mask", {}) if aux is not None else {}
    metrics = {}
    for branch, pred in preds.items():
        target = targets.get(branch)
        if target is None:
            continue
        mask = masks.get(branch)
        if mask is None:
            valid = torch.ones(
                pred.shape[:-1], device=pred.device, dtype=torch.bool
            )
        else:
            valid = mask.to(device=pred.device, dtype=torch.bool)
        if branch == "rhy":
            valid = _valid_ppg_mask(target, valid)
            pred_eval = torch.softmax(pred.detach(), dim=-1)
            target_eval = _normalize_ppg_distribution(target.detach())
        else:
            pred_eval = pred.detach()
            target_eval = target.detach()
        valid_frame_count = int(valid.sum().item())
        valid = valid.unsqueeze(-1).expand_as(pred_eval)
        pred_flat = pred_eval[valid].float().cpu().numpy()
        target_flat = target_eval[valid].float().cpu().numpy()
        count = int(pred_flat.size)
        if count == 0:
            metrics[branch] = {"mse": 0.0, "mae": 0.0, "corr": 0.0, "count": 0}
            continue

        diff = pred_flat - target_flat
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        if np.std(pred_flat) < 1e-8 or np.std(target_flat) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(pred_flat, target_flat)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        branch_metrics = {"mse": mse, "mae": mae, "corr": corr, "count": count}
        if branch == "rhy":
            frame_valid = valid[..., 0]
            pred_prob = pred_eval[frame_valid]
            target_prob = target_eval[frame_valid]
            kl = (
                target_prob
                * (
                    torch.log(target_prob.clamp_min(1e-8))
                    - torch.log(pred_prob.clamp_min(1e-8))
                )
            ).sum(dim=-1)
            ce = -(target_prob * torch.log(pred_prob.clamp_min(1e-8))).sum(dim=-1)
            top1_acc = (
                pred_prob.argmax(dim=-1) == target_prob.argmax(dim=-1)
            ).float()
            branch_metrics.update(
                {
                    "kl": float(kl.mean().cpu().item()) if valid_frame_count else 0.0,
                    "ce": float(ce.mean().cpu().item()) if valid_frame_count else 0.0,
                    "top1_acc": float(top1_acc.mean().cpu().item())
                    if valid_frame_count
                    else 0.0,
                    "frame_count": valid_frame_count,
                }
            )
        metrics[branch] = branch_metrics
    return metrics


def aggregate_hc_aux_metrics(metrics_list: list) -> dict:
    branch_totals = {}
    for metrics in metrics_list:
        for branch, values in metrics.items():
            total = branch_totals.setdefault(branch, {"count": 0})
            count = values.get("count", 0)
            total["count"] += count
            if "frame_count" in values:
                total["frame_count"] = total.get("frame_count", 0) + values.get(
                    "frame_count", 0
                )
            for key, value in values.items():
                if key in ["count", "frame_count"]:
                    continue
                weight = (
                    values.get("frame_count", count)
                    if key in ["kl", "ce", "top1_acc"]
                    else count
                )
                total[key] = total.get(key, 0.0) + value * weight

    aggregated = {}
    for branch, total in branch_totals.items():
        count = max(1, total["count"])
        aggregated[branch] = {"count": total["count"]}
        if "frame_count" in total:
            aggregated[branch]["frame_count"] = total["frame_count"]
        for key, value in total.items():
            if key in ["count", "frame_count"]:
                continue
            denom = (
                max(1, total.get("frame_count", count))
                if key in ["kl", "ce", "top1_acc"]
                else count
            )
            aggregated[branch][key] = value / denom
    return aggregated


def format_hc_aux_metrics(metrics: dict) -> str:
    if not metrics:
        return ""
    return " | ".join(
        (
            f"{branch}: ce={values['ce']:.4f}, kl={values['kl']:.4f}, "
            f"top1={values['top1_acc']:.3f}, prob_mse={values['mse']:.4f}"
        )
        if branch == "rhy" and {"ce", "kl", "top1_acc"}.issubset(values)
        else f"{branch}: mse={values['mse']:.4f}, mae={values['mae']:.4f}, corr={values['corr']:.3f}"
        for branch, values in metrics.items()
    )


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_layer_weight_regularization(
    layer_weights: torch.Tensor,
    entropy_weight: float = 0.0,
    diversity_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reg = layer_weights.new_zeros(())
    entropy_loss = layer_weights.new_zeros(())
    diversity_loss = layer_weights.new_zeros(())

    if entropy_weight != 0.0:
        entropy_loss = (
            -(layer_weights * torch.log(layer_weights.clamp_min(1e-8)))
            .sum(dim=-1)
            .mean()
        )
        reg = reg + entropy_weight * entropy_loss

    if diversity_weight != 0.0 and layer_weights.size(0) > 1:
        sims = []
        normalized = F.normalize(layer_weights, p=2, dim=-1)
        for i in range(normalized.size(0)):
            for j in range(i + 1, normalized.size(0)):
                sims.append((normalized[i] * normalized[j]).sum())
        if sims:
            diversity_loss = torch.stack(sims).mean()
            reg = reg + diversity_weight * diversity_loss

    return reg, entropy_loss, diversity_loss


def layer_weights_to_dict(audio_model) -> dict | None:
    if not hasattr(audio_model, "layer_logits") or not hasattr(audio_model, "branches"):
        return None
    with torch.no_grad():
        weights = (
            torch.softmax(audio_model.layer_logits, dim=-1).detach().cpu().tolist()
        )
    return {branch: weights[i] for i, branch in enumerate(audio_model.branches)}


def save_layer_weights(audio_model, exp_dir: str, epoch: int, is_best: bool = False):
    weights = layer_weights_to_dict(audio_model)
    if weights is None:
        return

    history_path = os.path.join(exp_dir, "layer_weights_epoch.jsonl")
    with open(history_path, "a") as f:
        f.write(json.dumps({"epoch": epoch, "weights": weights}) + "\n")

    if is_best:
        with open(os.path.join(exp_dir, "layer_weights_best.json"), "w") as f:
            json.dump(weights, f, indent=2)

    for branch, values in weights.items():
        top = np.argsort(values)[::-1][:3]
        top_str = ", ".join(f"{int(i)}:{values[int(i)]:.3f}" for i in top)
        print(f"  {branch} top layers: {top_str}")


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on " + str(device))

    # Get aspect information early for use throughout training
    aspect_names = args.aspect
    num_aspects = len(aspect_names)

    train_mse_values, train_corr_values = [], []
    val_mse_values, val_corr_values = [], []
    epochs_list = []

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    # Save hyperparameters to exp_dir
    hyperparams = vars(args).copy()
    hyperparams_file = os.path.join(exp_dir, "hyperparams.json")
    with open(hyperparams_file, "w") as f:
        # Convert non-serializable types to strings
        hyperparams_serializable = {}
        for k, v in hyperparams.items():
            try:
                json.dumps(v)
                hyperparams_serializable[k] = v
            except (TypeError, ValueError):
                hyperparams_serializable[k] = str(v)
        json.dump(hyperparams_serializable, f, indent=2)
    print(f"✅ Hyperparameters saved to: {hyperparams_file}")

    audio_model = audio_model.to(device)

    if args.model == "ClusterScorer" or (
        args.model == "TransformerScorer" and args.feature_type != "handcrafted"
    ):
        kmeans_model = joblib.load(args.kmeans_model)
    else:
        kmeans_model = None

    if args.model == "FDMPAScorer":
        main_params = [p for p in audio_model.non_mine_parameters() if p.requires_grad]
        mine_params = [p for p in audio_model.mine_parameters() if p.requires_grad]
        trainables = main_params + mine_params
        optimizer = torch.optim.Adam(
            main_params, args.lr, weight_decay=5e-7, betas=(0.95, 0.999)
        )
        mine_optimizer = torch.optim.Adam(
            mine_params,
            args.lr * args.mine_lr_scale,
            weight_decay=5e-7,
            betas=(0.95, 0.999),  # Reduced LR for stability
        )
    else:
        trainables = [p for p in audio_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999)
        )
        mine_optimizer = None

    print(
        "Total parameter number is : {:.3f} k".format(
            sum(p.numel() for p in audio_model.parameters()) / 1e3
        )
    )
    print(
        "Total trainable parameter number is : {:.3f} k".format(
            sum(p.numel() for p in trainables) / 1e3
        )
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1
    )

    loss_fn = nn.MSELoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    # Calculate result array size based on number of aspects
    if num_aspects > 1:
        # epoch, lr, avg(train_mse, train_corr, test_mse, test_corr), per-aspect(train_mse, train_corr, test_mse, test_corr)
        result_cols = 2 + 4 + (4 * num_aspects)
    else:
        # epoch, lr, train_mse, train_corr, test_mse, test_corr
        result_cols = 2 + (4 * num_aspects)
    result = np.zeros([args.n_epochs, result_cols])
    prev_avg_mine_loss = None
    mine_bad_epochs = 0
    hc_aux_ema_state = {}
    warm_up_step = max(0, args.warmup_steps)
    warmup_started = False

    # Stage 1: MINE Pre-training
    if args.model == "FDMPAScorer" and args.mine_epochs > 0:
        print(
            f"--- Starting Stage 1: Pre-training MINE for {args.mine_epochs} epochs ---"
        )

        # In Stage 1, we fix everything except MINE
        for p in audio_model.non_mine_parameters():
            p.requires_grad = False
        for p in audio_model.mine_parameters():
            p.requires_grad = True

        for mine_epoch in range(args.mine_epochs):
            audio_model.train()
            epoch_mine_losses = []
            epoch_mi_values = []
            epoch_mi_neg_count = 0
            epoch_skipped_batches = 0

            mine_progress = tqdm(
                train_loader,
                desc=f"MINE warmup {mine_epoch + 1}/{args.mine_epochs}",
                total=len(train_loader),
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )
            for batch_idx, data in enumerate(mine_progress):
                # Unpack: path, label, feats, hc_feats, index
                _, _, feats, hc_feats, _ = unpack_batch(data)

                if hc_feats is None:
                    epoch_skipped_batches += 1
                    continue

                feats = feats.to(device)
                hc_feats = hc_feats.to(device)

                # Forward pass but don't track gradients for the main model
                with torch.no_grad():
                    _, aux = audio_model(feats, hc_feats)

                # Compute MI terms using the detached tokens (local token MI only)
                # update_ema=True to stabilize the MINE denominator
                mi_local, _, _, _ = audio_model.compute_mi_terms(
                    aux["ssl_tokens"],
                    aux["hc_tokens"],
                    aux["ssl_global"],
                    aux["hc_global"],
                    update_ema=True,
                )

                # MINE Loss calculation (maximize MI = minimize -MI)
                mi_local_for_mine = torch.clamp(mi_local, min=-5.0, max=5.0)

                # Penalty for negative MI (helps stability) — global term removed
                mi_neg_penalty = F.relu(-mi_local_for_mine)

                mine_loss = (
                    -(mi_local_for_mine) + args.mi_neg_penalty_weight * mi_neg_penalty
                )

                if not torch.isfinite(mine_loss):
                    epoch_skipped_batches += 1
                    continue

                mine_optimizer.zero_grad()
                mine_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    audio_model.mine_parameters(), max_norm=5.0
                )
                mine_optimizer.step()

                mi_value = mi_local_for_mine.detach().cpu().item()
                epoch_mine_losses.append(mine_loss.item())
                epoch_mi_values.append(mi_value)
                if mi_value < 0:
                    epoch_mi_neg_count += 1
                mine_progress.set_postfix(
                    updates=len(epoch_mine_losses),
                    skipped=epoch_skipped_batches,
                    avg_mi=f"{np.mean(epoch_mi_values):.4f}",
                    neg=f"{epoch_mi_neg_count / max(1, len(epoch_mi_values)):.1%}",
                    lr=f"{mine_optimizer.param_groups[0]['lr']:.2e}",
                )

            if epoch_mine_losses:
                avg_mi = float(np.mean(epoch_mi_values))
                neg_ratio = epoch_mi_neg_count / max(1, len(epoch_mi_values))
                tqdm.write(
                    f"MINE warmup epoch {mine_epoch + 1}/{args.mine_epochs}: "
                    f"updates={len(epoch_mine_losses)}, skipped={epoch_skipped_batches}, "
                    f"avg_clamped_mi={avg_mi:.4f}, neg_mi_ratio={neg_ratio:.2%}, "
                    f"mine_lr={mine_optimizer.param_groups[0]['lr']:.2e}"
                )
            else:
                tqdm.write(
                    f"MINE Epoch {mine_epoch + 1}/{args.mine_epochs}, No valid losses found."
                )

        print("--- Stage 1 Complete: MINE is now initialized ---")

        # Important: Stage 2 starts, we need to track gradients for the main model
        for p in audio_model.non_mine_parameters():
            p.requires_grad = True
        # For FDMPAScorer, we typically keep MINE training in Stage 2 as well (alternating),
        # but the main while loop below handles that.

    while epoch < args.n_epochs:
        audio_model.train()

        # Initialize MI tracking for this epoch (FDMPAScorer)
        epoch_mi_local_values = []
        epoch_mi_global_values = []
        epoch_mine_loss_values = []
        epoch_mi_neg_penalty_values = []
        epoch_mi_local_neg_count = 0
        epoch_mi_global_neg_count = 0
        epoch_mi_count = 0
        # Per-branch MI tracking: {branch: [list of values]}
        epoch_mi_local_per_branch = {"int": [], "rhy": [], "pro": []}
        epoch_mi_global_per_branch = {"int": [], "rhy": [], "pro": []}
        epoch_train_hc_aux_metrics = []

        train_progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch + 1}/{args.n_epochs}",
            total=len(train_loader),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_idx, data in enumerate(train_progress):
            audio_paths, utt_label, feats, hc_feats, indexs = unpack_batch(data)
            cluster_index = None
            if indexs is not None:
                cluster_index = (indexs + 1).to(device)

            if warm_up_step > 0 and global_step <= warm_up_step:
                if not warmup_started:
                    warmup_started = True
                    tqdm.write(
                        f"[LR_WARMUP] linear warmup: steps=0..{warm_up_step}, "
                        f"target_lr={args.lr:.2e}, scheduler starts after warmup"
                    )
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warm_lr
                if global_step == warm_up_step:
                    tqdm.write(
                        "[LR_WARMUP] complete; scheduler will step at epoch end"
                    )

            feats = feats.to(device)
            if args.model == "FDMPAScorer":
                if hc_feats is None:
                    raise ValueError("FDMPAScorer requires handcrafted features.")
                hc_feats = hc_feats.to(device)
                batch_mine_losses = []

                for p in audio_model.mine_parameters():
                    p.requires_grad = True

                for _ in range(max(1, args.mine_steps)):
                    with torch.no_grad():
                        _, aux_detached = audio_model(feats, hc_feats)
                    (
                        mi_local,
                        _,
                        mi_local_dict,
                        _,
                    ) = audio_model.compute_mi_terms(
                        aux_detached["ssl_tokens"],
                        aux_detached["hc_tokens"],
                        aux_detached["ssl_global"],
                        aux_detached["hc_global"],
                        update_ema=True,
                    )
                    mi_local_for_mine = torch.clamp(mi_local, min=-5.0, max=5.0)
                    mi_neg_penalty = F.relu(-mi_local_for_mine)
                    mine_loss = (
                        -(mi_local_for_mine)
                        + args.mi_neg_penalty_weight * mi_neg_penalty
                    )
                    if not torch.isfinite(mine_loss):
                        print(
                            "Warning: non-finite MINE loss detected; skipping MINE update for this batch"
                        )
                        continue
                    mine_optimizer.zero_grad()
                    mine_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        audio_model.mine_parameters(), max_norm=5.0
                    )
                    mine_optimizer.step()
                    batch_mine_losses.append(mine_loss.detach().item())

                # Stage 2: Train Scorer with MI Maximization
                pred, aux = audio_model(feats, hc_feats)
                # Compute MI terms using attached SSL tokens (to pass gradient to encoder)
                # and detached HC tokens (HC is fixed ground truth)
                mi_local, _, mi_local_dict, _ = audio_model.compute_mi_terms(
                    aux["ssl_tokens"],
                    aux["hc_tokens"],
                    aux["ssl_global"],
                    aux["hc_global"],
                    update_ema=False,  # Don't update EMA during Scorer pass
                )
                if not torch.isfinite(mi_local):
                    mi_local = torch.zeros((), device=device)
                # Global MI removed; use zero placeholder for compatibility
                mi_global = torch.zeros((), device=device)

                for p in audio_model.mine_parameters():
                    p.requires_grad = False

                # Track MI values for epoch statistics (global MI removed)
                epoch_mi_local_values.append(mi_local.item())
                epoch_mi_global_values.append(mi_global.item())
                mi_neg_penalty_eval = F.relu(-mi_local)
                epoch_mi_neg_penalty_values.append(mi_neg_penalty_eval.item())
                epoch_mi_count += 1
                if mi_local.item() < 0:
                    epoch_mi_local_neg_count += 1
                # global neg count remains based on zero placeholder
                if batch_mine_losses:
                    epoch_mine_loss_values.append(float(np.mean(batch_mine_losses)))
                else:
                    epoch_mine_loss_values.append((-mi_local).item())

                # Track per-branch MI values
                for branch in ["int", "rhy", "pro"]:
                    if branch in mi_local_dict:
                        local_val = mi_local_dict[branch].item()
                        # global MI per-branch is removed; use 0.0 placeholder
                        global_val = 0.0
                        epoch_mi_local_per_branch[branch].append(local_val)
                        epoch_mi_global_per_branch[branch].append(global_val)
            elif args.model == "CrossAttnHCSSLScorer":
                if hc_feats is None:
                    raise ValueError(
                        "CrossAttnHCSSLScorer requires handcrafted features."
                    )
                hc_feats = hc_feats.to(device)
                pred = audio_model(feats, hc_feats)
            elif args.model in ["LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]:
                if hc_feats is None:
                    raise ValueError(
                        f"{args.model} requires handcrafted features."
                    )
                hc_feats = hc_feats.to(device)
                pred, aux = audio_model(feats, hc_feats)
            elif args.model == "ClusterScorer":
                if cluster_index is None:
                    raise ValueError(f"Model {args.model} requires cluster indices.")
                pred = audio_model(feats, cluster_index)
            elif args.model == "TransformerScorer":
                pred = audio_model(feats, cluster_index)
            elif (
                args.model == "NonClusterScorer"
                or args.model == "SimpleRegressionScorer"
            ):
                pred = audio_model(feats)
            else:
                raise ValueError(f"Model {args.model} not recognized.")

            # Labels are already filtered by the dataset, no need to index again
            labels = utt_label
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.to(device, non_blocking=True)

            # Compute branch auxiliary loss for FDMPAScorer
            branch_aux_loss = torch.zeros((), device=device)
            hc_aux_loss_details = {}
            if args.model == "FDMPAScorer":
                branch_preds = aux["branch_preds"]
                for branch_pred in branch_preds.values():
                    branch_aux_loss = branch_aux_loss + loss_fn(branch_pred, labels)
                branch_aux_loss = branch_aux_loss / len(branch_preds)
            elif args.model in ["LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]:
                branch_aux_loss, hc_aux_loss_details = compute_hc_aux_loss(
                    aux,
                    loss_fn,
                    device,
                    ema_state=hc_aux_ema_state,
                    ema_decay=args.hc_aux_ema_decay,
                    return_details=True,
                )

            mse_loss = loss_fn(pred, labels)
            if args.model == "FDMPAScorer":
                # Global MI removed; only use local token MI
                mi_neg_penalty_main = F.relu(-mi_local)
                mi_main_term = args.mine_weight * mi_local
                mi_main_scale = torch.clamp(
                    mse_loss.detach() / (mi_main_term.detach().abs() + 1e-6),
                    min=1.0,
                    max=args.mi_main_scale_max,
                )
                loss = (
                    mse_loss
                    - mi_main_scale * mi_main_term
                    + args.mi_neg_penalty_weight * mi_neg_penalty_main
                    + 0.1 * branch_aux_loss
                )
            elif args.model in ["LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]:
                if args.model == "LayerWeightedHCSSLScorer":
                    layer_reg, layer_entropy_loss, layer_diversity_loss = (
                        compute_layer_weight_regularization(
                            aux["layer_weights"],
                            entropy_weight=args.layer_weight_entropy,
                            diversity_weight=args.layer_weight_diversity,
                        )
                    )
                else:
                    layer_reg = torch.zeros((), device=device)
                    layer_entropy_loss = torch.zeros((), device=device)
                    layer_diversity_loss = torch.zeros((), device=device)
                loss = mse_loss + args.hc_aux_weight * branch_aux_loss + layer_reg
            else:
                loss = mse_loss

            # Print detailed loss breakdown when requested
            if args.model == "FDMPAScorer" and getattr(
                args, "print_loss_details", False
            ):
                try:
                    mse_val = float(mse_loss.detach().cpu().item())
                except Exception:
                    mse_val = float(mse_loss.detach().cpu().numpy())
                mi_local_val = (
                    mi_local.detach().cpu().item()
                    if torch.isfinite(mi_local)
                    else float("nan")
                )
                mi_main_term_val = mi_main_term.detach().cpu().item()
                mi_main_scale_val = mi_main_scale.detach().cpu().item()
                mi_neg_penalty_main_val = mi_neg_penalty_main.detach().cpu().item()
                branch_aux_loss_val = branch_aux_loss.detach().cpu().item()
                ssl_commit_loss_val = None
                if aux is not None and "ssl_commit_loss" in aux:
                    try:
                        ssl_commit_loss_val = float(
                            aux["ssl_commit_loss"].detach().cpu().item()
                        )
                    except Exception:
                        ssl_commit_loss_val = None

                print(
                    f"[LOSS_DETAILS] mse={mse_val:.6f}, mi_local={mi_local_val:.6f}, "
                    f"mine_weight={args.mine_weight}, mi_main_term={mi_main_term_val:.6f}, mi_main_scale={mi_main_scale_val:.6f}, "
                    f"mi_neg_penalty_weight={args.mi_neg_penalty_weight}, mi_neg_penalty_main={mi_neg_penalty_main_val:.6f}, branch_aux_w=0.1, branch_aux_loss={branch_aux_loss_val:.6f}, ssl_commit_loss={ssl_commit_loss_val}"
                )
            elif args.model in [
                "LayerWeightedHCSSLScorer",
                "SingleLayerHCSSLScorer",
            ] and getattr(
                args, "print_loss_details", False
            ):
                hc_aux_branch_parts = []
                for branch, values in hc_aux_loss_details.items():
                    hc_aux_branch_parts.append(
                        f"{branch}_raw={values['raw']:.6f}, "
                        f"{branch}_ema={values['ema']:.6f}, "
                        f"{branch}_norm={values['normalized']:.6f}"
                    )
                hc_aux_branch_details = "; ".join(hc_aux_branch_parts)
                print(
                    f"[LOSS_DETAILS] mse={mse_loss.detach().cpu().item():.6f}, "
                    f"hc_aux_w={args.hc_aux_weight}, hc_aux_loss={branch_aux_loss.detach().cpu().item():.6f}, "
                    f"hc_aux_ema_decay={args.hc_aux_ema_decay}, "
                    f"hc_aux_branches=[{hc_aux_branch_details}], "
                    f"layer_entropy={layer_entropy_loss.detach().cpu().item():.6f}, "
                    f"layer_diversity={layer_diversity_loss.detach().cpu().item():.6f}"
                )

            if not torch.isfinite(loss):
                print("Warning: non-finite total loss detected; skipping this batch")
                if args.model == "FDMPAScorer":
                    for p in audio_model.mine_parameters():
                        p.requires_grad = True
                continue

            if args.model in ["LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]:
                hc_aux_metrics = compute_hc_aux_metrics(aux)
                if hc_aux_metrics:
                    epoch_train_hc_aux_metrics.append(hc_aux_metrics)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(audio_model.parameters(), max_norm=5.0)
            optimizer.step()
            if args.model == "FDMPAScorer":
                for p in audio_model.mine_parameters():
                    p.requires_grad = True
            global_step += 1
            postfix = {
                "step": global_step,
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if warm_up_step > 0 and global_step <= warm_up_step:
                postfix["warmup"] = f"{global_step / warm_up_step:.0%}"
            train_progress.set_postfix(postfix)

        # Print MI statistics for FDMPAScorer
        if args.model == "FDMPAScorer" and epoch_mi_local_values:
            avg_mi_local = np.mean(epoch_mi_local_values)
            avg_mine_loss = np.mean(epoch_mine_loss_values)
            avg_mi_neg_penalty = (
                np.mean(epoch_mi_neg_penalty_values)
                if epoch_mi_neg_penalty_values
                else 0.0
            )
            local_neg_ratio = epoch_mi_local_neg_count / max(1, epoch_mi_count)
            print(
                f"Epoch {epoch} - MI(HC,SSL) Stats: local_token_sum={avg_mi_local:.4f}, mine_loss={avg_mine_loss:.4f}, neg_penalty={avg_mi_neg_penalty:.4f}, neg_ratio(local)={local_neg_ratio:.2%}"
            )

            # Print per-branch MI statistics (global MI removed)
            for branch in ["int", "rhy", "pro"]:
                if epoch_mi_local_per_branch[branch]:
                    avg_local = np.mean(epoch_mi_local_per_branch[branch])
                    print(f"  - {branch}: MI_token(local)={avg_local:.4f}")

            if prev_avg_mine_loss is not None:
                worsen_threshold = prev_avg_mine_loss + args.mine_lr_rel_tol * abs(
                    prev_avg_mine_loss
                )
                if avg_mine_loss > worsen_threshold:
                    mine_bad_epochs += 1
                else:
                    mine_bad_epochs = 0
                if mine_bad_epochs >= args.mine_lr_patience:
                    for param_group in mine_optimizer.param_groups:
                        param_group["lr"] = max(param_group["lr"] * 0.5, 5e-6)
                    print(
                        f"MINE loss worsened for {mine_bad_epochs} epochs ({prev_avg_mine_loss:.4f} -> {avg_mine_loss:.4f}); reducing MINE lr to {mine_optimizer.param_groups[0]['lr']:.2e}"
                    )
                    mine_bad_epochs = 0
            prev_avg_mine_loss = avg_mine_loss

        train_hc_aux_metrics = aggregate_hc_aux_metrics(epoch_train_hc_aux_metrics)
        if train_hc_aux_metrics:
            print(
                f"[HC_AUX_EPOCH] epoch={epoch} split=train "
                f"{format_hc_aux_metrics(train_hc_aux_metrics)}"
            )

        print("start validation")

        # ensemble results
        # don't save prediction for the training set
        (
            tr_mse,
            tr_corr,
            tr_mse_list,
            tr_corr_list,
            tr_branch_div,
            tr_hc_aux_metrics,
        ) = validate(
            audio_model,
            train_loader,
            args,
            -1,
            kmeans_model,
            split_name="train_eval",
        )
        (
            te_mse,
            te_corr,
            te_mse_list,
            te_corr_list,
            te_branch_div,
            te_hc_aux_metrics,
        ) = validate(
            audio_model,
            test_loader,
            args,
            best_mse,
            kmeans_model,
            split_name="test",
        )

        train_mse_values.append(tr_mse)
        train_corr_values.append(tr_corr)
        val_mse_values.append(te_mse)
        val_corr_values.append(te_corr)
        epochs_list.append(epoch)

        # Handle both tensor and scalar returns from validate
        tr_mse_val = tr_mse.item() if isinstance(tr_mse, torch.Tensor) else tr_mse
        te_mse_val = te_mse.item() if isinstance(te_mse, torch.Tensor) else te_mse

        # Print overall metrics
        print("Overall: Train MSE: {:.3f}, CORR: {:.3f}".format(tr_mse_val, tr_corr))
        print(
            f"Overall: Test MSE: {te_mse_val:.3f}, {bcolors.YELLOW}CORR: {te_corr:.3f}{bcolors.ENDC}"
        )

        # Print per-aspect metrics (always available as lists)
        if len(aspect_names) > 1:
            print("\nPer-aspect metrics:")
        for i, aspect in enumerate(aspect_names):
            full_name = aspect_name_map.get(aspect, aspect.upper())
            if len(aspect_names) > 1:
                print(
                    f"  {full_name}: Train MSE: {tr_mse_list[i]:.3f}, CORR: {tr_corr_list[i]:.3f} | "
                    f"Test MSE: {te_mse_list[i]:.3f}, {bcolors.CYAN}CORR: {te_corr_list[i]:.3f}{bcolors.ENDC}"
                )

        if train_hc_aux_metrics or tr_hc_aux_metrics or te_hc_aux_metrics:
            hc_aux_epoch_record = {
                "epoch": epoch,
                "train": train_hc_aux_metrics,
                "train_eval": tr_hc_aux_metrics,
                "test": te_hc_aux_metrics,
            }
            append_jsonl(
                os.path.join(exp_dir, "hc_aux_epoch_metrics.jsonl"),
                hc_aux_epoch_record,
            )
            if tr_hc_aux_metrics:
                print(
                    f"[HC_AUX_EPOCH] epoch={epoch} split=train_eval "
                    f"{format_hc_aux_metrics(tr_hc_aux_metrics)}"
                )
            if te_hc_aux_metrics:
                print(
                    f"[HC_AUX_EPOCH] epoch={epoch} split=test "
                    f"{format_hc_aux_metrics(te_hc_aux_metrics)}"
                )

        # Save branch diversity info if available
        if (
            args.model
            in ["FDMPAScorer", "LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]
            and te_branch_div is not None
        ):
            branch_div_log = {
                "epoch": epoch,
                "train": tr_branch_div,
                "test": te_branch_div,
            }
            with open(os.path.join(exp_dir, "branch_diversity.json"), "w") as f:
                json.dump(branch_div_log, f, indent=2)
            if te_branch_div.get("mean", 1.0) > 0.95:
                print(
                    f"{bcolors.YELLOW}WARNING: Test branch mean similarity = {te_branch_div['mean']:.3f} (>0.95)—branches may be collapsing!{bcolors.ENDC}"
                )

            # Save MI statistics (global MINE removed — only token-level/local MI retained)
            if epoch_mi_local_values:
                mi_stats = {
                    "epoch": epoch,
                    "mi_local_mean": np.mean(epoch_mi_local_values),
                    "mi_local_std": np.std(epoch_mi_local_values),
                    "mi_negative_penalty_mean": np.mean(epoch_mi_neg_penalty_values)
                    if epoch_mi_neg_penalty_values
                    else 0.0,
                    "mi_local_negative_ratio": epoch_mi_local_neg_count
                    / max(1, epoch_mi_count),
                    "mi_local_semantics": "token-level MI(HC,SSL), summed over branches",
                    "mi_main_scale_max": args.mi_main_scale_max,
                    "mine_loss_mean": np.mean(epoch_mine_loss_values),
                    "mine_loss_std": np.std(epoch_mine_loss_values),
                    # Per-branch statistics
                    "per_branch": {
                        "int": {
                            "mi_local_mean": np.mean(epoch_mi_local_per_branch["int"])
                            if epoch_mi_local_per_branch["int"]
                            else 0,
                            "mi_local_std": np.std(epoch_mi_local_per_branch["int"])
                            if epoch_mi_local_per_branch["int"]
                            else 0,
                        },
                        "rhy": {
                            "mi_local_mean": np.mean(epoch_mi_local_per_branch["rhy"])
                            if epoch_mi_local_per_branch["rhy"]
                            else 0,
                            "mi_local_std": np.std(epoch_mi_local_per_branch["rhy"])
                            if epoch_mi_local_per_branch["rhy"]
                            else 0,
                        },
                        "pro": {
                            "mi_local_mean": np.mean(epoch_mi_local_per_branch["pro"])
                            if epoch_mi_local_per_branch["pro"]
                            else 0,
                            "mi_local_std": np.std(epoch_mi_local_per_branch["pro"])
                            if epoch_mi_local_per_branch["pro"]
                            else 0,
                        },
                    },
                }
                # Append to mi_stats.json (or create if not exists)
                mi_stats_file = os.path.join(exp_dir, "mi_stats.json")
                if os.path.exists(mi_stats_file):
                    with open(mi_stats_file, "r") as f:
                        mi_history = json.load(f)
                else:
                    mi_history = []
                mi_history.append(mi_stats)
                with open(mi_stats_file, "w") as f:
                    json.dump(mi_history, f, indent=2)

        # Save results to array
        if len(aspect_names) > 1:
            # Save: epoch, lr, avg metrics, then per-aspect metrics
            result_row = [
                epoch,
                optimizer.param_groups[0]["lr"],
                tr_mse_val,
                tr_corr,
                te_mse_val,
                te_corr,
            ]
            # Add per-aspect metrics in order: train_mse, train_corr, test_mse, test_corr for each aspect
            for i in range(num_aspects):
                result_row.extend(
                    [tr_mse_list[i], tr_corr_list[i], te_mse_list[i], te_corr_list[i]]
                )
            result[epoch, :] = result_row
        else:
            # Single aspect: epoch, lr, train_mse, train_corr, test_mse, test_corr
            result[epoch, :] = [
                epoch,
                optimizer.param_groups[0]["lr"],
                tr_mse_list[0],
                tr_corr_list[0],
                te_mse_list[0],
                te_corr_list[0],
            ]

        print("-------------------validation finished-------------------")

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(
                audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir)
            )

        if args.model == "LayerWeightedHCSSLScorer":
            save_layer_weights(
                audio_model, exp_dir, epoch, is_best=(best_epoch == epoch)
            )

        if global_step > warm_up_step:
            scheduler.step()

        print("Epoch-{0} lr: {1}".format(epoch, optimizer.param_groups[0]["lr"]))
        epoch += 1

    draw_train_fig(
        train_mse_values,
        val_mse_values,
        train_corr_values,
        val_corr_values,
        epochs_list,
        exp_dir,
        aspect_names,
    )
    header = ",".join(gen_result_header(aspect_names))
    np.savetxt(
        exp_dir + "/result.csv", result, delimiter=",", header=header, comments=""
    )


def validate(
    audio_model,
    val_loader,
    args,
    best_mse,
    kmeans_model=None,
    split_name: str = "eval",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_pred, A_target = [], []
    branch_diversities = []
    hc_aux_metrics_list = []

    with torch.no_grad():
        val_progress = tqdm(
            val_loader,
            desc=f"Validate {split_name}",
            total=len(val_loader),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_idx, data in enumerate(val_progress):
            audio_paths, utt_label, feats, hc_feats, indexs = unpack_batch(data)
            cluster_index = None
            if indexs is not None:
                cluster_index = (indexs + 1).to(device)

            feats = feats.to(device)
            if args.model == "FDMPAScorer":
                if hc_feats is None:
                    raise ValueError("FDMPAScorer requires handcrafted features.")
                hc_feats = hc_feats.to(device)
                score, aux = audio_model(feats, hc_feats)
                branch_diversities.append(aux["branch_diversity"])
            elif args.model == "CrossAttnHCSSLScorer":
                if hc_feats is None:
                    raise ValueError(
                        "CrossAttnHCSSLScorer requires handcrafted features."
                    )
                hc_feats = hc_feats.to(device)
                score = audio_model(feats, hc_feats)
            elif args.model in ["LayerWeightedHCSSLScorer", "SingleLayerHCSSLScorer"]:
                if hc_feats is None:
                    raise ValueError(
                        f"{args.model} requires handcrafted features."
                    )
                hc_feats = hc_feats.to(device)
                score, aux = audio_model(feats, hc_feats)
                branch_diversities.append(aux["branch_diversity"])
                hc_aux_metrics = compute_hc_aux_metrics(aux)
                if hc_aux_metrics:
                    hc_aux_metrics_list.append(hc_aux_metrics)
            elif args.model == "ClusterScorer":
                if cluster_index is None:
                    raise ValueError(f"Model {args.model} requires cluster indices.")
                score = audio_model(feats, cluster_index)
            elif args.model == "TransformerScorer":
                score = audio_model(feats, cluster_index)
            elif (
                args.model == "NonClusterScorer"
                or args.model == "SimpleRegressionScorer"
            ):
                score = audio_model(feats)

            score = score.to("cpu").detach()
            # Labels are already filtered by the dataset, no need to index again
            labels = utt_label
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            A_pred.append(score)
            A_target.append(labels)

        A_pred, A_target = torch.cat(A_pred), torch.cat(A_target)

        # get the scores
        avg_mse, avg_corr, mse_list, corr_list = valid_scores(A_pred, A_target)

        if avg_mse < best_mse:
            # Generate aspect name for logging
            aspect_str = "+".join(args.aspect)
            print(
                f"\033[94mnew best {aspect_str} mse {avg_mse:.3f}, now saving predictions.\033[0m"
            )
            print(args.exp_dir)
            # create the directory
            if os.path.exists(args.exp_dir + "/preds") == False:
                os.mkdir(args.exp_dir + "/preds")

            # saving the target, only do once
            if os.path.exists(args.exp_dir + "/preds/target.npy") == False:
                np.save(args.exp_dir + "/preds/target.npy", A_target)

            np.save(args.exp_dir + "/preds/pred.npy", A_pred)

    # Aggregate branch diversity if available
    branch_div_avg = None
    if branch_diversities and len(branch_diversities) > 0:
        keys = branch_diversities[0].keys()
        branch_div_avg = {
            k: sum(d[k] for d in branch_diversities) / len(branch_diversities)
            for k in keys
        }
        print(
            f"  Branch Diversity (Cosine Similarity): int-rhy={branch_div_avg.get('int-rhy', 0):.3f}, "
            f"int-pro={branch_div_avg.get('int-pro', 0):.3f}, rhy-pro={branch_div_avg.get('rhy-pro', 0):.3f}, "
            f"mean={branch_div_avg.get('mean', 0):.3f}"
        )

    hc_aux_metrics_avg = aggregate_hc_aux_metrics(hc_aux_metrics_list)
    if hc_aux_metrics_avg:
        print(
            f"  HC Aux Prediction ({split_name}): "
            f"{format_hc_aux_metrics(hc_aux_metrics_avg)}"
        )

    return avg_mse, avg_corr, mse_list, corr_list, branch_div_avg, hc_aux_metrics_avg


def valid_scores(audio_output, target):
    """
    Validate score predictions, supporting multiple aspects.

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
        pred_std = np.std(pred)
        tgt_std = np.std(tgt)
        if pred_std < 1e-12 or tgt_std < 1e-12:
            aspect_corr = 0.0
        else:
            corr_matrix = np.corrcoef(pred, tgt)
            aspect_corr = corr_matrix[0, 1].item()
            if not np.isfinite(aspect_corr):
                aspect_corr = 0.0

        if not np.isfinite(aspect_mse):
            aspect_mse = 1e6

        mse_list.append(aspect_mse)
        corr_list.append(aspect_corr)

    # Return average MSE and correlation across all aspects, plus individual lists
    avg_mse = np.mean(mse_list)
    avg_corr = np.mean(corr_list)

    return avg_mse, avg_corr, mse_list, corr_list


def main():
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = set_arg(parser)
    print(
        "I am process %s, running on %s: starting (%s)"
        % (os.getpid(), os.uname()[1], time.asctime())
    )
    args = parser.parse_args()
    if os.path.exists(args.exp_dir) == False:
        os.mkdir(args.exp_dir)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Aspect mapping is now handled in the dataset classes
    # Keep this for backward compatibility with old code paths
    aspect_map = {
        "accuracy": 0,
        "completeness": 1,
        "fluency": 2,
        "prosodic": 3,
        "total": 4,
    }
    args.aspect_indices = [aspect_map[aspect] for aspect in args.aspect]
    if len(args.aspect_indices) == 1:
        args.aspect_indices = args.aspect_indices[0]

    print(f"Training aspects: {args.aspect}")

    print("Preparing datasets...")

    # Load kmeans model if using cluster-based models
    kmeans_model = None
    if args.model == "ClusterScorer" or (
        args.model == "TransformerScorer" and args.feature_type != "handcrafted"
    ):
        if os.path.exists(args.kmeans_model):
            kmeans_model = joblib.load(args.kmeans_model)
            print(f"Loaded kmeans model from: {args.kmeans_model}")
        else:
            print(f"Warning: kmeans model not found at {args.kmeans_model}")

    # Determine device for feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Inform penn about preferred GPU (if any) so penn.from_audio can use it
    try:
        import penn as _penn
        if device.type == 'cuda':
            _penn._DEFAULT_GPU = device.index if device.index is not None else 0
        else:
            _penn._DEFAULT_GPU = None
    except Exception:
        # penn may not be installed in some environments; ignore silently
        pass

    # Override feature_type for models that require HC features
    if args.model in ["FDMPAScorer", "CrossAttnHCSSLScorer", "SingleLayerHCSSLScorer"]:
        args.feature_type = "fdmpa"
        print(f"Setting feature_type to 'fdmpa' for {args.model}")
    elif args.model == "LayerWeightedHCSSLScorer":
        args.feature_type = "all_layers_fdmpa"
        print(f"Setting feature_type to 'all_layers_fdmpa' for {args.model}")

    # Create datasets using the factory function
    print(f"Dataset type: {args.dataset_type}")
    print(f"Train split: {args.train_split}, Test split: {args.test_split}")
    print("Using on-the-fly feature extraction with per-batch waveform padding.")

    tr_dataset = create_dataset(
        dataset_type=args.dataset_type,
        split=args.train_split,
        aspects=args.aspect,
        kmeans_model=kmeans_model,
        device=device,
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        dataset_name=args.dataset_type,  # For HuggingFace datasets
        max_duration_sec=args.max_duration_sec,
        on_the_fly_features=True,
    )

    te_dataset = create_dataset(
        dataset_type=args.dataset_type,
        split=args.test_split,
        aspects=args.aspect,
        kmeans_model=kmeans_model,
        device=device,
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        dataset_name=args.dataset_type,  # For HuggingFace datasets
        max_duration_sec=args.max_duration_sec,
        on_the_fly_features=True,
    )

    train_collate_fn = OnTheFlyFeatureCollator(
        feature_type=args.feature_type,
        device=device,
        kmeans_model=kmeans_model,
    )
    test_collate_fn = OnTheFlyFeatureCollator(
        feature_type=args.feature_type,
        device=device,
        kmeans_model=kmeans_model,
    )

    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )

    te_dataloader = DataLoader(
        te_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
    )

    input_dim = 50 if args.feature_type == "handcrafted" else 1024
    num_layers = 24 if args.model == "LayerWeightedHCSSLScorer" else None

    print(f"Dataset prepared. Input dimension: {input_dim}")
    print(f"Training samples: {len(tr_dataset)}, Test samples: {len(te_dataset)}")

    # Create model
    if args.model == "ClusterScorer":
        print("Training ClusterScorer model")
        audio_model = ClusterScorer(
            input_dim=input_dim,
            embed_dim=args.hidden_dim,
            clustering_dim=6,
            num_clusters=args.num_clusters,
            scorers=args.aspect,
        )
    elif args.model == "NonClusterScorer":
        print("Training NonClusterScorer model (no clustering)")
        audio_model = NonClusterScorer(
            input_dim=input_dim,
            embed_dim=args.hidden_dim,
            scorers=args.aspect,
        )
    elif args.model == "SimpleRegressionScorer":
        print("Training SimpleRegressionScorer model")
        audio_model = SimpleRegressionScorer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
        )
    elif args.model == "TransformerScorer":
        print("Training TransformerScorer model")
        audio_model = TransformerScorer(
            input_dim=input_dim,
            dropout_prob=args.dropout_prob,
            num_heads=args.num_heads,
            depth=args.depth,
            hidden_dim=args.hidden_dim,
            clustering_dim=0 if args.feature_type == "handcrafted" else 6,
            scorers=args.aspect,
        )
    elif args.model == "FDMPAScorer":
        print("Training FDMPAScorer model")
        audio_model = FDMPAScorer(
            ssl_input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
            mine_hidden_dim=args.mine_hidden_dim,
            mine_ema_decay=args.mine_ema_decay,
            num_tokens=args.fdmpa_num_tokens,
            dropout_prob=args.dropout_prob,
        )
    elif args.model == "CrossAttnHCSSLScorer":
        print("Training CrossAttnHCSSLScorer model (HC-SSL cross-attention, no MINE)")
        audio_model = CrossAttnHCSSLScorer(
            ssl_input_dim=input_dim,
            hc_input_dim=50,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
            num_heads=args.num_heads,
            depth=args.depth,
            dropout_prob=args.dropout_prob,
        )
    elif args.model == "SingleLayerHCSSLScorer":
        print(
            "Training SingleLayerHCSSLScorer model (single-layer HC auxiliary, no MINE)"
        )
        audio_model = SingleLayerHCSSLScorer(
            ssl_input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
            num_tokens=args.fdmpa_num_tokens,
            dropout_prob=args.dropout_prob,
        )
    elif args.model == "LayerWeightedHCSSLScorer":
        print(
            "Training LayerWeightedHCSSLScorer model (all-layer HC-guided SSL, no MINE)"
        )
        audio_model = LayerWeightedHCSSLScorer(
            ssl_input_dim=input_dim,
            num_layers=num_layers,
            hidden_dim=args.hidden_dim,
            scorers=args.aspect,
            num_tokens=args.fdmpa_num_tokens,
            dropout_prob=args.dropout_prob,
            hc_aux_weight=args.hc_aux_weight,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    train(audio_model, tr_dataloader, te_dataloader, args)


if __name__ == "__main__":
    main()
