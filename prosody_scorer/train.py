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

from prosody_scorer.models import (
    ClusterScorer,
    CrossAttnHCSSLScorer,
    FDMPAScorer,
    NonClusterScorer,
    SimpleRegressionScorer,
    TransformerScorer,
)
from prosody_scorer.speech_datasets import (
    create_dataset,
    custom_collate_fn,
    fdmpa_collate_fn,
    hcssl_collate_fn,
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
        choices=["ssl", "handcrafted", "fdmpa"],
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
        "--print_loss_details",
        action="store_true",
        help="Print detailed loss components each batch for FDMPAScorer",
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

            for batch_idx, data in enumerate(train_loader):
                # Unpack: path, label, feats, hc_feats, index
                _, _, feats, hc_feats, _ = unpack_batch(data)

                if hc_feats is None:
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
                    continue

                mine_optimizer.zero_grad()
                mine_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    audio_model.mine_parameters(), max_norm=5.0
                )
                mine_optimizer.step()

                epoch_mine_losses.append(mine_loss.item())

            if epoch_mine_losses:
                print(
                    f"MINE Epoch {mine_epoch + 1}/{args.mine_epochs}, Avg MINE Loss: {np.mean(epoch_mine_losses):.4f}"
                )
            else:
                print(
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

        for batch_idx, data in enumerate(train_loader):
            audio_paths, utt_label, feats, hc_feats, indexs = unpack_batch(data)
            cluster_index = None
            if indexs is not None:
                cluster_index = (indexs + 1).to(device)

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warm_lr
                print(
                    "warm-up learning rate is {:f}".format(
                        optimizer.param_groups[0]["lr"]
                    )
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
            if args.model == "FDMPAScorer":
                branch_preds = aux["branch_preds"]
                for branch_pred in branch_preds.values():
                    branch_aux_loss = branch_aux_loss + loss_fn(branch_pred, labels)
                branch_aux_loss = branch_aux_loss / len(branch_preds)

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

            if not torch.isfinite(loss):
                print("Warning: non-finite total loss detected; skipping this batch")
                if args.model == "FDMPAScorer":
                    for p in audio_model.mine_parameters():
                        p.requires_grad = True
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(audio_model.parameters(), max_norm=5.0)
            optimizer.step()
            if args.model == "FDMPAScorer":
                for p in audio_model.mine_parameters():
                    p.requires_grad = True
            global_step += 1

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

        print("start validation")

        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr, tr_mse_list, tr_corr_list, tr_branch_div = validate(
            audio_model, train_loader, args, -1, kmeans_model
        )
        te_mse, te_corr, te_mse_list, te_corr_list, te_branch_div = validate(
            audio_model, test_loader, args, best_mse, kmeans_model
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

        # Save branch diversity info if FDMPAScorer
        if args.model == "FDMPAScorer" and te_branch_div is not None:
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


def validate(audio_model, val_loader, args, best_mse, kmeans_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_pred, A_target = [], []
    branch_diversities = []

    with torch.no_grad():
        for _, data in enumerate(val_loader):
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

    return avg_mse, avg_corr, mse_list, corr_list, branch_div_avg


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

    # Override feature_type for models that require HC features
    if args.model in ["FDMPAScorer", "CrossAttnHCSSLScorer"]:
        args.feature_type = "fdmpa"
        print(f"Setting feature_type to 'fdmpa' for {args.model}")

    # Create datasets using the factory function
    print(f"Dataset type: {args.dataset_type}")
    print(f"Train split: {args.train_split}, Test split: {args.test_split}")

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
    )

    # Create data loaders
    # Select appropriate collate function based on model
    if args.model == "FDMPAScorer":
        train_collate_fn = fdmpa_collate_fn
        test_collate_fn = fdmpa_collate_fn
    elif args.model == "CrossAttnHCSSLScorer":
        train_collate_fn = hcssl_collate_fn
        test_collate_fn = hcssl_collate_fn
    else:
        train_collate_fn = custom_collate_fn
        test_collate_fn = custom_collate_fn

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

    # Get input dimension from first sample
    first_sample = tr_dataset[0]
    input_dim = first_sample[2].shape[1]  # features are at index 2

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
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    train(audio_model, tr_dataloader, te_dataloader, args)


if __name__ == "__main__":
    main()
