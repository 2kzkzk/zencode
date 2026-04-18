from __future__ import annotations

import argparse
import random
import warnings
from time import time
from typing import List, Tuple

import numpy as np
import torch

try:
    from splits import create_splits as official_create_splits
except Exception:
    official_create_splits = None

from causal_model_structcorr import (
    RoutingHGConfig,
    SparseRoutingHyperProto,
    build_graph_cache,
    compute_routing_losses,
    episode_accuracy,
)

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def remap_labels_zero_based(y: torch.Tensor) -> torch.Tensor:
    uniq = sorted(int(v) for v in torch.unique(y).detach().cpu().tolist())
    mapping = {old: new for new, old in enumerate(uniq)}
    if uniq == list(range(len(uniq))):
        return y
    y_cpu = y.detach().cpu()
    y_new = y_cpu.clone()
    for old, new in mapping.items():
        y_new[y_cpu == old] = new
    return y_new.to(y.device)


@torch.no_grad()
def sanitize_features(X: torch.Tensor, standardize: bool = False) -> torch.Tensor:
    X = torch.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    if standardize and X.numel() > 0:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        X = (X - mean) / std
    return X.clamp(min=-1e4, max=1e4)


def create_splits(y: torch.Tensor, per_class: int) -> Tuple[List[int], List[int], List[int]]:
    if official_create_splits is None:
        raise RuntimeError(
            "Official splits are required for reproducible experiments. Could not import splits.create_splits."
        )
    return official_create_splits(y, per_class)


@torch.no_grad()
def split_episode_classes(y: torch.Tensor, train_idx: List[int], val_idx: List[int], test_idx: List[int]) -> torch.Tensor:
    classes = sorted({int(y[i].item()) for i in train_idx + val_idx + test_idx})
    return torch.tensor(classes, dtype=torch.long, device=y.device)


@torch.no_grad()
def evaluate_fixed_split(model, H, graph_cache, X, Y, support_idx, eval_idx, episode_classes):
    model.eval()
    out = model.forward_episode(H_raw=H, graph_cache=graph_cache, X=X, support_idx=support_idx, query_idx=eval_idx, Y=Y, episode_classes=episode_classes)
    covered = out["edge_covered"]
    purity_mean = float(out["edge_purity"][covered].mean().item()) if covered.any() else 0.0
    kept_ratio = float(out["edge_delete_mask"].float().mean().item())
    return {
        "acc": episode_accuracy(out, on="query"),
        "hop0": float(out["hop_mix"][0].item()),
        "hop1": float(out["hop_mix"][1].item()),
        "hop2": float(out["hop_mix"][2].item()),
        "edge_weight_mean": float(out["edge_weight"].mean().item()),
        "edge_gate_mean": float(out["edge_gate"].mean().item()),
        "edge_align_mean": float(out["edge_align"].mean().item()),
        "edge_purity_mean": purity_mean,
        "edge_support_frac_mean": float(out["edge_support_frac"].mean().item()),
        "edge_cover_mean": float(covered.float().mean().item()),
        "kept_ratio": kept_ratio,
        "num_added": float(out["num_added_edges"].item()),
        "added_w": float(out["added_edge_weight_mean"].item()),
    }


def main():
    parser = argparse.ArgumentParser("routing-hyper-proto-fair")
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("-k", "--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10, help="Ignored in fixed split mode.")
    parser.add_argument("--n_way", type=int, default=None, help="Ignored in fixed split mode.")
    parser.add_argument("--run", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler_patience", type=int, default=8)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--standardize_x", action="store_true")

    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--proto_temp", type=float, default=1.0)
    parser.add_argument("--route_hidden", type=int, default=64)
    parser.add_argument("--min_edge_weight", type=float, default=0.0)
    parser.add_argument("--max_edge_weight", type=float, default=1.0)
    parser.add_argument("--route_topk", type=int, default=64)
    parser.add_argument("--route_temp", type=float, default=0.7)
    parser.add_argument("--support_keep_ratio", type=float, default=0.8)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--query_weight", type=float, default=1.0)
    parser.add_argument("--support_weight", type=float, default=0.5)
    parser.add_argument("--strong_edge_weight", type=float, default=0.5)
    parser.add_argument("--weak_edge_weight", type=float, default=0.05)
    parser.add_argument("--stab_weight", type=float, default=0.2)
    parser.add_argument("--rank_weight", type=float, default=0.2)
    parser.add_argument("--sparse_weight", type=float, default=1e-3)
    parser.add_argument("--hop_entropy_weight", type=float, default=1e-3)

    # structure correction module
    parser.add_argument("--disable_struct_corr", action="store_true")
    parser.add_argument("--disable_delete", action="store_true")
    parser.add_argument("--enable_add", action="store_true")
    parser.add_argument("--delete_tau", type=float, default=0.10)
    parser.add_argument("--add_topm_nodes", type=int, default=12)
    parser.add_argument("--add_per_class", type=int, default=1)
    parser.add_argument("--add_min_score", type=float, default=0.70)

    args = parser.parse_args()

    set_seed(args.seed)
    H = torch.load(f"data/{args.data}/H.pt").to(args.device)
    X = sanitize_features(torch.load(f"data/{args.data}/X.pt").to(args.device), standardize=args.standardize_x)
    Y = remap_labels_zero_based(torch.load(f"data/{args.data}/Y.pt").to(args.device))
    num_classes = int(Y.max().item()) + 1
    graph_cache = build_graph_cache(H, X.size(0))

    print("Using standardized input features with support-aware structural statistics." if args.standardize_x else "Using raw input features with support-aware structural statistics.")
    print(f"Dataset '{args.data}' loaded. Nodes={X.size(0)} | Feat={X.size(1)} | Classes={num_classes}")
    print("Setting: transductive few-shot node classification on hypergraphs")
    print("Protocol: ZEN-compatible fixed split with k train / k val / rest test per class")
    print("Method: SparseRoutingHyperProto + optional Structure Correction Module")
    print(f"Ignoring --n_way={args.n_way} and --q_query={args.q_query}; using fixed split protocol.")

    test_accs = []
    for split_id in range(args.run):
        split_seed = args.seed + split_id
        set_seed(split_seed)

        train_idx, val_idx, test_idx = create_splits(Y, args.k_shot)
        episode_classes = split_episode_classes(Y, train_idx, val_idx, test_idx)
        support_idx = torch.tensor(train_idx, dtype=torch.long, device=args.device)
        val_tensor = torch.tensor(val_idx, dtype=torch.long, device=args.device)
        test_tensor = torch.tensor(test_idx, dtype=torch.long, device=args.device)

        cfg = RoutingHGConfig(
            in_dim=X.size(1),
            num_classes=num_classes,
            hid_dim=args.hid_dim,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            metric=args.metric,
            proto_temp=args.proto_temp,
            route_hidden=args.route_hidden,
            min_edge_weight=args.min_edge_weight,
            max_edge_weight=args.max_edge_weight,
            route_topk=args.route_topk,
            route_temp=args.route_temp,
            support_keep_ratio=args.support_keep_ratio,
            rank_margin=args.rank_margin,
            query_weight=args.query_weight,
            support_weight=args.support_weight,
            strong_edge_weight=args.strong_edge_weight,
            weak_edge_weight=args.weak_edge_weight,
            stab_weight=args.stab_weight,
            rank_weight=args.rank_weight,
            sparse_weight=args.sparse_weight,
            hop_entropy_weight=args.hop_entropy_weight,
            enable_struct_corr=not args.disable_struct_corr,
            enable_delete=not args.disable_delete,
            enable_add=args.enable_add,
            delete_tau=args.delete_tau,
            add_topm_nodes=args.add_topm_nodes,
            add_per_class=args.add_per_class,
            add_min_score=args.add_min_score,
        )
        model = SparseRoutingHyperProto(cfg).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience)

        best_val, best_state, bad, best_epoch = -1.0, None, 0, 0
        split_t0 = time()
        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.reset_peak_memory_stats(args.device)

        train_logs = {"loss_total": 0.0, "loss_query": 0.0, "loss_support": 0.0, "loss_edge": 0.0, "loss_weak": 0.0, "loss_stab": 0.0, "loss_rank": 0.0, "loss_sparse": 0.0}

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_episode(H_raw=H, graph_cache=graph_cache, X=X, support_idx=support_idx, query_idx=support_idx, Y=Y, episode_classes=episode_classes)
            loss, logs = compute_routing_losses(out, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite loss during training.")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_logs = {k: float(v) for k, v in logs.items()}

            if epoch % args.eval_every == 0:
                val_metrics = evaluate_fixed_split(model, H, graph_cache, X, Y, support_idx, val_tensor, episode_classes)
                scheduler.step(val_metrics["acc"])
                if val_metrics["acc"] > best_val:
                    best_val = val_metrics["acc"]
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += args.eval_every
                    if bad >= args.patience:
                        break

        if best_state is not None:
            model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})

        test_metrics = evaluate_fixed_split(model, H, graph_cache, X, Y, support_idx, test_tensor, episode_classes)
        test_accs.append(test_metrics["acc"])
        split_time = time() - split_t0
        max_mem_mb = 0.0
        if torch.cuda.is_available() and "cuda" in args.device:
            max_mem_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)

        print(
            f"[Split {split_id+1:02d}/{args.run}] Classes={episode_classes.numel()} | BestEpoch={best_epoch} | "
            f"BestVal={best_val*100:.2f} | Test={test_metrics['acc']*100:.2f} | "
            f"Route(w={test_metrics['edge_weight_mean']:.3f}, gate={test_metrics['edge_gate_mean']:.3f}, "
            f"align={test_metrics['edge_align_mean']:.3f}, purity={test_metrics['edge_purity_mean']:.3f}, "
            f"supfrac={test_metrics['edge_support_frac_mean']:.3f}, cover={test_metrics['edge_cover_mean']:.3f}, kept={test_metrics['kept_ratio']:.3f}) | "
            f"StructCorr(add={test_metrics['num_added']:.0f}, addw={test_metrics['added_w']:.3f}) | "
            f"Hops(h0={test_metrics['hop0']:.2f}, h1={test_metrics['hop1']:.2f}, h2={test_metrics['hop2']:.2f}) | "
            f"Train(loss={train_logs['loss_total']:.3f}, q={train_logs['loss_query']:.3f}, sup={train_logs['loss_support']:.3f}, edge={train_logs['loss_edge']:.3f}, weak={train_logs['loss_weak']:.3f}, stab={train_logs['loss_stab']:.3f}, rank={train_logs['loss_rank']:.3f}, sparse={train_logs['loss_sparse']:.3f}) | "
            f"time={split_time:.2f}s | mem={max_mem_mb:.0f}MB"
        )

    mean_acc, std_acc = np.mean(test_accs), np.std(test_accs)
    print(f"{args.data} | RoutingHG Accuracy: {mean_acc*100:.1f}+-{std_acc*100:.1f}")


if __name__ == "__main__":
    main()
