import argparse
import random
from time import time
from typing import List, Tuple

import numpy as np
import torch

from causal_model import LIFTHG, set_seed, train_one_split


def create_splits(Y: torch.Tensor, k: int) -> Tuple[List[int], List[int], List[int]]:
    label_to_indices = {}
    for idx, label in enumerate(Y.tolist()):
        label_to_indices.setdefault(int(label), []).append(idx)

    train_indices = set()
    val_indices = set()
    for _, indices in label_to_indices.items():
        take = min(len(indices), 2 * k)
        sampled = random.sample(indices, take)
        train = sampled[: min(k, len(sampled))]
        val = sampled[min(k, len(sampled)):]
        train_indices.update(train)
        val_indices.update(val)

    all_indices = set(range(len(Y)))
    test_indices = all_indices - train_indices - val_indices
    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def load_dataset(data_name: str, device: torch.device):
    H = torch.load(f"data/{data_name}/H.pt", map_location=device).long()
    X = torch.load(f"data/{data_name}/X.pt", map_location=device).float()
    Y = torch.load(f"data/{data_name}/Y.pt", map_location=device).long()
    return H, X, Y


def maybe_standardize_features(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min_(1e-6)
    return (X - mean) / std


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LIFT-HG")
    parser.add_argument("-data", "--data", type=str, default="cora")
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("-run", "--run", type=int, default=10)
    parser.add_argument("-device", "--device", type=str, default="cuda:0")
    parser.add_argument("-n", "--n", type=int, default=9, help="kept for compatibility")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--incidence_chunk_size", type=int, default=131072)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--proto_reg_weight", type=float, default=0.1)
    parser.add_argument("--recon_weight", type=float, default=0.0)
    parser.add_argument("--recon_samples", type=int, default=20000)

    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--refine_tau", type=float, default=0.7)
    parser.add_argument("--standardize_x", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = resolve_device(args.device)

    H, X, Y = load_dataset(args.data, device)
    if args.standardize_x:
        X = maybe_standardize_features(X)

    num_classes = int(Y.max().item()) + 1
    print(
        f"Dataset '{args.data}' loaded | nodes={X.size(0)} incidences={H.size(1)} "
        f"classes={num_classes} feat_dim={X.size(1)}"
    )

    all_results = []
    all_times = []
    for run_idx in range(args.run):
        train_idx, val_idx, test_idx = create_splits(Y.cpu(), args.k)
        train_idx = torch.tensor(train_idx, device=device, dtype=torch.long)
        val_idx = torch.tensor(val_idx, device=device, dtype=torch.long)
        test_idx = torch.tensor(test_idx, device=device, dtype=torch.long)

        model = LIFTHG(
            in_dim=X.size(1),
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_slots=args.num_slots,
            dropout=args.dropout,
            incidence_chunk_size=args.incidence_chunk_size,
        ).to(device)

        start = time()
        result = train_one_split(
            model=model,
            X=X,
            H=H,
            y=Y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            patience=args.patience,
            proto_reg_weight=args.proto_reg_weight,
            recon_weight=args.recon_weight,
            recon_samples=args.recon_samples,
            refine_steps=args.refine_steps,
            refine_tau=args.refine_tau,
        )
        elapsed = time() - start
        all_results.append(result)
        all_times.append(elapsed)
        print(
            f"Run {run_idx + 1:02d} | val={result.best_val * 100:.2f} | "
            f"test={result.test_acc * 100:.2f} | best_epoch={result.best_epoch} | "
            f"init_gate={result.init_gate:.3f} | route_high={result.route_high:.3f} | "
            f"att={result.att_mean:.4f} | time={elapsed:.2f}s"
        )

    test_accs = np.array([r.test_acc for r in all_results])
    val_accs = np.array([r.best_val for r in all_results])
    gate_vals = np.array([r.init_gate for r in all_results])
    route_vals = np.array([r.route_high for r in all_results])
    att_vals = np.array([r.att_mean for r in all_results])

    print(
        f"{args.data} | Val: {val_accs.mean() * 100:.2f}+-{val_accs.std() * 100:.2f} | "
        f"Test: {test_accs.mean() * 100:.2f}+-{test_accs.std() * 100:.2f} | "
        f"InitGate: {gate_vals.mean():.3f} | RouteHigh: {route_vals.mean():.3f} | "
        f"Att: {att_vals.mean():.4f} | Time: {np.mean(all_times):.2f}s"
    )
