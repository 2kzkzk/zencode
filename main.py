from __future__ import annotations

import argparse
import copy
import random
import warnings
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from model import UERConfig, UERHGFast, precompute_states


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def create_splits(Y: torch.Tensor, k: int, val_k: int | None = None):
    if val_k is None:
        val_k = k
    label_to_indices = {}
    for idx, label in enumerate(Y.tolist()):
        label_to_indices.setdefault(int(label), []).append(idx)
    train_indices, val_indices = [], []
    for idxs in label_to_indices.values():
        idxs = idxs.copy()
        random.shuffle(idxs)
        train = idxs[: min(k, len(idxs))]
        valid = idxs[len(train): len(train) + min(val_k, max(len(idxs) - len(train), 0))]
        train_indices.extend(train)
        val_indices.extend(valid)
    all_idx = set(range(len(Y)))
    test_indices = sorted(list(all_idx - set(train_indices) - set(val_indices)))
    return sorted(train_indices), sorted(val_indices), test_indices


@torch.no_grad()
def evaluate(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor | List[int]) -> float:
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx, device=logits.device, dtype=torch.long)
    if idx.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return float((pred[idx] == y[idx]).float().mean().item())


@torch.no_grad()
def describe_dataset(data_name: str, H: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> None:
    num_nodes = X.shape[0]
    num_edges = int(H[1].max().item()) + 1
    num_classes = int(Y.max().item()) + 1
    incidence = H.shape[1]
    print(
        f"Dataset '{data_name}' loaded. Nodes={num_nodes} | Feat={X.shape[1]} | "
        f"Classes={num_classes} | Edges={num_edges} | Incidence={incidence}"
    )
    print('Method: UER-HG v2-fast (unified evidence backbone + weighted retrieval + shortlist val selection)')
    print('Setting: transductive few-shot node classification on hypergraphs | Unified body, no mode switching')


def alpha_templates_fast() -> List[Tuple[float, float, float]]:
    # reduced search space for speed
    return [
        (0.60, 0.30, 0.10),
        (0.55, 0.35, 0.10),
        (0.50, 0.35, 0.15),
        (0.45, 0.40, 0.15),
        (0.35, 0.40, 0.25),
        (0.25, 0.45, 0.30),
        (0.20, 0.45, 0.35),
        (0.20, 0.40, 0.40),
        (0.15, 0.45, 0.40),
    ]


def shrink_values_fast() -> List[float]:
    return [0.00, 0.10, 0.20]


def topk_values_fast() -> List[int]:
    return [1, 3]


def build_cfg(X: torch.Tensor, Y: torch.Tensor, alpha: Tuple[float, float, float], shrink: float, topk: int, args) -> UERConfig:
    return UERConfig(
        input_dim=X.shape[1],
        num_classes=int(Y.max().item()) + 1,
        alpha=alpha,
        base_shrink=shrink,
        topk=topk,
        dropout=args.dropout,
        hidden=args.calibrator_hidden,
        retr_temp=args.retr_temp,
        max_class_shrink=args.max_class_shrink,
    )


def stability_score(val_acc: float, train_acc: float, gap_lambda: float) -> float:
    gap = max(0.0, train_acc - val_acc)
    return val_acc - gap_lambda * gap


def train_model_for_candidate(states: Dict[str, torch.Tensor], Y: torch.Tensor,
                              support_idx_t: torch.Tensor, val_idx_t: torch.Tensor,
                              cfg: UERConfig, args, epochs: int, patience_limit: int):
    device = Y.device
    model = UERHGFast(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_score = -1.0
    best_epoch = 0
    best_train = 0.0
    patience = 0
    start = time()

    for epoch in range(1, epochs + 1):
        model.train()
        out = model(states, Y, support_idx_t)
        loss = model.loss(out, Y, support_idx_t)['total']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(states, Y, support_idx_t)
            logits = out['logits']
            val_acc = evaluate(logits, Y, val_idx_t)
            train_acc = evaluate(logits, Y, support_idx_t)
            score = stability_score(val_acc, train_acc, args.gap_lambda)
            if score > best_score:
                best_score = score
                best_val = val_acc
                best_epoch = epoch
                best_train = train_acc
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break

    model.load_state_dict(best_state)
    return model, {
        'best_val': float(best_val),
        'best_epoch': int(best_epoch),
        'best_train': float(best_train),
        'best_score': float(best_score),
        'cand_time': float(time() - start),
    }


def train_one_split(H: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, split, args) -> Dict[str, float]:
    split_start = time()
    support_idx, val_idx, test_idx = split
    device = X.device
    support_idx_t = torch.tensor(support_idx, device=device, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, device=device, dtype=torch.long)
    test_idx_t = torch.tensor(test_idx, device=device, dtype=torch.long)

    states = precompute_states(H, X)
    candidates = []
    # Stage 1: quick shortlist
    for alpha in alpha_templates_fast():
        for shrink in shrink_values_fast():
            for topk in topk_values_fast():
                cfg = build_cfg(X, Y, alpha, shrink, topk, args)
                _, metrics = train_model_for_candidate(states, Y, support_idx_t, val_idx_t, cfg, args,
                                                      epochs=args.quick_epochs, patience_limit=args.quick_patience)
                candidates.append({
                    'alpha': alpha,
                    'shrink': shrink,
                    'topk': topk,
                    **metrics,
                })

    candidates.sort(key=lambda x: x['best_score'], reverse=True)
    shortlist = candidates[: min(args.shortlist, len(candidates))]

    best = None
    # Stage 2: full train only shortlisted candidates
    for cand in shortlist:
        cfg = build_cfg(X, Y, cand['alpha'], cand['shrink'], cand['topk'], args)
        model, metrics = train_model_for_candidate(states, Y, support_idx_t, val_idx_t, cfg, args,
                                                  epochs=args.epochs, patience_limit=args.patience)
        if best is None or metrics['best_score'] > best['best_score']:
            best = {
                'model': model,
                'alpha': cand['alpha'],
                'shrink': cand['shrink'],
                'topk': cand['topk'],
                **metrics,
            }

    model = best['model']
    model.eval()
    with torch.no_grad():
        out = model(states, Y, support_idx_t)
        logits = out['logits']
    return {
        'best_epoch': int(best['best_epoch']),
        'best_val': float(best['best_val']),
        'train': evaluate(logits, Y, support_idx_t),
        'test': evaluate(logits, Y, test_idx_t),
        'alpha': list(best['alpha']),
        'shrink': float(best['shrink']),
        'topk': int(best['topk']),
        'split_time': float(time() - split_start),
        'best_cand_time': float(best['cand_time']),
        'searched': len(candidates),
        'shortlist': len(shortlist),
    }


def main() -> None:
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser('uer_hg_v2_fast')
    parser.add_argument('--data', '-data', type=str, default='cora')
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('--val_k', type=int, default=None)
    parser.add_argument('-run', '--run', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--quick_epochs', type=int, default=2)
    parser.add_argument('--quick_patience', type=int, default=1)
    parser.add_argument('--shortlist', type=int, default=6)
    parser.add_argument('--lr', type=float, default=8e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--calibrator_hidden', type=int, default=16)
    parser.add_argument('--retr_temp', type=float, default=3.0)
    parser.add_argument('--max_class_shrink', type=float, default=0.45)
    parser.add_argument('--gap_lambda', type=float, default=0.35)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dir = Path('data') / args.data
    H = torch.load(data_dir / 'H.pt', map_location='cpu').to(device).long()
    X = torch.load(data_dir / 'X.pt', map_location='cpu').to(device).float()
    Y = torch.load(data_dir / 'Y.pt', map_location='cpu').to(device).long()

    describe_dataset(args.data, H, X, Y)
    splits = [create_splits(Y.cpu(), args.k, args.val_k) for _ in range(args.run)]

    vals, tests, times = [], [], []
    for run_idx, split in enumerate(splits, start=1):
        result = train_one_split(H, X, Y, split, args)
        vals.append(result['best_val'])
        tests.append(result['test'])
        times.append(result['split_time'])
        alpha_fmt = f"({result['alpha'][0]:.2f}, {result['alpha'][1]:.2f}, {result['alpha'][2]:.2f})"
        print(
            f"[Split {run_idx:02d}/{args.run}] BestEpoch={result['best_epoch']} | "
            f"BestVal={result['best_val'] * 100:.2f} | Test={result['test'] * 100:.2f} | "
            f"Train={result['train'] * 100:.2f} | alpha={alpha_fmt} | shrink={result['shrink']:.2f} | "
            f"topk={result['topk']} | searched={result['searched']} | shortlist={result['shortlist']} | "
            f"split_time={result['split_time']:.2f}s | best_cand={result['best_cand_time']:.2f}s"
        )

    test_accs = np.array(vals if False else tests) * 100.0
    val_accs = np.array(vals) * 100.0
    times_arr = np.array(times)
    print('-' * 124)
    print(
        f"{args.data} | Val={val_accs.mean():.2f}±{val_accs.std():.2f} | "
        f"Test={test_accs.mean():.2f}±{test_accs.std():.2f} | Time={times_arr.mean():.2f}s"
    )


if __name__ == '__main__':
    main()
