from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UERConfig:
    input_dim: int
    num_classes: int
    alpha: Tuple[float, float, float]
    base_shrink: float
    topk: int
    hidden: int = 16
    dropout: float = 0.0
    retr_temp: float = 3.0
    max_class_shrink: float = 0.45


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def scatter_count(index: torch.Tensor, dim_size: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(dim_size, device=device, dtype=torch.float32)
    out.index_add_(0, index, torch.ones(index.numel(), device=device, dtype=torch.float32))
    return out


def compute_degrees(H: torch.Tensor):
    nodes, edges = H[0].long(), H[1].long()
    num_nodes = int(nodes.max().item()) + 1
    num_edges = int(edges.max().item()) + 1
    dv = scatter_count(nodes, num_nodes, H.device).clamp_min(1.0)
    de = scatter_count(edges, num_edges, H.device).clamp_min(1.0)
    return dv, de, num_nodes, num_edges


def hypergraph_propagate(H: torch.Tensor, X: torch.Tensor, dv: torch.Tensor, de: torch.Tensor) -> torch.Tensor:
    nodes, edges = H[0].long(), H[1].long()
    num_nodes = X.shape[0]
    x = safe_normalize(X, dim=1)
    node_scaled = x / dv.sqrt().unsqueeze(1)
    edge_msg = scatter_sum(node_scaled[nodes], edges, int(de.numel())) / de.unsqueeze(1)
    node_msg = scatter_sum(edge_msg[edges], nodes, num_nodes) / dv.sqrt().unsqueeze(1)
    return safe_normalize(node_msg, dim=1)


@torch.no_grad()
def precompute_states(H: torch.Tensor, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    dv, de, _, _ = compute_degrees(H)
    x0 = safe_normalize(X, dim=1)
    p1 = hypergraph_propagate(H, x0, dv, de)
    p2 = hypergraph_propagate(H, p1, dv, de)
    return {'x0': x0, 'p1': p1, 'p2': p2}


def class_adaptive_prototypes(z: torch.Tensor, y: torch.Tensor, support_idx: torch.Tensor,
                              num_classes: int, base_shrink: float, max_shrink: float) -> Tuple[torch.Tensor, torch.Tensor]:
    support_y = y[support_idx]
    d = z.shape[1]
    protos = z.new_zeros(num_classes, d)
    counts = z.new_zeros(num_classes)
    protos.index_add_(0, support_y, z[support_idx])
    counts.index_add_(0, support_y, torch.ones_like(support_y, dtype=z.dtype))
    protos = protos / counts.clamp_min(1.0).unsqueeze(1)
    protos = safe_normalize(protos, dim=1)

    global_proto = safe_normalize(z[support_idx].mean(dim=0, keepdim=True), dim=1)
    dispersions = z.new_full((num_classes,), 1.0)
    for c in range(num_classes):
        cls_idx = support_idx[support_y == c]
        if cls_idx.numel() > 0:
            sims = (z[cls_idx] @ protos[c].unsqueeze(1)).squeeze(1)
            dispersions[c] = (1.0 - sims.mean()).clamp_min(0.0)
    mean_disp = dispersions.mean().clamp_min(1e-6)
    class_shrink = (base_shrink * (dispersions / mean_disp)).clamp(0.0, max_shrink)
    protos = safe_normalize((1.0 - class_shrink).unsqueeze(1) * protos + class_shrink.unsqueeze(1) * global_proto, dim=1)
    return protos, class_shrink


class EvidenceCalibrator(nn.Module):
    def __init__(self, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(4, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(4, 1)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.net(feats).squeeze(-1)
        temp = self.log_temp.exp().clamp(0.25, 10.0)
        return logits * temp


class UERHGFast(nn.Module):
    def __init__(self, cfg: UERConfig):
        super().__init__()
        self.cfg = cfg
        self.alpha = tuple(float(v) for v in cfg.alpha)
        self.base_shrink = float(cfg.base_shrink)
        self.topk = int(cfg.topk)
        self.retr_temp = float(cfg.retr_temp)
        self.max_class_shrink = float(cfg.max_class_shrink)
        self.calibrator = EvidenceCalibrator(hidden=cfg.hidden, dropout=cfg.dropout)

    def fuse(self, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0, p1, p2 = states['x0'], states['p1'], states['p2']
        z = x0 + self.alpha[1] * (p1 - x0) + self.alpha[2] * (p2 - p1)
        z = safe_normalize(z, dim=1)
        return {'x0': x0, 'p1': p1, 'p2': p2, 'z': z}

    def _weighted_retrieval_scores(self, states: Dict[str, torch.Tensor], y: torch.Tensor, support_idx: torch.Tensor) -> torch.Tensor:
        z, x0, p1 = states['z'], states['x0'], states['p1']
        num_nodes = z.shape[0]
        num_classes = self.cfg.num_classes
        support_y = y[support_idx]
        retr = z.new_zeros(num_nodes, num_classes)
        for c in range(num_classes):
            cls_idx = support_idx[support_y == c]
            if cls_idx.numel() == 0:
                continue
            kk = min(self.topk, cls_idx.numel())
            sim_z = z @ z[cls_idx].t()
            sim_x = x0 @ x0[cls_idx].t()
            sim_p = p1 @ p1[cls_idx].t()
            # lightweight evidence contribution calibration
            contrib = 0.60 * sim_z + 0.25 * sim_x + 0.15 * sim_p
            top_vals, top_pos = contrib.topk(kk, dim=1)
            chosen_sim_z = sim_z.gather(1, top_pos)
            weights = F.softmax(self.retr_temp * top_vals, dim=1)
            retr[:, c] = (weights * chosen_sim_z).sum(dim=1)
        return retr

    def evidence_features(self, states: Dict[str, torch.Tensor], y: torch.Tensor, support_idx: torch.Tensor):
        fused = self.fuse(states)
        x0, p1, z = fused['x0'], fused['p1'], fused['z']
        proto_z, class_shrink = class_adaptive_prototypes(z, y, support_idx, self.cfg.num_classes, self.base_shrink, self.max_class_shrink)
        proto_x, _ = class_adaptive_prototypes(x0, y, support_idx, self.cfg.num_classes, self.base_shrink, self.max_class_shrink)
        proto_p1, _ = class_adaptive_prototypes(p1, y, support_idx, self.cfg.num_classes, self.base_shrink, self.max_class_shrink)
        proto_score = z @ proto_z.t()
        struct_score = p1 @ proto_p1.t()
        sem_score = x0 @ proto_x.t()
        retr_score = self._weighted_retrieval_scores(fused, y, support_idx)
        feats = torch.stack([proto_score, struct_score, retr_score, sem_score], dim=2)
        aux = {'proto_score': proto_score, 'retr_score': retr_score, 'class_shrink': class_shrink}
        return feats, aux

    def forward(self, states: Dict[str, torch.Tensor], y: torch.Tensor, support_idx: torch.Tensor):
        feats, aux = self.evidence_features(states, y, support_idx)
        logits = self.calibrator(feats)
        return {
            'logits': logits,
            'alpha': torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            'class_shrink': aux['class_shrink'],
            'proto_logits': aux['proto_score'],
            'retr_logits': aux['retr_score'],
        }

    def loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, train_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = F.cross_entropy(out['logits'][train_idx], y[train_idx])
        return {'total': ce, 'ce': ce}
