import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IncidenceCache:
    def __init__(self, H: torch.Tensor, num_nodes: int):
        if H.ndim != 2 or H.shape[0] != 2:
            raise ValueError("H must have shape [2, nnz].")
        self.node_idx = H[0].long()
        self.edge_idx = H[1].long()
        self.num_nodes = int(num_nodes)
        self.num_edges = int(self.edge_idx.max().item()) + 1
        self.nnz = int(H.shape[1])
        self.node_deg = torch.bincount(self.node_idx, minlength=self.num_nodes).float().clamp_min_(1.0)
        self.edge_size = torch.bincount(self.edge_idx, minlength=self.num_edges).float().clamp_min_(1.0)
        self.node_inv_deg = 1.0 / self.node_deg
        self.edge_inv_size = 1.0 / self.edge_size
        self.edge_log_size = torch.log1p(self.edge_size)
        inc_col = torch.arange(self.nnz, device=self.node_idx.device)
        ones = torch.ones(self.nnz, device=self.node_idx.device)
        self.node_inc_mat = torch.sparse_coo_tensor(
            torch.stack([self.node_idx, inc_col]), ones, (self.num_nodes, self.nnz)
        ).coalesce()

    def to(self, device: torch.device) -> "IncidenceCache":
        new = object.__new__(IncidenceCache)
        for name in [
            "node_idx", "edge_idx", "node_deg", "edge_size",
            "node_inv_deg", "edge_inv_size", "edge_log_size", "node_inc_mat"
        ]:
            setattr(new, name, getattr(self, name).to(device))
        new.num_nodes = self.num_nodes
        new.num_edges = self.num_edges
        new.nnz = self.nnz
        return new


def aggregate_to_nodes(cache: IncidenceCache, src: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(cache.node_inc_mat, src)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructFeatureBuilder(nn.Module):
    """Simple structural signatures for each node."""

    def forward(self, cache: IncidenceCache) -> torch.Tensor:
        device = cache.node_idx.device
        n = cache.num_nodes
        inc_sizes = cache.edge_size[cache.edge_idx]
        inc_log_sizes = cache.edge_log_size[cache.edge_idx]

        mean_size = torch.zeros(n, device=device)
        mean_size.index_add_(0, cache.node_idx, inc_sizes)
        mean_size = mean_size * cache.node_inv_deg

        mean_sq = torch.zeros(n, device=device)
        mean_sq.index_add_(0, cache.node_idx, inc_sizes * inc_sizes)
        mean_sq = mean_sq * cache.node_inv_deg
        std_size = torch.sqrt((mean_sq - mean_size * mean_size).clamp_min(0.0))

        mean_log = torch.zeros(n, device=device)
        mean_log.index_add_(0, cache.node_idx, inc_log_sizes)
        mean_log = mean_log * cache.node_inv_deg

        inv_size_sum = torch.zeros(n, device=device)
        inv_size_sum.index_add_(0, cache.node_idx, 1.0 / inc_sizes)

        struct = torch.stack([
            torch.log1p(cache.node_deg),
            mean_size,
            std_size,
            mean_log,
            inv_size_sum,
        ], dim=-1)
        return struct


class NodeInitializer(nn.Module):
    def __init__(self, x_dim: int, s_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.s_proj = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = MLP(hidden_dim * 2, hidden_dim, 1, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = self.x_proj(x)
        hs = self.s_proj(s)
        gate = torch.sigmoid(self.gate(torch.cat([hx, hs], dim=-1)))
        h = gate * hx + (1.0 - gate) * hs
        return self.norm(h), gate.squeeze(-1)


class HyperedgeSlotDecomposer(nn.Module):
    """
    Lightweight latent event decomposition.
    For each hyperedge, infer K latent slots and incidence-slot assignments.

    This version supports chunked incidence processing to reduce peak memory.
    """

    def __init__(self, hidden_dim: int, num_slots: int, dropout: float, chunk_size: int = 131072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.chunk_size = chunk_size
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_dim) / math.sqrt(hidden_dim))
        self.edge_mlp = MLP(hidden_dim + 1, hidden_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cache: IncidenceCache) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_idx = cache.node_idx
        edge_idx = cache.edge_idx
        e = cache.num_edges
        d = h.size(-1)

        edge_mean = h.new_zeros((e, d))
        edge_mean.index_add_(0, edge_idx, h[node_idx])
        edge_mean = edge_mean * cache.edge_inv_size.unsqueeze(-1)
        edge_context = self.edge_mlp(torch.cat([edge_mean, cache.edge_log_size.unsqueeze(-1)], dim=-1))

        slot_seed = torch.tanh(edge_context.unsqueeze(1) + self.slot_queries.unsqueeze(0))  # [E,K,D]

        assign = h.new_empty((cache.nnz, self.num_slots))
        flat_slots = h.new_zeros((e * self.num_slots, d))
        flat_den = h.new_zeros((e * self.num_slots, 1))

        for start in range(0, cache.nnz, self.chunk_size):
            end = min(start + self.chunk_size, cache.nnz)
            ni = node_idx[start:end]
            ei = edge_idx[start:end]

            node_chunk = h[ni]
            node_proj_chunk = self.node_proj(node_chunk)

            score_chunks = []
            for k in range(self.num_slots):
                slot_inc = self.slot_proj(slot_seed[ei, k])
                score = ((node_proj_chunk * slot_inc).sum(dim=-1) / math.sqrt(self.hidden_dim)).unsqueeze(-1)
                score_chunks.append(score)

            assign_chunk = torch.softmax(torch.cat(score_chunks, dim=-1), dim=-1)
            assign[start:end] = assign_chunk

            for k in range(self.num_slots):
                flat_idx = ei * self.num_slots + k
                weight = assign_chunk[:, k:k+1]
                flat_slots.index_add_(0, flat_idx, weight * node_chunk)
                flat_den.index_add_(0, flat_idx, weight)

        slot_repr = flat_slots / flat_den.clamp_min(1e-6)
        slot_repr = self.dropout(slot_repr.view(e, self.num_slots, d))
        return edge_mean, slot_repr, assign


class TaskRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.num_classes = num_classes
        self.task_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.default_task = nn.Parameter(torch.zeros(hidden_dim))
        self.router = MLP(hidden_dim * 2 + 1, hidden_dim, 3, dropout)

    def task_vector(self, h: torch.Tensor, support_idx: Optional[torch.Tensor], support_y: Optional[torch.Tensor]) -> torch.Tensor:
        if support_idx is None or support_y is None or support_idx.numel() == 0:
            return self.default_task
        support_h = h[support_idx]
        global_mean = support_h.mean(dim=0)
        onehot = F.one_hot(support_y, num_classes=self.num_classes).float()
        counts = onehot.sum(dim=0).clamp_min_(1.0)
        protos = onehot.t() @ support_h
        protos = protos / counts.unsqueeze(-1)
        active = counts > 0
        proto_mean = protos[active].mean(dim=0) if active.any() else global_mean
        return self.task_mlp(torch.cat([global_mean, proto_mean], dim=-1))

    def route(self, slot_repr: torch.Tensor, edge_log_size: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        task_expand = task_vec.view(1, 1, -1).expand(slot_repr.size(0), slot_repr.size(1), -1)
        size_feat = edge_log_size.view(-1, 1, 1).expand(-1, slot_repr.size(1), 1)
        return torch.softmax(self.router(torch.cat([slot_repr, task_expand, size_feat], dim=-1)), dim=-1)


class FrequencyMixLayer(nn.Module):
    """Retain + low-pass + contrastive/high-pass mixture on slot messages.

    This version processes incidences in chunks to avoid building full nnz x hidden tensors.
    """

    def __init__(self, hidden_dim: int, num_slots: int, dropout: float, chunk_size: int = 131072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.chunk_size = chunk_size
        self.low_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.high_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.att_mlp = MLP(hidden_dim * 3 + 1, hidden_dim, 1, dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        cache: IncidenceCache,
        edge_mean: torch.Tensor,
        slot_repr: torch.Tensor,
        assign: torch.Tensor,
        route_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_idx = cache.node_idx
        edge_idx = cache.edge_idx
        n = cache.num_nodes
        d = h.size(-1)

        out = h.new_zeros((n, d))
        den = h.new_zeros((n, 1))
        att_sum = h.new_zeros(())

        for start in range(0, cache.nnz, self.chunk_size):
            end = min(start + self.chunk_size, cache.nnz)
            ni = node_idx[start:end]
            ei = edge_idx[start:end]

            node_inc = h[ni]
            edge_inc = edge_mean[ei]
            edge_size_inc = cache.edge_log_size[ei].unsqueeze(-1)
            abs_diff = torch.abs(node_inc - edge_inc)

            inc_att = F.softplus(
                self.att_mlp(torch.cat([node_inc, edge_inc, abs_diff, edge_size_inc], dim=-1))
            ) + 1e-6
            inc_att = inc_att / torch.sqrt(edge_size_inc + 1.0)

            inc_msg = node_inc.new_zeros((node_inc.size(0), d))
            assign_chunk = assign[start:end]

            for k in range(self.num_slots):
                slot_inc = slot_repr[ei, k]
                low_msg = self.low_mlp(torch.cat([node_inc, slot_inc], dim=-1))
                high_msg = self.high_mlp(torch.cat([node_inc, slot_inc - node_inc], dim=-1))
                alpha = route_w[ei, k]  # [chunk,3]
                msg = alpha[:, 0:1] * node_inc + alpha[:, 1:2] * low_msg + alpha[:, 2:3] * high_msg
                inc_msg = inc_msg + assign_chunk[:, k:k+1] * msg

            weighted_msg = inc_att * inc_msg
            out.index_add_(0, ni, weighted_msg)
            den.index_add_(0, ni, inc_att)
            att_sum = att_sum + inc_att.sum()

        out = out / den.clamp_min(1e-6)
        out = self.out_proj(out)
        att_mean = att_sum / max(cache.nnz, 1)
        return self.norm(h + self.dropout(out)), att_mean


class StructuredInferenceHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_slots: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.slot_cls = MLP(hidden_dim, hidden_dim, num_classes, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        self.slot_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.refine_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def _scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(min=math.log(1.0), max=math.log(100.0)).exp()

    def compute_prototypes(self, h: torch.Tensor, support_idx: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        support_h = h[support_idx]
        onehot = F.one_hot(support_y, num_classes=self.num_classes).float()
        counts = onehot.sum(dim=0).clamp_min_(1.0)
        protos = onehot.t() @ support_h
        protos = protos / counts.unsqueeze(-1)
        return F.normalize(protos, p=2, dim=-1)

    def slot_prior_logits(self, slot_repr: torch.Tensor, assign: torch.Tensor, cache: IncidenceCache, num_nodes: int) -> torch.Tensor:
        node_idx = cache.node_idx
        edge_idx = cache.edge_idx
        slot_logits = self.slot_cls(slot_repr)
        inc_logits = slot_logits.new_zeros((cache.nnz, self.num_classes))
        for k in range(self.num_slots):
            inc_logits = inc_logits + assign[:, k:k+1] * slot_logits[edge_idx, k]
        out = aggregate_to_nodes(cache, inc_logits)
        den = cache.node_deg.view(-1, 1).to(out.device)
        return out / den.clamp_min(1e-6)

    def inference(
        self,
        unary: torch.Tensor,
        slot_prior: torch.Tensor,
        assign: torch.Tensor,
        cache: IncidenceCache,
        support_idx: Optional[torch.Tensor],
        support_y: Optional[torch.Tensor],
        refine_steps: int,
        refine_tau: float,
    ) -> torch.Tensor:
        num_nodes = unary.size(0)
        node_idx = cache.node_idx
        edge_idx = cache.edge_idx
        num_edge_slots = cache.num_edges * self.num_slots
        slot_mix = torch.sigmoid(self.slot_weight)
        refine_mix = torch.sigmoid(self.refine_weight)

        logits = unary + slot_mix * slot_prior
        q = torch.softmax(logits / refine_tau, dim=-1)
        support_mask = None
        support_onehot = None
        if support_idx is not None and support_y is not None and support_idx.numel() > 0:
            support_mask = torch.zeros(num_nodes, dtype=torch.bool, device=unary.device)
            support_mask[support_idx] = True
            support_onehot = F.one_hot(support_y, num_classes=self.num_classes).float()
            q[support_idx] = support_onehot

        for _ in range(refine_steps):
            edge_slot_sum = q.new_zeros((num_edge_slots, self.num_classes))
            edge_slot_den = q.new_zeros((num_edge_slots, 1))
            for k in range(self.num_slots):
                flat_idx = edge_idx * self.num_slots + k
                weight = assign[:, k:k+1]
                edge_slot_sum.index_add_(0, flat_idx, weight * q[node_idx])
                edge_slot_den.index_add_(0, flat_idx, weight)
            edge_slot_post = edge_slot_sum / edge_slot_den.clamp_min(1e-6)
            edge_slot_post = edge_slot_post.view(cache.num_edges, self.num_slots, self.num_classes)

            inc_cls = q.new_zeros((cache.nnz, self.num_classes))
            for k in range(self.num_slots):
                inc_cls = inc_cls + assign[:, k:k+1] * edge_slot_post[edge_idx, k]
            node_msg = aggregate_to_nodes(cache, inc_cls)
            node_msg = node_msg / cache.node_deg.view(-1, 1).to(node_msg.device).clamp_min(1e-6)

            logits = unary + slot_mix * slot_prior + refine_mix * torch.log(node_msg.clamp_min(1e-8))
            q = torch.softmax(logits / refine_tau, dim=-1)
            if support_mask is not None:
                q[support_idx] = support_onehot

        return torch.log(q.clamp_min(1e-8))

    def forward(
        self,
        h: torch.Tensor,
        slot_repr: torch.Tensor,
        assign: torch.Tensor,
        cache: IncidenceCache,
        support_idx: torch.Tensor,
        support_y: torch.Tensor,
        refine_steps: int,
        refine_tau: float,
        hard_clamp: bool,
    ) -> torch.Tensor:
        protos = self.compute_prototypes(h, support_idx, support_y)
        unary = self._scale() * (h @ protos.t())
        slot_prior = self.slot_prior_logits(slot_repr, assign, cache, h.size(0))
        if not hard_clamp:
            return F.log_softmax(unary + torch.sigmoid(self.slot_weight) * slot_prior, dim=-1)
        with torch.no_grad():
            return self.inference(unary, slot_prior, assign, cache, support_idx, support_y, refine_steps, refine_tau)


class LIFTHG(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_slots: int = 3,
        dropout: float = 0.2,
        incidence_chunk_size: int = 131072,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_slots = num_slots

        self.struct_builder = StructFeatureBuilder()
        self.node_init = NodeInitializer(in_dim, 5, hidden_dim, dropout)
        self.router = TaskRouter(hidden_dim, num_classes, dropout)
        self.decomposer = HyperedgeSlotDecomposer(hidden_dim, num_slots, dropout, chunk_size=incidence_chunk_size)
        self.layers = nn.ModuleList(
            [FrequencyMixLayer(hidden_dim, num_slots, dropout, chunk_size=incidence_chunk_size) for _ in range(num_layers)]
        )
        self.head = StructuredInferenceHead(hidden_dim, num_classes, num_slots, dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.recon_decoder = MLP(hidden_dim * 2, hidden_dim, 1, dropout)

    def build_cache(self, H: torch.Tensor, num_nodes: int, device: Optional[torch.device] = None) -> IncidenceCache:
        cache = IncidenceCache(H, num_nodes)
        return cache.to(device) if device is not None else cache

    def encode(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        cache: Optional[IncidenceCache] = None,
        support_idx: Optional[torch.Tensor] = None,
        support_y: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        if cache is None:
            cache = self.build_cache(H, X.size(0), X.device)
        elif cache.node_idx.device != X.device:
            cache = cache.to(X.device)

        struct_feat = self.struct_builder(cache)
        h, init_gate = self.node_init(X, struct_feat)
        task_vec = self.router.task_vector(h, support_idx, support_y)

        att_vals = []
        route_vals = []
        edge_mean = None
        slot_repr = None
        assign = None
        for layer in self.layers:
            edge_mean, slot_repr, assign = self.decomposer(h, cache)
            route_w = self.router.route(slot_repr, cache.edge_log_size, task_vec)
            h, att_mean = layer(h, cache, edge_mean, slot_repr, assign, route_w)
            att_vals.append(att_mean)
            route_vals.append(route_w[..., 2].mean())

        h = F.normalize(self.final_norm(h), p=2, dim=-1)
        if return_stats:
            return h, edge_mean, slot_repr, assign, init_gate.mean(), torch.stack(route_vals).mean(), torch.stack(att_vals).mean()
        return h, edge_mean, slot_repr, assign

    def forward(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        support_idx: torch.Tensor,
        support_y: torch.Tensor,
        cache: Optional[IncidenceCache] = None,
        refine_steps: int = 4,
        refine_tau: float = 0.7,
        hard_clamp: bool = True,
        return_stats: bool = False,
    ):
        if return_stats:
            h, edge_mean, slot_repr, assign, init_gate, route_mean, att_mean = self.encode(
                X, H, cache=cache, support_idx=support_idx, support_y=support_y, return_stats=True
            )
            logits = self.head(h, slot_repr, assign, cache if cache is not None else self.build_cache(H, X.size(0), X.device), support_idx, support_y, refine_steps, refine_tau, hard_clamp)
            return logits, init_gate, route_mean, att_mean
        h, edge_mean, slot_repr, assign = self.encode(X, H, cache=cache, support_idx=support_idx, support_y=support_y, return_stats=False)
        return self.head(h, slot_repr, assign, cache if cache is not None else self.build_cache(H, X.size(0), X.device), support_idx, support_y, refine_steps, refine_tau, hard_clamp)

    def prototype_regularizer(self, logits: torch.Tensor, y: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits[train_idx], dim=-1)
        target = F.one_hot(y[train_idx], num_classes=self.num_classes).float()
        return F.mse_loss(probs, target)

    def reconstruction_loss(self, h: torch.Tensor, edge_mean: torch.Tensor, cache: IncidenceCache, num_samples: int = 20000) -> torch.Tensor:
        node_idx = cache.node_idx
        edge_idx = cache.edge_idx
        num_pos = min(num_samples, cache.nnz)
        perm = torch.randperm(cache.nnz, device=h.device)[:num_pos]
        pos_nodes = node_idx[perm]
        pos_edges = edge_idx[perm]
        neg_nodes = torch.randint(0, cache.num_nodes, (num_pos,), device=h.device)
        neg_edges = torch.randint(0, cache.num_edges, (num_pos,), device=h.device)
        pos = self.recon_decoder(torch.cat([h[pos_nodes], edge_mean[pos_edges]], dim=-1)).squeeze(-1)
        neg = self.recon_decoder(torch.cat([h[neg_nodes], edge_mean[neg_edges]], dim=-1)).squeeze(-1)
        return 0.5 * (
            F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
            F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        )


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return float("nan")
    pred = logits[idx].argmax(dim=-1)
    return (pred == y[idx]).float().mean().item()


@dataclass
class SplitResult:
    best_val: float
    test_acc: float
    best_epoch: int
    init_gate: float
    route_high: float
    att_mean: float


@torch.no_grad()
def evaluate_model(
    model: LIFTHG,
    X: torch.Tensor,
    H: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    cache: Optional[IncidenceCache] = None,
    refine_steps: int = 4,
    refine_tau: float = 0.7,
) -> Tuple[float, float, float, float]:
    model.eval()
    logits, init_gate, route_high, att_mean = model(
        X, H, train_idx, y[train_idx], cache=cache,
        refine_steps=refine_steps, refine_tau=refine_tau,
        hard_clamp=True, return_stats=True,
    )
    return accuracy(logits, y, eval_idx), float(init_gate.item()), float(route_high.item()), float(att_mean.item())


def train_one_split(
    model: LIFTHG,
    X: torch.Tensor,
    H: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    pretrain_epochs: int = 10,
    finetune_epochs: int = 200,
    patience: int = 50,
    proto_reg_weight: float = 0.1,
    recon_weight: float = 0.0,
    recon_samples: int = 20000,
    refine_steps: int = 4,
    refine_tau: float = 0.7,
) -> SplitResult:
    cache = model.build_cache(H, X.size(0), X.device)

    if pretrain_epochs > 0 and recon_weight > 0:
        pre_opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(pretrain_epochs):
            model.train()
            pre_opt.zero_grad(set_to_none=True)
            h, edge_mean, slot_repr, assign = model.encode(X, H, cache=cache, support_idx=None, support_y=None)
            loss = recon_weight * model.reconstruction_loss(h, edge_mean, cache, num_samples=recon_samples)
            loss.backward()
            pre_opt.step()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0
    best_epoch = -1
    wait = 0

    for epoch in range(finetune_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(
            X, H, train_idx, y[train_idx], cache=cache,
            refine_steps=0, refine_tau=refine_tau,
            hard_clamp=False, return_stats=False,
        )
        loss = F.nll_loss(logits[train_idx], y[train_idx]) + proto_reg_weight * model.prototype_regularizer(logits, y, train_idx)
        loss.backward()
        opt.step()

        val_acc, _, _, _ = evaluate_model(model, X, H, y, train_idx, val_idx, cache=cache, refine_steps=refine_steps, refine_tau=refine_tau)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, init_gate, route_high, att_mean = evaluate_model(
        model, X, H, y, train_idx, test_idx, cache=cache, refine_steps=refine_steps, refine_tau=refine_tau
    )
    return SplitResult(best_val, test_acc, best_epoch, init_gate, route_high, att_mean)
