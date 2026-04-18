from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------


def _sanitize(x: torch.Tensor, clamp: float = 1e4) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)
    return x.clamp(min=-clamp, max=clamp)


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = _sanitize(x)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _edge_count(H: torch.Tensor) -> int:
    if H.numel() == 0:
        return 0
    return int(H[1].max().item()) + 1


@torch.no_grad()
def build_graph_cache(H: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
    device = H.device
    dtype = torch.float32
    node_idx = H[0].long()
    edge_idx = H[1].long()
    num_edges = _edge_count(H)

    sizes = torch.zeros(num_edges, device=device, dtype=dtype)
    sizes.index_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=dtype))
    sizes = sizes.clamp_min(1.0)

    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.index_add_(0, node_idx, torch.ones_like(node_idx, dtype=dtype))
    deg = deg.clamp_min(1.0)

    return {
        "node_idx": node_idx,
        "edge_idx": edge_idx,
        "num_nodes": torch.tensor(num_nodes, device=device),
        "num_edges": torch.tensor(num_edges, device=device),
        "sizes": sizes,
        "deg": deg,
    }


@torch.no_grad()
def compute_support_node_stats(graph_cache: Dict[str, torch.Tensor], support_idx: torch.Tensor) -> torch.Tensor:
    device = support_idx.device
    dtype = torch.float32
    num_nodes = int(graph_cache["num_nodes"].item())
    num_edges = int(graph_cache["num_edges"].item())
    node_idx = graph_cache["node_idx"]
    edge_idx = graph_cache["edge_idx"]
    sizes = graph_cache["sizes"]
    deg = graph_cache["deg"]

    support_flag = torch.zeros(num_nodes, device=device, dtype=dtype)
    if support_idx.numel() > 0:
        support_flag[support_idx] = 1.0

    edge_support = torch.zeros(num_edges, device=device, dtype=dtype)
    edge_support.index_add_(0, edge_idx, support_flag[node_idx])
    edge_support_frac = edge_support / sizes

    node_support_context = torch.zeros(num_nodes, device=device, dtype=dtype)
    node_support_context.index_add_(0, node_idx, edge_support_frac[edge_idx])
    node_support_context = node_support_context / deg

    node_support_touch = torch.zeros(num_nodes, device=device, dtype=dtype)
    node_support_touch.index_add_(0, node_idx, support_flag[node_idx])
    node_support_touch = node_support_touch / deg

    mean_edge_size = torch.zeros(num_nodes, device=device, dtype=dtype)
    mean_edge_size.index_add_(0, node_idx, sizes[edge_idx])
    mean_edge_size = mean_edge_size / deg
    log_mean_edge_size = torch.log1p(mean_edge_size)

    log_deg = torch.log1p(deg)
    return torch.stack([log_deg, log_mean_edge_size, node_support_context, node_support_touch], dim=1)


def hypergraph_weighted_mean_propagate(
    graph_cache: Dict[str, torch.Tensor],
    X: torch.Tensor,
    edge_weight: torch.Tensor,
    added_edges: List[torch.Tensor] | None = None,
    added_edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    node_idx = graph_cache["node_idx"]
    edge_idx = graph_cache["edge_idx"]
    sizes = graph_cache["sizes"].to(X.dtype)
    num_nodes = int(graph_cache["num_nodes"].item())
    num_edges = int(graph_cache["num_edges"].item())
    feat_dim = X.size(1)

    edge_sum = torch.zeros(num_edges, feat_dim, device=X.device, dtype=X.dtype)
    edge_sum.index_add_(0, edge_idx, X[node_idx])
    edge_feat = edge_sum / sizes.unsqueeze(1)
    edge_feat = edge_feat * edge_weight.unsqueeze(1)

    node_sum = torch.zeros(num_nodes, feat_dim, device=X.device, dtype=X.dtype)
    node_sum.index_add_(0, node_idx, edge_feat[edge_idx])

    node_den = torch.zeros(num_nodes, device=X.device, dtype=X.dtype)
    node_den.index_add_(0, node_idx, edge_weight[edge_idx])

    if added_edges is not None and len(added_edges) > 0 and added_edge_weights is not None:
        for e_nodes, w in zip(added_edges, added_edge_weights):
            if e_nodes.numel() == 0:
                continue
            e_repr = X[e_nodes].mean(dim=0, keepdim=True)
            node_sum[e_nodes] += w * e_repr
            node_den[e_nodes] += w

    out = node_sum / node_den.clamp_min(1e-6).unsqueeze(1)
    return _sanitize(out)


def _compute_prototypes(z: torch.Tensor, y_local: torch.Tensor, n_way: int) -> torch.Tensor:
    protos = []
    for c in range(n_way):
        mask = y_local == c
        if mask.any():
            protos.append(z[mask].mean(dim=0))
        else:
            protos.append(z.new_zeros(z.size(1)))
    return _sanitize(torch.stack(protos, dim=0))


def _pairwise_logits(z: torch.Tensor, protos: torch.Tensor, metric: str, temp: float) -> torch.Tensor:
    z = _sanitize(z)
    protos = _sanitize(protos)
    temp = max(float(temp), 1e-6)
    if metric == "cosine":
        return (_normalize(z) @ _normalize(protos).t()) / temp
    return -torch.cdist(z, protos, p=2).pow(2) / temp


@dataclass
class RoutingHGConfig:
    in_dim: int
    num_classes: int
    hid_dim: int = 128
    emb_dim: int = 64
    dropout: float = 0.2
    metric: str = "cosine"
    proto_temp: float = 1.0
    route_hidden: int = 64
    min_edge_weight: float = 0.0
    max_edge_weight: float = 1.0
    route_topk: int = 64
    route_temp: float = 0.7
    support_keep_ratio: float = 0.8
    rank_margin: float = 0.2
    query_weight: float = 1.0
    support_weight: float = 0.5
    strong_edge_weight: float = 0.5
    weak_edge_weight: float = 0.05
    stab_weight: float = 0.2
    rank_weight: float = 0.2
    sparse_weight: float = 1e-3
    hop_entropy_weight: float = 1e-3
    # Structure correction module
    enable_struct_corr: bool = True
    enable_delete: bool = True
    enable_add: bool = False
    delete_tau: float = 0.10
    add_topm_nodes: int = 12
    add_per_class: int = 1
    add_min_score: float = 0.70


class SparseRoutingHyperProto(nn.Module):
    def __init__(self, cfg: RoutingHGConfig):
        super().__init__()
        self.cfg = cfg
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.in_dim + 4, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hid_dim, cfg.emb_dim),
        )
        self.node_norm = nn.LayerNorm(cfg.emb_dim)
        self.route_mlp = nn.Sequential(
            nn.Linear(6 + cfg.num_classes, cfg.route_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.route_hidden, 1),
        )
        self.hop_logits = nn.Parameter(torch.tensor([1.2, 0.8, 0.3], dtype=torch.float32))

    def _local_targets(self, Y: torch.Tensor, indices: torch.Tensor, episode_classes: torch.Tensor) -> torch.Tensor:
        out = torch.empty(indices.size(0), dtype=torch.long, device=indices.device)
        for i, c in enumerate(episode_classes.tolist()):
            out[Y[indices] == c] = i
        return out

    def _node_embeddings(self, graph_cache: Dict[str, torch.Tensor], X: torch.Tensor, support_idx: torch.Tensor) -> torch.Tensor:
        X = _sanitize(X)
        stats = compute_support_node_stats(graph_cache, support_idx).to(X.dtype)
        inp = torch.cat([X, stats], dim=1)
        z = self.node_norm(self.node_encoder(inp))
        return _sanitize(z)

    def _edge_features(
        self,
        graph_cache: Dict[str, torch.Tensor],
        node_repr: torch.Tensor,
        support_idx: torch.Tensor,
        support_y_local: torch.Tensor,
        proto: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device, dtype = node_repr.device, node_repr.dtype
        node_idx = graph_cache["node_idx"]
        edge_idx = graph_cache["edge_idx"]
        sizes = graph_cache["sizes"].to(dtype)
        num_edges = int(graph_cache["num_edges"].item())
        n_way = proto.size(0)

        edge_sum = torch.zeros(num_edges, node_repr.size(1), device=device, dtype=dtype)
        edge_sum.index_add_(0, edge_idx, node_repr[node_idx])
        edge_repr = edge_sum / sizes.unsqueeze(1)

        node_logits = _pairwise_logits(node_repr, proto, self.cfg.metric, self.cfg.proto_temp)
        node_prob = F.softmax(node_logits, dim=1)
        node_prob = _sanitize(node_prob)

        edge_prob_sum = torch.zeros(num_edges, n_way, device=device, dtype=dtype)
        edge_prob_sum.index_add_(0, edge_idx, node_prob[node_idx])
        edge_prob = edge_prob_sum / sizes.unsqueeze(1)
        edge_align, edge_pred = edge_prob.max(dim=1)
        top2 = torch.topk(edge_prob, k=min(2, n_way), dim=1).values
        if n_way > 1:
            edge_margin = top2[:, 0] - top2[:, 1]
        else:
            edge_margin = top2[:, 0]
        edge_entropy = -(edge_prob * edge_prob.clamp_min(1e-8).log()).sum(dim=1)
        edge_entropy = edge_entropy / max(float(torch.log(torch.tensor(float(max(n_way, 2))))), 1e-6)

        support_flag = torch.zeros(node_repr.size(0), device=device, dtype=torch.bool)
        support_flag[support_idx] = True
        incidence_support = support_flag[node_idx]

        counts = torch.zeros(num_edges, n_way, device=device, dtype=dtype)
        support_node_idx = node_idx[incidence_support]
        support_edge_idx = edge_idx[incidence_support]
        if support_edge_idx.numel() > 0:
            support_label_map = torch.full((node_repr.size(0),), -1, device=device, dtype=torch.long)
            support_label_map[support_idx] = support_y_local
            touched_labels = support_label_map[support_node_idx]
            one_hot = F.one_hot(touched_labels.clamp_min(0), num_classes=n_way).to(dtype)
            counts.index_add_(0, support_edge_idx, one_hot)

        support_count = counts.sum(dim=1)
        support_frac = support_count / sizes
        purity = torch.where(
            support_count > 0,
            counts.max(dim=1).values / support_count.clamp_min(1.0),
            torch.zeros(num_edges, device=device, dtype=dtype),
        )
        covered = support_count > 0

        size_norm = torch.log1p(sizes)
        if num_edges > 1:
            size_norm = (size_norm - size_norm.mean()) / size_norm.std().clamp_min(1e-6)
        else:
            size_norm = torch.zeros_like(size_norm)

        # 6 scalar features + class distribution as semantic fingerprint
        base_feat = torch.stack([
            edge_align,
            edge_margin,
            1.0 - edge_entropy,
            support_frac.clamp(0.0, 1.0),
            purity,
            size_norm.tanh(),
        ], dim=1)
        feat = torch.cat([base_feat, edge_prob], dim=1)

        return {
            "edge_repr": edge_repr,
            "edge_prob": edge_prob,
            "edge_pred": edge_pred,
            "edge_align": edge_align,
            "edge_margin": edge_margin,
            "edge_entropy": edge_entropy,
            "support_frac": support_frac,
            "purity": purity,
            "covered": covered,
            "feat": feat,
        }

    def _route_scores(self, feat: torch.Tensor) -> torch.Tensor:
        scores = self.route_mlp(feat).squeeze(1)
        return _sanitize(scores, clamp=100.0)

    def _competitive_routing(self, scores: torch.Tensor, edge_pred: torch.Tensor, n_way: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = scores.device, scores.dtype
        gates = torch.zeros_like(scores)
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for c in range(n_way):
            idx = (edge_pred == c).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            k = min(self.cfg.route_topk, idx.numel())
            cls_scores = scores[idx]
            top_local = torch.topk(cls_scores, k=k, largest=True).indices
            chosen = idx[top_local]
            keep_mask[chosen] = True
            chosen_scores = scores[chosen] / max(self.cfg.route_temp, 1e-6)
            soft = F.softmax(chosen_scores, dim=0)
            # rescale to [0,1] with relative competition preserved
            gates[chosen] = soft / soft.max().clamp_min(1e-6)
        return gates, keep_mask

    def _structure_correction(
        self,
        gates: torch.Tensor,
        keep_mask: torch.Tensor,
        edge_info: Dict[str, torch.Tensor],
        support_idx: torch.Tensor,
        node_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # observed edges: delete only by zeroing out gates below threshold / not kept.
        delete_mask = torch.ones_like(gates, dtype=torch.bool)
        if self.cfg.enable_struct_corr and self.cfg.enable_delete:
            delete_mask = keep_mask & (gates >= self.cfg.delete_tau)
        obs_gates = gates * delete_mask.to(gates.dtype)

        added_edges: List[torch.Tensor] = []
        added_weights: List[torch.Tensor] = []

        if not (self.cfg.enable_struct_corr and self.cfg.enable_add):
            return obs_gates, delete_mask, added_edges, gates.new_zeros(0)

        n_way = node_prob.size(1)
        support_set = set(support_idx.detach().cpu().tolist())
        for c in range(n_way):
            cls_prob = node_prob[:, c]
            m = min(self.cfg.add_topm_nodes, cls_prob.numel())
            top_nodes = torch.topk(cls_prob, k=m, largest=True).indices
            # ensure supports of class can anchor the added edge if present
            support_anchor = [i for i in top_nodes.detach().cpu().tolist() if i in support_set]
            if len(support_anchor) == 0:
                continue
            score = cls_prob[top_nodes].mean()
            if float(score.item()) < self.cfg.add_min_score:
                continue
            added_edges.append(top_nodes)
            added_weights.append(score.clamp(0.0, 1.0))
            if len(added_edges) >= self.cfg.add_per_class * n_way:
                break

        if len(added_weights) == 0:
            return obs_gates, delete_mask, added_edges, gates.new_zeros(0)
        return obs_gates, delete_mask, added_edges, torch.stack(added_weights)

    def _perturbed_support(self, support_idx: torch.Tensor, support_y_local: torch.Tensor, n_way: int) -> torch.Tensor:
        keep = []
        for c in range(n_way):
            cls_nodes = support_idx[support_y_local == c]
            if cls_nodes.numel() == 0:
                continue
            k = max(1, int(round(cls_nodes.numel() * self.cfg.support_keep_ratio)))
            perm = torch.randperm(cls_nodes.numel(), device=cls_nodes.device)
            keep.append(cls_nodes[perm[:k]])
        if len(keep) == 0:
            return support_idx
        return torch.cat(keep, dim=0)

    def forward_episode(
        self,
        H_raw: torch.Tensor,
        graph_cache: Dict[str, torch.Tensor],
        X: torch.Tensor,
        support_idx: torch.Tensor,
        query_idx: torch.Tensor,
        Y: torch.Tensor,
        episode_classes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_way = episode_classes.numel()
        support_y_local = self._local_targets(Y, support_idx, episode_classes)
        query_y_local = self._local_targets(Y, query_idx, episode_classes)

        h0 = self._node_embeddings(graph_cache, X, support_idx)
        proto0 = _compute_prototypes(h0[support_idx], support_y_local, n_way)

        edge_info = self._edge_features(graph_cache, h0, support_idx, support_y_local, proto0)
        node_logits0 = _pairwise_logits(h0, proto0, self.cfg.metric, self.cfg.proto_temp)
        node_prob0 = F.softmax(node_logits0, dim=1)

        base_scores = self._route_scores(edge_info["feat"])
        base_gates, keep_mask = self._competitive_routing(base_scores, edge_info["edge_pred"], n_way)
        obs_gates, delete_mask, added_edges, added_weights = self._structure_correction(
            base_gates, keep_mask, edge_info, support_idx, node_prob0
        )
        edge_weight = self.cfg.min_edge_weight + (self.cfg.max_edge_weight - self.cfg.min_edge_weight) * obs_gates

        # Stability under support perturbation.
        pert_support = self._perturbed_support(support_idx, support_y_local, n_way)
        pert_y_local = self._local_targets(Y, pert_support, episode_classes)
        pert_proto = _compute_prototypes(h0[pert_support], pert_y_local, n_way)
        pert_edge_info = self._edge_features(graph_cache, h0, pert_support, pert_y_local, pert_proto)
        pert_scores = self._route_scores(pert_edge_info["feat"])

        # Multi-hop propagation on corrected graph.
        h1 = hypergraph_weighted_mean_propagate(graph_cache, h0, edge_weight, added_edges, added_weights)
        h2 = hypergraph_weighted_mean_propagate(graph_cache, h1, edge_weight, added_edges, added_weights)
        hop_mix = F.softmax(self.hop_logits, dim=0)
        z = hop_mix[0] * h0 + hop_mix[1] * h1 + hop_mix[2] * h2

        proto = _compute_prototypes(z[support_idx], support_y_local, n_way)
        logits_support = _pairwise_logits(z[support_idx], proto, self.cfg.metric, self.cfg.proto_temp)
        logits_query = _pairwise_logits(z[query_idx], proto, self.cfg.metric, self.cfg.proto_temp)

        return {
            "logits_support": logits_support,
            "logits_query": logits_query,
            "support_y_local": support_y_local,
            "query_y_local": query_y_local,
            "route_scores": base_scores,
            "route_scores_pert": pert_scores,
            "edge_gate": obs_gates,
            "edge_weight": edge_weight,
            "edge_keep_mask": keep_mask,
            "edge_delete_mask": delete_mask,
            "edge_align": edge_info["edge_align"],
            "edge_margin": edge_info["edge_margin"],
            "edge_entropy": edge_info["edge_entropy"],
            "edge_support_frac": edge_info["support_frac"],
            "edge_purity": edge_info["purity"],
            "edge_covered": edge_info["covered"],
            "edge_pred": edge_info["edge_pred"],
            "hop_mix": hop_mix,
            "node_prob": node_prob0,
            "num_added_edges": torch.tensor(len(added_edges), device=h0.device, dtype=torch.float32),
            "added_edge_weight_mean": added_weights.mean() if added_weights.numel() > 0 else h0.new_tensor(0.0),
        }


def compute_routing_losses(out: Dict[str, torch.Tensor], cfg: RoutingHGConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_q = F.cross_entropy(out["logits_query"], out["query_y_local"])
    loss_s = F.cross_entropy(out["logits_support"], out["support_y_local"])

    gates = out["edge_gate"].clamp(1e-5, 1 - 1e-5)
    covered = out["edge_covered"]
    uncovered = ~covered

    edge_align = out["edge_align"].detach().clamp(0.0, 1.0)
    purity = out["edge_purity"].detach().clamp(0.0, 1.0)
    strong_target = (0.5 * edge_align + 0.5 * purity).clamp(0.0, 1.0)
    weak_target = edge_align

    if covered.any():
        loss_edge_strong = F.binary_cross_entropy(gates[covered], strong_target[covered])
    else:
        loss_edge_strong = gates.new_tensor(0.0)

    if uncovered.any():
        loss_edge_weak = F.binary_cross_entropy(gates[uncovered], weak_target[uncovered])
    else:
        loss_edge_weak = gates.new_tensor(0.0)

    # support-stable routing loss
    loss_stab = F.mse_loss(out["route_scores"], out["route_scores_pert"])

    # ranking loss on covered edges: high target vs low target
    if covered.sum() >= 2:
        target_cov = strong_target[covered]
        score_cov = out["route_scores"][covered]
        pos_k = max(1, int(target_cov.numel() * 0.25))
        neg_k = max(1, int(target_cov.numel() * 0.25))
        pos_idx = torch.topk(target_cov, k=pos_k, largest=True).indices
        neg_idx = torch.topk(target_cov, k=neg_k, largest=False).indices
        pos_scores = score_cov[pos_idx]
        neg_scores = score_cov[neg_idx]
        pairs = min(pos_scores.numel(), neg_scores.numel())
        pos_scores = pos_scores[:pairs]
        neg_scores = neg_scores[:pairs]
        loss_rank = F.relu(cfg.rank_margin - pos_scores + neg_scores).mean()
    else:
        loss_rank = gates.new_tensor(0.0)

    loss_sparse = out["edge_gate"].mean() + 0.05 * out["added_edge_weight_mean"] + 0.01 * out["num_added_edges"]
    hop = out["hop_mix"].clamp_min(1e-8)
    hop_entropy = -(hop * hop.log()).sum() / max(float(torch.log(torch.tensor(3.0))), 1e-6)
    loss_hop = -hop_entropy

    total = (
        cfg.query_weight * loss_q
        + cfg.support_weight * loss_s
        + cfg.strong_edge_weight * loss_edge_strong
        + cfg.weak_edge_weight * loss_edge_weak
        + cfg.stab_weight * loss_stab
        + cfg.rank_weight * loss_rank
        + cfg.sparse_weight * loss_sparse
        + cfg.hop_entropy_weight * loss_hop
    )

    logs = {
        "loss_total": float(total.item()),
        "loss_query": float(loss_q.item()),
        "loss_support": float(loss_s.item()),
        "loss_edge": float(loss_edge_strong.item()),
        "loss_weak": float(loss_edge_weak.item()),
        "loss_stab": float(loss_stab.item()),
        "loss_rank": float(loss_rank.item()),
        "loss_sparse": float(loss_sparse.item()),
    }
    return total, logs


@torch.no_grad()
def episode_accuracy(out: Dict[str, torch.Tensor], on: str = "query") -> float:
    if on == "query":
        logits, y = out["logits_query"], out["query_y_local"]
    else:
        logits, y = out["logits_support"], out["support_y_local"]
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())
