#!/usr/bin/env python3
"""Final submission generator for MLNS DSBA 2026.

This script writes exactly two files:
1. submission_crossrun_rank_meta_r2_w50_30_20_proba.csv
2. submission_crossrun_rank_meta_r2_aggr_w40_cat60_proba.csv

Methodology kept in this refactor:
- Base predictors on text, graph heuristics, SVD text embeddings, Node2Vec, and CatBoost.
- Final leaderboard files are rank-ensembles of these predictors.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

EPS = 1e-9

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def start_timeout(seconds: int) -> threading.Event:
    """Hard-stop process after `seconds` unless cancelled."""
    stop_event = threading.Event()
    if seconds <= 0:
        return stop_event

    def _killer() -> None:
        time.sleep(seconds)
        if not stop_event.is_set():
            print(f"Hard runtime cap reached ({seconds}s). Exiting.")
            os._exit(124)

    thread = threading.Thread(target=_killer, daemon=True)
    thread.start()
    return stop_event

# Rank average function, useful later to combine predictions
def rank_average(pred_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if not pred_list:
        raise ValueError("pred_list cannot be empty")

    n = len(pred_list[0])
    for p in pred_list:
        if len(p) != n:
            raise ValueError("All prediction arrays must have the same length")

    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + EPS)

    rank_preds: List[np.ndarray] = []
    for p in pred_list:
        order = np.argsort(p, kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        r /= max(n - 1, 1)
        rank_preds.append(r)

    out = np.zeros(n, dtype=np.float64)
    for wi, ri in zip(w, rank_preds):
        out += wi * ri
    return out.astype(np.float32)


@dataclass
class Scaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

# Prediction models for linear and mlp
class TorchBinaryClassifier:
    def __init__(
        self,
        input_dim: int,
        model_name: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        seed: int,
    ) -> None:
        self.input_dim = input_dim
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.model = self._build_model()
        self.scaler: Scaler | None = None

    def _build_model(self) -> torch.nn.Module:
        if self.model_name == "linear":
            return torch.nn.Linear(self.input_dim, 1)
        if self.model_name == "mlp":
            return torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.20),
                torch.nn.Linear(128, 1),
            )
        raise ValueError(f"Unknown model_name={self.model_name}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        set_seed(self.seed)
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.scaler = Scaler(mean_=mean, std_=std)
        x_scaled = self.scaler.transform(x).astype(np.float32)

        x_tensor = torch.from_numpy(x_scaled)
        y_tensor = torch.from_numpy(y.astype(np.float32))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x_tensor).squeeze(-1)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Model must be fit before predict_proba")

        x_scaled = self.scaler.transform(x).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor).squeeze(-1)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba.astype(np.float32)


def train_and_predict_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> np.ndarray:
    model = TorchBinaryClassifier(
        input_dim=x_train.shape[1],
        model_name="linear",
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)


def train_and_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> np.ndarray:
    model = TorchBinaryClassifier(
        input_dim=x_train.shape[1],
        model_name="mlp",
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)


def train_and_predict_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    subsample: float,
    rsm: float,
    random_strength: float,
    thread_count: int,
) -> np.ndarray:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        rsm=rsm,
        random_strength=random_strength,
        random_seed=seed,
        verbose=False,
        thread_count=thread_count,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1].astype(np.float32)


def load_inputs(
    train_path: Path,
    test_path: Path,
    node_info_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(train_path, sep=" ", header=None, names=["u", "v", "y"])
    test_df = pd.read_csv(test_path, sep=" ", header=None, names=["u", "v"])
    node_df = pd.read_csv(node_info_path, header=None)

    node_ids = node_df.iloc[:, 0].astype(np.int64).to_numpy()
    node_features = node_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    id_to_local = {nid: i for i, nid in enumerate(node_ids)}

    train_pairs = train_df[["u", "v"]].to_numpy(dtype=np.int64)
    test_pairs = test_df[["u", "v"]].to_numpy(dtype=np.int64)
    y = train_df["y"].to_numpy(dtype=np.float32)

    train_local = np.empty_like(train_pairs, dtype=np.int32)
    test_local = np.empty_like(test_pairs, dtype=np.int32)

    train_local[:, 0] = np.array([id_to_local[int(x)] for x in train_pairs[:, 0]], dtype=np.int32)
    train_local[:, 1] = np.array([id_to_local[int(x)] for x in train_pairs[:, 1]], dtype=np.int32)
    test_local[:, 0] = np.array([id_to_local[int(x)] for x in test_pairs[:, 0]], dtype=np.int32)
    test_local[:, 1] = np.array([id_to_local[int(x)] for x in test_pairs[:, 1]], dtype=np.int32)

    return train_local, test_local, y, node_features

# Utility function for graph features n°1
def build_adjacency(n_nodes: int, positive_edges_local: np.ndarray) -> List[set]:
    adj: List[set] = [set() for _ in range(n_nodes)]
    for u, v in positive_edges_local:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj

# Utility function for graph features n°2
def connected_components(adj: List[set]) -> np.ndarray:
    component = -np.ones(len(adj), dtype=np.int32)
    component_id = 0

    for node in range(len(adj)):
        if component[node] != -1:
            continue
        stack = [node]
        component[node] = component_id
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if component[nb] == -1:
                    component[nb] = component_id
                    stack.append(nb)
        component_id += 1

    return component

# Utility function for graph features n°3
def clustering_coefficients(adj: List[set]) -> np.ndarray:
    coeff = np.zeros(len(adj), dtype=np.float32)
    for node in range(len(adj)):
        neighbors = list(adj[node])
        d = len(neighbors)
        if d < 2:
            continue
        triangles = 0
        for i in range(d):
            ni = neighbors[i]
            sni = adj[ni]
            for j in range(i + 1, d):
                if neighbors[j] in sni:
                    triangles += 1
        coeff[node] = 2.0 * triangles / (d * (d - 1))
    return coeff

# Utility function for graph features n°4
def shortest_path_cutoff(adj: List[set], src: int, dst: int, cutoff: int = 4) -> int:
    if src == dst:
        return 0
    if dst in adj[src]:
        return 1

    frontier = {src}
    visited = {src}
    for depth in range(1, cutoff + 1):
        nxt = set()
        for node in frontier:
            for nb in adj[node]:
                if nb in visited:
                    continue
                if nb == dst:
                    return depth + 1
                visited.add(nb)
                nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return cutoff + 1

# Create the graph features (using the utility functions)
def graph_pair_features(
    pairs_local: np.ndarray,
    positive_edges_local: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    adj = build_adjacency(n_nodes=n_nodes, positive_edges_local=positive_edges_local)
    degree = np.array([len(nbs) for nbs in adj], dtype=np.int32)

    inv_log_deg = np.zeros(n_nodes, dtype=np.float32)
    inv_deg = np.zeros(n_nodes, dtype=np.float32)
    log_deg = np.log1p(degree.astype(np.float32))

    for i, d in enumerate(degree):
        if d > 1:
            inv_log_deg[i] = 1.0 / np.log(float(d))
        if d > 0:
            inv_deg[i] = 1.0 / float(d)

    component = connected_components(adj)
    clust = clustering_coefficients(adj)

    features = np.zeros((len(pairs_local), 19), dtype=np.float32)
    for i, (u, v) in enumerate(pairs_local):
        nu = adj[u]
        nv = adj[v]
        common = nu & nv
        cn = len(common)

        du = len(nu)
        dv = len(nv)
        denom = du + dv - cn + EPS

        features[i, 0] = cn
        features[i, 1] = cn / denom
        features[i, 2] = float(sum(inv_log_deg[w] for w in common))
        features[i, 3] = float(sum(inv_deg[w] for w in common))
        features[i, 4] = du * dv
        features[i, 5] = du
        features[i, 6] = dv
        features[i, 7] = abs(du - dv)
        features[i, 8] = clust[u]
        features[i, 9] = clust[v]
        features[i, 10] = abs(clust[u] - clust[v])
        features[i, 11] = float(component[u] == component[v])
        features[i, 12] = float(cn > 0)
        features[i, 13] = float(shortest_path_cutoff(adj, int(u), int(v), cutoff=4))
        features[i, 14] = float(cn)
        features[i, 15] = log_deg[u]
        features[i, 16] = log_deg[v]
        features[i, 17] = abs(log_deg[u] - log_deg[v])

        two_hop_overlap_count = 0
        for neighbor in nu:
            two_hop_overlap_count += len(adj[neighbor] & nv)
        features[i, 18] = float(two_hop_overlap_count)

    return features

# Create the text features
def text_pair_features(pairs_local: np.ndarray, node_features: np.ndarray) -> np.ndarray:
    node_norm = np.linalg.norm(node_features, axis=1).astype(np.float32) + EPS
    node_nnz = np.count_nonzero(node_features, axis=1).astype(np.float32)

    u = pairs_local[:, 0]
    v = pairs_local[:, 1]
    xu = node_features[u]
    xv = node_features[v]

    had = xu * xv
    abs_diff = np.abs(xu - xv)
    dot = np.sum(had, axis=1, keepdims=True)
    cos = dot / (node_norm[u, None] * node_norm[v, None] + EPS)

    nnz_u = node_nnz[u, None]
    nnz_v = node_nnz[v, None]
    nnz_diff = np.abs(nnz_u - nnz_v)

    return np.hstack([dot, cos, nnz_u, nnz_v, nnz_diff, abs_diff, had]).astype(np.float32)

# Create the embedding features
def embedding_pair_features(pairs_local: np.ndarray, node_embedding: np.ndarray) -> np.ndarray:
    u = pairs_local[:, 0]
    v = pairs_local[:, 1]
    eu = node_embedding[u]
    ev = node_embedding[v]

    had = eu * ev
    abs_diff = np.abs(eu - ev)
    dot = np.sum(had, axis=1, keepdims=True)

    norm_u = np.linalg.norm(eu, axis=1, keepdims=True)
    norm_v = np.linalg.norm(ev, axis=1, keepdims=True)
    cos = dot / (norm_u * norm_v + EPS)
    l1 = np.sum(abs_diff, axis=1, keepdims=True)
    l2 = np.linalg.norm(eu - ev, axis=1, keepdims=True)

    return np.hstack([dot, cos, l1, l2, abs_diff, had]).astype(np.float32)

# Create SVD embeddings
def compute_text_svd_embeddings(node_features: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    max_comp = min(node_features.shape[0] - 1, node_features.shape[1] - 1)
    n_components = max(2, min(n_components, max_comp))

    x_sparse = sparse.csr_matrix(node_features)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    z = svd.fit_transform(x_sparse).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + EPS
    return z

# Utility function for node2vec embeddings
def node2vec_random_walk(
    start: int,
    adj_list: List[List[int]],
    adj_set: List[set],
    walk_length: int,
    p: float,
    q: float,
    rng: np.random.Generator,
) -> List[int]:
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = adj_list[cur]
        if not neighbors:
            break

        if len(walk) == 1:
            nxt = int(rng.choice(neighbors))
        else:
            prev = walk[-2]
            weights = np.empty(len(neighbors), dtype=np.float64)
            for i, nb in enumerate(neighbors):
                if nb == prev:
                    weights[i] = 1.0 / max(p, EPS)
                elif prev in adj_set[nb]:
                    weights[i] = 1.0
                else:
                    weights[i] = 1.0 / max(q, EPS)
            probs = weights / (weights.sum() + EPS)
            nxt = int(rng.choice(neighbors, p=probs))

        walk.append(nxt)

    return walk

# Create node2vec embdeddings
def compute_node2vec_embeddings(
    positive_edges_local: np.ndarray,
    n_nodes: int,
    dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    context_window: int,
    seed: int,
) -> np.ndarray:
    adj = build_adjacency(n_nodes=n_nodes, positive_edges_local=positive_edges_local)
    adj_list = [list(s) for s in adj]

    rng = np.random.default_rng(seed)
    counts: Dict[Tuple[int, int], float] = defaultdict(float)

    nodes = np.arange(n_nodes, dtype=np.int32)
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = node2vec_random_walk(
                start=int(start),
                adj_list=adj_list,
                adj_set=adj,
                walk_length=walk_length,
                p=p,
                q=q,
                rng=rng,
            )
            m = len(walk)
            for i, u in enumerate(walk):
                lo = max(0, i - context_window)
                hi = min(m, i + context_window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    v = walk[j]
                    counts[(u, v)] += 1.0

    if not counts:
        return np.zeros((n_nodes, dim), dtype=np.float32)

    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int32)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int32)
    vals = np.fromiter((v for v in counts.values()), dtype=np.float32)

    cooc = sparse.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    cooc = cooc + sparse.eye(n_nodes, format="csr", dtype=np.float32)
    cooc.data = np.log1p(cooc.data)

    max_dim = min(cooc.shape[0] - 1, cooc.shape[1] - 1)
    dim = max(2, min(dim, max_dim))

    svd = TruncatedSVD(n_components=dim, random_state=seed)
    emb = svd.fit_transform(cooc).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + EPS
    return emb


def write_submission(path: Path, values: np.ndarray) -> None:
    out = pd.DataFrame({"ID": np.arange(len(values), dtype=np.int64), "Predicted": values})
    out.to_csv(path, index=False)


def validate_submission_csv(path: Path, expected_rows: int) -> None:
    df = pd.read_csv(path)
    if list(df.columns) != ["ID", "Predicted"]:
        raise ValueError(f"{path} has invalid columns")
    if len(df) != expected_rows:
        raise ValueError(f"{path} has {len(df)} rows; expected {expected_rows}")
    if not np.array_equal(df["ID"].to_numpy(), np.arange(expected_rows)):
        raise ValueError(f"{path} has invalid ID ordering/range")
    pred = df["Predicted"].to_numpy(dtype=np.float64)
    if np.isnan(pred).any() or np.isinf(pred).any():
        raise ValueError(f"{path} has NaN/Inf predictions")

# Create all the 3 types of representations (text features, graph features, embeddings)
def build_feature_blocks(
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    y: np.ndarray,
    node_features: np.ndarray,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n_nodes = node_features.shape[0]
    positive_edges = train_pairs[y == 1]

    text_train = text_pair_features(train_pairs, node_features)
    text_test = text_pair_features(test_pairs, node_features)

    graph_train = graph_pair_features(train_pairs, positive_edges, n_nodes=n_nodes)
    graph_test = graph_pair_features(test_pairs, positive_edges, n_nodes=n_nodes)

    svd_emb = compute_text_svd_embeddings(node_features, n_components=128, seed=seed)
    svd_train = embedding_pair_features(train_pairs, svd_emb)
    svd_test = embedding_pair_features(test_pairs, svd_emb)

    n2v_emb = compute_node2vec_embeddings(
        positive_edges_local=positive_edges,
        n_nodes=n_nodes,
        dim=160,
        walk_length=48,
        num_walks=28,
        p=1.0,
        q=0.35,
        context_window=5,
        seed=seed + 100,
    )
    n2v_train = embedding_pair_features(train_pairs, n2v_emb)
    n2v_test = embedding_pair_features(test_pairs, n2v_emb)

    cat_train = np.hstack([text_train, graph_train, svd_train, n2v_train]).astype(np.float32)
    cat_test = np.hstack([text_test, graph_test, svd_test, n2v_test]).astype(np.float32)

    return {
        "text": (text_train, text_test),
        "graph": (graph_train, graph_test),
        "svd": (svd_train, svd_test),
        "node2vec": (n2v_train, n2v_test),
        "catboost": (cat_train, cat_test),
    }

# Create the base prediction models (to be combined later)
def train_base_predictors(
    feature_blocks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}

    linear_lr = 2e-3
    linear_weight_decay = 3e-4
    linear_epochs = 160

    mlp_lr = 6e-4
    mlp_weight_decay = 5e-4
    mlp_epochs = 220

    text_train, text_test = feature_blocks["text"]
    graph_train, graph_test = feature_blocks["graph"]
    svd_train, svd_test = feature_blocks["svd"]
    n2v_train, n2v_test = feature_blocks["node2vec"]
    cat_train, cat_test = feature_blocks["catboost"]

    predictions["text_linear"] = train_and_predict_linear(
        text_train, y, text_test,
        seed=seed + 10, lr=linear_lr, weight_decay=linear_weight_decay, epochs=linear_epochs,
    )
    predictions["text_mlp"] = train_and_predict_mlp(
        text_train, y, text_test,
        seed=seed + 20, lr=mlp_lr, weight_decay=mlp_weight_decay, epochs=mlp_epochs,
    )
    predictions["graph_linear"] = train_and_predict_linear(
        graph_train, y, graph_test,
        seed=seed + 30, lr=linear_lr, weight_decay=linear_weight_decay, epochs=linear_epochs,
    )
    predictions["svd_linear"] = train_and_predict_linear(
        svd_train, y, svd_test,
        seed=seed + 40, lr=linear_lr, weight_decay=linear_weight_decay, epochs=linear_epochs,
    )
    predictions["node2vec_linear"] = train_and_predict_linear(
        n2v_train, y, n2v_test,
        seed=seed + 50, lr=linear_lr, weight_decay=linear_weight_decay, epochs=linear_epochs,
    )

    predictions["catboost"] = train_and_predict_catboost(
        x_train=cat_train,
        y_train=y,
        x_eval=cat_test,
        seed=seed + 60,
        iterations=900,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=4.0,
        subsample=0.85,
        rsm=0.5,
        random_strength=1.0,
        thread_count=4,
    )

    return predictions

# Combine the base prediction models
def build_final_predictions(base_pred: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # No-CatBoost (text + graph + embedding predictors).
    no_catboost = rank_average(
        [
            base_pred["text_linear"],
            base_pred["text_mlp"],
            base_pred["graph_linear"],
            base_pred["svd_linear"],
            base_pred["node2vec_linear"],
        ],
        [0.35, 0.20, 0.15, 0.10, 0.20],
    )

    # CatBoost + Node2Vec + text_mlp
    cat_n2v_textmlp = rank_average(
        [base_pred["catboost"], base_pred["node2vec_linear"], base_pred["text_mlp"]],
        [0.55, 0.30, 0.15],
    )

    # CatBoost + Node2Vec + graph
    cat_n2v_graph = rank_average(
        [base_pred["catboost"], base_pred["node2vec_linear"], base_pred["graph_linear"]],
        [0.70, 0.20, 0.10],
    )

    model_1 = rank_average(
        [no_catboost, cat_n2v_textmlp, cat_n2v_graph],
        [0.50, 0.30, 0.20],
    )

    model_2 = rank_average(
        [model_1, cat_n2v_graph],
        [0.40, 0.60],
    )

    return {
        "submission_1.csv": model_1,
        "submission_2": model_2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final submission generator")
    parser.add_argument("--max-runtime-minutes", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-path", type=Path, default=Path("train.txt"))
    parser.add_argument("--test-path", type=Path, default=Path("test.txt"))
    parser.add_argument("--node-info-path", type=Path, default=Path("node_information.csv"))
    parser.add_argument("--torch-num-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(args.torch_num_threads, 1))
    set_seed(args.seed)

    timeout_seconds = int(args.max_runtime_minutes * 60)
    stop_event = start_timeout(timeout_seconds)
    try:
        train_pairs, test_pairs, y, node_features = load_inputs(
            train_path=args.train_path,
            test_path=args.test_path,
            node_info_path=args.node_info_path,
        )

        feature_blocks = build_feature_blocks(
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            y=y,
            node_features=node_features,
            seed=args.seed,
        )
        base_pred = train_base_predictors(feature_blocks=feature_blocks, y=y, seed=args.seed)
        final_pred = build_final_predictions(base_pred)

        expected_rows = len(test_pairs)
        for filename, values in final_pred.items():
            path = Path(filename)
            write_submission(path, values)
            validate_submission_csv(path, expected_rows=expected_rows)
            print(f"Generated: {filename}")
    finally:
        stop_event.set()

    print("Done.")


if __name__ == "__main__":
    main()
