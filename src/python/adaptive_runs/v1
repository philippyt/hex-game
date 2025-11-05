from __future__ import annotations
import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from parse_games import parse_csv_games, create_graphs_from_games
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

# first implementation

# NOTE: suboptimal implementation for demonstration purposes
# this ends with 3 symbols out of 18 left after 60 epochs, which is just 17% of representation
# this also uses raw TA literal frequency, which is unweighted by clause and doesn't account for which features stay active across epochs
# and then low activity features are not used yet these features are semantically important to hex logic
# tldr: gets geometric metadata, not game semantics

def compute_symbol_activity(tm: MultiClassGraphTsetlinMachine, graphs: Graphs) -> np.ndarray:
    clause_literals = tm.get_clause_literals(graphs.hypervectors)
    num_symbols = len(graphs.symbol_id)
    activity = np.zeros(num_symbols, dtype=np.float32)
    for c in range(clause_literals.shape[0]):
        for i in range(num_symbols):
            if clause_literals[c, i] == 1:
                activity[i] += 1
            elif clause_literals[c, i + num_symbols] == 1:
                activity[i] += 0.5
    return activity


def run_training_with_adaptive_selection(
    csv_path: str,
    board_dim: int = 5,
    epochs: int = 60,
    clause_count: int = 10000,
    T_value: int = 8000,
    s_value: float = 4.0,
    depth: int = 2,
    selection_interval: int = 10,
    retain_ratio: float = 0.8
) -> Dict[str, Any]:
    print(f"Loading games from {csv_path}")
    games = parse_csv_games(csv_path, board_dim = board_dim)
    n = len(games)
    indices = np.random.permutation(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_games = [games[i] for i in indices[:train_end]]
    val_games = [games[i] for i in indices[train_end:val_end]]
    test_games = [games[i] for i in indices[val_end:]]

    print("Creating graphs...")
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim = board_dim, hypervector_size = 256, hypervector_bits = 2)
    val_graphs, val_labels = create_graphs_from_games(val_games, board_dim = board_dim, init_with = train_graphs)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim = board_dim, init_with = train_graphs)
    print("Finished graph creation")

    print("Initializing model...")
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses = clause_count,
        T = T_value,
        s = s_value,
        depth = depth,
        message_size = 256,
        message_bits = 2,
        grid = (16 * 13, 1, 1),
        block = (128, 1, 1)
    )

    history: List[Dict[str, Any]] = []
    active_symbols = list(train_graphs.symbol_id.keys())

    for epoch in range(epochs):
        tm.fit(train_graphs, train_labels, epochs = 1, incremental = (epoch > 0))
        train_acc = 100 * np.mean(tm.predict(train_graphs) == train_labels)
        val_acc = 100 * np.mean(tm.predict(val_graphs) == val_labels)
        test_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
        print(f"Epoch {epoch + 1:02d}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}%")

        if (epoch + 1) % selection_interval == 0:
            print("Computing symbol activity...")
            activity = compute_symbol_activity(tm, train_graphs)
            symbol_names = [s for s in active_symbols if s in train_graphs.symbol_id]
            activity_map = dict(zip(symbol_names, activity[:len(symbol_names)]))
            sorted_symbols = sorted(activity_map.items(), key = lambda x: x[1], reverse = True)
            keep_count = int(len(sorted_symbols) * retain_ratio)
            kept = [s for s, _ in sorted_symbols[:keep_count]]
            dropped = [s for s, _ in sorted_symbols[keep_count:]]
            print(f"Rebuilding graphs with {len(kept)} / {len(sorted_symbols)} symbols retained")
            print(f"Kept symbols: {kept}")
            print(f"Dropped symbols: {dropped}")
            active_symbols = kept
            train_graphs, train_labels = create_graphs_from_games(train_games, board_dim = board_dim, hypervector_size = 256, hypervector_bits = 2)
            val_graphs, val_labels = create_graphs_from_games(val_games, board_dim = board_dim, init_with = train_graphs)
            test_graphs, test_labels = create_graphs_from_games(test_games, board_dim = board_dim, init_with = train_graphs)
            train_graphs.symbol_id = {k: v for k, v in train_graphs.symbol_id.items() if k in active_symbols}
            val_graphs.symbol_id = {k: v for k, v in val_graphs.symbol_id.items() if k in active_symbols}
            test_graphs.symbol_id = {k: v for k, v in test_graphs.symbol_id.items() if k in active_symbols}
            train_graphs.encode()
            val_graphs.encode()
            test_graphs.encode()

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "symbols": len(active_symbols)
        })

    return {
        "history": history,
        "tm": tm,
        "train_graphs": train_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "active_symbols": active_symbols,
        "val_games": val_games
    }


def run_ROAD_ROAR(
    tm: MultiClassGraphTsetlinMachine,
    graphs: Graphs,
    labels: np.ndarray,
    original_games: List[Tuple[np.ndarray, int]],
    board_dim: int = 5,
    keep_ratio: float = 0.5
) -> Dict[str, float]:
    print("Running ROAD / ROAR evaluation...")
    activity = compute_symbol_activity(tm, graphs)
    symbol_names = list(graphs.symbol_id.keys())
    activity_map = dict(zip(symbol_names, activity))
    sorted_symbols = sorted(activity_map.items(), key = lambda x: x[1], reverse = True)
    num_keep = int(len(sorted_symbols) * keep_ratio)
    top_symbols = [s for s, _ in sorted_symbols[:num_keep]]
    print("Rebuilding ROAD graphs (removing top features)...")
    road_graphs, _ = create_graphs_from_games(original_games, board_dim = board_dim, hypervector_size = 256, hypervector_bits = 2)
    road_graphs.symbol_id = {k: v for k, v in road_graphs.symbol_id.items() if k not in top_symbols}
    road_graphs.encode()
    road_acc = 100 * np.mean(tm.predict(road_graphs) == labels)
    print("Rebuilding ROAR graphs (retaining top features)...")
    roar_graphs, _ = create_graphs_from_games(original_games, board_dim = board_dim, hypervector_size = 256, hypervector_bits = 2)
    roar_graphs.symbol_id = {k: v for k, v in roar_graphs.symbol_id.items() if k in top_symbols}
    roar_graphs.encode()
    roar_acc = 100 * np.mean(tm.predict(roar_graphs) == labels)
    print(f"ROAD accuracy (remove top): {road_acc:.2f}%")
    print(f"ROAR accuracy (retain top): {roar_acc:.2f}%")
    return {"ROAD": road_acc, "ROAR": roar_acc}


if __name__ == "__main__":
    np.random.seed(42)
    result = run_training_with_adaptive_selection(csv_path = "hex_games_5.csv", board_dim = 5, epochs = 60, clause_count = 10000, T_value = 8000, s_value = 4.0, depth = 2, selection_interval = 10, retain_ratio = 0.8)
    tm = result["tm"]
    val_graphs = result["val_graphs"]
    val_labels = result["val_labels"]
    val_games = result["val_games"]
    run_ROAD_ROAR(tm = tm, graphs = val_graphs, labels = val_labels, original_games = val_games, board_dim = 5)
    os.makedirs("adaptive_results", exist_ok = True)
    with open("adaptive_results/history.json", "w") as f:
        json.dump(result["history"], f, indent = 2)
    print("Saved training history.")
