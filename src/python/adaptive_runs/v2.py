from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from parse_games import parse_csv_games, create_graphs_from_games
import json
import os

# NOTE: much better, the adaptive puning is working as intended
# training stabilized and recovered after pruning, meaning TM is reorganizing around the reduced symbol set effectively
# the 2% difference in ROAD and ROAR accuracy shows the most active symbols carry predictive value
# but pruning threshold is too aggressive, ROAR/ROAD results show model has redudancy

# TODO: warm up phase before pruning, smooth activity trends, switch ROAD/ROAR to val set, raise threshold, make plots

def compute_symbol_activity(tm: MultiClassGraphTsetlinMachine, graphs: Graphs, prev_activity: Dict[str, float] | None = None, alpha: float = 0.3, num_classes: int | None = None) -> Dict[str, float]:
    num_symbols = len(graphs.symbol_id)
    symbol_names = list(graphs.symbol_id.keys())
    activity = np.zeros(num_symbols, dtype = np.float32)
    if num_classes is None:
        num_classes = getattr(tm, "number_of_classes", 2)
    try:
        clause_literals = tm.get_clause_literals(graphs.hypervectors)
    except Exception:
        clause_literals = None
    if clause_literals is not None:
        for c in range(clause_literals.shape[0]):
            for i in range(num_symbols):
                if clause_literals[c, i] == 1:
                    activity[i] += 1.0
                elif clause_literals[c, i + num_symbols] == 1:
                    activity[i] += 0.5
    activity /= max(1, tm.number_of_clauses * num_classes)
    if prev_activity:
        for sym in symbol_names:
            idx = symbol_names.index(sym)
            old_val = prev_activity.get(sym, activity[idx])
            activity[idx] = alpha * activity[idx] + (1 - alpha) * old_val
    return dict(zip(symbol_names, activity.tolist()))

def run_training_with_adaptive_selection(csv_path: str, board_dim: int = 5, epochs: int = 60, clause_count: int = 10000, T_value: int = 8000, s_value: float = 4.0, depth: int = 2, selection_interval: int = 7, activity_threshold: float = 0.05, min_symbols: int = 10) -> Dict[str, Any]:
    print(f"Loading games from {csv_path}")
    games = parse_csv_games(csv_path, board_dim = board_dim)
    total = len(games)
    indices = np.random.permutation(total)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)
    train_games = [games[i] for i in indices[:train_end]]
    val_games = [games[i] for i in indices[train_end:val_end]]
    test_games = [games[i] for i in indices[val_end:]]
    print("Creating graphs...")
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim = board_dim, hypervector_size = 256, hypervector_bits = 2)
    val_graphs, val_labels = create_graphs_from_games(val_games, board_dim = board_dim, init_with = train_graphs)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim = board_dim, init_with = train_graphs)
    print("Finished graph creation")
    tm = MultiClassGraphTsetlinMachine(number_of_clauses = clause_count, T = T_value, s = s_value, depth = depth, message_size = 256, message_bits = 2, grid = (16 * 13, 1, 1), block = (128, 1, 1))
    print("Initializing model...")
    history: List[Dict[str, Any]] = []
    stability: Dict[str, List[float]] = {}
    prev_activity: Dict[str, float] | None = None
    active_symbols = list(train_graphs.symbol_id.keys())
    semantic_protect = {"Adjacent", "SameColor_Pos", "SameColor_Neg"}
    for epoch in range(epochs):
        tm.fit(train_graphs, train_labels, epochs = 1, incremental = (epoch > 0))
        train_acc = 100 * np.mean(tm.predict(train_graphs) == train_labels)
        val_acc = 100 * np.mean(tm.predict(val_graphs) == val_labels)
        test_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
        print(f"Epoch {epoch + 1:02d}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}%")
        if (epoch + 1) % selection_interval == 0 and len(active_symbols) > min_symbols:
            print("Computing symbol activity...")
            activity = compute_symbol_activity(tm, train_graphs, prev_activity)
            prev_activity = activity.copy()
            for sym, val in activity.items():
                if sym not in stability:
                    stability[sym] = []
                stability[sym].append(val)
                if len(stability[sym]) > 3:
                    stability[sym].pop(0)
            avg_activity = {sym: np.mean(vals) for sym, vals in stability.items()}
            inactive = [s for s, val in avg_activity.items() if val < activity_threshold and s not in semantic_protect]
            recently_inactive = [s for s in inactive if len(stability[s]) >= 2 and all(v < activity_threshold for v in stability[s][-2:])]
            if recently_inactive:
                print(f"Dropping {len(recently_inactive)} symbols due to low sustained activity (< {activity_threshold})")
                dropped = [s for s in recently_inactive if s in active_symbols]
                kept = [s for s in active_symbols if s not in dropped]
            else:
                print("No symbols consistently inactive. Retaining current set.")
                kept = active_symbols
                dropped = []
            if len(kept) < min_symbols:
                kept = active_symbols[:min_symbols]
            kept = list(set(kept) | semantic_protect)
            print(f"Rebuilding graphs with {len(kept)} / {len(active_symbols)} symbols retained")
            print(f"Kept symbols: {sorted(kept)}")
            print(f"Dropped symbols: {sorted(dropped)}")
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
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc})
    return {"history": history, "stability": stability, "train_graphs": train_graphs, "test_graphs": test_graphs, "train_labels": train_labels, "test_labels": test_labels, "tm": tm}

def run_ROAD_ROAR(tm: MultiClassGraphTsetlinMachine, graphs: Graphs, labels: np.ndarray, original_games: List[Tuple[np.ndarray, int]], board_dim: int = 5) -> Dict[str, float]:
    print("Running ROAD / ROAR evaluation...")
    activity = compute_symbol_activity(tm, graphs)
    if isinstance(activity, dict):
        activity_items = list(activity.items())
    else:
        symbol_names = list(graphs.symbol_id.keys())
        activity_items = list(zip(symbol_names, activity.tolist()))
    sorted_symbols = sorted(activity_items, key = lambda x: x[1], reverse = True)
    top_symbols = [s for s, _ in sorted_symbols[:5]]
    print(f"Top active symbols: {top_symbols}")
    print("Rebuilding ROAD graphs (zeroing top features)...")
    road_graphs, _ = create_graphs_from_games(original_games[-len(labels):], board_dim = board_dim)
    for sym in top_symbols:
        if sym in road_graphs.symbol_id:
            road_graphs.hypervectors[road_graphs.symbol_id[sym], :] = 0
    road_graphs.encode()
    road_preds = tm.predict(road_graphs)
    road_acc = 100 * np.mean(road_preds == labels)
    print("Rebuilding ROAR graphs (keeping only top features)...")
    roar_graphs, _ = create_graphs_from_games(original_games[-len(labels):], board_dim = board_dim)
    for sym in roar_graphs.symbol_id:
        if sym not in top_symbols:
            roar_graphs.hypervectors[roar_graphs.symbol_id[sym], :] = 0
    roar_graphs.encode()
    roar_preds = tm.predict(roar_graphs)
    roar_acc = 100 * np.mean(roar_preds == labels)
    print(f"ROAD accuracy (remove top): {road_acc:.2f}%")
    print(f"ROAR accuracy (retain top): {roar_acc:.2f}%")
    return {"ROAD": road_acc, "ROAR": roar_acc}

if __name__ == "__main__":
    np.random.seed(42)
    result = run_training_with_adaptive_selection(csv_path = "hex_games_5.csv", board_dim = 5, epochs = 60, clause_count = 10000, T_value = 8000, s_value = 4.0, depth = 2, selection_interval = 7, activity_threshold = 0.05, min_symbols = 10)
    tm = result["tm"]
    test_graphs = result["test_graphs"]
    test_labels = result["test_labels"]
    all_games = parse_csv_games("hex_games_5.csv", board_dim = 5)
    run_ROAD_ROAR(tm = tm, graphs = test_graphs, labels = test_labels, original_games = all_games, board_dim = 5)
    os.makedirs("adaptive_results", exist_ok = True)
    with open("adaptive_results/history.json", "w") as f:
        json.dump(result["history"], f)
    print("Saved training history.")