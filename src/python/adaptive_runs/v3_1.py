from __future__ import annotations
import os, math, time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from parse_games import parse_csv_games, create_graphs_from_games

@dataclass
class TMConfig:
    board_dim: int = 11
    csv_path: str = ""
    epochs: int = 80
    clause_count: int = 55000
    T_value: int = 60000
    s_value: float = 2.0
    depth: int = 3
    message_size: int = 256
    message_bits: int = 5
    percentile: float = 0.4
    seed: int = 42
    warmup_epochs: int = 5
    selection_interval: int = 3
    stability_window: int = 2
    min_symbols_scale: float = 0.3
    early_stop_patience: int = 5
    def __post_init__(self):
        if not self.csv_path:
            self.csv_path = f"hex_games_{self.board_dim}.csv"

def compute_symbol_activity(tm, graphs, prev_activity: Dict[str, float] | None = None, alpha: float = 0.3, num_classes: int | None = None) -> Dict[str, float]:
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
        for idx, sym in enumerate(symbol_names):
            old_val = prev_activity.get(sym, activity[idx])
            activity[idx] = alpha * activity[idx] + (1 - alpha) * old_val
    return dict(zip(symbol_names, activity.tolist()))

def prune_symbols(activity: Dict[str, float], stability: Dict[str, List[float]], active_symbols: List[str], protected: set[str], board_dim: int, min_symbols: int, percentile: float = 0.4, stability_window: int = 2) -> Tuple[List[str], List[str]]:
    if len(active_symbols) <= min_symbols:
        return active_symbols, []
    avg_activity = {sym: np.mean(vals) for sym, vals in stability.items() if sym in activity}
    if not avg_activity:
        return active_symbols, []
    values = np.array(list(avg_activity.values()))
    cutoff = np.percentile(values, percentile * 100)
    inactive = [s for s, v in avg_activity.items() if v <= cutoff and s not in protected]
    dropped = []
    for s in inactive:
        hist = stability.get(s, [])
        if len(hist) >= stability_window and all(v <= cutoff for v in hist[-stability_window:]):
            dropped.append(s)
    kept = [s for s in active_symbols if s not in dropped]
    if len(kept) < min_symbols:
        dropped = []
        kept = active_symbols
    kept = sorted(set(kept) | protected)
    return kept, dropped

def run_ROAD_ROAR(tm, test_graphs, test_labels, all_games, cfg: TMConfig, top_k: int = 10):
    activity = compute_symbol_activity(tm, test_graphs)
    sorted_syms = sorted(activity.items(), key = lambda x: x[1], reverse = True)
    syms = [s for s, _ in sorted_syms]
    results = []
    original_hv = test_graphs.hypervectors.copy()
    for k in range(1, min(top_k, len(syms)) + 1):
        top_syms = set(syms[:k])
        sym_id = test_graphs.symbol_id
        test_graphs.hypervectors[:] = original_hv
        for s in top_syms:
            if s in sym_id:
                test_graphs.hypervectors[sym_id[s], :] = 0
        test_graphs.encode()
        road_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
        test_graphs.hypervectors[:] = original_hv
        for s in sym_id:
            if s not in top_syms:
                test_graphs.hypervectors[sym_id[s], :] = 0
        test_graphs.encode()
        roar_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
        results.append((k, road_acc, roar_acc))
    test_graphs.hypervectors[:] = original_hv
    test_graphs.encode()
    return results

def pretty_print_clauses(tm, graphs, top_n = 10, out_path = None):
    clause_literals = tm.get_clause_literals(graphs.hypervectors)
    symbols = list(graphs.symbol_id.keys())
    lines = []
    for c in range(min(top_n, clause_literals.shape[0])):
        included = []
        for i, sym in enumerate(symbols):
            if clause_literals[c, i] == 1:
                included.append(sym)
            elif clause_literals[c, i + len(symbols)] == 1:
                included.append("¬" + sym)
        line = f"Clause {c}: " + " ∧ ".join(included)
        lines.append(line)
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text

def run_training_with_adaptive_selection(cfg: TMConfig) -> Dict[str, Any]:
    np.random.seed(cfg.seed)
    print(f"Loading {cfg.csv_path} (board = {cfg.board_dim}x{cfg.board_dim})")
    games = parse_csv_games(cfg.csv_path, board_dim = cfg.board_dim)
    total = len(games)
    indices = np.random.permutation(total)
    train_end, val_end = int(total * 0.7), int(total * 0.9)
    train_games = [games[i] for i in indices[:train_end]]
    val_games = [games[i] for i in indices[train_end:val_end]]
    test_games = [games[i] for i in indices[val_end:]]
    print("Creating graphs...")
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim = cfg.board_dim, hypervector_size = cfg.message_size, hypervector_bits = cfg.message_bits)
    val_graphs, val_labels = create_graphs_from_games(val_games, board_dim = cfg.board_dim, init_with = train_graphs)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim = cfg.board_dim, init_with = train_graphs)
    print("Finished graph creation.")
    tm = MultiClassGraphTsetlinMachine(number_of_clauses = cfg.clause_count, T = cfg.T_value, s = cfg.s_value, depth = cfg.depth, message_size = cfg.message_size, message_bits = cfg.message_bits, grid = (16 * 13, 1, 1), block = (128, 1, 1))
    min_symbols = 6 + int(cfg.min_symbols_scale * cfg.board_dim)
    protected = {"Adjacent", "SameColor_Pos", "SameColor_Neg"}
    print(f"[Pruning schedule] Warm-up = {cfg.warmup_epochs}, every = {cfg.selection_interval} ep, window = {cfg.stability_window}, min_symbols = {min_symbols}")
    history = []
    prev_activity = None
    stability: Dict[str, List[float]] = {}
    active_symbols = list(train_graphs.symbol_id.keys())
    no_change_epochs = 0
    last_train, last_val = None, None
    road_roar_history = []

    start_time = time.time()

    for epoch in range(cfg.epochs):
        tm.fit(train_graphs, train_labels, epochs = 1, incremental = (epoch > 0))
        train_acc = 100 * np.mean(tm.predict(train_graphs) == train_labels)
        val_acc = 100 * np.mean(tm.predict(val_graphs) == val_labels)
        elapsed = int(time.time() - start_time)
        print(f"Epoch {epoch + 1:02d}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Time: {elapsed} sec")
        if last_train is not None and last_val is not None:
            if abs(train_acc - last_train) < 1e-6 and abs(val_acc - last_val) < 1e-6:
                no_change_epochs += 1
            else:
                no_change_epochs = 0
        last_train, last_val = train_acc, val_acc
        if no_change_epochs >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} (no change for {cfg.early_stop_patience} epochs).")
            break
        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            rr = run_ROAD_ROAR(tm, val_graphs, val_labels, val_games, cfg, top_k = 10)
            avg_road = np.mean([r[1] for r in rr])
            avg_roar = np.mean([r[2] for r in rr])
            road_roar_history.append((epoch + 1, avg_road, avg_roar))
        if (epoch + 1) >= cfg.warmup_epochs and (epoch + 1 - cfg.warmup_epochs) % cfg.selection_interval == 0 and len(active_symbols) > min_symbols:
            print("Computing symbol activity...")
            activity = compute_symbol_activity(tm, train_graphs, prev_activity)
            prev_activity = activity.copy()
            for sym, val in activity.items():
                stability.setdefault(sym, []).append(val)
                if len(stability[sym]) > cfg.stability_window:
                    stability[sym].pop(0)
            kept, dropped = prune_symbols(activity, stability, active_symbols, protected, cfg.board_dim, min_symbols, cfg.percentile, cfg.stability_window)
            print(f"Pruning {len(dropped)} symbols. Keeping {len(kept)} / {len(active_symbols)}.")
            print("Dropped symbols:", sorted(dropped))
            print("Retained symbols:", sorted(kept))
            if dropped:
                for gset in (train_graphs, val_graphs):
                    for sym in dropped:
                        if sym in gset.symbol_id:
                            gset.hypervectors[gset.symbol_id[sym], :] = 0
                    gset.encode()
            active_symbols = kept
        history.append({"epoch": epoch + 1, "train": train_acc, "val": val_acc})
    total_time = int(time.time() - start_time)
    test_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(f"Total training time: {total_time} sec ({total_time / 60:.1f} min)")
    return {"history": history, "road_roar": road_roar_history, "tm": tm, "test_graphs": test_graphs, "test_labels": test_labels, "test_acc": test_acc, "train_graphs": train_graphs, "train_labels": train_labels}

def plot_results(history: List[Dict[str, float]], road_roar_results: List[Tuple[int, float, float]], outdir: str):
    os.makedirs(outdir, exist_ok = True)
    epochs = [h["epoch"] for h in history]
    train = [h["train"] for h in history]
    val = [h["val"] for h in history]
    plt.figure(figsize = (7, 5))
    plt.plot(epochs, train, color = "navy", marker = "o", label = "Train accuracy")
    plt.plot(epochs, val, color = "deepskyblue", marker = "s", label = "Validation accuracy")
    plt.xticks(range(0, max(epochs) + 1, 5))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()
    if road_roar_results:
        plt.figure(figsize = (7, 5))
        epochs_rr = [r[0] for r in road_roar_results]
        road_vals = [r[1] for r in road_roar_results]
        roar_vals = [r[2] for r in road_roar_results]
        plt.plot(epochs_rr, road_vals, color = "navy", marker = "o", label = "ROAD (remove top)")
        plt.plot(epochs_rr, roar_vals, color = "deepskyblue", marker = "s", label = "ROAR (keep top)")
        plt.xticks(range(0, max(epochs_rr) + 1, 5))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("ROAD / ROAR across training")
        plt.legend()
        plt.grid(True, linestyle = "--", alpha = 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "road_roar_curve.png"))
        plt.close()

if __name__ == "__main__":
    cfg = TMConfig(board_dim = 11)
    result = run_training_with_adaptive_selection(cfg)
    tm = result["tm"]
    test_graphs = result["test_graphs"]
    test_labels = result["test_labels"]
    all_games = parse_csv_games(cfg.csv_path, board_dim = cfg.board_dim)
    outdir = f"results/{cfg.board_dim}"
    plot_results(result["history"], result["road_roar"], outdir)
    pretty_print_clauses(tm, result["train_graphs"], top_n = 100, out_path = os.path.join(outdir, "clauses.txt"))
    print(f"\nAll results saved under: {outdir}/")