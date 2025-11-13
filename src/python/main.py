from __future__ import annotations
import numpy as np, time
from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from typing import NoReturn
from pathlib import Path

def main() -> NoReturn:
    np.random.seed(42)
    repo_root = Path(__file__).resolve().parents[2]
    dataset_fp = repo_root / "datasets" / "hex_games_5.csv"
    games = parse_csv_games(str(dataset_fp), board_dim = 5)
    total = len(games)
    indices = np.random.permutation(total)
    train_end, val_end = int(total * 0.7), int(total * 0.9)
    train_games = [games[i] for i in indices[:train_end]]
    val_games = [games[i] for i in indices[train_end:val_end]]
    test_games = [games[i] for i in indices[val_end:]]
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim = 5, hypervector_size = 256, hypervector_bits = 2)
    val_graphs, val_labels = create_graphs_from_games(val_games, board_dim = 5, init_with = train_graphs)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim = 5, init_with = train_graphs)
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses = 20000,
        T = 11000,
        s = 2,
        depth = 2,
        message_size = 256,
        message_bits = 2,
        grid = (16 * 13, 1, 1),
        block = (128, 1, 1)
    )

    best_val = 0
    patience = 0
    patience_limit = 5
    cumulative_time = 0
    for epoch in range(60):
        t0 = time.perf_counter()
        tm.fit(train_graphs, train_labels, epochs = 1, incremental = (epoch > 0))
        train_pred = tm.predict(train_graphs)
        val_pred = tm.predict(val_graphs)
        train_acc = 100 * np.mean(train_pred == train_labels)
        val_acc = 100 * np.mean(val_pred == val_labels)
        elapsed = int(time.perf_counter() - t0)
        cumulative_time += elapsed

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}: Train {train_acc:.1f}% | Val {val_acc:.1f}% | Time: {cumulative_time} sec")

        if patience >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    test_predictions = tm.predict(test_graphs)
    test_acc = 100 * np.mean(test_predictions == test_labels)
    print(f"\nFinal test accuracy: {test_acc:.1f}%")
    print("\nConfusion matrix:")
    print("                Predicted P0  Predicted P1")
    p0_as_p0 = np.sum((test_predictions == 0) & (test_labels == 0))
    p0_as_p1 = np.sum((test_predictions == 1) & (test_labels == 0))
    p1_as_p0 = np.sum((test_predictions == 0) & (test_labels == 1))
    p1_as_p1 = np.sum((test_predictions == 1) & (test_labels == 1))
    print(f"Actual P0:      {p0_as_p0:6d}        {p0_as_p1:6d}")
    print(f"Actual P1:      {p1_as_p0:6d}        {p1_as_p1:6d}")

if __name__ == "__main__":
    main()