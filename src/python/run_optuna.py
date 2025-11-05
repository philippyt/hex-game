from __future__ import annotations
import optuna
import numpy as np
from pathlib import Path
from typing import NoReturn

from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

board_size = 11

def objective(trial: optuna.Trial) -> float:
    # Fixed dataset / split (seeded inside main flow when called)
    repo_root = Path(__file__).resolve().parents[2]
    dataset_fp = repo_root / "datasets" / f"hex_games_{board_size}.csv"
    games = parse_csv_games(str(dataset_fp), board_dim=5)

    # Simple 80/20 split with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(games))
    train_size = int(0.8 * len(games))
    train_games = [games[i] for i in indices[:train_size]]
    val_games = [games[i] for i in indices[train_size:]]

    # Suggest hyperparameters (high-priority set)
    number_of_clauses = trial.suggest_categorical("number_of_clauses", [2000, 5000, 10000, 20000])
    T = trial.suggest_int("T", 1000, 15000)
    s = trial.suggest_float("s", 1.0, 10.0, log=True)
    depth = trial.suggest_categorical("depth", [1, 2, 3])
    message_size = trial.suggest_categorical("message_size", [128, 256, 512])
    message_bits = trial.suggest_categorical("message_bits", [1, 2])

    # Build graphs (use same hypervector params for train and val)
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim=5, hypervector_size=message_size, hypervector_bits=message_bits)
    val_graphs, val_labels = create_graphs_from_games(val_games, board_dim=5, init_with=train_graphs)

    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        depth=depth,
        message_size=message_size,
        message_bits=message_bits,
        grid=(16*13, 1, 1),
        block=(128, 1, 1),
    )

    max_epochs = 60

    for epoch in range(max_epochs):
        # incremental training same as original main: start incremental after first epoch
        tm.fit(train_graphs, train_labels, epochs=1, incremental=(epoch > 0))

        # Evaluate on validation set each epoch and report for pruning
        val_pred = tm.predict(val_graphs)
        val_acc = float(np.mean(val_pred == val_labels))

        # Report intermediate objective value to Optuna
        trial.report(val_acc, epoch)

        # Prune trial if needed
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return final validation accuracy (Optuna maximizes this)
    return val_acc


def main() -> NoReturn:
    # Study settings: TPESampler with MedianPruner; persistent sqlite storage
    storage = "sqlite:///optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage, study_name=f"hex_tm_opt_{board_size}", load_if_exists=True)

    n_trials = 50
    print(f"Starting Optuna study with {n_trials} trials (storage={storage})")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Study finished.")
    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()