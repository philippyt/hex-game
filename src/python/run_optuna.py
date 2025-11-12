from __future__ import annotations
import optuna
import numpy as np
from pathlib import Path
from typing import NoReturn

from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

BOARD_SIZE = 11
BOARD_SUFFIX = "" 

def objective(trial: optuna.Trial) -> tuple[float, int]:
    """
    Multi-objective optimization:
    1. Maximize validation accuracy
    2. Minimize number of clauses used
    
    Returns (accuracy, num_clauses) where Optuna will maximize accuracy
    and minimize num_clauses simultaneously.
    """
    repo_root = Path(__file__).resolve().parents[2]
    dataset_fp = repo_root / "datasets" / f"hex_games_{BOARD_SIZE}{BOARD_SUFFIX}.csv"
    games = parse_csv_games(str(dataset_fp), board_dim=5)

    # Simple 80/20 split with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(games))
    train_size = int(0.8 * len(games))
    train_games = [games[i] for i in indices[:train_size]]
    val_games = [games[i] for i in indices[train_size:]]

    # Hyperparameter intervals
    number_of_clauses = trial.suggest_int("number_of_clauses", 5000, 80000, log=True) 
    T = trial.suggest_int("T", 1000, 70000, log=True) 
    s = trial.suggest_float("s", 1.0, 10.0, log=True)
    depth = trial.suggest_categorical("depth", [1, 2, 3, 4])
    message_size = trial.suggest_categorical("message_size", [128, 256, 512])
    message_bits = trial.suggest_categorical("message_bits", [1, 2, 3, 4, 5])


    """number_of_clauses = trial.suggest_int("number_of_clauses", 5000, 80000, step=5000)
    T = trial.suggest_int("T", 5000, 80000, step=5000)
    s = trial.suggest_float("s", 2.0, 10.0, step=0.5)
    depth = trial.suggest_categorical("depth", [1, 2, 3, 4])
    message_size = trial.suggest_categorical("message_size", [128, 256, 512, 1024])
    message_bits = trial.suggest_categorical("message_bits", [1, 2, 3, 4, 5, 6])"""

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

    # Return both validation accuracy and number of clauses for multi-objective optimization
    # Optuna will maximize accuracy and minimize num_clauses
    return val_acc, number_of_clauses


def main() -> NoReturn:
    # Study settings: TPESampler with MedianPruner; persistent sqlite storage
    storage = "sqlite:///optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    # Multi-objective study: maximize accuracy, minimize number of clauses
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # maximize accuracy, minimize clauses
        sampler=sampler, 
        pruner=pruner, 
        storage=storage, 
        study_name=f"hex_tm_multiobjective_{BOARD_SIZE}{BOARD_SUFFIX}", 
        load_if_exists=True
    )

    n_trials = 50
    print(f"Starting Optuna multi-objective study with {n_trials} trials")
    print(f"Dataset: hex_games_{BOARD_SIZE}{BOARD_SUFFIX}.csv")
    print(f"Storage: {storage}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("\nStudy finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    
    # In multi-objective optimization, there's no single "best" trial
    # Instead, we have a Pareto front of non-dominated solutions
    print("\n=== Pareto Front (Non-dominated solutions) ===")
    pareto_trials = study.best_trials
    
    print(f"Found {len(pareto_trials)} non-dominated solutions on the Pareto front:")
    for i, trial in enumerate(pareto_trials, 1):
        accuracy = trial.values[0]
        num_clauses = trial.values[1]
        print(f"\nSolution {i}:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Number of clauses: {num_clauses}")
        print(f"  Parameters:")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")
    
    # Find the solution with 100% accuracy (if any) and minimum clauses
    perfect_accuracy_trials = [t for t in pareto_trials if t.values[0] >= 0.999]
    if perfect_accuracy_trials:
        best_perfect = min(perfect_accuracy_trials, key=lambda t: t.values[1])
        print(f"\n=== Best solution with ~100% accuracy ===")
        print(f"  Accuracy: {best_perfect.values[0]:.4f} ({best_perfect.values[0]*100:.2f}%)")
        print(f"  Number of clauses: {best_perfect.values[1]}")
        print(f"  Parameters:")
        for k, v in best_perfect.params.items():
            print(f"    {k}: {v}")
    else:
        print("\nNo solution achieved 100% accuracy yet.")
        best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])
        print(f"Best accuracy achieved: {best_acc_trial.values[0]:.4f} ({best_acc_trial.values[0]*100:.2f}%) with {best_acc_trial.values[1]} clauses")


if __name__ == "__main__":
    main()