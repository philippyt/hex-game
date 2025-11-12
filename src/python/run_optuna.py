from __future__ import annotations
from pathlib import Path

import numpy as np
import optuna

from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def objective(trial: optuna.Trial, train_games: list, val_games: list, board_size: int) -> tuple[float, int]:
    """
    Multi-objective optimization function for Optuna.
    
    Optimizes:
        1. Maximize validation accuracy
        2. Minimize number of clauses used
    
    Args:
        trial: Optuna trial object
        train_games: Pre-parsed training games
        val_games: Pre-parsed validation games
        board_size: Size of the hex board
        
    Returns:
        Tuple of (validation_accuracy, number_of_clauses)
    """
    # Suggest hyperparameters
    number_of_clauses = trial.suggest_int("number_of_clauses", 5000, 80000, log=True)
    T = trial.suggest_int("T", 1000, 70000, log=True)
    s = trial.suggest_float("s", 1.0, 10.0, log=True)
    depth = trial.suggest_categorical("depth", [1, 2, 3, 4])
    message_size = trial.suggest_categorical("message_size", [128, 256, 512])
    message_bits = trial.suggest_categorical("message_bits", [1, 2, 3, 4, 5])

    # Build graphs (use same hypervector params for train and val)
    train_graphs, train_labels = create_graphs_from_games(
        train_games,
        board_dim=board_size,
        hypervector_size=message_size,
        hypervector_bits=message_bits
    )
    val_graphs, val_labels = create_graphs_from_games(
        val_games,
        board_dim=board_size,
        init_with=train_graphs
    )

    # Initialize Tsetlin Machine
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        depth=depth,
        message_size=message_size,
        message_bits=message_bits,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    )

    max_epochs = 60

    for epoch in range(max_epochs):
        # Incremental training starts after first epoch
        tm.fit(train_graphs, train_labels, epochs=1, incremental=(epoch > 0))

    # Evaluate on validation set after all training epochs
    val_pred = tm.predict(val_graphs)
    val_acc = float(np.mean(val_pred == val_labels))

    # Return validation accuracy and number of clauses
    # Optuna will maximize accuracy and minimize num_clauses
    return val_acc, number_of_clauses


def run_optimization(board_size: int, board_suffix: str, n_trials: int = 50):
    """
    Run Optuna optimization for a specific board configuration.
    
    Args:
        board_size: Size of the hex board
        board_suffix: Suffix for dataset filename
        n_trials: Number of optimization trials to run
    """
    # Parse games and create train/validation split ONCE before all trials
    repo_root = Path(__file__).resolve().parents[2]
    dataset_fp = repo_root / "datasets" / f"hex_games_{board_size}{board_suffix}.csv"
    
    print(f"\n{'=' * 80}")
    print(f"Loading dataset: hex_games_{board_size}{board_suffix}.csv")
    games = parse_csv_games(str(dataset_fp), board_dim=board_size)
    
    # Simple 80/20 train/validation split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(games))
    train_size = int(0.8 * len(games))
    train_games = [games[i] for i in indices[:train_size]]
    val_games = [games[i] for i in indices[train_size:]]
    
    print(f"Total games: {len(games)}")
    print(f"Training games: {len(train_games)}")
    print(f"Validation games: {len(val_games)}")
    
    # Study settings: TPESampler with persistent SQLite storage
    # Note: Pruning is not supported for multi-objective optimization
    storage = "sqlite:///optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=42)

    # Create multi-objective study: maximize accuracy, minimize clauses
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=storage,
        study_name=f"hex_tm_multiobjective_{board_size}{board_suffix}",
        load_if_exists=True
    )

    print(f"Starting Optuna multi-objective study with {n_trials} trials")
    print(f"Storage: {storage}")
    print(f"{'=' * 80}\n")

    # Run optimization with pre-parsed games
    study.optimize(
        lambda trial: objective(trial, train_games, val_games, board_size),
        n_trials=n_trials,
        gc_after_trial=True
    )

    print("\nStudy finished.")
    print(f"Number of finished trials: {len(study.trials)}")

    # Display Pareto front solutions
    print("\n=== Pareto Front (Non-dominated solutions) ===")
    pareto_trials = study.best_trials

    print(f"Found {len(pareto_trials)} non-dominated solutions on the Pareto front:")
    for i, trial in enumerate(pareto_trials, 1):
        accuracy = trial.values[0]
        num_clauses = trial.values[1]
        print(f"\nSolution {i}:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Number of clauses: {num_clauses}")
        print(f"  Parameters:")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")

    # Find the best solution with perfect accuracy
    perfect_accuracy_trials = [t for t in pareto_trials if t.values[0] >= 0.999]
    if perfect_accuracy_trials:
        best_perfect = min(perfect_accuracy_trials, key=lambda t: t.values[1])
        print(f"\n=== Best solution with ~100% accuracy ===")
        print(f"  Accuracy: {best_perfect.values[0]:.4f} ({best_perfect.values[0] * 100:.2f}%)")
        print(f"  Number of clauses: {best_perfect.values[1]}")
        print(f"  Parameters:")
        for k, v in best_perfect.params.items():
            print(f"    {k}: {v}")
    else:
        print("\nNo solution achieved 100% accuracy yet.")
        best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])
        print(f"Best accuracy achieved: {best_acc_trial.values[0]:.4f} "
              f"({best_acc_trial.values[0] * 100:.2f}%) with {best_acc_trial.values[1]} clauses")


def main() -> None:
    """Main function to run optimization across all datasets."""
    # Define all dataset configurations to test
    configurations = [
        # Board size 5
        (5, ""),
        (5, "_minus2"),
        (5, "_minus5"),
        # Board size 7
        (7, ""),
        (7, "_minus2"),
        (7, "_minus5"),
        # Board size 9
        (9, ""),
        (9, "_minus2"),
        (9, "_minus5"),
        # Board size 11
        (11, ""),
        (11, "_minus2"),
        (11, "_minus5"),
    ]

    n_trials = 50  # Number of trials per dataset

    print(f"Running optimization for {len(configurations)} dataset configurations")
    print(f"Each configuration will run {n_trials} trials\n")

    for board_size, board_suffix in configurations:
        try:
            run_optimization(board_size, board_suffix, n_trials)
        except FileNotFoundError:
            print(f"⚠ Warning: Dataset hex_games_{board_size}{board_suffix}.csv not found, skipping...")
        except Exception as e:
            print(f"⚠ Error processing hex_games_{board_size}{board_suffix}.csv: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("All optimizations completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
