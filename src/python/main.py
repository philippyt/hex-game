import numpy as np
import argparse
from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def main(num_clauses=1000, threshold=2000, specificity=2.0, num_epochs=60, depth=3):
    print("="*60)
    print("HEX GAME WINNER PREDICTION - Graph Tsetlin Machine")
    print("="*60 + "\n")
    
    # Step 1: Parse the games (get raw game data first)
    print("Step 1: Loading game data...")
    print("Parsing games from datasets/hex_games.csv...")
    games = parse_csv_games('datasets/hex_games.csv', board_dim=9)
    
    if not games:
        print("Error: Failed to load games!")
        return
    
    print(f"Found {len(games)} games")
    
    # Get all labels
    all_labels = np.array([winner for board, winner in games], dtype=np.int32)
    print(f"Winner distribution: Player 0: {np.sum(all_labels == 0)}, Player 1: {np.sum(all_labels == 1)}")
    
    # Step 2: Split data into training and testing sets
    print("\nStep 2: Splitting data into train/test sets...")
    num_games = len(games)
    
    # Use 80% for training, 20% for testing
    train_size = int(0.8 * num_games)
    
    # Shuffle the data
    indices = np.random.permutation(num_games)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Split the games
    train_games = [games[i] for i in train_indices]
    test_games = [games[i] for i in test_indices]
    
    print(f"Training set: {len(train_games)} games")
    print(f"Test set: {len(test_games)} games")
    
    # Convert to graph format
    print("\nConverting training set to graph format...")
    train_graphs, train_labels = create_graphs_from_games(
        train_games, 
        board_dim=9,
        hypervector_size=128, 
        hypervector_bits=2
    )
    
    print("Converting test set to graph format...")
    test_graphs, test_labels = create_graphs_from_games(
        test_games,
        board_dim=9,
        hypervector_size=128, 
        hypervector_bits=2
    )
    
    print(f"Train - Player 0: {np.sum(train_labels == 0)}, Player 1: {np.sum(train_labels == 1)}")
    print(f"Test  - Player 0: {np.sum(test_labels == 0)}, Player 1: {np.sum(test_labels == 1)}")
    
    # Step 3: Initialize the Graph Tsetlin Machine
    print("\nStep 3: Initializing Graph Tsetlin Machine...")
    print(f"Hyperparameters: clauses={num_clauses}, T={threshold}, s={specificity}, depth={depth}")
    
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=num_clauses,
        T=threshold,
        s=specificity,
        depth=depth,
        message_size=128,            # Size of messages passed between nodes
        message_bits=2,              # Bits per message
        max_included_literals=32,    # Max features per clause
        grid=(16*13,1,1),           # GPU grid size (adjust for your GPU)
        block=(128,1,1)             # GPU block size
    )
    
    print(f"Clauses: {tm.number_of_clauses}")
    print(f"Threshold: {tm.T}")
    print(f"Specificity: {tm.s}")
    print(f"Depth: {tm.depth}")
    print(f"Message size: {tm.message_size} bits")
    
    # Step 4: Train the model
    print("\nStep 4: Training the model...")
    print("-" * 60)
    
    epochs = num_epochs
    best_test_accuracy = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train on training set only (without incremental to reset between epochs)
        if epoch == 0:
            tm.fit(train_graphs, train_labels, epochs=1, incremental=False)
        else:
            tm.fit(train_graphs, train_labels, epochs=1, incremental=True)
        
        # Evaluate on both sets periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_predictions = tm.predict(train_graphs)
            train_accuracy = 100 * np.sum(train_predictions == train_labels) / len(train_labels)
            
            test_predictions = tm.predict(test_graphs)
            test_accuracy = 100 * np.sum(test_predictions == test_labels) / len(test_labels)
            
            # Calculate per-class accuracy on test set
            test_p0_correct = np.sum((test_predictions == 0) & (test_labels == 0))
            test_p0_total = np.sum(test_labels == 0)
            test_p1_correct = np.sum((test_predictions == 1) & (test_labels == 1))
            test_p1_total = np.sum(test_labels == 1)
            
            # Track best model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:3d}/{epochs} - Train: {train_accuracy:.2f}% | Test: {test_accuracy:.2f}% | P0: {100*test_p0_correct/max(test_p0_total,1):.1f}% P1: {100*test_p1_correct/max(test_p1_total,1):.1f}%")
    
    print(f"\nBest test accuracy: {best_test_accuracy:.2f}% at epoch {best_epoch}")
    
    print("-" * 60)
    
    # Step 5: Final evaluation
    print("\nStep 5: Final Evaluation")
    print("="*60)
    
    # Training set evaluation
    train_predictions = tm.predict(train_graphs)
    train_accuracy = 100 * np.sum(train_predictions == train_labels) / len(train_labels)
    
    train_p0_correct = np.sum((train_predictions == 0) & (train_labels == 0))
    train_p0_total = np.sum(train_labels == 0)
    train_p1_correct = np.sum((train_predictions == 1) & (train_labels == 1))
    train_p1_total = np.sum(train_labels == 1)
    
    # Test set evaluation
    test_predictions = tm.predict(test_graphs)
    test_accuracy = 100 * np.sum(test_predictions == test_labels) / len(test_labels)
    
    test_p0_correct = np.sum((test_predictions == 0) & (test_labels == 0))
    test_p0_total = np.sum(test_labels == 0)
    test_p1_correct = np.sum((test_predictions == 1) & (test_labels == 1))
    test_p1_total = np.sum(test_labels == 1)
    
    print(f"\n{'TRAINING SET':^60}")
    print(f"Overall Accuracy: {train_accuracy:.2f}%")
    print(f"\nPer-Class Performance:")
    print(f"  Player 0 wins: {train_p0_correct}/{train_p0_total} ({100*train_p0_correct/train_p0_total:.2f}%)")
    print(f"  Player 1 wins: {train_p1_correct}/{train_p1_total} ({100*train_p1_correct/train_p1_total:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted P0  Predicted P1")
    print(f"Actual P0:      {np.sum((train_predictions == 0) & (train_labels == 0)):6d}        {np.sum((train_predictions == 1) & (train_labels == 0)):6d}")
    print(f"Actual P1:      {np.sum((train_predictions == 0) & (train_labels == 1)):6d}        {np.sum((train_predictions == 1) & (train_labels == 1)):6d}")
    
    print(f"\n{'TEST SET':^60}")
    print(f"Overall Accuracy: {test_accuracy:.2f}%")
    print(f"\nPer-Class Performance:")
    print(f"  Player 0 wins: {test_p0_correct}/{test_p0_total} ({100*test_p0_correct/test_p0_total:.2f}%)")
    print(f"  Player 1 wins: {test_p1_correct}/{test_p1_total} ({100*test_p1_correct/test_p1_total:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted P0  Predicted P1")
    print(f"Actual P0:      {np.sum((test_predictions == 0) & (test_labels == 0)):6d}        {np.sum((test_predictions == 1) & (test_labels == 0)):6d}")
    print(f"Actual P1:      {np.sum((test_predictions == 0) & (test_labels == 1)):6d}        {np.sum((test_predictions == 1) & (test_labels == 1)):6d}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Graph Tsetlin Machine on Hex game data')
    parser.add_argument('--clauses', type=int, default=1000, help='Number of clauses (default: 1000)')
    parser.add_argument('--threshold', '-T', type=int, default=2000, help='Threshold T (default: 2000)')
    parser.add_argument('--specificity', '-s', type=float, default=2.0, help='Specificity s (default: 2.0)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs (default: 60)')
    parser.add_argument('--depth', type=int, default=3, help='Graph depth (default: 3)')
    
    args = parser.parse_args()
    
    main(
        num_clauses=args.clauses,
        threshold=args.threshold,
        specificity=args.specificity,
        num_epochs=args.epochs,
        depth=args.depth
    )