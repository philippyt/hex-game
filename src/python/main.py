import numpy as np
from parse_games import parse_csv_games, create_graphs_from_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def main():
    # Load games
    games = parse_csv_games('datasets/hex_games_5.csv', board_dim=5)
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(games))
    indices = np.random.permutation(len(games))
    train_games = [games[i] for i in indices[:train_size]]
    test_games = [games[i] for i in indices[train_size:]]
    
    # Convert to graphs
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim=5)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim=5)
    
    # Initialize model
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=1000,
        T=2000,
        s=2.0,
        depth=3,
        message_size=128,
        message_bits=2,
        max_included_literals=32,
        grid=(16*13, 1, 1),
        block=(128, 1, 1)
    )
    
    # Train
    for epoch in range(60):
        tm.fit(train_graphs, train_labels, epochs=1, incremental=(epoch > 0))
        
        if (epoch + 1) % 10 == 0:
            train_acc = 100 * np.mean(tm.predict(train_graphs) == train_labels)
            test_acc = 100 * np.mean(tm.predict(test_graphs) == test_labels)
            print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%")
    
    # Final evaluation
    test_predictions = tm.predict(test_graphs)
    test_acc = 100 * np.mean(test_predictions == test_labels)
    print(f"\nFinal Test Accuracy: {test_acc:.1f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                Predicted P0  Predicted P1")
    p0_as_p0 = np.sum((test_predictions == 0) & (test_labels == 0))
    p0_as_p1 = np.sum((test_predictions == 1) & (test_labels == 0))
    p1_as_p0 = np.sum((test_predictions == 0) & (test_labels == 1))
    p1_as_p1 = np.sum((test_predictions == 1) & (test_labels == 1))
    print(f"Actual P0:      {p0_as_p0:6d}        {p0_as_p1:6d}")
    print(f"Actual P1:      {p1_as_p0:6d}        {p1_as_p1:6d}")

if __name__ == "__main__":
    main()