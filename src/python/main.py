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
    
    # Convert to graphs (test must use same hypervector mappings as train!)
    # Use larger hypervectors for better discrimination
    train_graphs, train_labels = create_graphs_from_games(train_games, board_dim=5, hypervector_size=256, hypervector_bits=2)
    test_graphs, test_labels = create_graphs_from_games(test_games, board_dim=5, init_with=train_graphs)
    
    # Initialize model - using parameters similar to working examples
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=8000,  # Match MNIST convolution example
        T=3000,  # Higher threshold
        s=10.0,  # Higher specificity for sparse patterns
        depth=3,  # Message passing depth
        message_size=256,  # Match hypervector size  
        message_bits=2,  # Keep consistent with hypervector_bits
        grid=(16*13, 1, 1),
        block=(128, 1, 1)
    )
    
    # Train
    for epoch in range(60):
        tm.fit(train_graphs, train_labels, epochs=1, incremental=(epoch > 0))
        
        if (epoch + 1) % 10 == 0:
            train_pred = tm.predict(train_graphs)
            test_pred = tm.predict(test_graphs)
            
            train_acc = 100 * np.mean(train_pred == train_labels)
            test_acc = 100 * np.mean(test_pred == test_labels)
            
            # Check if model is predicting both classes
            train_p0 = np.sum(train_pred == 0)
            train_p1 = np.sum(train_pred == 1)
            test_p0 = np.sum(test_pred == 0)
            test_p1 = np.sum(test_pred == 1)
            
            print(f"Epoch {epoch+1}: Train {train_acc:.1f}% (P0:{train_p0}, P1:{train_p1}) | Test {test_acc:.1f}% (P0:{test_p0}, P1:{test_p1})")
    
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