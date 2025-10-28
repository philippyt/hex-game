import numpy as np
from parse_games import parse_games
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def main():
    print("="*60)
    print("HEX GAME WINNER PREDICTION - Graph Tsetlin Machine")
    print("="*60 + "\n")
    
    # Step 1: Parse the games
    print("Step 1: Loading and parsing games...")
    graphs, labels = parse_games(
        'datasets/hex_games.csv',
        board_dim=9,
        hypervector_size=128, 
        hypervector_bits=2,  
        verbose=True
    )
    
    if graphs is None:
        print("Error: Failed to load games!")
        return
    
    # Step 2: Split data into training and testing sets
    print("\nStep 2: Splitting data into train/test sets...")
    num_games = len(labels)
    
    # Use 80% for training, 20% for testing
    train_size = int(0.8 * num_games)
    
    # Shuffle the data
    indices = np.random.permutation(num_games)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    print(f"Training set: {len(train_indices)} games")
    print(f"Test set: {len(test_indices)} games")
    
    # Step 3: Initialize the Graph Tsetlin Machine
    print("\nStep 3: Initializing Graph Tsetlin Machine...")
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=500,      # Number of clauses (patterns to learn)
        T=1000,                      # Threshold for voting
        s=1.0,                       # Specificity parameter
        depth=3,                     # Depth of graph patterns
        message_size=128,            # Size of messages passed between nodes
        message_bits=2,              # Bits per message
        max_included_literals=32,    # Max features per clause
        grid=(16*13,1,1),           # GPU grid size (adjust for your GPU)
        block=(128,1,1)             # GPU block size
    )
    
    print(f"Clauses: {tm.number_of_clauses}")
    print(f"Threshold: {tm.T}")
    print(f"Depth: {tm.depth}")
    print(f"Message size: {tm.message_size} bits")
    
    # Step 4: Train the model
    print("\nStep 4: Training the model...")
    print("-" * 60)
    
    epochs = 50  # Number of training epochs
    
    for epoch in range(epochs):
        # Train on training set
        tm.fit(graphs, labels, epoch=1, incremental=True)
        
        # Evaluate on training set
        train_predictions = tm.predict(graphs)
        train_accuracy = 100 * np.sum(train_predictions == labels) / len(labels)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Train Accuracy: {train_accuracy:.2f}%")
    
    print("-" * 60)
    
    # Step 5: Final evaluation
    print("\nStep 5: Final Evaluation")
    print("="*60)
    
    # Training accuracy
    train_predictions = tm.predict(graphs)
    train_accuracy = 100 * np.sum(train_predictions == labels) / len(labels)
    
    # Calculate per-class accuracy
    player0_correct = np.sum((train_predictions == 0) & (labels == 0))
    player0_total = np.sum(labels == 0)
    player1_correct = np.sum((train_predictions == 1) & (labels == 1))
    player1_total = np.sum(labels == 1)
    
    print(f"\nOverall Accuracy: {train_accuracy:.2f}%")
    print(f"\nPer-Class Performance:")
    print(f"  Player 0 wins: {player0_correct}/{player0_total} ({100*player0_correct/player0_total:.2f}%)")
    print(f"  Player 1 wins: {player1_correct}/{player1_total} ({100*player1_correct/player1_total:.2f}%)")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"                Predicted P0  Predicted P1")
    print(f"Actual P0:      {np.sum((train_predictions == 0) & (labels == 0)):6d}        {np.sum((train_predictions == 1) & (labels == 0)):6d}")
    print(f"Actual P1:      {np.sum((train_predictions == 0) & (labels == 1)):6d}        {np.sum((train_predictions == 1) & (labels == 1)):6d}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()