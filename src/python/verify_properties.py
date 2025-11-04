"""Verify that properties are being added correctly"""
import sys
sys.path.append('src/python')
from parse_games import parse_csv_games, create_graphs_from_games
import numpy as np

# Load just 2 games
games = parse_csv_games('datasets/hex_games_5.csv', board_dim=5)[:2]

# Create graphs
graphs, labels = create_graphs_from_games(games, board_dim=5)

print("Checking if properties (including position info) are visible in graphs...\n")

for graph_id in range(len(games)):
    board, winner = games[graph_id]
    print(f"=== Game {graph_id} (Winner: {winner}) ===")
    
    # Count pieces
    player_neg = sum(row.count(-1) for row in board)
    player_pos = sum(row.count(1) for row in board)
    empty = sum(row.count(0) for row in board)
    
    print(f"Board counts: PlayerNeg={player_neg}, PlayerPos={player_pos}, Empty={empty}")
    
    # Print just first 3 nodes to see if position info is there
    print("First 3 nodes:")
    for node_id in range(3):
        print(f"  Node {node_id} (should have piece type + Row{node_id//5} + Col{node_id%5}):")
    
    # Use the built-in print function
    graphs.print_graph_nodes(graph_id)
    print()
