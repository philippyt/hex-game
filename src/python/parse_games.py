import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs

_BOARD_DIM = 9  # Default: 9x9 board

def set_board_dimension(board_dim):
    """
    Set the board dimension for parsing.
    
    Args:
        board_dim: Size of the board (e.g., 9 for 9x9 board)
    """
    global _BOARD_DIM
    _BOARD_DIM = board_dim

def get_board_dimension():
    """Get the current board dimension."""
    return _BOARD_DIM

def create_hex_neighbors(board_dim):
    """Create neighbor offsets for hex board (same as in C code)
    
    Args:
        board_dim: Size of the board
    
    Returns a list of integer offsets corresponding to the 6 neighbors
    """
    return [-(board_dim+2) + 1, -(board_dim+2), -1, 1, (board_dim+2), (board_dim+2) - 1]

def parse_csv_games(csv_file, board_dim=None):
    """Parse CSV file containing hex game data.
    
    Args:
        csv_file: Path to CSV file with game states
        board_dim: Board dimension (if None, uses global _BOARD_DIM)
    
    Returns:
        List of (board, winner) tuples where board is 2D array
    """
    if board_dim is None:
        board_dim = _BOARD_DIM
    
    df = pd.read_csv(csv_file)
    games = []
    
    for idx, row in df.iterrows():
        # Extract board state
        board = [] # board is a 2D array
        for i in range(board_dim):
            row_cells = []
            for j in range(board_dim):
                cell_value = row[f'cell_{i}_{j}']
                row_cells.append(int(cell_value))
            board.append(row_cells)
        
        # Extract winner
        winner = int(row['winner'])
        games.append((board, winner))
    
    return games

def convert_board_to_graph(board, graph_id, graphs_obj, board_dim):
    """
    Convert a single Hex board state to graph representation.
    
    Args:
        board: 2D array of board state (-1=empty, 0=Player 0, 1=Player 1)
        graph_id: Index of this graph
        graphs_obj: Graphs object to add to
        board_dim: Size of the board
    """
    neighbors = create_hex_neighbors(board_dim)
    
    # Each board position is a node
    num_nodes = board_dim * board_dim
    
    # Set number of nodes for this graph
    graphs_obj.set_number_of_graph_nodes(graph_id, num_nodes)
    
    # Calculate edges for each position (each position has up to 6 neighbors)
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            
            # Count valid neighbors
            position_in_padded = (i+1) * (board_dim+2) + (j+1)
            valid_neighbors = 0
            
            for neighbor_offset in neighbors:
                neighbor_pos = position_in_padded + neighbor_offset
                # Convert back to 2D coordinates
                neighbor_i = (neighbor_pos // (board_dim+2)) - 1
                neighbor_j = (neighbor_pos % (board_dim+2)) - 1
                
                # Check if neighbor is within board bounds
                if 0 <= neighbor_i < board_dim and 0 <= neighbor_j < board_dim:
                    valid_neighbors += 1
            
            # Add node with number of edges (using integer node_id)
            graphs_obj.add_graph_node(graph_id, node_id, valid_neighbors)

def add_edges_to_graph(graph_id, graphs_obj, board_dim):
    """Add edges between neighboring cells.
    
    Args:
        graph_id: Index of this graph
        graphs_obj: Graphs object to add to
        board_dim: Size of the board
    """
    neighbors = create_hex_neighbors(board_dim)
    
    for i in range(board_dim):
        for j in range(board_dim):
            source_node_id = i * board_dim + j
            position_in_padded = (i+1) * (board_dim+2) + (j+1)
            
            for neighbor_offset in neighbors:
                neighbor_pos = position_in_padded + neighbor_offset
                neighbor_i = (neighbor_pos // (board_dim+2)) - 1
                neighbor_j = (neighbor_pos % (board_dim+2)) - 1
                
                if 0 <= neighbor_i < board_dim and 0 <= neighbor_j < board_dim:
                    dest_node_id = neighbor_i * board_dim + neighbor_j
                    edge_type = 'Adjacent'
                    graphs_obj.add_graph_node_edge(graph_id, source_node_id, dest_node_id, edge_type)

def add_properties_to_graph(board, graph_id, graphs_obj, board_dim):
    """Add properties to nodes (1, 0, or -1).
    
    Args:
        board: 2D array of board state
        graph_id: Index of this graph
        graphs_obj: Graphs object to add to
        board_dim: Size of the board
    """
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            
            # CSV encoding: 1=Player 1, 0=Player 0, -1=Empty
            if board[i][j] == 1:  # Player 1
                graphs_obj.add_graph_node_property(graph_id, node_id, '1')
            elif board[i][j] == 0:  # Player 0
                graphs_obj.add_graph_node_property(graph_id, node_id, '0')
            else:  # Empty (-1)
                graphs_obj.add_graph_node_property(graph_id, node_id, '-1')

def create_graphs_from_games(games, board_dim=None, hypervector_size=128, hypervector_bits=2):
    """
    THE MAIN ORCHESTRATION FUNCTION

    Convert list of games to Graphs object.
    
    Args:
        games: List of (board, winner) tuples
        board_dim: Board dimension (if None, uses global _BOARD_DIM)
        hypervector_size: Size of hypervectors for encoding (default: 128)
        hypervector_bits: Bits per hypervector (default: 2)
    
    Returns:
        graphs: Graphs object
        labels: numpy array of winners
    """
    if board_dim is None:
        board_dim = _BOARD_DIM
    
    num_games = len(games)
    symbols = ['1', '0', '-1']  # Simple symbols matching CSV values
    num_nodes = board_dim * board_dim
    
    # Create Graphs object
    graphs = Graphs(
        number_of_graphs=num_games,
        symbols=symbols,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits
    )
    
    # Step 1: Set number of nodes for each graph
    for graph_id in range(num_games):
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    
    # Step 2: Prepare node configuration
    # The system allocates memory for all nodes aka it needs to know how many nodes per graph
    graphs.prepare_node_configuration()

    # Step 3: Add all nodes
    # Sets up the node structure in the Graphs object
    for graph_id, (board, winner) in enumerate(games):
        convert_board_to_graph(board, graph_id, graphs, board_dim)

    # Step 4: Prepare edge configuration
    # Allocates memory for edges based on previously added nodes
    graphs.prepare_edge_configuration()
    
    # Step 5: Add all edges
    # Adds all connections between neighboring cells to the graph
    for graph_id, (board, winner) in enumerate(games):
        add_edges_to_graph(graph_id, graphs, board_dim)
    
    # Step 6: Add properties to nodes
    # Each node now knows what player's piece is on it (or if it's empty)
    for graph_id, (board, winner) in enumerate(games):
        add_properties_to_graph(board, graph_id, graphs, board_dim)
    
    # Step 7: Encode the graphs
    # Converts everything to hypervector representation
    graphs.encode()
    
    # Extract labels (winners)
    labels = np.array([winner for board, winner in games], dtype=np.int32)
    
    return graphs, labels

def parse_games(csv_file, board_dim=None, hypervector_size=128, hypervector_bits=2, verbose=True):
    """
    Main function to parse CSV game files and convert to graph format.
    
    Args:
        csv_file: Path to CSV file containing game states
        board_dim: Board dimension (if None, uses global _BOARD_DIM)
        hypervector_size: Size of hypervectors for encoding (default: 128)
        hypervector_bits: Bits per hypervector (default: 2)
        verbose: Print progress messages (default: True)
    
    Returns:
        graphs: Graphs object ready for training
        labels: Winner labels for each game
    """
    if board_dim is None:
        board_dim = _BOARD_DIM
    
    if verbose:
        print(f"Parsing games from {csv_file}...")
    
    games = parse_csv_games(csv_file, board_dim)
    
    if verbose:
        print(f"Found {len(games)} games")
    
    if len(games) == 0:
        if verbose:
            print("No games found in CSV file!")
        return None, None
    
    if verbose:
        print("Converting to graph format...")
    
    graphs, labels = create_graphs_from_games(games, board_dim, hypervector_size, hypervector_bits)
    
    if verbose:
        print(f"Created {graphs.number_of_graphs} graphs")
        print(f"Winner distribution: Player 0: {np.sum(labels == 0)}, Player 1: {np.sum(labels == 1)}")
    
    return graphs, labels

if __name__ == "__main__":

    csv_file = 'datasets/hex_games.csv'

    graphs, labels = parse_games(csv_file=csv_file)
