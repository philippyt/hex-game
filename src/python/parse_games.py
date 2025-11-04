import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs

def create_hex_neighbors(board_dim):
    """Create neighbor offsets for hex board"""
    return [-(board_dim+2) + 1, -(board_dim+2), -1, 1, (board_dim+2), (board_dim+2) - 1]

def parse_csv_games(csv_file, board_dim=9):
    """Parse CSV file containing hex game data"""
    df = pd.read_csv(csv_file)
    games = []
    
    for idx, row in df.iterrows():
        board = []
        for i in range(board_dim):
            row_cells = []
            for j in range(board_dim):
                row_cells.append(int(row[f'cell_{i}_{j}']))
            board.append(row_cells)
        
        winner = int(row['winner'])
        games.append((board, winner))
    
    return games

def convert_board_to_graph(board, graph_id, graphs_obj, board_dim):
    """Convert board state to graph representation"""
    neighbors = create_hex_neighbors(board_dim)
    # Note: set_number_of_graph_nodes is called BEFORE this function
    
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            position_in_padded = (i+1) * (board_dim+2) + (j+1)
            valid_neighbors = 0
            
            for neighbor_offset in neighbors:
                neighbor_pos = position_in_padded + neighbor_offset
                neighbor_i = (neighbor_pos // (board_dim+2)) - 1
                neighbor_j = (neighbor_pos % (board_dim+2)) - 1
                
                if 0 <= neighbor_i < board_dim and 0 <= neighbor_j < board_dim:
                    valid_neighbors += 1
            
            graphs_obj.add_graph_node(graph_id, node_id, valid_neighbors)

def add_edges_to_graph(graph_id, graphs_obj, board_dim):
    """Add edges between neighboring cells"""
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
                    graphs_obj.add_graph_node_edge(graph_id, source_node_id, dest_node_id, 'Adjacent')

def add_properties_to_graph(board, graph_id, graphs_obj, board_dim):
    """Add properties to nodes
    CSV encoding: -1 and 1 are the two players, 0 is empty
    Winner encoding: winner=0 means player with 1's wins, winner=1 means player with -1's wins
    
    Also adds position information to help model learn spatial patterns
    """
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            
            # Add piece type
            if board[i][j] == -1:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'PlayerNeg')
            elif board[i][j] == 1:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'PlayerPos')
            else:  # board[i][j] == 0
                graphs_obj.add_graph_node_property(graph_id, node_id, 'Empty')
            
            # Add position information (helps model learn spatial patterns)
            graphs_obj.add_graph_node_property(graph_id, node_id, f'Row{i}')
            graphs_obj.add_graph_node_property(graph_id, node_id, f'Col{j}')

def create_graphs_from_games(games, board_dim=9, hypervector_size=128, hypervector_bits=2, init_with=None):
    """Convert list of games to Graphs object
    
    Args:
        games: List of (board, winner) tuples
        board_dim: Dimension of the board
        hypervector_size: Size of hypervectors
        hypervector_bits: Bits per hypervector
        init_with: Optional Graphs object to initialize from (for test set)
    """
    num_games = len(games)
    
    # Create symbol list: piece types + position information
    symbols = ['PlayerNeg', 'PlayerPos', 'Empty']
    # Add row and column symbols for spatial awareness
    for i in range(board_dim):
        symbols.append(f'Row{i}')
        symbols.append(f'Col{i}')
    
    num_nodes = board_dim * board_dim
    
    if init_with is not None:
        # Use existing hypervector mappings from train set
        graphs = Graphs(
            number_of_graphs=num_games,
            init_with=init_with
        )
    else:
        # Create new hypervector mappings (for train set)
        graphs = Graphs(
            number_of_graphs=num_games,
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits
        )
    
    # Set number of nodes
    for graph_id in range(num_games):
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    
    graphs.prepare_node_configuration()
    
    # Add nodes
    for graph_id, (board, winner) in enumerate(games):
        convert_board_to_graph(board, graph_id, graphs, board_dim)
    
    graphs.prepare_edge_configuration()
    
    # Add edges
    for graph_id, (board, winner) in enumerate(games):
        add_edges_to_graph(graph_id, graphs, board_dim)
    
    # Add properties
    for graph_id, (board, winner) in enumerate(games):
        add_properties_to_graph(board, graph_id, graphs, board_dim)
    
    graphs.encode()
    
    labels = np.array([winner for board, winner in games], dtype=np.int32)
    
    return graphs, labels
