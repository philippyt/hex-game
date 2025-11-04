from __future__ import annotations
import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs
from typing import List, Tuple

RED = -1
EMPTY = 0
BLUE = 1

def create_hex_neighbors(board_dim: int) -> List[int]:
    return [-(board_dim + 2) + 1, -(board_dim + 2), -1, 1, (board_dim + 2), (board_dim + 2) - 1]

def parse_csv_games(csv_file: str, board_dim: int = 9) -> List[Tuple[np.ndarray, int]]:
    df = pd.read_csv(csv_file)
    games: List[Tuple[np.ndarray, int]] = []
    for _, row in df.iterrows():
        board = np.array([[int(row[f'cell_{i}_{j}']) for j in range(board_dim)] for i in range(board_dim)], dtype=np.int8)
        winner = int(row["winner"])
        games.append((board, winner))
    return games

def convert_board_to_graph(board: np.ndarray, graph_id: int, graphs_obj: Graphs, board_dim: int) -> None:
    neighbors = create_hex_neighbors(board_dim)
    num_nodes = board_dim * board_dim
    graphs_obj.set_number_of_graph_nodes(graph_id, num_nodes)
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            pos = (i + 1) * (board_dim + 2) + (j + 1)
            valid_neighbors = 0
            for off in neighbors:
                npos = pos + off
                ni = (npos // (board_dim + 2)) - 1
                nj = (npos % (board_dim + 2)) - 1
                if 0 <= ni < board_dim and 0 <= nj < board_dim:
                    valid_neighbors += 1
            graphs_obj.add_graph_node(graph_id, node_id, valid_neighbors)

def add_edges_to_graph(board: np.ndarray, graph_id: int, graphs_obj: Graphs, board_dim: int) -> None:
    neighbors = create_hex_neighbors(board_dim)
    for i in range(board_dim):
        for j in range(board_dim):
            src = i * board_dim + j
            pos = (i + 1) * (board_dim + 2) + (j + 1)
            src_val = board[i][j]
            for off in neighbors:
                npos = pos + off
                ni = (npos // (board_dim + 2)) - 1
                nj = (npos % (board_dim + 2)) - 1
                if 0 <= ni < board_dim and 0 <= nj < board_dim:
                    dst = ni * board_dim + nj
                    dst_val = board[ni][nj]
                    if src_val == dst_val and src_val != 0:
                        graphs_obj.add_graph_node_edge(graph_id, src, dst, f"SameColor_{'Pos' if src_val==1 else 'Neg'}")
                    else:
                        graphs_obj.add_graph_node_edge(graph_id, src, dst, 'Adjacent')

def add_properties_to_graph(board: np.ndarray, graph_id: int, graphs_obj: Graphs, board_dim: int) -> None:
    for i in range(board_dim):
        for j in range(board_dim):
            node_id = i * board_dim + j
            value = board[i][j]
            if value == RED:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'PlayerNeg')
            elif value == BLUE:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'PlayerPos')
            else:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'Empty')
            graphs_obj.add_graph_node_property(graph_id, node_id, f'Row{i}')
            graphs_obj.add_graph_node_property(graph_id, node_id, f'Col{j}')
            if i == 0 or j == 0 or i == board_dim - 1 or j == board_dim - 1:
                graphs_obj.add_graph_node_property(graph_id, node_id, 'EdgeNode')
            if (i in (0, board_dim - 1)) and (j in (0, board_dim - 1)):
                graphs_obj.add_graph_node_property(graph_id, node_id, 'CornerNode')

def create_graphs_from_games(games: List[Tuple[np.ndarray, int]], board_dim: int = 9, hypervector_size: int = 128, hypervector_bits: int = 2, init_with: Graphs | None = None) -> Tuple[Graphs, np.ndarray]:
    num_games = len(games)
    base_symbols = ['PlayerNeg', 'PlayerPos', 'Empty', 'EdgeNode', 'CornerNode', 'Adjacent', 'SameColor_Pos', 'SameColor_Neg']
    symbols = base_symbols + [f'Row{i}' for i in range(board_dim)] + [f'Col{i}' for i in range(board_dim)]
    if init_with is not None:
        graphs = Graphs(number_of_graphs=num_games, init_with=init_with)
    else:
        graphs = Graphs(number_of_graphs=num_games, symbols=symbols, hypervector_size=hypervector_size, hypervector_bits=hypervector_bits)
    graphs.prepare_node_configuration()
    for graph_id, (board, _) in enumerate(games):
        convert_board_to_graph(board, graph_id, graphs, board_dim)
    graphs.prepare_edge_configuration()
    for graph_id, (board, _) in enumerate(games):
        add_edges_to_graph(board, graph_id, graphs, board_dim)
    for graph_id, (board, _) in enumerate(games):
        add_properties_to_graph(board, graph_id, graphs, board_dim)
    graphs.encode()
    labels = np.array([(1 if winner == 1 else 0) for _, winner in games], dtype=np.int32)
    return graphs, labels