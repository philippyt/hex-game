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
        board = np.array([[int(row[f'cell_{i}_{j}']) for j in range(board_dim)] for i in range(board_dim)],dtype=np.int8,)
        winner = int(row["winner"])
        games.append((board, winner))
    return games

def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(a, b)

def majority_bundle(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Empty list for bundle.")
    stacked = np.stack(vectors)
    votes = stacked.sum(axis=0)
    return (votes > (len(vectors) / 2)).astype(np.uint8)

def initialize_vectors(symbols: List[str], board_dim: int, hypervector_size: int) -> Tuple[dict, dict]:
    rng = np.random.default_rng(42)
    node_vectors = {i: rng.integers(0, 2, hypervector_size, dtype=np.uint8) for i in range(board_dim * board_dim)}
    symbol_vectors = {s: rng.integers(0, 2, hypervector_size, dtype=np.uint8) for s in symbols}
    return node_vectors, symbol_vectors

def configure_nodes(graphs: Graphs, games: List[Tuple[np.ndarray, int]], board_dim: int):
    for g_id, _ in enumerate(games):
        graphs.set_number_of_graph_nodes(g_id, board_dim * board_dim)
    graphs.prepare_node_configuration()

    neighbors = create_hex_neighbors(board_dim)
    for g_id, _ in enumerate(games):
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
                graphs.add_graph_node(g_id, node_id, valid_neighbors)

def configure_edges(graphs: Graphs, games: List[Tuple[np.ndarray, int]], board_dim: int, node_vectors: dict, symbol_vectors: dict, use_binding: bool):
    graphs.prepare_edge_configuration()
    neighbors = create_hex_neighbors(board_dim)

    for g_id, (board, _) in enumerate(games):
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
                            edge_type = f"SameColor_{'Pos' if src_val == 1 else 'Neg'}"
                        else:
                            edge_type = "Adjacent"

                        if use_binding:
                            sym_vec = symbol_vectors[edge_type]
                            bound_vec = xor_bind(sym_vec, xor_bind(node_vectors[src], node_vectors[dst]))
                            symbol_vectors[edge_type] = majority_bundle([symbol_vectors[edge_type], bound_vec])

                        graphs.add_graph_node_edge(g_id, src, dst, edge_type)

def add_node_properties(graphs: Graphs, games: List[Tuple[np.ndarray, int]], board_dim: int):
    for g_id, (board, _) in enumerate(games):
        for i in range(board_dim):
            for j in range(board_dim):
                node_id = i * board_dim + j
                value = board[i][j]

                if value == RED:
                    graphs.add_graph_node_property(g_id, node_id, "PlayerNeg")
                elif value == BLUE:
                    graphs.add_graph_node_property(g_id, node_id, "PlayerPos")
                else:
                    graphs.add_graph_node_property(g_id, node_id, "Empty")

                graphs.add_graph_node_property(g_id, node_id, f"Row{i}")
                graphs.add_graph_node_property(g_id, node_id, f"Col{j}")

                if i == 0 or j == 0 or i == board_dim - 1 or j == board_dim - 1:
                    graphs.add_graph_node_property(g_id, node_id, "EdgeNode")
                if (i in (0, board_dim - 1)) and (j in (0, board_dim - 1)):
                    graphs.add_graph_node_property(g_id, node_id, "CornerNode")

def create_graphs_from_games(games: List[Tuple[np.ndarray, int]], board_dim: int = 9, hypervector_size: int = 128, hypervector_bits: int = 2, init_with: Graphs | None = None, use_binding: bool = True) -> Tuple[Graphs, np.ndarray]:
    num_games = len(games)
    base_symbols = [
        "PlayerNeg",
        "PlayerPos",
        "Empty",
        "EdgeNode",
        "CornerNode",
        "Adjacent",
        "SameColor_Pos",
        "SameColor_Neg",
    ]
    symbols = base_symbols + [f"Row{i}" for i in range(board_dim)] + [f"Col{i}" for i in range(board_dim)]

    if init_with is not None:
        graphs = Graphs(number_of_graphs=num_games, init_with=init_with)
    else:
        graphs = Graphs(
            number_of_graphs=num_games,
            symbols=symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
        )

    node_vectors, symbol_vectors = initialize_vectors(symbols, board_dim, hypervector_size)
    configure_nodes(graphs, games, board_dim)
    configure_edges(graphs, games, board_dim, node_vectors, symbol_vectors, use_binding)
    add_node_properties(graphs, games, board_dim)
    graphs.encode()
    graphs.bound_symbol_vectors = symbol_vectors

    labels = np.array([(1 if winner == 1 else 0) for _, winner in games], dtype=np.int32)
    return graphs, labels