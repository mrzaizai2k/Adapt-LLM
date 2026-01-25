import sys
sys.path.append("")


import sys
import networkx as nx
import json
from typing import Literal
from src.embedding.feather import FEATHERG


def get_embedding(
    graphs_nx_df,
    n_nodes: int,
    rounding_digits: int = 2,
    method: Literal["feather", "gnn", "graph2vec", "netlsd"] = "feather",
):
    # -----------------------
    # Deduplicate graphs
    # -----------------------
    combined_unique_graphs_df = (
        graphs_nx_df[["graph_id", "edgelist_json"]]
        .drop_duplicates()
    )

    # -----------------------
    # Build NetworkX graphs
    # -----------------------
    def create_weighted_graph_nx(w_elist):
        G = nx.Graph()
        G.add_weighted_edges_from(w_elist)
        return G

    combined_unique_graphs_df["edgelist_py_list"] = (
        combined_unique_graphs_df["edgelist_json"]
        .apply(lambda x: [
            (e[0] - 1, e[1] - 1, e[2]) for e in json.loads(x)
        ])
    )

    combined_unique_graphs_df["graph_nx"] = (
        combined_unique_graphs_df["edgelist_py_list"]
        .apply(create_weighted_graph_nx)
    )

    # -----------------------
    # Filter by node count
    # -----------------------
    graphs_nx_filt_names = []
    graphs_nx_filt_list = []

    for graph_id, g in zip(
        combined_unique_graphs_df["graph_id"],
        combined_unique_graphs_df["graph_nx"],
    ):
        if g.number_of_nodes() == n_nodes:
            graphs_nx_filt_names.append(graph_id)
            graphs_nx_filt_list.append(g)

    if not graphs_nx_filt_list:
        raise ValueError(f"No graphs found with n_nodes={n_nodes}")

    # -----------------------
    # Index mappings
    # -----------------------
    emb_graph_idx_to_id_dict = {
        idx: graph_id
        for idx, graph_id in enumerate(graphs_nx_filt_names)
    }

    # -----------------------
    # Embedding
    # -----------------------
    if method == "feather":
        model = FEATHERG()
        model.fit(graphs=graphs_nx_filt_list)
        emb = model.get_embedding()

    elif method == "gnn":
        raise NotImplementedError("GNN embedding not implemented yet")

    elif method == "graph2vec":
        raise NotImplementedError("Graph2Vec embedding not implemented yet")

    else:
        raise ValueError(f"Unknown embedding method: {method}")

    emb = emb.round(rounding_digits)

    return emb, emb_graph_idx_to_id_dict


if __name__ == "__main__":
    import pandas as pd

    # -----------------------
    # Create dummy test data
    # -----------------------
    # Example: 3 graphs, each with 3 nodes
    # edgelist_json format: [[u, v, weight], ...] (1-based indexing)
    data = {
        "graph_id": [1, 2, 3],
        "edgelist_json": [
            json.dumps([[1, 2, 1.0], [2, 3, 0.5], [1, 3, 0.8]]),
            json.dumps([[1, 2, 0.7], [2, 3, 1.2], [1, 3, 0.3]]),
            json.dumps([[1, 2, 1.5], [2, 3, 0.9], [1, 3, 1.1]]),
        ],
    }

    graphs_nx_df = pd.DataFrame(data)

    # -----------------------
    # Parameters
    # -----------------------
    n_nodes = 3
    rounding_digits = 2


    emb, idx_to_id = get_embedding(
        graphs_nx_df=graphs_nx_df,
        n_nodes=n_nodes,
        rounding_digits=rounding_digits,
        method='feather',
    )

    # -----------------------
    # Inspect results
    # -----------------------
    print("Embedding shape:", emb.shape)
    print("Embedding (first rows):")
    print(emb[:2])

    print("\nIndex → Graph ID mapping:")
    print(idx_to_id)

    