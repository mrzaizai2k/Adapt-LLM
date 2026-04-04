
import time
import yaml
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper



def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def add_weights_to_nx_graph(nx_graph):
    for u, v in nx_graph.edges():
        w = round(random.uniform(0, 1), 2)
        while w == 0:
            w = round(random.uniform(0, 1), 2)
        nx_graph[u][v]["weight"] = w
    return nx_graph


def generate_er_graphs(n_graphs, n_nodes):

    graphs = {}

    for i in range(n_graphs):

        p = random.randrange(6, 9) / 10

        g = nx.erdos_renyi_graph(
            n=n_nodes,
            p=p
        )

        g = add_weights_to_nx_graph(g)

        graphs[f"er_graph_{i}"] = g

    return graphs


import re
from typing import Tuple, List, Dict
 
import networkx as nx
import numpy as np
import pandas as pd
 
# ---------------------------------------------------------------------------
# ARCH / METHOD ALIASES
# ---------------------------------------------------------------------------
 
ARCH_ALIASES: Dict[str, str] = {
    "llama": "LLaMA",
    "gpt":   "NanoGPT",
}
 
METHOD_ALIASES: Dict[str, str] = {
    "netlsd":  "NetLSD",
    "feather": "Feather",
    "gnn":     "GNN",
}
 
# ---------------------------------------------------------------------------
# NAME EXTRACTION
# ---------------------------------------------------------------------------
 
def extract_arch(ckpt_path: str) -> str:
    """
    Extract the architecture token from a checkpoint filename.
 
    The filename is expected to follow the pattern:
        <arch>_ckpt_<step>_<method>_...
 
    Examples:
        "llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt" -> "LLaMA"
        "gpt_ckpt_3500_feather_ar_0_957__er_0_0.pt" -> "GPT"
 
    Returns the alias from ARCH_ALIASES if found, otherwise the raw token uppercased.
    """
    basename = ckpt_path.split("/")[-1]
    parts    = basename.split("_")
    try:
        raw = parts[0].lower()
        return ARCH_ALIASES.get(raw, raw.upper())
    except IndexError:
        return basename
 
 
def extract_method(ckpt_path: str) -> str:
    """
    Extract the embedding/method token from a checkpoint filename.
 
    The filename is expected to follow the pattern:
        <arch>_ckpt_<step>_<method>_...
 
    Examples:
        "llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt" -> "GNN"
        "gpt_ckpt_3500_netlsd_ar_0_957__er_0_0.pt"  -> "NetLSD"
 
    Returns the alias from METHOD_ALIASES if found, otherwise the raw token uppercased.
    """
    basename = ckpt_path.split("/")[-1]
    parts    = basename.split("_")
    try:
        raw = parts[3].lower()
        return METHOD_ALIASES.get(raw, raw.upper())
    except IndexError:
        return "UNKNOWN"
 
 
def extract_model_name(ckpt_path: str) -> str:
    """
    Auto-extract a human-readable "<Arch>-<Method>" name from a checkpoint path.
 
    Examples:
        "nanoGPT/out-10_nodes_gnn/llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt"
            -> "LLaMA-GNN"
        "nanoGPT/out-10_nodes_feather/gpt_ckpt_3500_feather_ar_0_957__er_0_0.pt"
            -> "GPT-Feather"
 
    Falls back to the raw basename if parsing fails.
    """
    try:
        arch   = extract_arch(ckpt_path)
        method = extract_method(ckpt_path)
        return f"{arch}-{method}"
    except Exception:
        return ckpt_path.split("/")[-1]
 
 
def resolve_model_name(cfg: dict) -> str:
    """
    Return cfg['name'] if explicitly set, otherwise auto-extract from cfg['ckpt'].
 
    This allows per-entry overrides:
        dict(name="My Custom Name", ckpt="...", data_dir="...")  -> "My Custom Name"
        dict(ckpt="llama_ckpt_5500_gnn...", data_dir="...")      -> "LLaMA-GNN"
    """
    return cfg.get("name") or extract_model_name(cfg["ckpt"])
 
 
def attach_resolved_names(model_configs: List[dict]) -> List[dict]:
    """
    Attach 'resolved_name', 'arch', and 'method' keys to every config in-place.
    Also prints a summary of resolved names.
    """
    for cfg in model_configs:
        cfg["resolved_name"] = resolve_model_name(cfg)
        cfg["arch"]          = extract_arch(cfg["ckpt"])
        cfg["method"]        = extract_method(cfg["ckpt"])
 
    print("Resolved model names:")
    for cfg in model_configs:
        print(f"  {cfg['resolved_name']}  (arch={cfg['arch']}, method={cfg['method']})")
 
    return model_configs
 
# ---------------------------------------------------------------------------
# GRAPH UTILITIES
# ---------------------------------------------------------------------------
 
def graph_name_to_num(graph_name: str) -> int:
    """
    Extract the trailing integer from a graph_name string.
 
    Examples:
        'graph_007'  -> 7
        'g42'        -> 42
        'graph_0042' -> 42
 
    Falls back to 0 if no number is found.
    """
    match = re.search(r"(\d+)$", str(graph_name))
    return int(match.group(1)) if match else 0
 
 
def edgelist_to_nx(edgelist, n_nodes: int) -> nx.Graph:
    """Convert a list of (u, v, w) 1-indexed edge tuples to a NetworkX Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in edgelist:
        G.add_edge(u - 1, v - 1, weight=w)
    return G
 
 
def load_graphs_from_adapt(adapt_df: pd.DataFrame) -> Tuple[List[nx.Graph], pd.DataFrame]:
    """
    Build NetworkX graphs from a (pre-deduplicated) ADAPT DataFrame.
 
    Returns:
        graphs  : list of nx.Graph, one per row
        meta_df : DataFrame with columns ['graph_name', 'graph_num']
    """
    graphs, meta = [], []
    for _, row in adapt_df.iterrows():
        G = edgelist_to_nx(row["edgelist_list"], row["n_nodes"])
        graphs.append(G)
        meta.append({
            "graph_name": row["graph_name"],
            "graph_num":  graph_name_to_num(row["graph_name"]),
        })
    return graphs, pd.DataFrame(meta)
 
# ---------------------------------------------------------------------------
# ADAPT AGGREGATION
# ---------------------------------------------------------------------------
 
def load_and_aggregate_adapt(
    data_input_path: str,
    debug_limit=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[nx.Graph], pd.DataFrame]:
    """
    Load raw ADAPT results, attach graph_num, aggregate per graph, and build
    unique graph list for model inference.
 
    Returns:
        adapt_df      : raw ADAPT DataFrame with 'graph_num' column
        adapt_agg     : per-graph aggregated ADAPT stats
        graphs_unique : list of unique nx.Graph instances
        meta_df       : DataFrame with ['graph_name', 'graph_num'] aligned
                        to graphs_unique
    """
    from src.adapt_utils import get_combined_res_df  # local import to keep utils portable
 
    adapt_df = get_combined_res_df(data_input_path, debug_limit=debug_limit)
    adapt_df["graph_num"] = adapt_df["graph_name"].apply(graph_name_to_num)
 
    print(f"Total ADAPT rows      : {len(adapt_df)}")
    print(f"Unique graphs         : {adapt_df['graph_num'].nunique()}")
    print(f"Runs per graph (mean) : {adapt_df.groupby('graph_num').size().mean():.2f}")
 
    adapt_agg = adapt_df.groupby("graph_num").agg(
        graph_name        = ("graph_name",   "first"),
        adapt_ar_mean     = ("approx_ratio", "mean"),
        adapt_ar_best     = ("approx_ratio", "max"),
        adapt_ar_std      = ("approx_ratio", "std"),
        adapt_layers_mean = ("n_layers",     "mean"),
        adapt_layers_best = ("n_layers",     "min"),
        adapt_n_runs      = ("run",          "count"),
    ).reset_index()
 
    adapt_agg["adapt_ar_std"] = adapt_agg["adapt_ar_std"].fillna(0)
 
    unique_adapt_df        = adapt_df.drop_duplicates(subset="graph_num").reset_index(drop=True)
    graphs_unique, meta_df = load_graphs_from_adapt(unique_adapt_df)
 
    print(f"\nAggregated ADAPT shape : {adapt_agg.shape}")
    print(f"Graphs fed to model    : {len(graphs_unique)}")
 
    return adapt_df, adapt_agg, graphs_unique, meta_df
 
# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------
 
# def compute_metrics_per_graph(
#     df: pd.DataFrame,
# ) -> Tuple[pd.Series, pd.Series, pd.Series]:
#     """
#     Compute evaluation metrics per graph instance.
 
#     Each row in `df` corresponds to a graph and contains lists of:
#         - q_circuits
#         - adapt_gpt_energies
#         - energy_gurobi (scalar)
 
#     Returns:
#         Tuple of three pd.Series (aligned with df.index):
#             - ar         : Mean approximation ratio per graph (NaN if no valid samples)
#             - layers     : Mean number of QAOA layers per graph
#             - error_rate : Fraction of invalid samples (energy == 999) per graph
#     """
#     full_index = df.index
 
#     # --- Layers ---
#     df_expl = df.explode(["adapt_gpt_energies", "q_circuits"])
#     layers = (
#         df_expl.groupby(level=0)["q_circuits"]
#         .apply(lambda xs: xs.apply(lambda x: x.count("new_layer_p")).mean())
#         .reindex(full_index)
#     )
 
#     # --- Energy & error rate ---
#     df_energy = df[["adapt_gpt_energies", "energy_gurobi"]].explode("adapt_gpt_energies")
 
#     error_rate = (
#         df_energy.groupby(level=0)["adapt_gpt_energies"]
#         .apply(lambda x: (x == 999).sum() / len(x))
#         .reindex(full_index, fill_value=0.0)
#     )
 
#     # --- AR (valid samples only) ---
#     df_valid       = df_energy[df_energy["adapt_gpt_energies"] != 999].copy()
#     df_valid["ar"] = df_valid["adapt_gpt_energies"] / df_valid["energy_gurobi"]
 
#     ar = (
#         df_valid.groupby(level=0)["ar"]
#         .mean()
#         .reindex(full_index, fill_value=np.nan)
#     )
 
#     return ar, layers, error_rate
 
 
def build_results_df(
    meta_df: pd.DataFrame,
    cfg: dict,
    ar: pd.Series,
    layers: pd.Series,
    error_rate: pd.Series,
) -> pd.DataFrame:
    """
    Assemble a per-graph results DataFrame from metric series and config metadata.
 
    Columns: graph_name, graph_num, model, arch, method,
             model_ar, model_layers, model_error_rate
    """
    return pd.DataFrame({
        "graph_name"       : meta_df["graph_name"].values,
        "graph_num"        : meta_df["graph_num"].values,
        "model"            : cfg["resolved_name"],
        "arch"             : cfg["arch"],
        "method"           : cfg["method"],
        "model_ar"         : ar.values,
        "model_layers"     : layers.values,
        "model_error_rate" : error_rate.values,
    })
 
 
def build_final_df(adapt_agg: pd.DataFrame, model_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ADAPT aggregates with model results and compute diff columns.
 
    Adds: ar_diff_vs_mean, ar_diff_vs_best, layer_diff
    """
    final_df = adapt_agg.merge(model_results_df, on="graph_num")
 
    if "graph_name_x" in final_df.columns:
        final_df = final_df.rename(columns={"graph_name_x": "graph_name"}).drop(
            columns=["graph_name_y"], errors="ignore"
        )
 
    final_df["ar_diff_vs_mean"] = final_df["model_ar"]     - final_df["adapt_ar_mean"]
    final_df["ar_diff_vs_best"] = final_df["model_ar"]     - final_df["adapt_ar_best"]
    final_df["layer_diff"]      = final_df["model_layers"] - final_df["adapt_layers_mean"]
 
    return final_df.sort_values("graph_num").reset_index(drop=True)
 
 
def build_summary_df(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate final_df to one row per model for high-level comparison.
    Includes arch and method columns for downstream grouping plots.
    """
    return (
        final_df.groupby("model").agg(
            arch             = ("arch",             "first"),
            method           = ("method",           "first"),
            adapt_ar_mean    = ("adapt_ar_mean",    "mean"),
            adapt_ar_best    = ("adapt_ar_best",    "mean"),
            model_ar         = ("model_ar",         "mean"),
            adapt_layers     = ("adapt_layers_mean","mean"),
            model_error_rate = ("model_error_rate", "mean"),
            model_layers     = ("model_layers",     "mean"),
            ar_diff_vs_mean  = ("ar_diff_vs_mean",  "mean"),
            ar_diff_vs_best  = ("ar_diff_vs_best",  "mean"),
            n_graphs         = ("graph_num",        "count"),
        )
        .reset_index()
    )