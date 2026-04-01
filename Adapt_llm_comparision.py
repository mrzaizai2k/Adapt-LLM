# %% [markdown]
# # QAOA Comparison Framework: ADAPT vs GPT (Scalable)

# %%
# ------------------------
# IMPORTS
# ------------------------

import random
import numpy as np
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path

from adapt_utils import get_combined_res_df
from model_interface import QAOA_GPT

pd.set_option("display.max_columns", None)

# %%
# ------------------------
# CONFIG
# ------------------------

SEED = 1337

# ADAPT
ADAPT_OUTPUT_DIR = "./ADAPT.jl_results/test/9_nodes"

# MODEL CONFIG (can extend later)
MODEL_CONFIGS = [
    dict(
        name="GPT-Feather",
        ckpt="nanoGPT/out-9_nodes_feather/gpt_ckpt_3500_feather_ar_0_95709__er_0_0.pt",
        data_dir="nanoGPT/data/9_nodes_feather",
    ),
]

# GENERATION
N_SAMPLES = 5
MAX_TOKENS = 150

# %%
# ------------------------
# SEED
# ------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
# ------------------------
# GRAPH UTILS
# ------------------------

def edgelist_to_nx(edgelist, n_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for u, v, w in edgelist:
        G.add_edge(u - 1, v - 1, weight=w)

    return G


def load_graphs_from_adapt(adapt_df):
    graphs, meta = [], []

    for _, row in adapt_df.iterrows():
        G = edgelist_to_nx(row["edgelist_list"], row["n_nodes"])
        graphs.append(G)

        meta.append({
            "graph_name": row["graph_name"],
            "adapt_ar": row["approx_ratio"],
            "adapt_layers": row["n_layers"],
        })

    return graphs, pd.DataFrame(meta)

# %%
# ------------------------
# MODEL UTILS
# ------------------------

def load_model(cfg):
    return QAOA_GPT(
        model_ckpt=cfg["ckpt"],
        data_dir=cfg["data_dir"],
        temp_folder="temp_data",
    )


def run_model(qaoa, graphs):
    df_model = qaoa.generate_circ_from_nx(
        graphs,
        num_samples=N_SAMPLES,
        max_new_tokens=MAX_TOKENS,
        temperature=0.1,
        top_k=200,
    )

    return qaoa.eval_circ_df_jl(df_model)


def compute_model_metrics(df):

    df_expl = df.explode(["adapt_gpt_energies", "q_circuits"])

    # layers
    layers = df_expl["q_circuits"].apply(
        lambda x: x.count("new_layer_p")
    )

    # energy
    df_energy = df[
        ["adapt_gpt_energies", "energy_gurobi"]
    ].explode("adapt_gpt_energies")

    df_corr = df_energy[
        df_energy["adapt_gpt_energies"] != 999
    ].copy()

    df_corr["ar"] = (
        df_corr["adapt_gpt_energies"]
        / df_corr["energy_gurobi"]
    )

    return (
        df_corr.groupby(level=0)["ar"].mean(),
        layers.groupby(level=0).mean(),
    )

# %%
# ------------------------
# LOAD ADAPT
# ------------------------

adapt_df = get_combined_res_df(
    ADAPT_OUTPUT_DIR,
    debug_limit=None,
)

graphs, meta_df = load_graphs_from_adapt(adapt_df)

# %%
# ------------------------
# RUN ALL MODELS
# ------------------------

all_results = []

for cfg in MODEL_CONFIGS:

    print(f"Running {cfg['name']}")

    model = load_model(cfg)
    df_eval = run_model(model, graphs)

    model_ar, model_layers = compute_model_metrics(df_eval)

    res_df = pd.DataFrame({
        "graph_name": meta_df["graph_name"],
        "model": cfg["name"],
        "model_ar": model_ar.values,
        "model_layers": model_layers.values,
    })

    all_results.append(res_df)

model_results_df = pd.concat(all_results, ignore_index=True)

# %%
# ------------------------
# MERGE WITH ADAPT
# ------------------------

final_df = meta_df.merge(
    model_results_df,
    on="graph_name"
)

final_df["ar_diff"] = final_df["model_ar"] - final_df["adapt_ar"]
final_df["layer_diff"] = final_df["model_layers"] - final_df["adapt_layers"]

final_df.head()

# %%
# ------------------------
# SUMMARY TABLE
# ------------------------

summary_df = final_df.groupby("model").agg({
    "adapt_ar": "mean",
    "model_ar": "mean",
    "adapt_layers": "mean",
    "model_layers": "mean",
    "ar_diff": "mean",
    "layer_diff": "mean",
}).reset_index()

summary_df

# %%
# ------------------------
# PLOTTING FUNCTIONS
# ------------------------

def plot_bar_avg(df):

    x = np.arange(len(df["model"]))

    plt.figure()
    plt.bar(x - 0.15, df["adapt_ar"], width=0.3, label="ADAPT")
    plt.bar(x + 0.15, df["model_ar"], width=0.3, label="MODEL")

    plt.xticks(x, df["model"])
    plt.ylabel("Approximation Ratio")
    plt.title("Average Approximation Ratio")

    plt.legend()
    plt.grid()
    plt.show()


def plot_bar_layers(df):

    x = np.arange(len(df["model"]))

    plt.figure()
    plt.bar(x - 0.15, df["adapt_layers"], width=0.3, label="ADAPT")
    plt.bar(x + 0.15, df["model_layers"], width=0.3, label="MODEL")

    plt.xticks(x, df["model"])
    plt.ylabel("Layers")
    plt.title("Average Layers")

    plt.legend()
    plt.grid()
    plt.show()


def plot_scatter_tradeoff(df):

    plt.figure()

    plt.scatter(
        df["adapt_layers"],
        df["adapt_ar"],
        label="ADAPT",
    )

    plt.scatter(
        df["model_layers"],
        df["model_ar"],
        label="MODEL",
    )

    plt.xlabel("Layers")
    plt.ylabel("Approximation Ratio")
    plt.title("AR vs Layers Tradeoff")

    plt.legend()
    plt.grid()
    plt.show()


def plot_per_graph(final_df):

    plt.figure()
    plt.plot(final_df["adapt_ar"].values, label="ADAPT")
    plt.plot(final_df["model_ar"].values, label="MODEL")
    plt.title("AR per graph")
    plt.legend()
    plt.grid()
    plt.show()

# %%
# ------------------------
# RUN PLOTS
# ------------------------

plot_bar_avg(summary_df)
plot_bar_layers(summary_df)
plot_scatter_tradeoff(summary_df)
plot_per_graph(final_df)

# %%
# ------------------------
# INSIGHTS
# ------------------------

print("===== INSIGHTS =====")

print("\nAvg ADAPT AR:", summary_df["adapt_ar"].mean())
print("Avg MODEL AR:", summary_df["model_ar"].mean())

print("\nAvg ADAPT Layers:", summary_df["adapt_layers"].mean())
print("Avg MODEL Layers:", summary_df["model_layers"].mean())

print("\nKey Observation:")

if summary_df["model_ar"].mean() > summary_df["adapt_ar"].mean():
    print("- Model outperforms ADAPT in approximation ratio")
else:
    print("- ADAPT still stronger in approximation ratio")

if summary_df["model_layers"].mean() < summary_df["adapt_layers"].mean():
    print("- Model produces shallower circuits")
else:
    print("- Model circuits are deeper")

print("\nTradeoff:")
print("- Check scatter plot: better models sit TOP-LEFT (high AR, low layers)")