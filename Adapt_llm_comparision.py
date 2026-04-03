# %%
# ------------------------
# IMPORTS
# ------------------------

import random
from typing import Tuple
import numpy as np
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.adapt_utils import get_combined_res_df, compute_metrics
from src.model_interface import QAOA_GPT

pd.set_option("display.max_columns", None)

# %%
# CONFIG
SEED = 1337
data_input_path = "./ADAPT.jl_results/test/9_nodes"
MODEL_CONFIGS = [
    dict(
        name="GPT-Feather",
        ckpt="nanoGPT/out-9_nodes_feather/gpt_ckpt_3500_feather_ar_0_95709__er_0_0.pt",
        data_dir="nanoGPT/data/9_nodes_feather",
    ),
]
N_SAMPLES = 5
MAX_TOKENS = 150

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
    """Load unique graphs from ADAPT df (drop_duplicates on graph_name before passing)."""
    graphs, meta = [], []
    for _, row in adapt_df.iterrows():
        G = edgelist_to_nx(row["edgelist_list"], row["n_nodes"])
        graphs.append(G)
        meta.append({"graph_name": row["graph_name"]})
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

# %%
# ------------------------
# LOAD & AGGREGATE ADAPT
# ------------------------

adapt_df = get_combined_res_df(data_input_path, debug_limit=None)

print(f"Total ADAPT rows      : {len(adapt_df)}")
print(f"Unique graphs         : {adapt_df['graph_name'].nunique()}")
print(f"Runs per graph (mean) : {adapt_df.groupby('graph_name').size().mean():.2f}")

# Aggregate ADAPT per graph — captures mean, best, std, and layer stats
adapt_agg = adapt_df.groupby("graph_name").agg(
    adapt_ar_mean  = ("approx_ratio", "mean"),
    adapt_ar_best  = ("approx_ratio", "max"),
    adapt_ar_std   = ("approx_ratio", "std"),
    adapt_layers_mean = ("n_layers",  "mean"),
    adapt_layers_best = ("n_layers",  "min"),   # min layers = most efficient run
    adapt_n_runs   = ("run",          "count"),
).reset_index()

adapt_agg["adapt_ar_std"] = adapt_agg["adapt_ar_std"].fillna(0)  # single-run graphs → std=NaN

print(f"\nAggregated ADAPT shape: {adapt_agg.shape}")

# %%
adapt_agg.head()

# %%
# Use only unique graphs for model generation
unique_adapt_df = adapt_df.drop_duplicates(subset="graph_name").reset_index(drop=True)
graphs_unique, meta_df = load_graphs_from_adapt(unique_adapt_df)

print(f"Graphs fed to model: {len(graphs_unique)}")

# %%
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
# RUN ALL MODELS
# ------------------------

all_results = []

for cfg in MODEL_CONFIGS:
    print(f"\nRunning {cfg['name']} ...")
    model   = load_model(cfg)
    df_eval = run_model(model, graphs_unique)

    model_ar, model_layers = compute_model_metrics(df_eval)

    res_df = pd.DataFrame({
        "graph_name"       : meta_df["graph_name"],
        "model"            : cfg["name"],
        "model_ar"         : model_ar.values,
        "model_layers"     : model_layers.values,
    })
    all_results.append(res_df)

model_results_df = pd.concat(all_results, ignore_index=True)

# %%
model_results_df.head()

# %%
# ------------------------
# MERGE
# ------------------------

final_df = adapt_agg.merge(model_results_df, on="graph_name")

# Diffs vs mean ADAPT and vs best ADAPT
final_df["ar_diff_vs_mean"] = final_df["model_ar"] - final_df["adapt_ar_mean"]
final_df["ar_diff_vs_best"] = final_df["model_ar"] - final_df["adapt_ar_best"]
final_df["layer_diff"]      = final_df["model_layers"] - final_df["adapt_layers_mean"]

print(f"\nFinal df shape: {final_df.shape}")

# %%
final_df.head(10)

# %%
# ------------------------
# SUMMARY TABLE
# ------------------------

summary_df = final_df.groupby("model").agg(
    adapt_ar_mean    = ("adapt_ar_mean",   "mean"),
    adapt_ar_best    = ("adapt_ar_best",   "mean"),
    model_ar         = ("model_ar",        "mean"),
    adapt_layers     = ("adapt_layers_mean","mean"),
    model_layers     = ("model_layers",    "mean"),
    ar_diff_vs_mean  = ("ar_diff_vs_mean", "mean"),
    ar_diff_vs_best  = ("ar_diff_vs_best", "mean"),
    n_graphs         = ("graph_name",      "count"),
).reset_index()

print(summary_df.to_string(index=False))

# %%
# ============================================================
# PLOTS  — 6-panel dashboard
# ============================================================

def make_dashboard(final_df, summary_df):

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("QAOA: ADAPT vs GPT-Feather — 9-node graphs", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors = {"ADAPT mean": "#4C72B0", "ADAPT best": "#55A868", "Model": "#C44E52"}

    # ── 1. Bar: Average AR comparison ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(summary_df))
    w   = 0.25
    ax1.bar(x - w,  summary_df["adapt_ar_mean"], width=w, label="ADAPT (mean)", color=colors["ADAPT mean"])
    ax1.bar(x,      summary_df["adapt_ar_best"], width=w, label="ADAPT (best)", color=colors["ADAPT best"])
    ax1.bar(x + w,  summary_df["model_ar"],      width=w, label="Model",        color=colors["Model"])
    ax1.set_xticks(x); ax1.set_xticklabels(summary_df["model"], rotation=10)
    ax1.set_ylabel("Approx. Ratio"); ax1.set_title("Avg Approximation Ratio")
    ax1.set_ylim(0.85, 1.02)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.4)

    # ── 2. Bar: Average Layers comparison ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w/2, summary_df["adapt_layers"], width=w, label="ADAPT (mean)", color=colors["ADAPT mean"])
    ax2.bar(x + w/2, summary_df["model_layers"], width=w, label="Model",        color=colors["Model"])
    ax2.set_xticks(x); ax2.set_xticklabels(summary_df["model"], rotation=10)
    ax2.set_ylabel("Layers"); ax2.set_title("Avg Number of Layers")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.4)

    # ── 3. Scatter: AR vs Layers trade-off ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(final_df["adapt_layers_mean"], final_df["adapt_ar_mean"],
                label="ADAPT mean", alpha=0.6, color=colors["ADAPT mean"])
    ax3.scatter(final_df["adapt_layers_best"], final_df["adapt_ar_best"],
                label="ADAPT best", alpha=0.6, color=colors["ADAPT best"], marker="^")
    ax3.scatter(final_df["model_layers"],      final_df["model_ar"],
                label="Model",      alpha=0.6, color=colors["Model"],      marker="s")
    ax3.set_xlabel("Layers"); ax3.set_ylabel("Approx. Ratio")
    ax3.set_title("AR vs Layers Trade-off")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.4)

    # ── 4. Per-graph AR line plot ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    idx = np.arange(len(final_df))
    ax4.plot(idx, final_df["adapt_ar_mean"], label="ADAPT (mean)", color=colors["ADAPT mean"])
    ax4.fill_between(idx,
                     final_df["adapt_ar_mean"] - final_df["adapt_ar_std"],
                     final_df["adapt_ar_mean"] + final_df["adapt_ar_std"],
                     alpha=0.15, color=colors["ADAPT mean"])
    ax4.plot(idx, final_df["adapt_ar_best"], label="ADAPT (best)", color=colors["ADAPT best"],
             linestyle="--", alpha=0.7)
    ax4.plot(idx, final_df["model_ar"],      label="Model",        color=colors["Model"])
    ax4.set_xlabel("Graph index"); ax4.set_ylabel("Approx. Ratio")
    ax4.set_title("Per-graph AR  (shaded = ADAPT ± 1 std)")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.4)

    # ── 5. Histogram: AR diff (model − ADAPT mean) ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    diffs = final_df["ar_diff_vs_mean"]
    ax5.hist(diffs, bins=20, color=colors["Model"], edgecolor="white", alpha=0.85)
    ax5.axvline(0,          color="black", linewidth=1.2, linestyle="--", label="Parity")
    ax5.axvline(diffs.mean(), color="gold", linewidth=1.5, linestyle="-",  label=f"Mean {diffs.mean():+.4f}")
    ax5.set_xlabel("Model AR − ADAPT mean AR")
    ax5.set_ylabel("Count")
    ax5.set_title("AR Gap Distribution\n(positive = model beats ADAPT mean)")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.4)

    plt.savefig("qaoa_comparison_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → qaoa_comparison_dashboard.png")


make_dashboard(final_df, summary_df)

# %%
# ------------------------
# EXTRA INSIGHTS
# ------------------------

print("=" * 55)
print("INSIGHTS")
print("=" * 55)

for _, row in summary_df.iterrows():
    print(f"\nModel : {row['model']}")
    print(f"  Graphs evaluated      : {int(row['n_graphs'])}")
    print(f"  ADAPT AR (mean / best): {row['adapt_ar_mean']:.4f} / {row['adapt_ar_best']:.4f}")
    print(f"  Model AR              : {row['model_ar']:.4f}")
    print(f"  AR diff vs mean       : {row['ar_diff_vs_mean']:+.4f}")
    print(f"  AR diff vs best       : {row['ar_diff_vs_best']:+.4f}")
    print(f"  ADAPT layers (mean)   : {row['adapt_layers']:.2f}")
    print(f"  Model layers          : {row['model_layers']:.2f}")
    print(f"  Layer reduction       : {row['adapt_layers'] - row['model_layers']:+.2f}")
    print(f"  Model error rate      : {row['model_error_rate']:.4f}")

n_model_wins_mean = (final_df["ar_diff_vs_mean"] > 0).sum()
n_model_wins_best = (final_df["ar_diff_vs_best"] > 0).sum()
print(f"\n  Graphs where model > ADAPT mean : {n_model_wins_mean} / {len(final_df)}")
print(f"  Graphs where model > ADAPT best : {n_model_wins_best} / {len(final_df)}")


