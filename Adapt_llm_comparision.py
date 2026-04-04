# %% [markdown]
# # QAOA Model Evaluation
# Compares ADAPT baseline against multiple GPT/LLaMA-based QAOA circuit generators.

# %%
# ------------------------
# IMPORTS
# ------------------------

import re
import random
import numpy as np
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from src.adapt_utils import get_combined_res_df
from src.model_interface import QAOA_GPT

pd.set_option("display.max_columns", None)

# %%
# ------------------------
# CONFIG
# ------------------------

SEED = 1337
data_input_path = "./ADAPT.jl_results/test/9_nodes"

# Model configs — each entry is a dict with required keys: ckpt, data_dir.
# Optional key: name (auto-extracted from ckpt filename if omitted).
# Auto-extraction: splits filename by "_", takes element[0] as arch and element[3] as method.
# Example: "llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt" → "LLaMA-GNN"
# If name is provided explicitly, it overrides auto-extraction.
MODEL_CONFIGS = [
    dict(
        name="GPT-Feather",                   # explicit name — overrides auto-extraction
        ckpt="nanoGPT/out-9_nodes_feather/gpt_ckpt_3500_feather_ar_0_95709__er_0_0.pt",
        data_dir="nanoGPT/data/9_nodes_feather",
    ),
    dict(
        # no name key → auto-extracted as "LLaMA-NetLSD"
        ckpt="nanoGPT/out-10_nodes_netlsd/llama_ckpt_6000_netlsd_ar_0_9436__er_0_026.pt",
        data_dir="nanoGPT/data/10_nodes_netlsd",
    ),
    dict(
        # no name key → auto-extracted as "LLaMA-GNN"
        ckpt="nanoGPT/out-10_nodes_gnn/llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt",
        data_dir="nanoGPT/data/10_nodes_gnn",
    ),
]

N_SAMPLES   = 5
MAX_TOKENS  = 150

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
# ------------------------
# MODEL NAME EXTRACTION
# ------------------------

def extract_model_name(ckpt_path: str) -> str:
    """
    Auto-extract a human-readable model name from a checkpoint filename.

    Splits the basename by '_' and reads:
      - element[0]  → architecture  (e.g. 'llama' → 'LLaMA', 'gpt' → 'GPT')
      - element[3]  → method        (e.g. 'gnn'   → 'GNN',  'netlsd' → 'NetLSD')

    Resulting name: "<Arch>-<METHOD>" (e.g. "LLaMA-GNN").

    If the filename does not follow the expected pattern the raw basename is
    returned as a fallback.
    """
    ARCH_ALIASES = {
        "llama": "LLaMA",
        "gpt":   "GPT",
    }

    basename = ckpt_path.split("/")[-1]          # keep only filename
    parts    = basename.split("_")

    try:
        arch   = ARCH_ALIASES.get(parts[0].lower(), parts[0].upper())
        method = parts[3].upper()
        return f"{arch}-{method}"
    except IndexError:
        return basename  # graceful fallback


def resolve_model_name(cfg: dict) -> str:
    """Return cfg['name'] if present, otherwise auto-extract from cfg['ckpt']."""
    return cfg.get("name") or extract_model_name(cfg["ckpt"])


# Attach resolved names back into configs for convenience
for cfg in MODEL_CONFIGS:
    cfg["resolved_name"] = resolve_model_name(cfg)

print("Resolved model names:")
for cfg in MODEL_CONFIGS:
    print(f"  {cfg['resolved_name']}")

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


def graph_name_to_num(graph_name: str) -> int:
    """
    Extract the trailing integer from a graph_name string.

    Examples:
        'graph_007'  → 7
        'g42'        → 42
        'graph_0042' → 42

    Falls back to 0 if no number is found (should not happen in practice).
    """
    match = re.search(r"(\d+)$", str(graph_name))
    return int(match.group(1)) if match else 0


def load_graphs_from_adapt(adapt_df):
    """Load unique graphs from ADAPT df (drop_duplicates on graph_name before passing)."""
    graphs, meta = [], []
    for _, row in adapt_df.iterrows():
        G = edgelist_to_nx(row["edgelist_list"], row["n_nodes"])
        graphs.append(G)
        meta.append({
            "graph_name": row["graph_name"],
            "graph_num":  graph_name_to_num(row["graph_name"]),
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

# %%
# ------------------------
# LOAD & AGGREGATE ADAPT
# ------------------------

adapt_df = get_combined_res_df(data_input_path, debug_limit=None)

# Attach graph_num to every ADAPT row
adapt_df["graph_num"] = adapt_df["graph_name"].apply(graph_name_to_num)

print(f"Total ADAPT rows      : {len(adapt_df)}")
print(f"Unique graphs         : {adapt_df['graph_num'].nunique()}")
print(f"Runs per graph (mean) : {adapt_df.groupby('graph_num').size().mean():.2f}")

# Aggregate ADAPT per graph_num
adapt_agg = adapt_df.groupby("graph_num").agg(
    graph_name        = ("graph_name",    "first"),   # keep for reference / merging
    adapt_ar_mean     = ("approx_ratio",  "mean"),
    adapt_ar_best     = ("approx_ratio",  "max"),
    adapt_ar_std      = ("approx_ratio",  "std"),
    adapt_layers_mean = ("n_layers",      "mean"),
    adapt_layers_best = ("n_layers",      "min"),     # min layers = most efficient run
    adapt_n_runs      = ("run",           "count"),
).reset_index()

adapt_agg["adapt_ar_std"] = adapt_agg["adapt_ar_std"].fillna(0)  # single-run → NaN → 0

print(f"\nAggregated ADAPT shape: {adapt_agg.shape}")

# %%
adapt_agg.head()

# %%
# Use only unique graphs for model generation
unique_adapt_df = adapt_df.drop_duplicates(subset="graph_num").reset_index(drop=True)
graphs_unique, meta_df = load_graphs_from_adapt(unique_adapt_df)

print(f"Graphs fed to model: {len(graphs_unique)}")

# %%
# ------------------------
# MODEL METRICS
# ------------------------

def compute_model_metrics(df):
    full_index = df.index  # original per-graph index — used to reindex all outputs

    df_expl = df.explode(["adapt_gpt_energies", "q_circuits"])

    # Average layers per graph
    layers = df_expl.groupby(level=0)["q_circuits"].apply(
        lambda xs: xs.apply(lambda x: x.count("new_layer_p")).mean()
    ).reindex(full_index)

    # Explode energy samples
    df_energy = df[["adapt_gpt_energies", "energy_gurobi"]].explode("adapt_gpt_energies")

    # Error rate per graph (sentinel value 999 = invalid circuit)
    error_rate = df_energy.groupby(level=0)["adapt_gpt_energies"].apply(
        lambda x: (x == 999).sum() / len(x)
    ).reindex(full_index, fill_value=0.0)

    # AR — valid samples only; graphs where ALL samples are invalid → NaN → fill 0
    df_corr = df_energy[df_energy["adapt_gpt_energies"] != 999].copy()
    df_corr["ar"] = df_corr["adapt_gpt_energies"] / df_corr["energy_gurobi"]
    ar = df_corr.groupby(level=0)["ar"].mean().reindex(full_index, fill_value=float("nan"))

    return ar, layers, error_rate

# %%
# ------------------------
# RUN ALL MODELS
# ------------------------

all_results = []

for cfg in MODEL_CONFIGS:
    model_name = cfg["resolved_name"]
    print(f"\nRunning {model_name} ...")

    model   = load_model(cfg)
    df_eval = run_model(model, graphs_unique)

    model_ar, model_layers, model_error_rate = compute_model_metrics(df_eval)

    res_df = pd.DataFrame({
        "graph_name"       : meta_df["graph_name"],
        "graph_num"        : meta_df["graph_num"],
        "model"            : model_name,
        "model_ar"         : model_ar.values,
        "model_layers"     : model_layers.values,
        "model_error_rate" : model_error_rate.values,
    })
    all_results.append(res_df)

model_results_df = pd.concat(all_results, ignore_index=True)

# %%
model_results_df.head()

# %%
# ------------------------
# MERGE
# ------------------------

# Merge on graph_num (primary key) — graph_name kept for reference
final_df = adapt_agg.merge(model_results_df, on="graph_num")

# Rename duplicate graph_name columns if present after merge
if "graph_name_x" in final_df.columns:
    final_df = final_df.rename(columns={"graph_name_x": "graph_name"}).drop(
        columns=["graph_name_y"], errors="ignore"
    )

# Diffs
final_df["ar_diff_vs_mean"] = final_df["model_ar"] - final_df["adapt_ar_mean"]
final_df["ar_diff_vs_best"] = final_df["model_ar"] - final_df["adapt_ar_best"]
final_df["layer_diff"]      = final_df["model_layers"] - final_df["adapt_layers_mean"]

# Sort for consistent plotting
final_df = final_df.sort_values("graph_num").reset_index(drop=True)

print(f"\nFinal df shape: {final_df.shape}")

# %%
final_df.head(10)

# %%
# ------------------------
# SUMMARY TABLE
# ------------------------

summary_df = final_df.groupby("model").agg(
    adapt_ar_mean    = ("adapt_ar_mean",    "mean"),
    adapt_ar_best    = ("adapt_ar_best",    "mean"),
    model_ar         = ("model_ar",         "mean"),
    adapt_layers     = ("adapt_layers_mean","mean"),
    model_error_rate = ("model_error_rate", "mean"),
    model_layers     = ("model_layers",     "mean"),
    ar_diff_vs_mean  = ("ar_diff_vs_mean",  "mean"),
    ar_diff_vs_best  = ("ar_diff_vs_best",  "mean"),
    n_graphs         = ("graph_num",        "count"),
).reset_index()

print(summary_df.to_string(index=False))

# %%
# ============================================================
# SCALABLE PLOTS — one figure per metric
# ============================================================

ADAPT_PALETTE = {"mean": "#4C72B0", "best": "#55A868", "std_fill": "#4C72B0"}


def _model_colors(models):
    """Assign a stable color per model name."""
    palette = ["#C44E52", "#DD8452", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def plot_ar_bar(summary_df):
    """Bar chart: average AR — ADAPT mean/best vs each model."""
    models = summary_df["model"].tolist()
    mc     = _model_colors(models)
    n      = len(models)
    x      = np.arange(n)
    w      = 0.22

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))

    ax.bar(x - w,  summary_df["adapt_ar_mean"], width=w, label="ADAPT (mean)",
           color=ADAPT_PALETTE["mean"])
    ax.bar(x,      summary_df["adapt_ar_best"], width=w, label="ADAPT (best)",
           color=ADAPT_PALETTE["best"])
    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.bar(x[i] + w, row["model_ar"], width=w, label=row["model"],
               color=mc[row["model"]])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Average Approximation Ratio: ADAPT vs Models")
    ax.set_ylim(
        min(summary_df["adapt_ar_mean"].min(), summary_df["model_ar"].min()) - 0.02,
        1.01,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_layers_bar(summary_df):
    """Bar chart: average layers — ADAPT mean vs each model."""
    models = summary_df["model"].tolist()
    mc     = _model_colors(models)
    n      = len(models)
    x      = np.arange(n)
    w      = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))

    ax.bar(x - w / 2, summary_df["adapt_layers"], width=w,
           label="ADAPT (mean)", color=ADAPT_PALETTE["mean"])
    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.bar(x[i] + w / 2, row["model_layers"], width=w,
               label=row["model"], color=mc[row["model"]])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Layers")
    ax.set_title("Average Number of QAOA Layers: ADAPT vs Models")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_error_rate_bar(summary_df):
    """Bar chart: model error rate per model."""
    models = summary_df["model"].tolist()
    mc     = _model_colors(models)
    n      = len(models)
    x      = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))

    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.bar(x[i], row["model_error_rate"], width=0.4,
               label=row["model"], color=mc[row["model"]])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Error Rate")
    ax.set_title("Model Circuit Error Rate (fraction of invalid outputs)")
    ax.set_ylim(0, min(1.0, summary_df["model_error_rate"].max() + 0.05))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_ar_vs_layers_scatter(final_df):
    """Scatter: AR vs Layers trade-off — ADAPT + all models."""
    models = final_df["model"].unique().tolist()
    mc     = _model_colors(models)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(final_df["adapt_layers_mean"], final_df["adapt_ar_mean"],
               label="ADAPT (mean)", alpha=0.55, color=ADAPT_PALETTE["mean"], zorder=2)
    ax.scatter(final_df["adapt_layers_best"], final_df["adapt_ar_best"],
               label="ADAPT (best)", alpha=0.55, color=ADAPT_PALETTE["best"],
               marker="^", zorder=2)

    for model, grp in final_df.groupby("model"):
        ax.scatter(grp["model_layers"], grp["model_ar"],
                   label=model, alpha=0.65, color=mc[model], marker="s", zorder=3)

    ax.set_xlabel("Layers")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("AR vs Layers Trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_per_graph_ar(final_df):
    """Line plot: per-graph AR for ADAPT + all models, x-axis = graph_num."""
    models      = final_df["model"].unique().tolist()
    mc          = _model_colors(models)
    graph_order = sorted(final_df["graph_num"].unique())

    # ADAPT baseline — use any model's rows (same ADAPT values for all)
    adapt_base = (
        final_df[final_df["model"] == models[0]]
        .set_index("graph_num")
        .reindex(graph_order)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(graph_order) // 5), 5))

    ax.plot(graph_order, adapt_base["adapt_ar_mean"], label="ADAPT (mean)",
            color=ADAPT_PALETTE["mean"], linewidth=1.5)
    ax.fill_between(
        graph_order,
        adapt_base["adapt_ar_mean"] - adapt_base["adapt_ar_std"],
        adapt_base["adapt_ar_mean"] + adapt_base["adapt_ar_std"],
        alpha=0.12, color=ADAPT_PALETTE["mean"],
    )
    ax.plot(graph_order, adapt_base["adapt_ar_best"], label="ADAPT (best)",
            color=ADAPT_PALETTE["best"], linestyle="--", linewidth=1.2, alpha=0.75)

    for model, grp in final_df.groupby("model"):
        grp_ordered = grp.set_index("graph_num").reindex(graph_order)
        ax.plot(graph_order, grp_ordered["model_ar"],
                label=model, color=mc[model], linewidth=1.5)

    ax.set_xlabel("Graph number")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Per-graph AR  (shaded band = ADAPT ± 1 std)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_ar_gap_hist(final_df):
    """Histogram: AR gap (model − ADAPT mean) per model."""
    models = final_df["model"].unique().tolist()
    mc     = _model_colors(models)

    fig, axes = plt.subplots(
        1, len(models),
        figsize=(max(6, 5 * len(models)), 4),
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        grp   = final_df[final_df["model"] == model]
        diffs = grp["ar_diff_vs_mean"]

        ax.hist(diffs, bins=20, color=mc[model], edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Parity")
        ax.axvline(diffs.mean(), color="gold", linewidth=1.5,
                   label=f"Mean {diffs.mean():+.4f}")
        ax.set_title(f"{model}\nAR Gap Distribution")
        ax.set_xlabel("Model AR − ADAPT mean AR")
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

    fig.suptitle("AR Gap: positive = model beats ADAPT mean", fontsize=11)
    fig.tight_layout()
    plt.show()

# %%
# ── Run all plots ──────────────────────────────────────────────────────────

plot_ar_bar(summary_df)
plot_layers_bar(summary_df)
plot_error_rate_bar(summary_df)
plot_ar_vs_layers_scatter(final_df)
plot_per_graph_ar(final_df)
plot_ar_gap_hist(final_df)

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