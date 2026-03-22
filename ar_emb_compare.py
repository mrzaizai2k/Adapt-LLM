# %% [markdown]
# QAOA-GPT embedding comparison

# %%
import random
import numpy as np
import torch

import pandas as pd
import matplotlib.pyplot as plt

from model_interface import QAOA_GPT

from src.utils import (
    generate_er_graphs,
    compute_ar,
    compute_layers,
)

pd.set_option("display.max_columns", None)


# %% ------------------------
# SEED
# ------------------------

SEED = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% ------------------------
# CONFIG
# ------------------------

N_GRAPHS = 5
N_NODES = 10

N_SAMPLES = 5
MAX_TOKENS = 150


EMBEDDINGS = ["feather", "netlsd", "gnn"]
MODELS = ["gpt", "llama"]


EMBEDDING_NAMES = dict(
    feather="Feather",
    netlsd="NetLSD",
    gnn="GNN",
)

MODEL_NAMES = dict(
    gpt="nanoGPT",
    llama="LLaMA",
)


# %% ------------------------
# PATH CONFIG
# ------------------------

PATHS = dict(

    gpt=dict(
        feather=(
            "nanoGPT/out-10_nodes_feather/gpt_ckpt_5000_feather_ar_0_96346__er_0_0.pt",
            "nanoGPT/data/10_nodes_feather",
        ),
        netlsd=(
            "nanoGPT/out-10_nodes_netlsd/gpt_ckpt_6500_netlsd_ar_0_95657__er_0_0.pt",
            "nanoGPT/data/10_nodes_netlsd",
        ),
        gnn=(
            "nanoGPT/out-10_nodes_gnn/gpt_ckpt_1500_gnn_ar_0_92952__er_0_0.pt",
            "nanoGPT/data/10_nodes_gnn",
        ),
    ),

    llama=dict(
        feather=(
            "nanoGPT/out-10_nodes_feather/llama_ckpt_7500_feather_ar_0_89038__er_0_0.pt",
            "nanoGPT/data/10_nodes_feather",
        ),
        netlsd=(
            "nanoGPT/out-10_nodes_netlsd/llama_ckpt_5500_netlsd_ar_0_94951__er_0_0.pt",
            "nanoGPT/data/10_nodes_netlsd",
        ),
        gnn=(
            "nanoGPT/out-10_nodes_gnn/llama_ckpt_6250_gnn.pt",
            "nanoGPT/data/10_nodes_gnn",
        ),
    ),

)


# %% ------------------------
# ERROR RATE
# ------------------------

def compute_error_rate(df):

    df = df.explode(
        ["adapt_gpt_energies", "q_circuits"]
    )

    total = len(df)

    errors = (df["adapt_gpt_energies"] == 999).sum()

    return errors / total


# %% ------------------------
# RUN ONE
# ------------------------

def run_one(model, embedding):

    print(f"Running {model} + {embedding}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ckpt, data = PATHS[model][embedding]

    qaoa = QAOA_GPT(
        model_ckpt=ckpt,
        data_dir=data,
        temp_folder="temp_data",
    )

    graphs = generate_er_graphs(
        N_GRAPHS,
        N_NODES,
    )

    df = qaoa.generate_circ_from_nx(
        graphs,
        num_samples=N_SAMPLES,
        max_new_tokens=MAX_TOKENS,
        temperature=0.1,
        top_k=200,
    )

    df_eval = qaoa.eval_circ_df_jl(df)

    ar = compute_ar(df_eval)
    layers = compute_layers(df_eval)
    err = compute_error_rate(df_eval)

    return dict(
        model=model,
        embedding=embedding,
        AR=ar,
        layers=layers,
        error=err,
    )


# %% ------------------------
# RUN ALL
# ------------------------

results = []

for model in MODELS:
    for emb in EMBEDDINGS:

        res = run_one(model, emb)

        results.append(res)

result_df = pd.DataFrame(results)

result_df


# %% ------------------------
# FORMAT TABLE
# ------------------------

result_df["model"] = result_df["model"].map(MODEL_NAMES)
result_df["embedding"] = result_df["embedding"].map(EMBEDDING_NAMES)

result_df


# %% ------------------------
# PLOT AR (GROUPED)
# ------------------------

def plot_ar(df):

    fig, ax = plt.subplots(figsize=(7, 4))

    width = 0.25

    x = np.arange(len(EMBEDDINGS))

    for i, model in enumerate(df["model"].unique()):

        sub = df[df["model"] == model]

        ax.bar(
            x + i * width,
            sub["AR"],
            width,
            label=model,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df["embedding"].unique())

    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Embedding performance")

    ax.legend()
    ax.grid(True, axis="y")

    plt.show()


plot_ar(result_df)


# %% ------------------------
# PLOT ERROR
# ------------------------

def plot_error(df):

    fig, ax = plt.subplots(figsize=(7, 4))

    width = 0.25
    x = np.arange(len(EMBEDDINGS))

    for i, model in enumerate(df["model"].unique()):

        sub = df[df["model"] == model]

        ax.bar(
            x + i * width,
            sub["error"],
            width,
            label=model,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df["embedding"].unique())

    ax.set_ylabel("Error rate")
    ax.set_title("Circuit error rate")

    ax.legend()
    ax.grid(True, axis="y")

    plt.show()


plot_error(result_df)


# %% ------------------------
# PLOT LAYERS
# ------------------------

def plot_layers(df):

    fig, ax = plt.subplots(figsize=(7, 4))

    width = 0.25
    x = np.arange(len(EMBEDDINGS))

    for i, model in enumerate(df["model"].unique()):

        sub = df[df["model"] == model]

        ax.bar(
            x + i * width,
            sub["layers"],
            width,
            label=model,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df["embedding"].unique())

    ax.set_ylabel("Layers")
    ax.set_title("Average circuit depth")

    ax.legend()
    ax.grid(True, axis="y")

    plt.show()


plot_layers(result_df)


# %% ------------------------
# REPORT
# ------------------------

def report(df):

    print("====== REPORT ======")

    for model in df["model"].unique():

        print("\nMODEL:", model)

        sub = df[df["model"] == model]

        best = sub.sort_values(
            "AR",
            ascending=False,
        ).iloc[0]

        print("Best embedding:", best["embedding"])
        print("Best AR:", best["AR"])

        print("\nTable:")
        print(sub)


report(result_df)