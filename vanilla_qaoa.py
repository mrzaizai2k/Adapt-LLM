# %% [markdown]
# # QAOA Vanilla - MaxCut on Weighted ER Graphs
# 
# This notebook:
# - Generates 5 weighted Erdős–Rényi graphs
# - Runs vanilla QAOA on each graph
# - Collects expectation, variance, and optimal parameters

# %%
# !pip install qaoa

# %%
import networkx as nx
from qaoa import QAOA, problems, mixers, initialstates

from src.utils import generate_er_graphs, edgelist_to_nx

import pandas as pd

# %%
# Configuration
n_graphs = 5
n_nodes = 10
depth = 10   # QAOA layers (p)

# %%
# Generate graphs (already weighted from your function)
graphs = generate_er_graphs(n_graphs=n_graphs, n_nodes=n_nodes)

print(f"Generated {len(graphs)} graphs")

# %%
# Inspect one graph
name, G = list(graphs.items())[0]
print("Example graph:", name)
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# %%
# Storage for results
results = []

# %%
# Run QAOA on each graph
for graph_name, G in graphs.items():

    print(f"\nRunning QAOA on {graph_name}")

    # Initialize QAOA
    qaoa = QAOA(
        problem=problems.MaxCut(G),
        mixer=mixers.X(),
        initialstate=initialstates.Plus()
    )

    # --- Step 1: Landscape (p=1 only, optional but useful)
    qaoa.sample_cost_landscape()

    # --- Step 2: Optimization
    qaoa.optimize(depth=depth)

    # --- Step 3: Extract results
    exp_val = qaoa.get_Exp(depth=depth)
    var_val = qaoa.get_Var(depth=depth)

    gamma = qaoa.get_gamma(depth=depth)
    beta = qaoa.get_beta(depth=depth)

    # Histogram (top states insight)
    hist = qaoa.hist

    # Store results
    results.append({
        "graph_name": graph_name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "depth_p": depth,
        "expectation": exp_val,
        "variance": var_val,
        "gamma": gamma,
        "beta": beta,
        "hist": hist
    })

    print("Expectation:", exp_val)
    print("Variance:", var_val)

# %%
# Convert to DataFrame
df = pd.DataFrame(results)

df

# %%
# Sort by best expectation (MaxCut → usually minimize energy)
df_sorted = df.sort_values(by="expectation")

df_sorted

# %%
# Inspect one result in detail
row = df_sorted.iloc[0]

print("Best graph:", row["graph_name"])
print("Gamma:", row["gamma"])
print("Beta:", row["beta"])
print("Expectation:", row["expectation"])
print("Variance:", row["variance"])

# %%
# Optional: analyze parameter patterns
import numpy as np

all_gamma = np.array(df["gamma"].tolist())
all_beta = np.array(df["beta"].tolist())

print("Mean gamma:", all_gamma.mean(axis=0))
print("Mean beta:", all_beta.mean(axis=0))

# %%
# Done
print("QAOA experiment complete ✅")