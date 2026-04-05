# %% [markdown]
# # QAOA Vanilla - MaxCut on Weighted ER Graphs
# 
# This notebook:
# - Generates 5 weighted Erdős–Rényi graphs
# - Runs vanilla QAOA on each graph
# - Collects expectation, variance, and optimal parameters

# %%
from qaoa import QAOA, problems, mixers, initialstates
from qiskit_algorithms.optimizers import L_BFGS_B
import time
from src.utils import generate_er_graphs, maxcut_bruteforce

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %%
# Configuration
n_graphs = 1
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

def graph_to_edgelist(G):
    """Convert NetworkX graph → [[u, v, w], ...]"""
    return [[u, v, d.get("weight", 1.0)] for u, v, d in G.edges(data=True)]


def run_qaoa_on_graph(graph_name, G, depth):
    print(f"\nRunning QAOA on {graph_name}")
    start = time.time()

    # --- Initialize QAOA
    qaoa = QAOA(
        problem=problems.MaxCut(G),
        mixer=mixers.X(),
        initialstate=initialstates.Plus(),
        interpolate=True,
        optimizer=[L_BFGS_B, {
            "maxiter": 50,
            "ftol": 1e-9
        }]
    )

    # --- Step 1: Landscape (optional, only meaningful for p=1)
    if depth == 1:
        qaoa.sample_cost_landscape()

    # --- Step 2: Optimization
    qaoa.optimize(depth=depth)

    # --- Step 3: Metrics
    exp_val = qaoa.get_Exp(depth=depth)
    var_val = qaoa.get_Var(depth=depth)

    gamma = qaoa.get_gamma(depth=depth)
    beta = qaoa.get_beta(depth=depth)

    # --- Step 4: Optimal (bruteforce)
    energy_opt, best_state = maxcut_bruteforce(G)

    # --- Step 5: Approximation ratio
    approx_ratio = exp_val / energy_opt if energy_opt != 0 else None

    # --- Step 6: Edge list
    edgelist_list = graph_to_edgelist(G)

    end = time.time()

    result = {
        "graph_name": graph_name,
        "n_nodes": G.number_of_nodes(),
        "edgelist_list_len": G.number_of_edges(),
        "n_layers": depth,
        "expected_energy": exp_val,
        "variance": var_val,
        "γ_coeff": gamma,
        "β_coeff": beta,
        "approx_ratio": approx_ratio,
        "energy_mqlib": energy_opt,
        "edgelist_list": edgelist_list,
        "took_time": round(end - start, 3),
        "method": "vanilla_qaoa",
        "optimizer": "BFGS"
    }

    print(f"Expectation: {exp_val}")
    print(f"Variance: {var_val}")

    return result

# %%
results = []

for graph_name, G in graphs.items():
    result = run_qaoa_on_graph(graph_name, G, depth)
    results.append(result)

# %%
# for x in dir(qaoa):
#     if not x.startswith("_"):    
#         print(x)

# %%
# Convert to DataFrame
df = pd.DataFrame(results)

df

# %%
# Sort by best expectation (MaxCut → usually minimize energy)
df_sorted = df.sort_values(by="expected_energy")

df_sorted

# %%
# Done
print("QAOA experiment complete ✅")
