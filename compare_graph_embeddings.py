# %%
# ============================================================
# Graph Embedding Comparison Notebook
# NetLSD vs FEATHERG
# ============================================================

# ------------------------
# GLOBAL CONFIGURATION
# ------------------------

SEED = 42
NUM_GRAPHS = 6
NUM_NODES = 8
EDGE_PROB = 0.4

EMBEDDING_DIM = 500

# NetLSD
NETLSD_SCALE_STEPS = 500
NETLSD_APPROX = 50

# FEATHER
FEATHER_ORDER = 5
FEATHER_EVAL_POINTS = 25
FEATHER_THETA_MAX = 2.5
FEATHER_POOLING = "mean"

# Perturbation
PERTURB_RATE = 0.1

np.random.seed(SEED)


# %%
# ------------------------
# IMPORTS
# ------------------------

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from src.embedding.netlsd import NetLSD
from src.embedding.feather import FEATHERG


# %%
# ------------------------
# UTILITY FUNCTIONS
# ------------------------

def force_dim(embeddings, target_dim):
    """Force embeddings to a fixed dimensionality."""
    fixed = []
    for emb in embeddings:
        if emb.shape[0] > target_dim:
            emb = emb[:target_dim]
        elif emb.shape[0] < target_dim:
            emb = np.pad(emb, (0, target_dim - emb.shape[0]))
        fixed.append(emb)
    return np.array(fixed)


def perturb_graph(G, p=0.1):
    """Random edge perturbation."""
    Gp = G.copy()
    nodes = list(Gp.nodes())

    for u, v in list(Gp.edges()):
        if np.random.rand() < p:
            Gp.remove_edge(u, v)

    for _ in range(int(p * len(nodes))):
        u, v = np.random.choice(nodes, 2, replace=False)
        Gp.add_edge(u, v)

    return Gp


# %%
# ------------------------
# GRAPH GENERATION
# ------------------------

graphs = []

for _ in range(NUM_GRAPHS):
    G = nx.erdos_renyi_graph(NUM_NODES, EDGE_PROB)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.uniform(0.5, 2.0)
    graphs.append(G)

print(f"Generated {NUM_GRAPHS} graphs with {NUM_NODES} nodes each.")


# %%
# ------------------------
# NetLSD EMBEDDINGS
# ------------------------

netlsd = NetLSD(
    scale_steps=NETLSD_SCALE_STEPS,
    approximations=NETLSD_APPROX,
    seed=SEED,
)

netlsd.fit(graphs)
netlsd_emb = force_dim(netlsd.get_embedding(), EMBEDDING_DIM)

print("NetLSD shape:", netlsd_emb.shape)


# %%
# ------------------------
# FEATHERG EMBEDDINGS
# ------------------------

feather = FEATHERG(
    theta_max=FEATHER_THETA_MAX,
    eval_points=FEATHER_EVAL_POINTS,
    order=FEATHER_ORDER,
    pooling=FEATHER_POOLING,
)

feather.fit(graphs)
feather_emb = force_dim(feather.get_embedding(), EMBEDDING_DIM)

print("FEATHER shape:", feather_emb.shape)


# %%
# ------------------------
# PAIRWISE DISTANCE HEATMAPS
# ------------------------

methods = {
    "NetLSD": netlsd_emb,
    "FEATHER": feather_emb,
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, emb) in zip(axes, methods.items()):
    dist = squareform(pdist(emb, metric="cosine"))
    im = ax.imshow(dist)
    ax.set_title(name)
    plt.colorbar(im, ax=ax)

plt.suptitle("Cosine Distance Between Graph Embeddings")
plt.tight_layout()
plt.show()


# %%
# ------------------------
# PCA VISUALIZATION (2D)
# ------------------------

plt.figure(figsize=(6, 5))

for name, emb in methods.items():
    emb_2d = PCA(n_components=2).fit_transform(emb)
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], label=name, s=70)

plt.legend()
plt.title("PCA Projection of Graph Embeddings")
plt.grid(True)
plt.show()


# %%
# ------------------------
# EMBEDDING VARIANCE ANALYSIS
# ------------------------

for name, emb in methods.items():
    var = np.mean(np.std(emb, axis=0))
    print(f"{name} mean per-dimension std: {var:.6f}")


# %%
# ------------------------
# STABILITY UNDER PERTURBATION
# ------------------------

perturbed_graphs = [perturb_graph(G, PERTURB_RATE) for G in graphs]

# NetLSD
netlsd_p = NetLSD(
    scale_steps=NETLSD_SCALE_STEPS,
    approximations=NETLSD_APPROX,
    seed=SEED,
)
netlsd_p.fit(perturbed_graphs)

netlsd_shift = np.linalg.norm(
    netlsd_emb - force_dim(netlsd_p.get_embedding(), EMBEDDING_DIM),
    axis=1
)

# FEATHER
feather_p = FEATHERG(
    theta_max=FEATHER_THETA_MAX,
    eval_points=FEATHER_EVAL_POINTS,
    order=FEATHER_ORDER,
    pooling=FEATHER_POOLING,
)
feather_p.fit(perturbed_graphs)

feather_shift = np.linalg.norm(
    feather_emb - force_dim(feather_p.get_embedding(), EMBEDDING_DIM),
    axis=1
)


# %%
# ------------------------
# STABILITY PLOT
# ------------------------

plt.figure(figsize=(6, 4))
plt.plot(netlsd_shift, marker="o", label="NetLSD")
plt.plot(feather_shift, marker="^", label="FEATHER")
plt.xlabel("Graph index")
plt.ylabel("Embedding shift (L2)")
plt.title("Stability Under Edge Perturbation")
plt.legend()
plt.grid(True)
plt.show()
