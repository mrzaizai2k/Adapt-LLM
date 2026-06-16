import os
import streamlit as st
import pandas as pd
import networkx as nx
import random
from src.utils import plot_maxcut_compare_fig

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QAOA-LLM Demo",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light-theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre, .stCode        { font-family: 'JetBrains Mono', monospace !important; }

/* ── App background ── */
.stApp { background: #ffffff !important; color: #1a1a2e; }
.main .block-container { background: #ffffff !important; }

header[data-testid="stHeader"] {
    background: #ffffff !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8f9fc !important;
    border-right: 1px solid #e2e8f0;
}

/* Sidebar H2 */
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #1e40af;
    font-weight: 800;
    font-size: 2rem !important;   /* increase size */
}

/* Sidebar H3 */
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e3a8a;
    font-size: 1.5rem !important; /* increase size */
}
            
            
/* ── Force all input/widget backgrounds to white ── */
input, textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] div,
[data-baseweb="base-input"],
.stTextInput > div > div,
.stNumberInput > div > div > input,
[data-testid="stNumberInputField"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* Select/dropdown background */
[data-baseweb="select"] [data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* Slider track & thumb */
[data-testid="stSlider"] { color: #1e293b !important; }

/* Number input spin buttons */
.stNumberInput button { background: #f1f5f9 !important; color: #1e40af !important; border-color: #cbd5e1 !important; }

/* ── Dataframe / table ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] iframe,
.stDataFrame { background: #ffffff !important; color: #1e293b !important; }
[data-testid="stDataFrame"] * { color: #1e293b !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] { background: #f0f5ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: #1e40af !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; }
[data-testid="stMetricValue"] { color: #1e293b !important; font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; }

/* ── Headers ── */
h1 { color: #0f172a; font-weight: 800; letter-spacing: -1px; }
h2 { color: #1e40af; font-weight: 700; border-bottom: 2px solid #bfdbfe; padding-bottom: 6px; }
h3 { color: #1e3a8a; font-weight: 600; }
h4 { color: #334155; }

/* ── Buttons ── */
.stButton > button {
    background: #ffffff; color: #1e40af;
    border: 1.5px solid #3b82f6; border-radius: 8px;
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
    letter-spacing: 0.04em; transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #eff6ff; border-color: #2563eb; color: #1d4ed8;
    transform: translateY(-1px); box-shadow: 0 4px 14px rgba(59,130,246,0.2);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);
    color: #ffffff; border-color: #2563eb; font-size: 1rem;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#1d4ed8 0%,#1e40af 100%);
    box-shadow: 0 4px 16px rgba(37,99,235,0.35);
}

/* ── Alerts ── */
.stAlert { border-radius: 8px; background: #f8fafc !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Input labels ── */
.stSlider label, .stNumberInput label, .stSelectbox label,
.stTextInput label { color: #374151 !important; font-size: 2.5rem; font-weight: 600; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #f1f5f9; border-radius: 10px 10px 0 0; border-bottom: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { color: #475569; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { color: #1e40af !important; background: #dbeafe !important; }

/* ── Expander ── */
[data-testid="stExpander"] { background: #f8fafc !important; border: 1px solid #e2e8f0; border-radius: 8px; }

/* ── Hero ── */
.hero-thesis {
    font-size: 2.2rem !important;
    font-weight: 900;
    color: #1e40af;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0 0 16px 0;
    text-align: center;
}

.hero-supervisor {
    color: #475569;
    font-size: 1rem;
    font-style: italic;
    margin: 0 0 6px 0;
    text-align: center;
}
            
.hero-banner {
    background: linear-gradient(135deg,#eff6ff 0%,#f0f9ff 50%,#f5f3ff 100%);
    border: 1.5px solid #bfdbfe;
    border-radius: 16px;
    padding: 60px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;

    /* center content */
    text-align: center;
}

.hero-banner::before {
    content:'';
    position:absolute;
    top:-40%;
    right:-5%;
    width:300px;
    height:300px;
    background:radial-gradient(circle,rgba(59,130,246,.06) 0%,transparent 70%);
    pointer-events:none;
}

.hero-title {
    font-size: 30px !important;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -1.5px;
    line-height: 1.3;
    margin: 0 0 20px 0;
    text-align: center;
}

.hero-author {
    color: #1e40af;
    font-size: 30px !important;        /* bigger author */
    font-weight: 700;
    margin: 0;
    text-align: center;
}

.hero-sub {
    color: #475569;
    font-size: 1rem;
    font-family:'JetBrains Mono',monospace;
    margin-top:12px;
    text-align:center;
}
            
/* ── Status badges ── */
.status-ok  { display:inline-block; background:rgba(16,185,129,.1);  border:1px solid #6ee7b7; border-radius:6px; padding:2px 10px; font-size:1rem; font-family:'JetBrains Mono',monospace; color:#065f46; }
.status-err { display:inline-block; background:rgba(239,68,68,.1);   border:1px solid #fca5a5; border-radius:6px; padding:2px 10px; font-size:1rem; font-family:'JetBrains Mono',monospace; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-thesis">MASTER'S THESIS</p>
    <p class="hero-title">
        Investigating a Hybrid LLM-GNN Model to Enhance the Efficiency<br>
        of ADAPT-QAOA for Quantum Circuit Optimization
    </p>
    <p class="hero-supervisor">Supervisor: Assoc. Prof. Dr. Thoại Nam</p>
    <p class="hero-author">
        Mai Chi Bao &nbsp;·&nbsp; 2370691
    </p>
</div>
""", unsafe_allow_html=True)

# ── File-browser helper ───────────────────────────────────────────────────────
def file_browser(label, default_value, file_type, key, help_text=""):
    # _value is the source of truth (never bound to a widget key)
    if f"{key}_value" not in st.session_state:
        st.session_state[f"{key}_value"] = default_value
    if f"{key}_browsing" not in st.session_state:
        st.session_state[f"{key}_browsing"] = False

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        # No `key=` on the text input — use `value=` from our own state only
        typed = st.text_input(
            label,
            value=st.session_state[f"{key}_value"],
            help=help_text,
        )
        st.session_state[f"{key}_value"] = typed  # reflect manual edits

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("📂 Browse", key=f"{key}_browse_btn"):
            st.session_state[f"{key}_browsing"] = True

    if st.session_state[f"{key}_browsing"]:
        if file_type == "file":
            start_dir = "nanoGPT" if os.path.isdir("nanoGPT") else (
                os.path.dirname(st.session_state[f"{key}_value"]) or "."
            )
        else:
            start_dir = os.path.dirname(st.session_state[f"{key}_value"]) or "."

        if not os.path.isdir(start_dir):
            start_dir = "."

        try:
            if file_type == "file":
                entries = []
                for root, dirs, files in os.walk(start_dir):
                    dirs.sort()
                    for f in sorted(files):
                        if f.endswith(".pt"):
                            entries.append(os.path.join(root, f))
            else:
                entries = sorted([
                    os.path.join(start_dir, f)
                    for f in os.listdir(start_dir)
                    if os.path.isdir(os.path.join(start_dir, f))
                ])
        except Exception:
            entries = []

        if entries:
            chosen = st.selectbox(
                f"{'Checkpoint files (.pt)' if file_type == 'file' else 'Directories'} under `{start_dir}`",
                options=entries,
                key=f"{key}_picker",
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✓ Use this", key=f"{key}_use"):
                    st.session_state[f"{key}_value"] = chosen  # update source of truth only
                    st.session_state[f"{key}_browsing"] = False
                    st.rerun()
            with col_cancel:
                if st.button("✗ Cancel", key=f"{key}_cancel"):
                    st.session_state[f"{key}_browsing"] = False
                    st.rerun()
        else:
            label_type = ".pt files" if file_type == "file" else "directories"
            st.caption(f"No {label_type} found under `{start_dir}`")
            if st.button("Close", key=f"{key}_close"):
                st.session_state[f"{key}_browsing"] = False
                st.rerun()

    return st.session_state[f"{key}_value"]

def infer_from_ckpt_path(ckpt_path: str) -> dict:
    """Try to extract n_nodes and embedding_method from checkpoint path."""
    try:
        # e.g. out-10_nodes_feather
        folder = os.path.basename(os.path.dirname(ckpt_path))
        import re
        m = re.search(r'out-(\d+)_nodes_(\w+)', folder)
        if m:
            n_nodes = int(m.group(1))
            emb     = m.group(2)
            return {
                "n_nodes": n_nodes,
                "data_dir": f"nanoGPT/data/{n_nodes}_nodes_{emb}",
            }
    except Exception:
        pass
    return {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 📂 Model")
    model_ckpt = file_browser(
        label="Checkpoint path (.pt)",
        default_value="nanoGPT/out-10_nodes_feather/gpt_ckpt_3500_feather_ar_0_96305__er_0_0.pt",
        file_type="file",
        key="model_ckpt",
        help_text="Path to the .pt model checkpoint file",
    )
    # ── Auto-fill from checkpoint path ───────────────────────────────────────
    _last = st.session_state.get("_last_ckpt_path")
    if model_ckpt != _last:
        st.session_state["_last_ckpt_path"] = model_ckpt
        _inferred = infer_from_ckpt_path(model_ckpt)
        if _inferred:
            st.session_state["data_dir_value"] = _inferred["data_dir"]
            st.session_state["_inferred_n_nodes"] = _inferred["n_nodes"]

    data_dir = file_browser(
        label="Data directory (meta.pkl)",
        default_value="nanoGPT/data/10_nodes_feather",
        file_type="dir",
        key="data_dir",
        help_text="Directory containing meta.pkl",
    )

    st.markdown("### 🔗 Graph")
    n_graphs = st.slider("Number of graphs", 1, 20, 5)
    # n_nodes  = st.number_input("Nodes per graph", min_value=2, max_value=20, value=10)

    n_nodes = st.number_input(
        "Nodes per graph",
        min_value=2, max_value=20,
        value=st.session_state.get("_inferred_n_nodes", 10),
    )

    st.markdown("### 🤖 Generation")
    n_samples_per_batch = st.slider("Samples per batch", 1, 100, 50)
    num_samples         = st.slider("Num samples (per graph)", 1, 20, 5)
    max_new_tokens      = st.number_input("Max new tokens", min_value=10, max_value=500, value=150)
    temperature         = st.slider("Temperature", 0.01, 2.0, 0.1, step=0.01)
    top_k               = st.number_input("Top-k", min_value=1, max_value=1000, value=200)

    st.markdown("---")
    if st.button("▶ Run Inference", type="primary", use_container_width=True):
        # Snapshot all current values at click time
        st.session_state["run_btn"] = True
        st.session_state["run_config"] = {
            "model_ckpt":         model_ckpt,
            "data_dir":           data_dir,
            "n_graphs":           int(n_graphs),
            "n_nodes":            int(n_nodes),
            "n_samples_per_batch": int(n_samples_per_batch),
            "num_samples":        int(num_samples),
            "max_new_tokens":     int(max_new_tokens),
            "temperature":        float(temperature),
            "top_k":              int(top_k),
        }
        for k in ("qaoa_gpt", "graphs", "eval_df", "run_key",
                  "compute_metrics", "maxcut_bruteforce", "edgelist_to_nx"):
            st.session_state.pop(k, None)

    run_btn = st.session_state.get("run_btn", False)




def safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            if out[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                out[col] = out[col].apply(lambda x: str(x) if x is not None else "")
    return out


# ── Main area ─────────────────────────────────────────────────────────────────

if not run_btn:
    st.info("👈  Configure model & graph parameters in the sidebar, then press **▶ Run Inference**.", icon="ℹ️")
    st.markdown("""
        #### What this demo does
        1. Loads the **QAOA-GPT** checkpoint you specify.
        2. Generates *n* random Erdős–Rényi graphs.
        3. Uses the language model to **predict QAOA circuits** for each graph.
        4. Evaluates each circuit's MaxCut energy via Julia.
        5. Displays the **metrics table** and a **side-by-side MaxCut figure**.
    """)

else:
    # Always use the snapshotted config, not live widget values
    cfg = st.session_state["run_config"]
    model_ckpt          = cfg["model_ckpt"]
    data_dir            = cfg["data_dir"]
    n_graphs            = cfg["n_graphs"]
    n_nodes             = cfg["n_nodes"]
    n_samples_per_batch = cfg["n_samples_per_batch"]
    num_samples         = cfg["num_samples"]
    max_new_tokens      = cfg["max_new_tokens"]
    temperature         = cfg["temperature"]
    top_k               = cfg["top_k"]

    run_key = (model_ckpt, data_dir, n_graphs, n_nodes,
               n_samples_per_batch, num_samples,
               max_new_tokens, temperature, top_k)

    if st.session_state.get("run_key") != run_key:
        # Clear stale cache
        for k in ("qaoa_gpt", "graphs", "eval_df", "run_key",
                  "maxcut_bruteforce", "edgelist_to_nx"):
            st.session_state.pop(k, None)

    # ── 1 · Load model ────────────────────────────────────────────────────────
    st.markdown("## 1 · Model Loading")

    if "qaoa_gpt" not in st.session_state:
        with st.spinner("Importing modules & loading checkpoint …"):
            try:
                from src.model_interface import QAOA_GPT
                from src.adapt_utils import compute_metrics
                from src.utils import maxcut_bruteforce, edgelist_to_nx

                qaoa_gpt = QAOA_GPT(model_ckpt=model_ckpt, data_dir=data_dir)
                st.session_state["qaoa_gpt"]           = qaoa_gpt
                st.session_state["compute_metrics"]    = compute_metrics
                st.session_state["maxcut_bruteforce"]  = maxcut_bruteforce
                st.session_state["edgelist_to_nx"]     = edgelist_to_nx
            except Exception as e:
                st.markdown('<span class="status-err">✗ Load failed</span>', unsafe_allow_html=True)
                st.error(f"**Error loading model:** {e}")
                st.stop()
    else:
        qaoa_gpt          = st.session_state["qaoa_gpt"]
        compute_metrics   = st.session_state["compute_metrics"]
        maxcut_bruteforce = st.session_state["maxcut_bruteforce"]
        edgelist_to_nx    = st.session_state["edgelist_to_nx"]

    st.markdown('<span class="status-ok">✓ Model loaded</span>', unsafe_allow_html=True)

    # ── 2 · Graph generation ──────────────────────────────────────────────────
    st.markdown("## 2 · Graph Generation")

    def add_weights(G):
        for u, v in G.edges():
            w = round(random.uniform(0, 1), 2)
            while w == 0:
                w = round(random.uniform(0, 1), 2)
            G[u][v]["weight"] = w
        return G

    if "graphs" not in st.session_state:
        graphs = {}
        with st.spinner(f"Generating {n_graphs} Erdős–Rényi graphs …"):
            for i in range(int(n_graphs)):
                p = random.randrange(6, 9) / 10
                g = nx.erdos_renyi_graph(n=int(n_nodes), p=p)
                graphs[f"er_graph_{i}"] = add_weights(g)
        st.session_state["graphs"] = graphs
    else:
        graphs = st.session_state["graphs"]

    st.markdown(f'<span class="status-ok">✓ Generated {len(graphs)} graphs with {n_nodes} nodes each.</span>', unsafe_allow_html=True)
    edge_summary = pd.DataFrame(
        [{"graph": k, "nodes": v.number_of_nodes(), "edges": v.number_of_edges()}
         for k, v in graphs.items()]
    )
    with st.expander("Graph edge summary", expanded=False):
        st.dataframe(edge_summary, use_container_width=True, hide_index=True)

    # ── 3 · Circuit generation ────────────────────────────────────────────────
    st.markdown("## 3 · Circuit Generation")

    if "eval_df" not in st.session_state:
        with st.spinner("Running QAOA-GPT inference …"):
            try:
                emb = qaoa_gpt.embedding_method
                st.caption(f"Embedding method: `{emb}`")
                circ_df = qaoa_gpt.generate_circ_from_nx(
                    graphs,
                    n_samples_per_batch=int(n_samples_per_batch),
                    num_samples=int(num_samples),
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    allow_larger_graphs=True,  # Disable graph size check to allow all generated graphs
                )
            except Exception as e:
                st.error(f"**Circuit generation failed:** {e}")
                st.stop()

        # ── 4 · Circuit evaluation ────────────────────────────────────────────
        st.markdown("## 4 · Circuit Evaluation")
        with st.spinner("Evaluating circuits via Julia …"):
            try:
                eval_df = qaoa_gpt.eval_circ_df_jl(circ_df)
                st.session_state["eval_df"] = eval_df
                st.session_state["run_key"] = run_key
            except Exception as e:
                st.error(f"**Evaluation failed:** {e}")
                st.stop()
    else:
        eval_df = st.session_state["eval_df"]

    st.markdown('<span class="status-ok">✓ Circuits generated & evaluated</span>', unsafe_allow_html=True)

    # ── 5 · Results ───────────────────────────────────────────────────────────
    st.markdown("## 5 · Results")

    ar, err, layers = compute_metrics(eval_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Approximation Ratio", f"{ar:.4f}",  help="Ratio of predicted energy to optimal (1 = perfect)")
    c2.metric("Avg Error Rate",          f"{err:.4f}", help="Fraction of circuits producing invalid / error states")
    c3.metric("Avg Circuit Layers",      f"{layers:.2f}", help="Mean number of QAOA layers across all samples")

    st.markdown("### Circuit evaluation table")
    eval_expl_df = eval_df.explode(["adapt_gpt_energies", "q_circuits"])

    tab1, tab2 = st.tabs(["📋 All rows", "⚠️ Invalid circuits only"])
    with tab1:
        st.dataframe(safe_df_for_display(eval_expl_df), use_container_width=True)
    with tab2:
        # Invalid = anything where adapt_gpt_energies is NOT > 0
        invalid_df = eval_expl_df[(eval_expl_df["adapt_gpt_energies"] > 0)]
        if len(invalid_df):
            st.dataframe(safe_df_for_display(invalid_df), use_container_width=True)
        else:
            st.success("No invalid circuits found 🎉")

    # ── 6 · MaxCut Visualisation ──────────────────────────────────────────────
    st.markdown("## 6 · MaxCut Visualisation")

    # Use authoritative df that always has adapt_gpt_bitstrings
    vis_df = qaoa_gpt.qaoa_gpt_circ_eval_df.copy()

    bs_col_candidates = [c for c in vis_df.columns if "bitstring" in c.lower()]
    if not bs_col_candidates:
        st.warning(f"No bitstring column found. Available: `{list(vis_df.columns)}`")
        st.stop()
    bs_col = bs_col_candidates[0]

    graph_names    = list(graphs.keys())
    selected_graph = st.selectbox("Select graph to visualise", graph_names, key="vis_graph_select")

    # Match row by graph_prefix column, then fall back to index scan
    row_idx = 0
    if "graph_prefix" in vis_df.columns:
        matches = vis_df.index[vis_df["graph_prefix"] == selected_graph].tolist()
        if matches:
            row_idx = vis_df.index.get_loc(matches[0])
    else:
        for i, idx_val in enumerate(vis_df.index):
            if selected_graph in str(idx_val):
                row_idx = i
                break

    row        = vis_df.iloc[row_idx]
    graph_raw  = row["graph"]
    bitstrings = row[bs_col]

    if bitstrings is not None and len(bitstrings) > 0:
        if len(bitstrings) > 1:
            bs_idx = st.slider("Sample index", 0, len(bitstrings) - 1, 0, key="vis_bs_slider")
        else:
            bs_idx = 0
            st.caption("Only 1 sample available.")
        bitstring = bitstrings[bs_idx]

        try:
            buf = plot_maxcut_compare_fig(graph_raw, bitstring, edgelist_to_nx, maxcut_bruteforce)
            st.image(buf, use_column_width=True)
        except Exception as e:
            st.error(f"**Figure rendering failed:** {e}")
            st.exception(e)
    else:
        st.warning("No valid bitstrings found for this graph.")